# -*- coding: utf-8 -*-
"""巨潮资讯网（cninfo.com.cn）公开接口封装。

使用巨潮前端页面自身调用的三个公开接口，无需登录、无 token：
  1. 全市场股票列表   GET  /new/data/szse_stock.json         -> code / orgId / 中文简称
  2. 关键词搜索       POST /new/information/topSearch/query  -> 取 orgId 的兜底方案
  3. 历史公告查询     POST /new/hisAnnouncement/query        -> 标题 / 披露时间 / PDF 相对路径
  PDF 实体            GET  http://static.cninfo.com.cn/{adjunctUrl}

设计要点：
  * 所有请求带 Referer + UA，否则会被 403；
  * 统一节流（config.REQUEST_INTERVAL）+ 重试（config.MAX_RETRY）；
  * 查询用 category 精确过滤半年报，若接口返回空则退化为本地标题过滤。
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

import requests

import config
from .common import get_logger

logger = get_logger("cninfo")

QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
STOCKLIST_URL = "http://www.cninfo.com.cn/new/data/szse_stock.json"
TOPSEARCH_URL = "http://www.cninfo.com.cn/new/information/topSearch/query"
STATIC_HOST = "http://static.cninfo.com.cn/"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "http://www.cninfo.com.cn",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
}


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass
class Announcement:
    sec_code: str
    sec_name: str
    title: str
    publish_ts: int          # 毫秒时间戳
    adjunct_url: str         # 形如 finalpage/2026-08-30/1234567890.PDF
    announcement_id: str = ""

    @property
    def pdf_url(self) -> str:
        return STATIC_HOST + self.adjunct_url.lstrip("/")

    @property
    def publish_date(self) -> str:
        return datetime.fromtimestamp(self.publish_ts / 1000).strftime("%Y-%m-%d")

    @property
    def report_year(self) -> int | None:
        """从标题解析报告期年份："2025年半年度报告" -> 2025。"""
        m = re.search(r"(20\d{2})\s*年", self.title)
        return int(m.group(1)) if m else None


@dataclass
class Stock:
    code: str
    name: str
    org_id: str
    category: str = field(default="")


# --------------------------------------------------------------------------
# 会话与底层请求
# --------------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _request(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response | None:
    """带重试与节流的请求；失败返回 None 而不抛异常，交由调用方决定。"""
    kwargs.setdefault("timeout", config.REQUEST_TIMEOUT)
    last_err = None
    for attempt in range(1, config.MAX_RETRY + 1):
        try:
            resp = session.request(method, url, **kwargs)
            if resp.status_code == 200:
                time.sleep(config.REQUEST_INTERVAL)
                return resp
            last_err = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            last_err = str(exc)
        # 指数退避 + 抖动
        time.sleep(config.REQUEST_INTERVAL * attempt + random.uniform(0, 0.5))
        logger.debug("重试 %s/%s %s（%s）", attempt, config.MAX_RETRY, url, last_err)
    logger.warning("请求失败：%s -> %s", url, last_err)
    return None


# --------------------------------------------------------------------------
# 1) 股票列表 / orgId 解析
# --------------------------------------------------------------------------
def fetch_stock_list(session: requests.Session) -> dict[str, Stock]:
    """拉取全市场股票列表，返回 {code: Stock}。巨潮查询公告必须带 orgId。"""
    resp = _request(session, "GET", STOCKLIST_URL)
    if resp is None:
        return {}
    try:
        raw = resp.json()
    except json.JSONDecodeError:
        logger.warning("股票列表返回非 JSON，前 200 字符：%s", resp.text[:200])
        return {}

    # 该接口返回结构可能是 {"stockList":[...]} 或直接是 list
    items = raw.get("stockList", raw) if isinstance(raw, dict) else raw
    out: dict[str, Stock] = {}
    for it in items or []:
        code = str(it.get("code", "")).strip()
        org_id = str(it.get("orgId", "")).strip()
        if code and org_id:
            out[code] = Stock(code=code, name=str(it.get("zwjc", "")).strip(),
                              org_id=org_id, category=str(it.get("category", "")))
    logger.info("股票列表载入 %d 条", len(out))
    return out


def resolve_org_id(session: requests.Session, code: str, name: str,
                   stock_map: dict[str, Stock] | None = None) -> str | None:
    """三级兜底拿 orgId：本地股票表 -> 关键词搜索 -> 拼接猜测。"""
    if stock_map and code in stock_map:
        return stock_map[code].org_id

    for keyword in (code, name):
        resp = _request(session, "POST", TOPSEARCH_URL,
                        data={"keyWord": keyword, "maxNum": "10"})
        if resp is None:
            continue
        try:
            arr = resp.json()
        except json.JSONDecodeError:
            continue
        for it in arr or []:
            if str(it.get("code", "")) == code:
                return str(it.get("orgId", "")).strip()

    # 最后兜底：沪市 gssh0{code} / 深市 gssz0{code}
    prefix = "gssh0" if code.startswith("6") else "gssz0"
    guess = f"{prefix}{code}"
    logger.warning("%s(%s) 未取到 orgId，使用猜测值 %s", name, code, guess)
    return guess


def exchange_params(code: str) -> dict[str, str]:
    """按交易所返回 column / plate 参数（沪市 6 开头，深市 0/3 开头）。"""
    if code.startswith("6"):
        return {"column": "sse", "plate": "sh"}
    return {"column": "szse", "plate": "sz"}


# --------------------------------------------------------------------------
# 2) 公告查询
# --------------------------------------------------------------------------
def query_announcements(session: requests.Session, code: str, org_id: str,
                        se_date: str, category: str | None = config.REPORT_CATEGORY,
                        page_size: int = 30) -> list[Announcement]:
    """查询某只股票在指定日期区间内的公告列表。"""
    params = {
        "pageNum": "1",
        "pageSize": str(page_size),
        "tabName": "fulltext",
        "searchkey": "",
        "secid": "",
        "stock": f"{code},{org_id}",
        "trade": "",
        "seDate": se_date,
        "sortName": "",
        "sortType": "",
        "isHLtitle": "false",
        **exchange_params(code),
    }
    if category:
        params["category"] = category

    resp = _request(session, "POST", QUERY_URL, data=params)
    if resp is None:
        return []
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        logger.warning("%s 公告查询返回非 JSON：%s", code, resp.text[:200])
        return []

    items = (payload or {}).get("announcements") or []
    result: list[Announcement] = []
    for it in items:
        adjunct = str(it.get("adjunctUrl", "") or "")
        if not adjunct.lower().endswith(".pdf"):
            continue
        result.append(Announcement(
            sec_code=str(it.get("secCode", code)),
            sec_name=str(it.get("secName", "")),
            title=re.sub(r"<[^>]+>", "", str(it.get("announcementTitle", ""))).strip(),
            publish_ts=int(it.get("announcementTime", 0) or 0),
            adjunct_url=adjunct,
            announcement_id=str(it.get("announcementId", "")),
        ))
    return result


def is_target_report(title: str, year: int | None = None) -> bool:
    """标题过滤：命中"半年度报告"且不含"摘要/英文/更正"等噪声词。"""
    if not any(k in title for k in config.TITLE_MUST_HAVE):
        return False
    if any(k in title for k in config.TITLE_MUST_NOT_HAVE):
        return False
    if year is not None and f"{year}年" not in title:
        return False
    return True


def pick_latest_report(anns: Sequence[Announcement], year: int | None = None) -> Announcement | None:
    """从候选公告中挑出目标半年报：优先指定年份，否则取报告期最新的一份。"""
    candidates = [a for a in anns if is_target_report(a.title, year)]
    if not candidates:
        candidates = [a for a in anns if "半年度报告" in a.title]
    if not candidates:
        return None
    # 排序键：报告期年份 -> 披露时间，取最大
    candidates.sort(key=lambda a: (a.report_year or 0, a.publish_ts))
    return candidates[-1]


# --------------------------------------------------------------------------
# 3) PDF 下载
# --------------------------------------------------------------------------
def download_pdf(session: requests.Session, ann: Announcement, dest_dir) -> tuple[bool, str]:
    """下载公告 PDF 到 dest_dir，文件名 {code}_{name}_{报告期}年半年度报告.pdf。

    返回 (是否成功, 本地路径或错误信息)。已存在且大小合理则跳过。
    """
    from pathlib import Path
    dest_dir = Path(dest_dir)

    year = ann.report_year or datetime.fromtimestamp(ann.publish_ts / 1000).year
    filename = f"{ann.sec_code}_{ann.sec_name}_{year}年半年度报告.pdf"
    dest = dest_dir / filename

    if dest.exists() and dest.stat().st_size > 100 * 1024:
        return True, str(dest)

    resp = _request(session, "GET", ann.pdf_url, headers={"Referer": HEADERS["Referer"]},
                    stream=True)
    if resp is None:
        return False, "下载请求失败"

    tmp = dest.with_suffix(".pdf.part")
    try:
        with open(tmp, "wb") as fh:
            first = True
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                if first:
                    if not chunk.startswith(b"%PDF"):
                        tmp.unlink(missing_ok=True)
                        return False, "返回内容不是 PDF（可能被拦截或需登录）"
                    first = False
                fh.write(chunk)
        if tmp.stat().st_size < 100 * 1024:
            size = tmp.stat().st_size
            tmp.unlink(missing_ok=True)
            return False, f"文件过小（{size} 字节），疑似无效公告"
        tmp.replace(dest)
        return True, str(dest)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        return False, f"写入失败：{exc}"
