# -*- coding: utf-8 -*-
"""PDF 解析：逐页抽取文本 + 按行列结构还原财务报表表格 + 页码留痕。

两种解析引擎：
  * pdfplumber（默认）：能拿到字符级坐标，可用 extract_tables() 还原表格，
    这是财务报表三张表（资产负债表/利润表/现金流量表）能被检索到的前提。
  * pypdf：纯文本流，速度快，但表格会退化成一行行挤在一起的数字，仅作备选。

输出的 PageContent 保留 page 字段（1-based），后续切块时据此回填页码标签。
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

import config
from .common import get_logger

logger = get_logger("pdf_parser")

# 纯页码行：如 "12"、"- 12 -"、"第 12 页"、"—12—"
PAGE_NO_LINE_RE = re.compile(r"^\s*(?:第\s*)?[-—–－\s]*\d{1,4}\s*[-—–－\s]*(?:页)?\s*$")
# 连续空白（含全角空格）压缩
MULTI_SPACE_RE = re.compile(r"[ \t　]{2,}")
# 表格单元格内的换行
CELL_NEWLINE_RE = re.compile(r"[\r\n]+")
# 章节标题（第X节 ...）：即使出现在页面边缘也不能当页眉删掉，否则章节识别会失效
SECTION_LIKE_RE = re.compile(r"^[ \t　]*第\s*[一二三四五六七八九十百〇零]{1,4}\s*节")


@dataclass
class PageContent:
    page: int                 # 1-based 页码
    text: str                 # 清洗后的正文（表格已按行列结构拼入）
    n_tables: int = 0         # 本页还原出的表格数
    engine: str = ""


# --------------------------------------------------------------------------
# 清洗
# --------------------------------------------------------------------------
def clean_page_text(raw: str) -> str:
    """页面级清洗：统一换行、压缩空白、去行首行尾空白，保留段落结构。"""
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("﻿", "")
    lines = [MULTI_SPACE_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    # 去掉连续空行，只保留单空行作为段落分隔
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


def strip_running_headers(pages: list[PageContent]) -> list[PageContent]:
    """跨页频次法剔除页眉/页脚。

    半年报每页顶部/底部都会重复"XX股份有限公司 2025 年半年度报告"和页码，
    这些行会污染 BM25 倒排索引（让所有块都命中公司名），必须去掉。

    规则：**只在每页首/末 HEADER_FOOTER_LINES 行内**统计候选，
    某候选行出现在 >DEDUP_HEADER_RATIO 的页面且总页数 >= DEDUP_HEADER_MIN_PAGES
    且长度 <= 40 字符 → 判定为页眉/页脚剔除。
    限定在页面边缘是刻意的：财务报表里"项目 | 期末余额 | 期初余额"这类行
    也可能跨页重复，但它们位于正文中部，不能被误杀。
    """
    if len(pages) < config.DEDUP_HEADER_MIN_PAGES:
        return pages

    n = config.HEADER_FOOTER_LINES
    counter: Counter[str] = Counter()
    for pg in pages:
        lines = pg.text.split("\n")
        edge = lines[:n] + (lines[-n:] if len(lines) > n else [])
        for ln in set(edge):
            if not (0 < len(ln) <= 40):
                continue
            if SECTION_LIKE_RE.match(ln):     # 章节标题：保留
                continue
            if " | " in ln:                   # 表格行：保留（财务三表行首列常重复）
                continue
            counter[ln] += 1

    threshold = max(2, int(len(pages) * config.DEDUP_HEADER_RATIO))
    boilerplate = {ln for ln, c in counter.items() if c >= threshold}
    if boilerplate:
        logger.info("识别到 %d 条重复页眉/页脚，例如：%s",
                    len(boilerplate), list(boilerplate)[:3])

    for pg in pages:
        lines = pg.text.split("\n")
        kept: list[str] = []
        for idx, ln in enumerate(lines):
            # 只在页面首尾做剔除，避免误删正文中重复出现的表头行/表格数字单元格
            at_edge = idx < n or idx >= len(lines) - n
            if at_edge and (ln in boilerplate or PAGE_NO_LINE_RE.match(ln)):
                continue
            kept.append(ln)
        pg.text = "\n".join(kept).strip()
    return pages


# --------------------------------------------------------------------------
# 表格结构还原
# --------------------------------------------------------------------------
def render_tables(tables: list) -> tuple[str, set[str]]:
    """把 pdfplumber 的二维表格渲染成「行 = 单元格 | 单元格」的文本。

    返回 (渲染文本, 已渲染行的归一化集合)，后者用于在正文里去掉重复行。
    """
    blocks: list[str] = []
    rendered_lines: set[str] = set()

    for table in tables or []:
        rows: list[list[str]] = []
        for row in table or []:
            if not row:
                continue
            cells = [CELL_NEWLINE_RE.sub(" ", (c or "")).strip() for c in row]
            if not any(cells):
                continue
            rows.append(cells)
        if len(rows) < 2:          # 单行"表"多半是误检
            continue

        width = max(len(r) for r in rows)
        lines = []
        for r in rows:
            r = r + [""] * (width - len(r))
            line = " | ".join(r).strip(" |")
            line = MULTI_SPACE_RE.sub(" ", line)
            if line:
                lines.append(line)
                rendered_lines.add(_norm(line))
        if lines:
            blocks.append("【表格】\n" + "\n".join(lines))

    return "\n\n".join(blocks), rendered_lines


def _norm(line: str) -> str:
    return re.sub(r"[\s|]", "", line)


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def parse_pdf(path: Path, engine: str | None = None, max_pages: int | None = None,
              extract_tables: bool | None = None,
              progress_cb: Callable[[int, int], None] | None = None) -> list[PageContent]:
    """解析 PDF，返回逐页内容列表。

    Args:
        path: PDF 路径
        engine: "pdfplumber"（默认，表格好）或 "pypdf"（快）
        max_pages: 只解析前 N 页（调试用，None = 全部）
        extract_tables: 是否做表格还原（默认取 config.EXTRACT_TABLES）
        progress_cb: 可选回调 (当前页, 总页数)
    """
    engine = (engine or config.PDF_ENGINE).lower()
    do_tables = config.EXTRACT_TABLES if extract_tables is None else extract_tables
    path = Path(path)

    if engine == "pypdf":
        pages = _parse_with_pypdf(path, max_pages, progress_cb)
    else:
        pages = _parse_with_pdfplumber(path, max_pages, do_tables, progress_cb)

    pages = strip_running_headers(pages)
    n_tables = sum(p.n_tables for p in pages)
    logger.info("%s 解析完成：%d 页，%d 张表格", path.name, len(pages), n_tables)
    return pages


def _parse_with_pdfplumber(path: Path, max_pages, do_tables: bool, progress_cb) -> list[PageContent]:
    import pdfplumber

    pages: list[PageContent] = []
    with pdfplumber.open(str(path)) as pdf:
        total = len(pdf.pages)
        limit = total if not max_pages else min(total, max_pages)
        for i in range(limit):
            page = pdf.pages[i]
            raw = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            text = clean_page_text(raw)
            n_tables = 0

            if do_tables:
                try:
                    tables = page.extract_tables()
                except Exception as exc:               # 个别页表格结构异常，跳过即可
                    logger.debug("%s 第 %d 页表格解析失败：%s", path.name, i + 1, exc)
                    tables = []
                table_text, rendered = render_tables(tables)
                if table_text:
                    n_tables = table_text.count("【表格】")
                    # 正文里与表格重复的行去掉，避免同一份数据进库两次
                    kept = [ln for ln in text.split("\n")
                            if _norm(ln) not in rendered]
                    text = "\n".join(kept).strip()
                    text = f"{text}\n\n{table_text}".strip()

            pages.append(PageContent(page=i + 1, text=text, n_tables=n_tables,
                                     engine="pdfplumber"))
            if progress_cb:
                progress_cb(i + 1, limit)
    return pages


def _parse_with_pypdf(path: Path, max_pages, progress_cb) -> list[PageContent]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    total = len(reader.pages)
    limit = total if not max_pages else min(total, max_pages)
    pages: list[PageContent] = []
    for i in range(limit):
        try:
            raw = reader.pages[i].extract_text() or ""
        except Exception as exc:
            logger.debug("%s 第 %d 页抽取失败：%s", path.name, i + 1, exc)
            raw = ""
        pages.append(PageContent(page=i + 1, text=clean_page_text(raw),
                                 n_tables=0, engine="pypdf"))
        if progress_cb:
            progress_cb(i + 1, limit)
    return pages


# --------------------------------------------------------------------------
# 落盘/读回（阶段三的断点续跑缓存）
# --------------------------------------------------------------------------
def pages_to_dicts(pages: list[PageContent]) -> list[dict]:
    return [asdict(p) for p in pages]


def dicts_to_pages(rows: Iterable[dict]) -> list[PageContent]:
    return [PageContent(**row) for row in rows]


def is_toc_like(text: str) -> bool:
    """判断一段文本是否像目录（大量点线），用于切块阶段丢弃。"""
    if not text:
        return True
    dots = len(re.findall(config.TOC_DOTS_RE, text))
    return dots >= 3
