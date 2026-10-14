# -*- coding: utf-8 -*-
"""阶段二：批量下载申万食品饮料 10 家公司的最新半年度报告全文 PDF。

用法
----
    python download.py                     # 全部 10 家，自动取最新一期半年报
    python download.py --year 2025         # 锁定 2025 年半年报
    python download.py --company 600519 000858
    python download.py --dry-run           # 只查询不下载，先看命中哪些公告
    python download.py --force             # 已存在也重新下载

产物
----
    data/pdfs/{code}_{name}_{year}年半年度报告.pdf
    data/manifest.json                     # 下载清单（可入库，记录取数口径）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import config
from utils import cninfo
from utils.common import get_logger, human_size, load_env, setup_console

logger = get_logger("download")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="下载申万食品饮料行业半年度报告 PDF（数据源：巨潮资讯网）")
    p.add_argument("--company", nargs="*", default=None,
                   help="只处理指定公司代码，如 --company 600519 000858；默认全部 10 家")
    p.add_argument("--year", type=int, default=config.TARGET_YEAR,
                   help="锁定报告期年份，如 2025；默认自动取最新一期半年报")
    p.add_argument("--outdir", type=Path, default=config.PDF_DIR, help="PDF 输出目录")
    p.add_argument("--interval", type=float, default=config.REQUEST_INTERVAL,
                   help="请求间隔秒数（文明爬虫）")
    p.add_argument("--dry-run", action="store_true", help="只查询、不下载")
    p.add_argument("--force", action="store_true", help="已存在也重新下载")
    return p.parse_args()


def main() -> int:
    setup_console()
    load_env()
    args = parse_args()
    config.REQUEST_INTERVAL = args.interval
    args.outdir.mkdir(parents=True, exist_ok=True)

    targets = config.COMPANIES
    if args.company:
        wanted = {c.strip() for c in args.company}
        targets = [c for c in config.COMPANIES if c["code"] in wanted]
        missing = wanted - {c["code"] for c in targets}
        if missing:
            logger.warning("以下代码不在配置文件清单中，已忽略：%s", sorted(missing))
    if not targets:
        logger.error("没有待处理的公司，退出。")
        return 1

    session = cninfo.make_session()

    logger.info("步骤 1/3：拉取巨潮股票列表以解析 orgId ...")
    stock_map = cninfo.fetch_stock_list(session)

    logger.info("步骤 2/3：逐家查询半年度报告公告 ...")
    plan: list[tuple[dict, cninfo.Announcement]] = []
    unresolved: list[dict] = []

    for comp in targets:
        code, name = comp["code"], comp["name"]
        org_id = cninfo.resolve_org_id(session, code, name, stock_map)
        if not org_id:
            unresolved.append(comp)
            continue

        # 优先用「目标年份」窗口；未指定年份时用近三个披露窗口倒序遍历
        years = [args.year] if args.year else [datetime.now().year, datetime.now().year - 1]
        picked: cninfo.Announcement | None = None
        for y in years:
            se_date = config.SEARCH_WINDOW.format(year=y)
            anns = cninfo.query_announcements(session, code, org_id, se_date)
            picked = cninfo.pick_latest_report(anns, year=args.year)
            if picked:
                break

        if not picked:
            logger.warning("%s(%s) 未查到半年报公告", name, code)
            unresolved.append(comp)
            continue

        picked.sec_name = name      # 统一用 config 里的简称，保证文件名可预测
        plan.append((comp, picked))
        logger.info("  ✓ %s(%s) -> 《%s》 披露日 %s", name, code,
                    picked.title, picked.publish_date)

    if not plan and not unresolved:
        logger.error("全部公司均未命中公告，请检查网络或 --year 参数。")
        return 2

    if args.dry_run:
        logger.info("--dry-run 已开启，共命中 %d 份公告，未执行下载：", len(plan))
        for comp, ann in plan:
            print(f"  {comp['name']:>6}({comp['code']})  {ann.report_year or '?'}H1  {ann.pdf_url}")
        return 0

    logger.info("步骤 3/3：下载 PDF 全文 ...")
    records: list[dict] = []
    ok = 0
    for comp, ann in plan:
        if args.force:
            # 删除同名旧文件，强制重下
            year = ann.report_year or "NA"
            old = args.outdir / f"{ann.sec_code}_{comp['name']}_{year}年半年度报告.pdf"
            old.unlink(missing_ok=True)

        success, info = cninfo.download_pdf(session, ann, args.outdir)
        if success:
            size = Path(info).stat().st_size
            ok += 1
            logger.info("  ✓ %s  %s", Path(info).name, human_size(size))
            records.append({
                "code": comp["code"], "name": comp["name"],
                "sw_industry": config.SW_INDUSTRY, "sw_l2": comp.get("sw_l2", ""),
                "report_year": ann.report_year, "period": f"{ann.report_year}H1",
                "title": ann.title, "publish_date": ann.publish_date,
                "file": Path(info).name, "bytes": size,
                "source": ann.pdf_url, "source_site": "cninfo.com.cn",
            })
        else:
            logger.error("  ✗ %s(%s) 下载失败：%s", comp["name"], comp["code"], info)
            records.append({
                "code": comp["code"], "name": comp["name"],
                "sw_industry": config.SW_INDUSTRY, "status": "failed", "error": info,
                "title": ann.title, "source": ann.pdf_url,
            })

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "industry": config.SW_INDUSTRY,
        "source": "巨潮资讯网 http://www.cninfo.com.cn",
        "category": config.REPORT_CATEGORY,
        "target_year": args.year,
        "company_count": len(config.COMPANIES),
        "downloaded": ok,
        "records": records,
        "unresolved": [c["code"] for c in unresolved],
    }
    config.MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"下载完成：成功 {ok} / 目标 {len(targets)} 家；PDF 目录：{args.outdir}")
    if unresolved:
        print("未命中：" + "、".join(f"{c['name']}({c['code']})" for c in unresolved))
    print(f"清单已写入：{config.MANIFEST_FILE}（该文件被 Git 追踪，PDF 目录被忽略）")
    print("=" * 68)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
