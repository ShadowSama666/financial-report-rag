# -*- coding: utf-8 -*-
"""阶段三：PDF 文本提取 -> 表格还原 -> 800/120 带标签切块 -> chunks.jsonl。

用法
----
    python build_kb.py                      # 处理 data/pdfs/ 下全部 PDF
    python build_kb.py --company 600519     # 只处理贵州茅台
    python build_kb.py --limit-pages 30     # 只解析前 30 页（快速冒烟测试）
    python build_kb.py --engine pypdf       # 换解析引擎（快，但表格结构丢失）
    python build_kb.py --rebuild            # 忽略 data/pages/ 缓存，重新解析

产物
----
    data/pages/{pdf名}.json                 # 逐页解析缓存（.gitignore）
    data/chunks/chunks.jsonl                # 知识库：带标签文本块（.gitignore，可再生）
    data/chunks/chunks_stats.csv            # 每家公司块数/页数/表格数统计
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import config
from utils import pdf_parser
from utils.chunker import build_chunks, parse_pdf_filename
from utils.common import get_logger, human_size, setup_console

logger = get_logger("build_kb")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="解析半年报 PDF 并切分成带元数据标签的文本块")
    p.add_argument("--company", nargs="*", default=None,
                   help="只处理指定公司代码，如 --company 600519 000858")
    p.add_argument("--pdf-dir", type=Path, default=config.PDF_DIR)
    p.add_argument("--engine", choices=["pdfplumber", "pypdf"], default=config.PDF_ENGINE)
    p.add_argument("--limit-pages", type=int, default=None, help="只解析前 N 页（调试用）")
    p.add_argument("--no-tables", action="store_true", help="关闭表格结构还原")
    p.add_argument("--rebuild", action="store_true", help="忽略逐页缓存，强制重新解析")
    return p.parse_args()


# --------------------------------------------------------------------------
def load_or_parse(pdf_path: Path, args: argparse.Namespace):
    """逐页解析，结果缓存到 data/pages/{stem}.json；有缓存且未 --rebuild 时直接读回。"""
    cache_file = config.PAGE_DIR / f"{pdf_path.stem}.json"

    if cache_file.exists() and not args.rebuild and args.limit_pages is None:
        try:
            rows = json.loads(cache_file.read_text(encoding="utf-8"))
            pages = pdf_parser.dicts_to_pages(rows["pages"])
            logger.info("  ↺ 命中解析缓存（%d 页）", len(pages))
            return pages, 0.0
        except Exception as exc:
            logger.warning("  缓存损坏（%s），重新解析", exc)

    t0 = time.time()
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    bar = None

    def cb(cur: int, tot: int) -> None:
        nonlocal bar
        if tqdm is None:
            return
        if bar is None:                       # 首帧才知道总页数，此时再建进度条
            bar = tqdm(total=tot, desc=f"  {pdf_path.stem[:20]}", unit="页",
                       leave=False, ncols=88)
        bar.update(1)

    pages = pdf_parser.parse_pdf(
        pdf_path,
        engine=args.engine,
        max_pages=args.limit_pages,
        extract_tables=not args.no_tables,
        progress_cb=cb,
    )
    if bar is not None:
        bar.close()

    elapsed = time.time() - t0
    if args.limit_pages is None:      # 只缓存完整解析结果
        cache_file.write_text(
            json.dumps({"file": pdf_path.name, "engine": args.engine,
                        "parsed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "pages": pdf_parser.pages_to_dicts(pages)},
                       ensure_ascii=False),
            encoding="utf-8")
    return pages, elapsed


def collect_pdfs(args: argparse.Namespace) -> list[Path]:
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    parsed: list[tuple[Path, dict]] = []
    for p in pdfs:
        meta = parse_pdf_filename(p.stem)
        if meta is None:
            logger.warning("文件名不符合 {code}_{name}_{year}年半年度报告.pdf 规范，跳过：%s", p.name)
            continue
        if args.company and meta["code"] not in set(args.company):
            continue
        parsed.append((p, meta))
    return parsed


# --------------------------------------------------------------------------
def main() -> int:
    setup_console()
    args = parse_args()

    if not args.pdf_dir.exists() or not any(args.pdf_dir.glob("*.pdf")):
        logger.error("未在 %s 找到 PDF，请先运行：python download.py", args.pdf_dir)
        return 1

    targets = collect_pdfs(args)
    if not targets:
        logger.error("按当前过滤条件没有可处理的 PDF。")
        return 1

    print(f"\n待处理 {len(targets)} 份半年报，解析引擎 = {args.engine}"
          f"{'（关闭表格还原）' if args.no_tables else '（启用表格还原）'}\n")

    all_chunks: list[dict] = []
    stats_rows: list[dict] = []

    for pdf_path, meta in targets:
        comp = config.COMPANY_BY_CODE.get(meta["code"])
        if comp is None:
            logger.warning("%s 不在 config.COMPANIES 清单中，跳过（行业口径需一致）", meta["name"])
            continue

        logger.info("处理 %s（%s）· %s", comp["name"], comp["code"], pdf_path.name)
        pages, elapsed = load_or_parse(pdf_path, args)
        if not pages:
            stats_rows.append({"company": comp["name"], "code": comp["code"],
                               "file": pdf_path.name, "pages": 0, "chunks": 0,
                               "tables": 0, "chars": 0, "seconds": round(elapsed, 1)})
            continue

        chunks = build_chunks(pages, comp, pdf_path.name, period=meta["period"])
        all_chunks.extend(c.to_dict() for c in chunks)

        stats_rows.append({
            "company": comp["name"], "code": comp["code"], "file": pdf_path.name,
            "period": meta["period"], "pages": len(pages),
            "tables": sum(p.n_tables for p in pages),
            "chars": sum(len(p.text) for p in pages),
            "chunks": len(chunks),
            "avg_chunk_len": round(sum(c.char_len for c in chunks) / len(chunks), 1) if chunks else 0,
            "size": human_size(pdf_path.stat().st_size),
            "seconds": round(elapsed, 1),
        })
        logger.info("  ✓ %d 页 -> %d 块（平均 %.0f 字）",
                    len(pages), len(chunks),
                    stats_rows[-1]["avg_chunk_len"])

    # ---- 写出知识库 ----
    config.CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNK_FILE, "w", encoding="utf-8") as fh:
        for row in all_chunks:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if stats_rows:
        import csv
        with open(config.STATS_FILE, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(stats_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stats_rows)

    # ---- 质检：抽一个块展示标签格式 ----
    print("\n" + "=" * 78)
    print(f"知识库构建完成：{len(all_chunks)} 个文本块")
    print(f"  chunk 文件：{config.CHUNK_FILE}")
    print(f"  统计报表  ：{config.STATS_FILE}")
    print("=" * 78)
    if all_chunks:
        sample = all_chunks[len(all_chunks) // 2]
        print("\n【标签格式抽样】")
        print(sample["tagged_text"].split("\n")[0])
        print("-" * 78)
        print(sample["text"][:260].replace("\n", " ") + " ...")
        print("-" * 78)
        print(f"text 字段长度 {sample['char_len']} 字，"
              f"覆盖页码 {sample['page_start']}-{sample['page_end']}，"
              f"章节「{sample['section']}」，来源 {sample['doc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
