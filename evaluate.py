# -*- coding: utf-8 -*-
"""阶段六：评测框架与测试题。

10 道测试题（含 2 道跨公司全景对比题）跑一遍完整链路，逐题记录三件事：

1. **召回了哪些文本块** —— 公司 + 页码 + 命中来源（向量/BM25/双路）+ 分数，
   逐块落盘到 `recall_detail.csv`，并统计是否命中人工核对过的"黄金页"。
2. **是否答对** —— 两级判定：
   * 机器判官（DeepSeek，按 `key_points` 逐条判是否覆盖）；
   * 人工列（`人工结论` / `人工失败原因`，课堂上手填，机器不覆盖）。
   没有 API Key 时机器判官自动跳过，只出召回指标，人工列照常留空。
3. **翻车原因分析** —— 机器判官按固定分类给出，另外把"检索漏召 / 召回噪声干扰"
   两类与召回指标绑定：黄金页命中率低 → 检索漏召；黄金页命中但答案错 → 生成侧问题。

为什么指标体系要分两级
----------------------
只看"答对没有"无法定位问题在检索还是在生成。所以每题同时记录：
* `黄金页命中率`：答案所在的原文页有没有被召回（检索侧）；
* `目标公司覆盖率`：跨公司题里 4~5 家公司是否都被召回（跨公司聚合的前置条件）；
* `判官结论` + `失败原因`：最终答案质量（生成侧）。
三者交叉才能说清一次失败到底是"没找到"还是"找到了但没用好"。

用法
----
    python evaluate.py                      # 全量 10 题，检索 + 生成 + 判官
    python evaluate.py --no-generate        # 只跑检索，测召回（不烧 token）
    python evaluate.py --no-judge           # 生成但不自动判分，留人工填
    python evaluate.py --only Q09 Q10       # 只跑跨公司对比题
    python evaluate.py --write-template     # 只刷新空白评分模板
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import config
from utils.common import get_logger, page_label, setup_console
from utils.llm import (DeepSeekClient, DeepSeekError, build_eval_messages,
                       build_messages, parse_citations)
from utils.retriever import HybridRetriever, IndexNotBuilt

logger = get_logger("evaluate")
setup_console()

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)

FAILURE_CLASSES = ["检索漏召", "召回噪声干扰", "生成幻觉",
                   "跨公司聚合失败", "数值口径错误", "其他"]

# 评分表字段。前 16 列由机器填写，最后 3 列留给人工（课堂上手填）。
SCORING_FIELDS = [
    "题号", "类型", "问题", "目标公司",
    "召回块数", "黄金页命中率", "目标公司覆盖率", "章节命中",
    "双路命中块数", "公司配额", "黄金页明细", "召回Top1(公司·页)",
    "生成", "角标合规", "判官结论", "失败原因",
    "人工结论", "人工失败原因", "人工备注",
]

RECALL_FIELDS = [
    "题号", "排序", "公司", "章节", "页码", "命中来源",
    "是否黄金页", "是否目标公司", "RRF分", "余弦", "BM25分", "块字数", "chunk_id",
]


# ==========================================================================
# 参数与载入
# ==========================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="申万食品饮料半年报知识库评测",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=config.EVAL_DIR / "questions.json",
                    help="测试题文件（默认 eval/questions.json）")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出目录（默认 eval/runs/<时间戳>/）")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定题号，例：--only Q09 Q10")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    ap.add_argument("--kv", type=int, default=config.TOP_K_VECTOR, help="向量通道 Top-K")
    ap.add_argument("--kb", type=int, default=config.TOP_K_BM25, help="BM25 通道 Top-K")
    ap.add_argument("--top-n", type=int, default=config.TOP_K_FINAL, help="最终送模型 Top-N")
    ap.add_argument("--rrf", type=int, default=config.RRF_K, help="RRF 常数 k")
    ap.add_argument("--no-generate", action="store_true", help="只跑检索，不调用 DeepSeek")
    ap.add_argument("--no-judge", action="store_true", help="生成但不自动判分")
    ap.add_argument("--write-template", action="store_true",
                    help="只刷新 eval/scoring_template.csv 后退出")
    return ap.parse_args()


def load_questions(path: Path) -> tuple[dict, list[dict]]:
    if not path.exists():
        raise SystemExit(f"未找到测试题文件 {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    qs = data["questions"]
    meta = data.get("meta", {})
    n_cross = sum(1 for q in qs if q["type"] == "跨公司")
    if n_cross < 2:
        raise SystemExit(f"测试题必须含至少 2 道跨公司对比题，当前只有 {n_cross} 道")
    return meta, qs


# ==========================================================================
# 指标
# ==========================================================================
def page_str(chunk: dict) -> str:
    """页码显示口径与前端共用（见 utils.common.page_label）。"""
    return page_label(chunk["page_start"], chunk["page_end"])


def is_gold(chunk: dict, gold: list[dict]) -> bool:
    """召回块是否落在人工核对过的黄金页上（公司一致 + 页码在集合内）。"""
    for g in gold:
        if chunk["company"] != g["company"]:
            continue
        lo, hi = min(g["pages"]), max(g["pages"])
        # 块可能跨页，与黄金页区间有交集即算命中
        if chunk["page_start"] <= hi and chunk["page_end"] >= lo:
            return True
    return False


def recall_metrics(q: dict, fused: list) -> dict:
    chunks = [r.chunk for r in fused]
    gold = q.get("gold", [])
    targets = q["companies"]

    gold_hits, gold_detail = 0, []
    for g in gold:
        lo, hi = min(g["pages"]), max(g["pages"])
        hit = any(c["company"] == g["company"]
                  and c["page_start"] <= hi and c["page_end"] >= lo for c in chunks)
        gold_hits += hit
        gold_detail.append(
            f"{g['company']}p{lo if lo == hi else f'{lo}-{hi}'}{'✓' if hit else '✗'}")

    recalled_companies = {c["company"] for c in chunks}
    covered = [c for c in targets if c in recalled_companies]
    expect_sections = q.get("expect_sections", [])
    section_hit = any(any(s in c["section"] for s in expect_sections) for c in chunks) \
        if expect_sections else None

    return {
        "gold_recall": gold_hits / len(gold) if gold else None,
        "gold_any": gold_hits > 0,
        "gold_detail": " ".join(gold_detail),
        "company_coverage": len(covered) / len(targets),
        "missing_companies": [c for c in targets if c not in recalled_companies],
        "section_hit": section_hit,
        "both_hits": sum(1 for r in fused if r.vector_rank and r.bm25_rank),
        "top1": f"{chunks[0]['company']}·p{page_str(chunks[0])}" if chunks else "—",
        "n_recalled": len(chunks),
    }


def citation_metrics(answer: str, chunks: list[dict], targets: list[str]) -> dict:
    """角标合规性：是否有角标、角标是否越界、目标公司是否都出现在被引用的块里。"""
    cited = parse_citations(answer)
    invalid = [c for c in cited if c < 1 or c > len(chunks)]
    valid = [c for c in cited if 1 <= c <= len(chunks)]
    cited_companies = {chunks[c - 1]["company"] for c in valid}
    missing = [c for c in targets if c not in cited_companies]
    return {
        "n_cited": len(valid),
        "invalid": invalid,
        "missing_companies": missing,
        # 跨公司题要求每家公司都被引用到；单公司题只要求 ≥1 个有效角标
        "ok": bool(valid) and not invalid and not missing,
    }


# ==========================================================================
# 判官
# ==========================================================================
def parse_judge_json(text: str) -> dict | None:
    m = JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def run_judge(client: DeepSeekClient, q: dict, answer: str) -> dict:
    try:
        raw = client.chat(build_eval_messages(q["question"], answer, q["key_points"],
                                              q.get("ref", "")),
                          temperature=0.0, max_tokens=1500)
    except DeepSeekError as exc:
        return {"verdict": "判官失败", "failure_reason": "其他",
                "error": str(exc)[:200]}
    parsed = parse_judge_json(raw)
    if not parsed:
        return {"verdict": "判官返回非 JSON", "failure_reason": "其他",
                "raw": raw[:300]}
    covered = parsed.get("key_points", [])
    n_ok = sum(1 for p in covered if isinstance(p, dict) and p.get("covered"))
    return {
        "verdict": parsed.get("verdict", "—"),
        "failure_reason": parsed.get("failure_reason", "其他"),
        "citation_ok": parsed.get("citation_ok"),
        "hallucination": parsed.get("hallucination"),
        "key_points_covered": f"{n_ok}/{len(q['key_points'])}",
        "points": covered,
    }


def attribute_failure(m: dict, judge: dict | None) -> str:
    """把召回指标与判官结论交叉，给出可定位的失败归类。

    这一步是评测的价值所在：不能只说"错了"，要说清错在检索还是生成。
    """
    if judge is None:
        return ""
    if judge.get("verdict") == "答对":
        return "无"
    if m["gold_recall"] is not None and not m["gold_any"]:
        return "检索漏召"
    if m.get("company_coverage", 1) < 1:
        return "跨公司聚合失败"
    if judge.get("hallucination"):
        return "生成幻觉"
    if judge.get("failure_reason") and judge["failure_reason"] != "无":
        return judge["failure_reason"]
    if m["gold_any"]:
        return "召回噪声干扰"
    return "其他"


# ==========================================================================
# 输出
# ==========================================================================
def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    # utf-8-sig：Windows 上 Excel 直接双击打开不乱码
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_template(path: Path, questions: list[dict]) -> None:
    """空白评分模板：题号/类型/问题/目标公司 预填，其余留空给人工判。

    列顺序**复用 SCORING_FIELDS**，不另抄一份：之前这里抄了一份硬编码列表，
    加了「公司配额」列之后立刻与正式输出漂移，手工填的表和机器填的表对不上。
    """
    rows = [{"题号": q["id"], "类型": q["type"], "问题": q["question"],
             "目标公司": "、".join(q["companies"])} for q in questions]
    write_csv(path, SCORING_FIELDS, rows)
    logger.info("评分模板已写入 %s（%d 题，%d 列）",
                path, len(rows), len(SCORING_FIELDS))


# ==========================================================================
# 主流程
# ==========================================================================
def main() -> None:
    args = parse_args()
    meta, questions = load_questions(args.questions)

    if args.write_template:
        write_template(config.EVAL_DIR / "scoring_template.csv", questions)
        return

    if args.only:
        wanted = {s.upper() for s in args.only}
        questions = [q for q in questions if q["id"].upper() in wanted]
        if not questions:
            raise SystemExit(f"--only 未匹配到任何题号：{args.only}")
    if args.limit:
        questions = questions[:args.limit]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else config.EVAL_DIR / "runs" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    client = DeepSeekClient()
    do_generate = not args.no_generate and client.available
    do_judge = do_generate and not args.no_judge

    logger.info("评测开始：%d 题 | 报告期 %s | 检索 kv=%d kb=%d top_n=%d rrf=%d",
                len(questions), meta.get("report_period", "?"),
                args.kv, args.kb, args.top_n, args.rrf)
    logger.info("生成：%s | 自动判官：%s | 输出目录：%s",
                "开" if do_generate else ("关（无 API Key）" if not client.available else "关"),
                "开" if do_judge else "关", out_dir)
    if not client.available:
        logger.warning("未配置 DEEPSEEK_API_KEY：本次只测召回，人工列照常留空待填。")

    try:
        retriever = HybridRetriever()
    except IndexNotBuilt as exc:
        raise SystemExit(f"{exc}\n请先运行：python download.py && python build_kb.py && "
                         f"python build_index.py")

    scoring, recall_rows, results = [], [], []
    t0 = time.time()

    for i, q in enumerate(questions, start=1):
        logger.info("[%d/%d] %s %s", i, len(questions), q["id"], q["question"][:44] + "…")
        t_q = time.time()

        # 跨公司对比题必须给每家留出名额：Top-N 小于公司数时，再好的配额轮转
        # 也凑不出"每家一块"。这里按 2 块/家 放宽，并把实际值记进结果，
        # 不让口径悄悄漂移（results.jsonl 的 param.top_k_final 是实测值）。
        top_n = args.top_n
        if len(q["companies"]) > 1 and config.BALANCE_MULTI_COMPANY:
            top_n = max(top_n, min(2 * len(q["companies"]), 16))
            if top_n != args.top_n:
                logger.info("  ↑ 跨公司题（%d 家）：Top-N 由 %d 放宽到 %d",
                            len(q["companies"]), args.top_n, top_n)

        trace = retriever.search_trace(q["question"], top_k_final=top_n,
                                       top_k_vector=args.kv, top_k_bm25=args.kb,
                                       rrf_k=args.rrf, companies=q["companies"])
        m = recall_metrics(q, trace.fused)
        chunks = [r.chunk for r in trace.fused]

        for j, r in enumerate(trace.fused, start=1):
            recall_rows.append({
                "题号": q["id"], "排序": j, "公司": r.chunk["company"],
                "章节": r.chunk["section"], "页码": page_str(r.chunk),
                "命中来源": r.source,
                "是否黄金页": "是" if is_gold(r.chunk, q.get("gold", [])) else "",
                "是否目标公司": "是" if r.chunk["company"] in q["companies"] else "",
                "RRF分": f"{r.rrf_score:.5f}",
                "余弦": "" if r.vector_score is None else f"{r.vector_score:.4f}",
                "BM25分": "" if r.bm25_score is None else f"{r.bm25_score:.3f}",
                "块字数": r.chunk["char_len"], "chunk_id": r.chunk["chunk_id"],
            })

        answer, gen_state, cite, judge = "", "未生成", None, None
        if do_generate:
            try:
                answer = client.chat(build_messages(q["question"], chunks),
                                     temperature=config.ANSWER_TEMPERATURE,
                                     max_tokens=config.ANSWER_MAX_TOKENS)
                gen_state = "成功"
            except DeepSeekError as exc:
                gen_state = f"失败：{str(exc)[:80]}"
                logger.error("  %s 生成失败：%s", q["id"], exc)

        if answer:
            cite = citation_metrics(answer, chunks, q["companies"])
            if do_judge:
                judge = run_judge(client, q, answer)
                logger.info("  → 判官：%s（要点 %s）", judge.get("verdict"),
                            judge.get("key_points_covered", "—"))

        scoring.append({
            "题号": q["id"], "类型": q["type"], "问题": q["question"],
            "目标公司": "、".join(q["companies"]),
            "召回块数": m["n_recalled"],
            "黄金页命中率": "" if m["gold_recall"] is None else f"{m['gold_recall']:.0%}",
            "目标公司覆盖率": f"{m['company_coverage']:.0%}",
            "章节命中": "—" if m["section_hit"] is None
                        else ("是" if m["section_hit"] else "否"),
            "双路命中块数": m["both_hits"],
            "公司配额": "、".join(f"{c}×{n}" for c, n in trace.company_counts.items()),
            "黄金页明细": m["gold_detail"],
            "召回Top1(公司·页)": m["top1"],
            "生成": gen_state,
            "角标合规": "—" if cite is None else ("是" if cite["ok"] else "否"),
            "判官结论": "—" if judge is None else judge.get("verdict", "—"),
            "失败原因": attribute_failure(m, judge),
            "人工结论": "", "人工失败原因": "", "人工备注": "",
        })

        results.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "target_companies": q["companies"], "expect_sections": q.get("expect_sections", []),
            "key_points": q["key_points"],
            "metrics": m,
            "balanced": trace.balanced,
            "per_company_channel": trace.per_company_channel,
            "company_counts": trace.company_counts,
            "pool_counts": trace.pool_counts,
            "param": {"top_k_vector": args.kv, "top_k_bm25": args.kb,
                      "top_k_final": top_n, "rrf_k": args.rrf},
            "bm25_tokens": trace.bm25_tokens,
            "query_xy": trace.query_xy,
            "recalled": [r.to_dict() for r in trace.fused],
            "answer": answer,
            "citation": cite,
            "judge": judge,
            "elapsed_sec": round(time.time() - t_q, 2),
        })

    # ---------------------------------------------------------------- 落盘
    write_csv(out_dir / "scoring.csv", SCORING_FIELDS, scoring)
    write_csv(out_dir / "recall_detail.csv", RECALL_FIELDS, recall_rows)
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 空白模板每次都刷新，保证字段与本次评测一致
    write_template(config.EVAL_DIR / "scoring_template.csv", questions)

    # ---------------------------------------------------------------- 汇总
    n = len(scoring)
    gold = [r["metrics"]["gold_recall"] for r in results if r["metrics"]["gold_recall"] is not None]
    cov = [r["metrics"]["company_coverage"] for r in results]
    judged = [r for r in results if r["judge"]]
    correct = sum(1 for r in judged if r["judge"].get("verdict") == "答对")
    partial = sum(1 for r in judged if r["judge"].get("verdict") == "部分答对")
    failures: dict[str, int] = {}
    for row in scoring:
        if row["失败原因"] and row["失败原因"] != "无":
            failures[row["失败原因"]] = failures.get(row["失败原因"], 0) + 1

    lines = [
        f"# 评测汇总 {stamp}",
        "",
        f"- 题量：{n}（跨公司 {sum(1 for r in results if r['type'] == '跨公司')} 道）",
        f"- 检索参数：向量 Top-{args.kv} / BM25 Top-{args.kb} / RRF k={args.rrf}"
        f" / 最终 Top-{args.top_n}（单公司题；跨公司题另见下行的放宽值，"
        f"每题实测值在 `results.jsonl` 的 `param.top_k_final`）",
        f"- 跨公司配额轮转：{'开启' if config.BALANCE_MULTI_COMPANY else '关闭'}"
        f"（每家最多 {config.PER_COMPANY_K} 块；跨公司题的 Top-N 自动放宽到 2 块/家）",
        f"- 黄金页命中：{sum(1 for x in gold if x > 0)}/{len(gold)} 题命中至少一处"
        f"，平均黄金页召回率 {sum(gold) / len(gold):.1%}" if gold else "- 黄金页命中：无数据",
        f"- 目标公司覆盖率均值：{sum(cov) / len(cov):.1%}",
        f"- 生成成功率：{sum(1 for r in results if r['answer'])}/{n}",
        f"- 判官结论：答对 {correct} / 部分答对 {partial} / 已判定 {len(judged)}",
        f"- 失败原因分布：{failures or '（无，或未判定）'}",
        f"- 总耗时：{time.time() - t0:.0f} 秒",
        "",
        "## 逐题",
        "",
        "| 题号 | 类型 | 黄金页召回 | 公司覆盖 | 角标 | 判官 | 失败原因 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in scoring:
        lines.append(
            f"| {row['题号']} | {row['类型']} | {row['黄金页命中率']} | "
            f"{row['目标公司覆盖率']} | {row['角标合规']} | {row['判官结论']} | "
            f"{row['失败原因'] or '—'} |")
    lines += [
        "",
        "## 怎么读这张表",
        "",
        "* `黄金页命中率`低而 `判官结论`错 → **检索侧**问题：先调 Top-K / RRF k，",
        "  或检查该题问法是否绕开了原文用词（向量通道该补，BM25 命中不了同义改写）。",
        "* `黄金页命中率`高而 `判官结论`仍错 → **生成侧**问题：块进来了但模型没用对，",
        "  多为跨公司聚合失败或数值口径错误，要看 `results.jsonl` 里的 `recalled` 是否被角标引用。",
        "* `目标公司覆盖率` < 100% 的跨公司题，答案必然缺项。本框架已用**跨公司配额轮转**"
        "（`BALANCE_MULTI_COMPANY`）把覆盖率兜住：先按公司分桶再轮转取用，"
        "并对跨公司题自动放宽 Top-N 到 2 块/家。所以这里若仍 < 100%，"
        "说明某家公司在候选池里**一块都没被召回**（两路都没捞到），属于召回问题而非选取问题。",
        "* 跨公司题的 `黄金页命中率` 偏低但 `目标公司覆盖率` 为 100%，是**配额不够**："
        "每家在 Top-N 里只分到 1~2 块，恰好没落在黄金页上。调大 `--top-n` "
        "或调大 `PER_COMPANY_K` 再复跑即可，`公司配额` 列能直接看出每家占了几块。",
        "* 看 `公司配额` 列时注意：开启配额轮转后，「送入序号」不再严格按 RRF 总分递减，"
        "这是选取策略而非打分异常（`RRF分` 列本身未改动）。",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info("评测完成，用时 %.0f 秒", time.time() - t0)
    print("\n" + "\n".join(lines[:10]))
    print(f"\n明细：{out_dir / 'scoring.csv'}")
    print(f"       {out_dir / 'recall_detail.csv'}")
    print(f"       {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
