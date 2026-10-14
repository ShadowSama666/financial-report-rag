# -*- coding: utf-8 -*-
"""带标签切块：800 字 / 重合 120 字 / 句末收尾 / 强制注入元数据标签。

核心思路
--------
1. 把逐页文本拼成一条长文本 `full_text`，同时记录「字符区间 -> 页码」映射表。
   这样切块在全局偏移上做，天然允许一个块跨页，页码再从映射表反查，
   既不会因逐页切块把句子切碎，也能给出精确的 页码:起始-结束。
2. 用正则扫描全文中所有"第X节 XXX"章节标题，得到「字符偏移 -> 章节名」表，
   每个块继承其起始位置所属的章节。
3. 切点优先落在句末标点（。！？；）上，其次换行，实在没有才硬切；
   下一块起点 = 切点 - 120，实现重叠。
4. 每块生成两段文本：
     text        —— 干净原文，用于展示与喂给 LLM
     tagged_text —— 前缀【公司:.. | 申万行业:.. | 章节:.. | 页码:.. | 块序号:..】
                   用于 BM25 与向量编码（把"这是谁家的、第几页"编码进去）
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, asdict

import config
from .pdf_parser import PageContent, is_toc_like
from .common import get_logger

logger = get_logger("chunker")

# 硬收尾标点：句子结束
HARD_TERM = set("。！？；!?;")
# 软收尾：换行（表格行、列表项）
SOFT_TERM = "\n"

# 章节标题：第一节 释义 / 第十节 财务报告
SECTION_RE = re.compile(r"^[ \t　]*第\s*([一二三四五六七八九十百〇零]{1,4})\s*节\s*([^\n]{0,30})",
                        re.MULTILINE)
CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8,
          "九": 9, "十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14,
          "十五": 15, "二十": 20}
DOTS_RE = re.compile(config.TOC_DOTS_RE)


@dataclass
class Chunk:
    chunk_id: str
    company: str
    code: str
    sw_industry: str
    sw_l2: str
    doc: str
    section: str
    page_start: int
    page_end: int
    chunk_index: int
    char_len: int
    text: str
    tagged_text: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# 1) 页码映射
# --------------------------------------------------------------------------
class PageMap:
    """把「全局字符偏移」映射到「页码」的查询表。"""

    def __init__(self, pages: list[PageContent]) -> None:
        self.starts: list[int] = []
        self.page_nos: list[int] = []
        self._parts: list[str] = []
        cursor = 0
        for pg in pages:
            text = pg.text
            if not text:
                continue
            self.starts.append(cursor)
            self.page_nos.append(pg.page)
            self._parts.append(text)
            cursor += len(text)
            cursor += 2                    # 页与页之间插入 "\n\n" 分隔
        self.text = "\n\n".join(self._parts)
        # 补一个哨兵，方便 bisect 落在最后一页
        self.starts.append(cursor)

    def page_of(self, offset: int) -> int:
        if not self.page_nos:
            return 0
        idx = bisect.bisect_right(self.starts, offset) - 1
        idx = min(max(idx, 0), len(self.page_nos) - 1)
        return self.page_nos[idx]

    def page_range(self, start: int, end: int) -> tuple[int, int]:
        return self.page_of(start), self.page_of(max(start, end - 1))


# --------------------------------------------------------------------------
# 2) 章节映射
# --------------------------------------------------------------------------
def build_section_map(full_text: str, default: str = "全文") -> list[tuple[int, str]]:
    """扫描章节标题，返回 [(字符偏移, "第三节 管理层讨论与分析"), ...]，按偏移升序。

    最容易踩的坑：半年报开头的「目录」也写着"第X节 XXX"。若把它当成正文标题，
    8 个节号会全部锚定在目录区，导致正文所有块都退化成最后一个章节——
    章节角标集体失真。目录排版各家不同，因此用三条互补规则判定目录条目
    （命中任意一条即剔除）：

      a) **点线**：``第一节 释义 ......... 5``
      b) **页码尾 + 后随标题**：raw 以页码结尾，且后 120 字内又是节标题。
         如 ``第二节 公司简介和主要财务指标 5``（古井贡酒这类无点线目录）
      c) **连续成串**：>=4 个节标题彼此间隔 <200 字 —— 目录就是把节名一条条排下来，
         而正文标题之间至少隔上千字（如第七节只有一页，紧接第八节，中间也隔了几十行正文）

    规则 b 必须带"页码尾"这个前提；规则 c 必须用"连续间隔"而不是"窗口密度"，
    否则正文标题只要离目录近一点就会被一起误杀。
    过滤后同一节号只保留首次出现（正文的交叉引用/题注会重复提到节名）。
    """
    TOC_GAP, TOC_RUN = 200, 4
    PAGE_TAIL_RE = re.compile(r"\d{1,3}\s*$")

    matches = [(m.start(), m.end(), m.group(1), m.group(0).strip())
               for m in SECTION_RE.finditer(full_text)]
    starts = [s for s, _, _, _ in matches]

    # 预标记规则 c：找出所有"连续成串"的节标题下标
    in_toc_run = [False] * len(matches)
    run_start = 0
    for i in range(1, len(matches) + 1):
        if i == len(matches) or starts[i] - starts[i - 1] > TOC_GAP:
            if i - run_start >= TOC_RUN:
                for t in range(run_start, i):
                    in_toc_run[t] = True
            run_start = i

    kept = []
    for i, (start, end, num_cn, raw) in enumerate(matches):
        tail = full_text[end:end + 120]
        if DOTS_RE.search(raw) or DOTS_RE.search(tail):           # a
            continue
        if PAGE_TAIL_RE.search(raw) and SECTION_RE.search(tail):  # b
            continue
        if in_toc_run[i]:                                         # c
            continue
        kept.append((start, end, num_cn, raw))

    out: list[tuple[int, str]] = []
    seen: set[str] = set()
    for start, _end, num_cn, _raw in kept:
        key = f"第{num_cn}节"
        if key in seen:
            continue
        seen.add(key)

        title_body = re.sub(r"[\s　]+", "", _raw)
        title_body = re.sub(r"^第[一二三四五六七八九十百〇零]{1,4}节", "", title_body)
        title_body = title_body.strip(" .．·…-—")
        # 无点线目录行会把页码黏在标题尾部（如"第八节 财务报告32"），去掉尾随数字
        title_body = re.sub(r"\d+$", "", title_body).strip(" .．·…-—、")
        title = f"第{num_cn}节 {title_body}" if title_body else f"第{num_cn}节"
        out.append((start, title[:40]))

    if not out:
        return [(0, default)]
    if out[0][0] > 0:
        out.insert(0, (0, "封面与重要提示"))
    return out


def section_at(section_map: list[tuple[int, str]], offset: int) -> str:
    if not section_map:
        return "全文"
    idx = bisect.bisect_right([s[0] for s in section_map], offset) - 1
    idx = min(max(idx, 0), len(section_map) - 1)
    return section_map[idx][1]


# --------------------------------------------------------------------------
# 3) 切块
# --------------------------------------------------------------------------
def _find_cut(text: str, start: int, target: int, floor: int) -> int:
    """在 [floor, target) 内从后往前找自然收尾点，返回切点（开区间右端）。"""
    for i in range(target - 1, floor - 1, -1):
        if text[i] in HARD_TERM:
            return i + 1
    for i in range(target - 1, floor - 1, -1):
        if text[i] in SOFT_TERM:
            return i + 1
    return target


def chunk_spans(text: str, size: int = config.CHUNK_SIZE,
                overlap: int = config.CHUNK_OVERLAP,
                min_size: int = config.MIN_CHUNK_SIZE) -> list[tuple[int, int]]:
    """按目标字数切块，返回 [(start, end), ...]。"""
    spans: list[tuple[int, int]] = []
    n = len(text)
    start = 0
    floor_back = int(size * 0.35)          # 最多向前回溯 35% 去找句末

    while start < n:
        target = min(start + size, n)
        if target >= n:
            end = n
        else:
            floor = max(start + min_size, target - floor_back)
            end = _find_cut(text, start, target, floor)

        if end <= start:                   # 保险：绝不死循环
            end = min(start + size, n)

        if text[start:end].strip():
            spans.append((start, end))

        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        start = next_start

    return spans


# --------------------------------------------------------------------------
# 4) 标签与主流程
# --------------------------------------------------------------------------
def make_tag(company: str, section: str, page_start: int, page_end: int,
             chunk_index: int) -> str:
    page = f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"
    return (f"【公司:{company} | 申万行业:{config.SW_INDUSTRY} | 章节:{section} | "
            f"页码:{page} | 块序号:{chunk_index}】")


def build_chunks(pages: list[PageContent], company: dict, doc_name: str,
                 period: str = "") -> list[Chunk]:
    """把一个公司的逐页文本切成带标签的块。

    Args:
        pages: PageContent 列表（页码 1-based）
        company: config.COMPANIES 中的一项 {code, name, exchange, sw_l2}
        doc_name: PDF 文件名（写入元数据，便于溯源回原文）
        period: 报告期标签，如 "2025H1"，参与 chunk_id 构造
    """
    pmap = PageMap(pages)
    full_text = pmap.text
    if not full_text.strip():
        logger.warning("%s 解析结果为空，跳过", doc_name)
        return []

    section_map = build_section_map(full_text)
    logger.debug("%s 识别章节 %d 个：%s", doc_name, len(section_map),
                 [s[1] for s in section_map[:5]])

    spans = chunk_spans(full_text)
    chunks: list[Chunk] = []
    dropped_toc = 0

    for start, end in spans:
        raw = full_text[start:end].strip()
        if not raw:
            continue
        if is_toc_like(raw):                    # 丢弃目录块
            dropped_toc += 1
            continue
        if len(raw) < config.MIN_CHUNK_SIZE and end < len(full_text) - 1:
            continue

        section = section_at(section_map, start)
        p_start, p_end = pmap.page_range(start, end)
        idx = len(chunks)
        tag = make_tag(company["name"], section, p_start, p_end, idx)

        chunks.append(Chunk(
            chunk_id=f"{company['code']}-{period or 'NA'}-{idx:04d}",
            company=company["name"],
            code=company["code"],
            sw_industry=config.SW_INDUSTRY,
            sw_l2=company.get("sw_l2", ""),
            doc=doc_name,
            section=section,
            page_start=p_start,
            page_end=p_end,
            chunk_index=idx,
            char_len=len(raw),
            text=raw,
            tagged_text=f"{tag}\n{raw}",
        ))

    if dropped_toc:
        logger.info("%s 丢弃目录块 %d 个", doc_name, dropped_toc)
    return chunks


# --------------------------------------------------------------------------
# 5) 文件名解析
# --------------------------------------------------------------------------
FN_RE = re.compile(r"^(\d{6})_(.+?)_(\d{4})年半年度报告$")


def parse_pdf_filename(stem: str) -> dict | None:
    """从 "600519_贵州茅台_2025年半年度报告" 解析出公司代码、名称、报告期。"""
    m = FN_RE.match(stem)
    if not m:
        return None
    code, name, year = m.group(1), m.group(2), m.group(3)
    return {"code": code, "name": name, "year": int(year), "period": f"{year}H1"}
