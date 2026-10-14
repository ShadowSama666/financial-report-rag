# -*- coding: utf-8 -*-
"""双通道混合检索器：向量召回 + BM25 倒排召回 + RRF 融合。

被 build_index.py / app.py / evaluate.py 共用，保证三处的检索逻辑完全一致
（否则评测结果无法解释线上行为）。

设计要点
--------
* **非对称编码**：Qwen3-Embedding 官方用法是文档端裸编码、查询端加
  `Instruct: ...\\nQuery: ...` 前缀。文档向量在 build_index 时固化，
  查询向量在检索时现算，两侧共用同一个 L2 归一化空间。
* **RRF 融合**：不混合两路的原始分数（余弦相似度与 BM25 分数尺度不可比），
  只用排名做倒数融合 score = Σ 1/(k + rank)，k=60。
* **可解释**：每个结果都带两路的排名与分数，前端"检索动态过程"直接读这些字段画图。
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import config
from .common import get_logger
from .tokenizer import tokenize

logger = get_logger("retriever")


class IndexNotBuilt(RuntimeError):
    """索引尚未构建，提示用户先跑 build_index.py。"""


@dataclass
class Retrieved:
    """一条检索结果及其双通道溯源信息。"""
    chunk: dict
    rrf_score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    bm25_score: float | None = None

    @property
    def idx(self) -> int:
        return self.chunk["_row"]

    @property
    def source(self) -> str:
        """命中来源，用于前端标注是向量、BM25 还是双路都命中。"""
        if self.vector_rank is not None and self.bm25_rank is not None:
            return "双路命中"
        return "向量" if self.vector_rank is not None else "BM25"

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.chunk.items() if not k.startswith("_")}
        d.update(rrf_score=round(self.rrf_score, 6), vector_rank=self.vector_rank,
                 bm25_rank=self.bm25_rank,
                 vector_score=None if self.vector_score is None else round(float(self.vector_score), 4),
                 bm25_score=None if self.bm25_score is None else round(float(self.bm25_score), 4),
                 source=self.source)
        return d


@dataclass
class RetrievalTrace:
    """一次检索的完整中间态，供阶段五做"检索动态过程"可视化。"""
    query: str
    query_vector: list[float] = field(default_factory=list)
    query_xy: tuple[float, float] | None = None
    vector_hits: list[dict] = field(default_factory=list)   # [{row, score, rank, xy}]
    bm25_hits: list[dict] = field(default_factory=list)
    fused: list[Retrieved] = field(default_factory=list)
    bm25_tokens: list[str] = field(default_factory=list)
    top_k_vector: int = 0
    top_k_bm25: int = 0
    rrf_k: int = 0
    balanced: bool = False          # 最后一轮是否走了"跨公司配额轮转"
    company_counts: dict = field(default_factory=dict)  # 最终 TopN 的公司分布
    pool_counts: dict = field(default_factory=dict)     # 融合候选池的公司分布（配额前）
    per_company_channel: bool = False                   # 两路是否按公司分别召回后并集


def merge_channel_hits(hit_lists: list[list[tuple[int, float]]]
                       ) -> list[tuple[int, float]]:
    """把"按公司分别取 Top-N"的若干路结果并成一个全局榜。

    同一行被多路取到时保留最高分（各公司行互不相交，这里主要是防御性处理），
    然后**按分数全局重排**再返回 —— 这一步不能省：如果只是简单首尾相接，
    各家自己的第 1 名会一起占到"全局第 1 名"的位置，RRF 的 1/(k+rank)
    就失去了可比性，融合分会失真。
    """
    best: dict[int, float] = {}
    for hits in hit_lists:
        for row, score in hits:
            if row not in best or score > best[row]:
                best[row] = score
    return sorted(best.items(), key=lambda kv: -kv[1])


def balance_by_company(cands, top_k_final: int, per_company_k: int) -> list:
    """跨公司配额轮转：先让每家公司各占 1 个名额，再发第 2 轮、第 3 轮…

    动机见 config.BALANCE_MULTI_COMPANY 的注释：对比题里 RRF 分数会被
    "命中词最多的那一家"吃掉，若不干预，Top-N 会被单公司垄断，其余公司
    一块都进不来，跨公司对比在检索阶段就已经失败了。

    注意：本函数**不动 rrf_score**，只决定入围顺序。所以界面上看到的分数
    仍是真实 RRF 分，但顺序不再严格单调 —— 这是配额选取的代价，须在 UI 里
    明确标注，不能让人误以为是算错了。
    """
    buckets: dict[str, list] = {}
    for r in cands:
        buckets.setdefault(r.chunk["company"], []).append(r)
    for b in buckets.values():
        b.sort(key=lambda x: -x.rrf_score)
    # 公司出场顺序 = 各桶最高分降序，使结果与输入顺序无关（可复现）
    order = sorted(buckets, key=lambda c: -buckets[c][0].rrf_score)

    picked: list = []
    quota: dict[str, int] = {}
    while len(picked) < top_k_final:
        progressed = False
        for c in order:
            if len(picked) >= top_k_final:
                break
            k = quota.get(c, 0)
            if per_company_k and k >= per_company_k:
                continue
            if k < len(buckets[c]):
                picked.append(buckets[c][k])
                quota[c] = k + 1
                progressed = True
        if not progressed:      # 所有桶都取空了
            break
    return picked


class HybridRetriever:
    """加载 index/ 下的双通道索引，提供检索接口。模型按需惰性加载。"""

    def __init__(self, index_dir: Path | None = None, model_name: str | None = None,
                 device: str | None = None, chunks_file: Path | None = None) -> None:
        self.index_dir = Path(index_dir or config.INDEX_DIR)
        self.model_name = model_name or config.EMBED_MODEL
        self.device = device
        self.chunks_file = Path(chunks_file or config.CHUNK_FILE)

        self._model = None
        self._bm25 = None
        self._chunks_by_id: dict[str, dict] | None = None

        self._load()

    # ------------------------------------------------------------------ 加载
    def _load(self) -> None:
        import numpy as np

        if not config.EMBEDDINGS_FILE.exists():
            raise IndexNotBuilt(
                f"未找到 {config.EMBEDDINGS_FILE}。请先运行：python build_index.py")

        self.embeddings = np.load(config.EMBEDDINGS_FILE, mmap_mode="r")
        self.chunk_ids: list[str] = json.loads(
            config.CHUNK_IDS_FILE.read_text(encoding="utf-8"))

        # PCA 底图参数（把查询向量投到同一张散点图上，必须复用建库时的成分）
        pca = np.load(config.PCA_FILE)
        self.pca_mean = pca["mean"]              # (D,)
        self.pca_components = pca["components"]  # (PCA_DIMS, D)
        self.proj = pca["proj"]                  # (N, PCA_DIMS)

        with open(config.BM25_FILE, "rb") as fh:
            bundle = pickle.load(fh)
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(bundle["tokens"], k1=bundle["k1"], b=bundle["b"])

        # 与知识库对齐检查：chunks.jsonl 变了而索引没重建，会静默错位
        if len(self.chunk_ids) != self.embeddings.shape[0]:
            raise IndexNotBuilt("索引元数据与向量矩阵行数不一致，请重新运行 build_index.py")
        if len(self.proj) != self.embeddings.shape[0]:
            raise IndexNotBuilt("PCA 底图与向量矩阵行数不一致，请重新运行 build_index.py")

        logger.info("索引载入：%d 个文本块，向量维度 %d，BM25 词表 %d",
                    len(self.chunk_ids), self.embeddings.shape[1], len(self._bm25.idf))

    def _ensure_chunks(self) -> dict[str, dict]:
        """按 chunk_id 建索引，并回填向量矩阵行号 `_row`。"""
        if self._chunks_by_id is not None:
            return self._chunks_by_id
        if not self.chunks_file.exists():
            raise IndexNotBuilt(f"未找到知识库 {self.chunks_file}，请先运行 build_kb.py")

        rows = [json.loads(line) for line in
                self.chunks_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {r["chunk_id"]: r for r in rows}

        missing = [cid for cid in self.chunk_ids if cid not in by_id]
        if missing:
            raise IndexNotBuilt(
                f"索引中有 {len(missing)} 个 chunk_id 在 chunks.jsonl 中不存在"
                f"（例：{missing[:3]}）。知识库已重建，请重新运行 build_index.py")

        for row_no, cid in enumerate(self.chunk_ids):
            by_id[cid]["_row"] = row_no
        self._chunks_by_id = by_id
        return by_id

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("加载本地向量模型 %s（首次运行会从 HuggingFace 下载权重）...",
                        self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = config.EMBED_MAX_LEN
            logger.info("向量模型就绪")
        return self._model

    # -------------------------------------------------------------- 编码工具
    def encode_query(self, query: str):
        import numpy as np
        model = self._ensure_model()
        text = config.EMBED_QUERY_TEMPLATE.format(
            instruction=config.EMBED_QUERY_INSTRUCTION, query=query)
        vec = model.encode([text], normalize_embeddings=True,
                           convert_to_numpy=True)[0]
        return np.asarray(vec, dtype="float32")

    def project(self, vector) -> tuple[float, float]:
        """把任意向量投到与语料散点图同一张 PCA 底图上，返回 2D 坐标。"""
        z = (vector - self.pca_mean) @ self.pca_components.T
        return float(z[0]), float(z[1])

    # ------------------------------------------------------------------ 检索
    def vector_search(self, query: str, top_k: int = config.TOP_K_VECTOR,
                      allowed: list[int] | None = None,
                      qvec=None) -> list[tuple[int, float]]:
        import numpy as np
        # qvec 可传入：多公司问题要按公司分别检索，同一问题编码 N 次纯属浪费
        q = self.encode_query(query) if qvec is None else qvec
        emb = self.embeddings
        if allowed is None:
            scores = np.asarray(emb @ q, dtype="float32")
            rows = np.arange(len(scores))
        else:
            rows = np.asarray(allowed, dtype="int64")
            # 归一化向量内积 = 余弦相似度，越大越相关
            scores = np.asarray(emb[rows] @ q, dtype="float32")
        top = min(top_k, len(rows))
        order = np.argsort(-scores, kind="stable")[:top]
        return [(int(rows[i]), float(scores[i])) for i in order]

    def bm25_search(self, query: str, top_k: int = config.TOP_K_BM25,
                    allowed: list[int] | None = None,
                    tokens: list[str] | None = None
                    ) -> tuple[list[tuple[int, float]], list[str]]:
        # tokens 同理可传入，避免按公司重复分词与重复算分
        tokens = tokenize(query) if tokens is None else tokens
        if not tokens:
            return [], []
        scores = self._bm25.get_scores(tokens)
        if allowed is None:
            rows = list(range(len(scores)))
        else:
            rows = list(allowed)
        ranked = sorted(((int(r), float(scores[r])) for r in rows),
                        key=lambda x: -x[1])
        ranked = [x for x in ranked[:top_k] if x[1] > 0]
        return ranked, tokens

    def search(self, query: str, top_k_final: int = config.TOP_K_FINAL,
               top_k_vector: int = config.TOP_K_VECTOR,
               top_k_bm25: int = config.TOP_K_BM25, rrf_k: int = config.RRF_K,
               companies: list[str] | None = None) -> list[Retrieved]:
        """双通道检索 + RRF 融合。"""
        return self.search_trace(query, top_k_final=top_k_final,
                                 top_k_vector=top_k_vector, top_k_bm25=top_k_bm25,
                                 rrf_k=rrf_k, companies=companies).fused

    def search_trace(self, query: str, top_k_final: int = config.TOP_K_FINAL,
                     top_k_vector: int = config.TOP_K_VECTOR,
                     top_k_bm25: int = config.TOP_K_BM25, rrf_k: int = config.RRF_K,
                     companies: list[str] | None = None) -> RetrievalTrace:
        """同 search()，但返回完整中间态（含查询向量与两路原始命中）供可视化。"""
        by_id = self._ensure_chunks()

        allowed: list[int] | None = None
        if companies:
            wanted = set(companies)
            allowed = [i for i, cid in enumerate(self.chunk_ids)
                       if by_id[cid]["company"] in wanted]
            if not allowed:
                logger.warning("公司过滤 %s 未命中任何文本块，回退到全库检索", companies)
                allowed = None

        # `companies` 这一个参数扛了两个角色，必须拆开：
        #   ① 检索白名单 —— 侧边栏「公司范围」，界面上默认 10 家全选；
        #   ② "这道题在问哪几家" —— 决定配额轮转该对着谁轮。
        # 之前把 ② 直接当成 ①，于是问「茅台、五粮液、老窖」时是**对着 10 家轮转**：
        # 最终只送 6 块，10 家人均 0.6 块，实测茅台/老窖/汾酒/今世缘一块都没进 ——
        # 模型只能回"检索结果中未出现贵州茅台任何数据"。看着像检索坏了，
        # 其实是配额把名额发给了根本没被问到的公司。
        asked = [c for c in (companies or []) if c in query]
        targets = asked or list(companies or [])
        multi_company = len(set(targets)) > 1

        qv = self.encode_query(query)
        tokens = tokenize(query)

        # 多公司问题：两个通道都改成"按公司各取 Top-N 再并集"，
        # 否则全局 top-k 截断会把排名靠后的公司整个挤出候选池，
        # 融合层的配额轮转就无从下手（它只能重排已有候选，造不出新候选）。
        per_company = bool(multi_company
                           and config.BALANCE_MULTI_COMPANY
                           and config.PER_COMPANY_CHANNEL_K > 0
                           and allowed is not None)
        buckets: dict[str, list[int]] = {}
        if per_company:
            # 只给**被问到的那几家**分桶取材。白名单里没被问到的公司仍然可以
            # 通过下面的普通检索进来，只是不占配额。
            wanted = set(targets)
            for i in allowed:
                c = by_id[self.chunk_ids[i]]["company"]
                if c in wanted:
                    buckets.setdefault(c, []).append(i)
            per_company = len(buckets) > 1        # 被问的公司没有块，退回普通检索

        if per_company:
            k = config.PER_COMPANY_CHANNEL_K
            vhits = merge_channel_hits(
                [self.vector_search(query, k, rows, qvec=qv) for rows in buckets.values()])
            bhits = merge_channel_hits(
                [self.bm25_search(query, k, rows, tokens=tokens)[0]
                 for rows in buckets.values()])
        else:
            vhits = self.vector_search(query, top_k_vector, allowed, qvec=qv)
            bhits, _ = self.bm25_search(query, top_k_bm25, allowed, tokens=tokens)

        trace = RetrievalTrace(
            query=query,
            query_vector=[float(x) for x in qv],
            query_xy=self.project(qv),
            bm25_tokens=tokens,
            top_k_vector=top_k_vector, top_k_bm25=top_k_bm25, rrf_k=rrf_k,
        )

        fused: dict[int, Retrieved] = {}

        def _ensure(row: int) -> Retrieved:
            if row not in fused:
                fused[row] = Retrieved(chunk=by_id[self.chunk_ids[row]])
            return fused[row]

        for rank, (row, score) in enumerate(vhits, start=1):
            r = _ensure(row)
            r.vector_rank, r.vector_score = rank, score
            r.rrf_score += 1.0 / (rrf_k + rank)
            trace.vector_hits.append({
                "row": row, "rank": rank, "score": score, "xy": self.xy_of(row),
                "chunk_id": self.chunk_ids[row], "company": r.chunk["company"],
                "section": r.chunk["section"], "page_start": r.chunk["page_start"],
                "page_end": r.chunk["page_end"]})

        for rank, (row, score) in enumerate(bhits, start=1):
            r = _ensure(row)
            r.bm25_rank, r.bm25_score = rank, score
            r.rrf_score += 1.0 / (rrf_k + rank)
            trace.bm25_hits.append({
                "row": row, "rank": rank, "score": score, "xy": self.xy_of(row),
                "chunk_id": self.chunk_ids[row], "company": r.chunk["company"],
                "section": r.chunk["section"], "page_start": r.chunk["page_start"],
                "page_end": r.chunk["page_end"]})

        ordered = sorted(fused.values(), key=lambda x: -x.rrf_score)

        # 候选池的公司分布（配额选取之前）。这是排查"某家公司一块都没进"的关键证据：
        #   池子里就没有  → 检索层问题，得调 PER_COMPANY_CHANNEL_K / 分词 / 编码
        #   池子里有但没选中 → 融合层问题，得调 PER_COMPANY_K / top_k_final
        # 两种情况的修法完全不同，所以在 trace 里把池子分布单独留下来。
        trace.pool_counts = {}
        for r in ordered:
            c = r.chunk["company"]
            trace.pool_counts[c] = trace.pool_counts.get(c, 0) + 1

        # 融合层轮转由 BALANCE_MULTI_COMPANY 单独控制，与上面候选池那层互不依赖：
        # PER_COMPANY_CHANNEL_K=0 关掉的是"每家公司都进池"，不是"最终每家公司都有配额"。
        # 轮转的判定同样按"被问到的公司"（targets），不按检索范围 —— 理由见上面 asked 那段。
        multi = bool(multi_company and config.BALANCE_MULTI_COMPANY)
        if multi:
            ordered = balance_by_company(ordered, top_k_final, config.PER_COMPANY_K)
        else:
            ordered = ordered[:top_k_final]

        trace.fused = ordered
        # balanced = 走了融合层轮转；per_company_channel = 两路也按公司分别召回。
        # 这是两个独立机制，分开记录，便于单独衡量各自的效果。
        trace.balanced = multi
        trace.per_company_channel = per_company
        trace.company_counts = {}
        for r in ordered:
            c = r.chunk["company"]
            trace.company_counts[c] = trace.company_counts.get(c, 0) + 1
        return trace

    # ------------------------------------------------------------------ 底图
    def xy_of(self, row: int) -> tuple[float, float]:
        return float(self.proj[row, 0]), float(self.proj[row, 1])

    def all_xy(self):
        return self.proj[:, 0], self.proj[:, 1]

    def chunk_at(self, row: int) -> dict:
        return self._ensure_chunks()[self.chunk_ids[row]]
