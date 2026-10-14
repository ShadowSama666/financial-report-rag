# -*- coding: utf-8 -*-
"""阶段四：构建双通道混合索引（本地向量索引 + BM25 倒排索引）。

用法
----
    python build_index.py                     # 全量构建
    python build_index.py --limit 200         # 只索引前 200 块（冒烟测试）
    python build_index.py --device cuda       # 有 GPU 时加速编码
    python build_index.py --no-vector         # 只建 BM25，跳过模型下载
    python build_index.py --rebuild           # 忽略向量缓存，全部重编码

产物（index/ 目录整体被 .gitignore 排除）
----
    index/embeddings.npy       (N, D) float32，已 L2 归一化 —— 内积即余弦相似度
    index/chunk_ids.json       第 i 行向量对应的 chunk_id（与 chunks.jsonl 对齐）
    index/bm25.pkl             jieba 分词后的倒排语料 + BM25 参数
    index/pca.npz              PCA 底图（mean / components / 50 维投影）
    index/embed_cache.npz      文本哈希 -> 向量 的增量缓存，改切块参数后可秒级重建
    index/index_manifest.json  本次构建的口径快照（模型名、维度、块数、耗时）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import config
from utils.common import get_logger, load_env, setup_console
from utils.tokenizer import init_jieba, tokenize

logger = get_logger("build_index")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建向量索引与 BM25 倒排索引")
    p.add_argument("--chunks", type=Path, default=config.CHUNK_FILE, help="知识库文件")
    p.add_argument("--model", default=config.EMBED_MODEL, help="本地向量模型名或路径")
    p.add_argument("--device", default=None, help="cpu / cuda / mps，默认自动")
    p.add_argument("--batch", type=int, default=config.EMBED_BATCH)
    p.add_argument("--limit", type=int, default=None, help="只索引前 N 块（调试用）")
    p.add_argument("--no-vector", action="store_true", help="只建 BM25 索引")
    p.add_argument("--rebuild", action="store_true", help="忽略向量缓存，全部重编码")
    p.add_argument("--threads", type=int, default=None,
                   help="torch CPU 线程数。默认交给 torch；混合架构 CPU（P核+E核）"
                        "上建议显式设成 P 核数，线程超订会明显拖慢")
    p.add_argument("--checkpoint-every", type=int, default=256,
                   help="每编码多少块就把缓存落盘一次（长任务防丢进度，默认 256）")
    return p.parse_args()


# --------------------------------------------------------------------------
def load_chunks(path: Path, limit: int | None) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"未找到知识库 {path}，请先运行：python build_kb.py")
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} 是空文件，请重新运行 build_kb.py")
    if limit:
        rows = rows[:limit]
    return rows


def corpus_fingerprint(rows: list[dict], model_name: str) -> str:
    """语料 + 模型指纹：块内容或模型变了，索引就必须重建。"""
    h = hashlib.sha1(model_name.encode("utf-8"))
    for r in rows:
        h.update(r["chunk_id"].encode("utf-8"))
        h.update(hashlib.sha1(r["text"].encode("utf-8")).digest())
    return h.hexdigest()[:16]


def text_key(text: str, model_name: str) -> str:
    return hashlib.sha1(f"{model_name}\x00{text}".encode("utf-8")).hexdigest()


def load_embed_cache(path: Path) -> dict[str, "object"]:
    import numpy as np
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception as exc:
        logger.warning("向量缓存读取失败（%s），将忽略缓存重新编码", exc)
        return {}


def save_embed_cache(path: Path, cache: dict) -> None:
    import numpy as np
    if cache:
        np.savez_compressed(path, **cache)


# --------------------------------------------------------------------------
def build_vector_index(rows: list[dict], args: argparse.Namespace):
    """编码文本块。返回 (embeddings, 命中缓存的块数)。"""
    import numpy as np

    field = "tagged_text" if config.USE_TAGGED_TEXT_FOR_EMBED else "text"
    texts = [r[field] for r in rows]
    keys = [text_key(t, args.model) for t in texts]

    cache_path = config.INDEX_DIR / "embed_cache.npz"
    cache = {} if args.rebuild else load_embed_cache(cache_path)
    hit = sum(1 for k in keys if k in cache)
    logger.info("向量缓存命中 %d / %d", hit, len(keys))

    missing = [i for i, k in enumerate(keys) if k not in cache]
    if missing:
        import torch
        if args.threads:
            # 混合架构 CPU（P核+E核）上线程超订会被 E 核拖住，显式设定更稳
            torch.set_num_threads(args.threads)
            try:
                torch.set_num_interop_threads(1)   # 小块推理用不上 inter-op 并行
            except RuntimeError:
                pass                              # 已在并行区之后调用，忽略
            logger.info("torch 线程数设为 %d（inter-op 1）", args.threads)

        from sentence_transformers import SentenceTransformer
        logger.info("加载本地向量模型 %s（首次运行会自动下载权重 ~1.2GB）...", args.model)
        t0 = time.time()
        model = SentenceTransformer(args.model, device=args.device)
        model.max_seq_length = config.EMBED_MAX_LEN
        logger.info("模型就绪（%.1fs），开始编码 %d 个文本块，batch=%d",
                    time.time() - t0, len(missing), args.batch)

        t0 = time.time()
        ckpt = max(1, args.checkpoint_every)
        n_done = last_ckpt = 0
        for s in range(0, len(missing), args.batch):
            idxs = missing[s:s + args.batch]
            vecs = model.encode([texts[i] for i in idxs],
                                batch_size=args.batch,
                                normalize_embeddings=True,
                                convert_to_numpy=True,
                                show_progress_bar=False)
            for i, v in zip(idxs, np.asarray(vecs, dtype="float32")):
                cache[keys[i]] = v
            n_done += len(idxs)

            # 落盘是长任务唯一的进度保障。整库编码在普通 CPU 上是小时级的，
            # 若只在全部结束后才写缓存，中途断电/关机就前功尽弃。
            if n_done - last_ckpt >= ckpt:
                save_embed_cache(cache_path, cache)
                last_ckpt = n_done
            el = time.time() - t0
            rate = n_done / max(el, 1e-6)
            eta = (len(missing) - n_done) / max(rate, 1e-6) / 60
            print(f"  编码进度 {n_done}/{len(missing)} ({n_done/len(missing):5.1%})"
                  f"  已用 {el/60:5.1f}分  速度 {rate:4.2f} 块/秒  预计还需 {eta:5.1f} 分",
                  flush=True)
        save_embed_cache(cache_path, cache)
        logger.info("编码完成，耗时 %.1f 分钟，缓存已写入 embed_cache.npz",
                    (time.time() - t0) / 60)

    emb = np.stack([cache[k] for k in keys]).astype("float32")
    # 双保险：缓存可能来自不同归一化设置，统一再归一化一次
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    np.divide(emb, np.maximum(norms, 1e-9), out=emb)
    return emb, hit


def build_pca(embeddings):
    """拟合 PCA 底图：既降低可视化维度，又给查询向量准备了可复用的投影矩阵。"""
    import numpy as np
    from sklearn.decomposition import PCA

    n_dims = min(config.PCA_DIMS, embeddings.shape[0] - 1, embeddings.shape[1])
    logger.info("拟合 PCA 底图：%d 维 -> %d 维", embeddings.shape[1], n_dims)
    pca = PCA(n_components=n_dims, random_state=42)
    proj = pca.fit_transform(embeddings).astype("float32")
    logger.info("前两维累计解释方差 %.1f%%", pca.explained_variance_ratio_[:2].sum() * 100)
    return pca, proj, n_dims


def build_bm25(rows: list[dict]):
    """jieba 分词建倒排。返回 (tokenized_corpus, 平均词数)。"""
    init_jieba()
    field = "tagged_text" if config.USE_TAGGED_TEXT_FOR_BM25 else "text"
    tokenized = []
    try:
        from tqdm import tqdm
        it = tqdm(rows, desc="  jieba 分词", unit="块", ncols=88, leave=False)
    except ImportError:
        it = rows
    for r in it:
        tokenized.append(tokenize(r[field]))
    avg = sum(len(t) for t in tokenized) / max(1, len(tokenized))
    vocab = len({w for t in tokenized for w in t})
    logger.info("分词完成：平均 %.0f 词/块，词表 %d", avg, vocab)
    return tokenized, avg


# --------------------------------------------------------------------------
def main() -> int:
    setup_console()
    load_env()
    args = parse_args()
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    rows = load_chunks(args.chunks, args.limit)
    print(f"\n知识库载入：{len(rows)} 个文本块")
    fingerprint = corpus_fingerprint(rows, args.model)
    logger.info("语料指纹 %s", fingerprint)

    chunk_ids = [r["chunk_id"] for r in rows]
    config.CHUNK_IDS_FILE.write_text(
        json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")

    dim, cache_hit, pca_dims = 0, 0, 0
    if args.no_vector:
        logger.info("--no-vector：跳过向量索引，仅构建 BM25")
        config.PCA_FILE.unlink(missing_ok=True)
    else:
        logger.info("步骤 1/3：构建向量索引（%s）...", args.model)
        embeddings, cache_hit = build_vector_index(rows, args)
        dim = embeddings.shape[1]
        import numpy as np
        np.save(config.EMBEDDINGS_FILE, embeddings)
        logger.info("向量矩阵已落盘：%s  shape=%s  %.1fMB",
                    config.EMBEDDINGS_FILE.name, embeddings.shape,
                    embeddings.nbytes / 1024 / 1024)

        logger.info("步骤 2/3：拟合 PCA 底图 ...")
        pca, proj, pca_dims = build_pca(embeddings)
        np.savez_compressed(config.PCA_FILE, mean=pca.mean_.astype("float32"),
                            components=pca.components_.astype("float32"), proj=proj)

    logger.info("步骤 3/3：构建 BM25 倒排索引 ...")
    import pickle
    tokenized, avg_tokens = build_bm25(rows)
    with open(config.BM25_FILE, "wb") as fh:
        pickle.dump({"tokens": tokenized, "ids": chunk_ids,
                     "k1": config.BM25_K1, "b": config.BM25_B,
                     "model": args.model}, fh)
    logger.info("BM25 索引已落盘：%s", config.BM25_FILE.name)

    elapsed = time.time() - t_start
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_fingerprint": fingerprint,
        "n_chunks": len(rows),
        "companies": sorted({r["company"] for r in rows}),
        "embed_model": None if args.no_vector else args.model,
        "embed_dim": dim,
        "embed_field": "tagged_text" if config.USE_TAGGED_TEXT_FOR_EMBED else "text",
        "embed_cache_hit": cache_hit,
        "pca_dims": pca_dims or None,
        "bm25": {"k1": config.BM25_K1, "b": config.BM25_B,
                 "field": "tagged_text" if config.USE_TAGGED_TEXT_FOR_BM25 else "text",
                 "avg_tokens": round(avg_tokens, 1),
                 "vocab": len({w for t in tokenized for w in t})},
        "elapsed_sec": round(elapsed, 1),
        "chunks_file": str(args.chunks),
    }
    config.INDEX_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"索引构建完成，耗时 {elapsed:.1f}s")
    print(f"  文本块    ：{len(rows)}")
    print(f"  向量维度  ：{dim or '（已跳过）'}"
          + (f"   覆盖 {len(manifest['companies'])} 家公司" if dim else ""))
    print(f"  BM25 词表 ：{manifest['bm25']['vocab']}（jieba + 行业自定义词典）")
    print(f"  BM25 参数 ：k1={config.BM25_K1}  b={config.BM25_B}")
    print(f"  索引目录  ：{config.INDEX_DIR}")
    print("=" * 70)
    print("\n下一步：python -m streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
