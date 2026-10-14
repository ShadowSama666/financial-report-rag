# -*- coding: utf-8 -*-
"""阶段五：申万食品饮料半年报知识库 —— 可视化问答前端（Streamlit）。

三个演示面板
------------
1. 问答            : 提问 -> 双通道检索 -> DeepSeek 流式作答 -> 带【角标】的严格溯源
2. 文本块星图      : 2717 个文本块降维散点图，点击任意点回显原文/公司/章节/页码
3. 检索动态过程    : 问题怎么变成向量、落到图的哪个位置、余弦邻居怎么被召回、
                     RRF 怎么把两路结果合成最终上下文，逐步动画呈现

配色遵循"按身份/量级分派 + 脚本校验"的方法：散点图是 all-pairs 形态，
默认色板在 all-pairs 下只有前 3 个槽位达标，故身份维度只用 3 个分类色
（白酒/啤酒/大众品），公司级身份交给分面、交互与明细表承载。详见 utils/palette.py。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from utils import palette as P
from utils.common import load_env, page_label
from utils.llm import (DeepSeekClient, DeepSeekError, build_messages,
                       parse_citations)
from utils.retriever import HybridRetriever, IndexNotBuilt, RetrievalTrace

load_env()

st.set_page_config(page_title="申万食品饮料·半年报问答知识库", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")


# ==========================================================================
# 数据加载（全部走缓存，避免每次交互重读磁盘 / 重载模型）
# ==========================================================================
@st.cache_resource(show_spinner=False)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    if config.INDEX_MANIFEST.exists():
        return json.loads(config.INDEX_MANIFEST.read_text(encoding="utf-8"))
    return {}


@st.cache_data(show_spinner="载入文本块与降维底图 ...")
def load_corpus(chunks_file: str, pca_file: str) -> pd.DataFrame:
    """把 chunks.jsonl 与 PCA 底图拼成一张分析用宽表。"""
    rows = [json.loads(l) for l in
            open(chunks_file, encoding="utf-8").read().splitlines() if l.strip()]
    proj = np.load(pca_file)["proj"]
    df = pd.DataFrame(rows)
    # `row` = chunks.jsonl 行序，与 build_index 写入的向量矩阵行号一一对应，
    # 是全应用唯一可靠的"文本块 ↔ 底图坐标 ↔ 索引行"关联键。
    df["row"] = np.arange(len(df))
    df["x"] = proj[:, 0]
    df["y"] = proj[:, 1]
    df["category"] = df["sw_l2"].map(P.category_of)
    # 页码口径走 utils.common.page_label，与出处角标、评测 recall_detail.csv 同一套
    df["page"] = [page_label(a, b) for a, b in zip(df["page_start"], df["page_end"])]
    return df


def t() -> dict:
    return P.tokens(st.session_state.get("theme", "light"))


def index_ready() -> bool:
    """索引四件套齐了才往下走。

    不用 `st.stop()`：在无会话上下文/被 try 包住的场景它不一定会中断脚本，
    会一路跑到 `load_corpus` 抛一个看不懂的 FileNotFoundError。这里显式 return。
    """
    missing = [p.name for p in (config.EMBEDDINGS_FILE, config.CHUNK_IDS_FILE,
                                config.PCA_FILE, config.BM25_FILE) if not p.exists()]
    if not missing:
        return True
    st.error(f"**索引不完整**，`index/` 下缺少：{'、'.join(missing)}")
    st.markdown("照顺序跑一遍即可（阶段二、三只需跑一次）：")
    st.code("python download.py\npython build_kb.py\npython build_index.py",
            language="bash")
    return False


# ---- Streamlit 版本兼容 ---------------------------------------------------
# 1.5x 起 `use_container_width=True` 被 `width="stretch"` 取代（旧写法先告警、后移除）。
# 课程环境的 Streamlit 版本不可控，这里第一次调用时两种写法各试一次并记住结果，
# 避免版本差异导致整页崩掉。
_STRETCH: dict[str, str] = {}


def stretch(fn, *args, **kwargs):
    kind = _STRETCH.get("kind")
    if kind == "old":
        return fn(*args, **kwargs, use_container_width=True)
    if kind == "new":
        return fn(*args, **kwargs, width="stretch")
    try:
        out = fn(*args, **kwargs, width="stretch")
        _STRETCH["kind"] = "new"
        return out
    except Exception:
        _STRETCH["kind"] = "old"
        return fn(*args, **kwargs, use_container_width=True)


# ==========================================================================
# 绘图
# ==========================================================================
def _corpus_customdata(sub: pd.DataFrame) -> np.ndarray:
    return np.stack([sub["chunk_id"], sub["company"], sub["page"],
                     sub["section"], sub["category"]], axis=-1)


CORPUS_HOVER = ("<b>%{customdata[1]}</b> · %{customdata[4]}<br>"
                "%{customdata[3]} · 第 %{customdata[2]} 页"
                "<extra>点击查看原文</extra>")


def build_explorer(df: pd.DataFrame, mode: str, color_by: str,
                   focus_company: str | None = None,
                   height: int = 580) -> go.Figure:
    """文本块星图。color_by='category' 用 3 大类着色；'company' 仅在范围内 <=3 家时启用。"""
    tok = t()
    smap = P.series_map(mode)
    fig = go.Figure()

    if color_by == "company" and focus_company:
        # 聚焦单家公司：底噪压成灰，该公司按大类色高亮（强调，而不是新增色相）
        rest = df[df["company"] != focus_company]
        sub = df[df["company"] == focus_company]
        fig.add_trace(go.Scattergl(
            x=rest["x"], y=rest["y"], mode="markers", name="其他公司",
            marker=dict(size=5, color=tok["corpus_muted"], opacity=0.5),
            customdata=_corpus_customdata(rest), hovertemplate=CORPUS_HOVER))
        fig.add_trace(go.Scattergl(
            x=sub["x"], y=sub["y"], mode="markers", name=focus_company,
            marker=dict(size=9, color=smap[sub["category"].iloc[0]], opacity=0.9,
                        line=dict(width=2, color=tok["surface"])),
            customdata=_corpus_customdata(sub), hovertemplate=CORPUS_HOVER))
    else:
        for name in P.CATEGORY_NAMES:
            sub = df[df["category"] == name]
            if sub.empty:
                continue
            fig.add_trace(go.Scattergl(
                x=sub["x"], y=sub["y"], mode="markers",
                name=f"{name} · {len(sub)} 块",
                marker=dict(size=6, color=smap[name], opacity=0.55),
                customdata=_corpus_customdata(sub), hovertemplate=CORPUS_HOVER))

    fig.update_layout(**P.plotly_layout(mode, height=height))
    fig.update_layout(margin=dict(l=40, r=20, t=52, b=40))
    fig.update_xaxes(title_text="PCA-1", showticklabels=False)
    fig.update_yaxes(title_text="PCA-2", showticklabels=False)
    return fig


def build_facets(df: pd.DataFrame, mode: str, ncols: int = 5, height: int = 520) -> go.Figure:
    """小倍数分面：每家公司一格。

    这就是"超过 3 类身份"的正解——不是造第 4 个色相，而是分面。
    每格里灰色是全库底噪，彩色是该公司的块，一眼就能看出
    "同一家公司的文本块在向量空间里自成团簇，且团簇位置与该大类相关"。
    """
    tok = t()
    smap = P.series_map(mode)
    companies = sorted(df["company"].unique())
    nrows = int(np.ceil(len(companies) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.015, vertical_spacing=0.045,
                        subplot_titles=companies)

    for i, comp in enumerate(companies):
        r, c = divmod(i, ncols)
        sub = df[df["company"] == comp]
        color = smap[sub["category"].iloc[0]]
        fig.add_trace(go.Scattergl(
            x=df["x"], y=df["y"], mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(size=3, color=tok["corpus_muted"], opacity=0.45)),
            row=r + 1, col=c + 1)
        fig.add_trace(go.Scattergl(
            x=sub["x"], y=sub["y"], mode="markers", showlegend=False,
            marker=dict(size=5, color=color, opacity=0.9),
            customdata=_corpus_customdata(sub), hovertemplate=CORPUS_HOVER),
            row=r + 1, col=c + 1)

    layout = P.plotly_layout(mode, height=height)
    layout.pop("legend")
    fig.update_layout(**layout)
    fig.update_layout(margin=dict(l=30, r=16, t=44, b=28),
                      title=dict(text="小倍数分面：每家公司的文本块分布（灰点为全库底噪）",
                                 font=dict(size=15, color=tok["text_primary"]), x=0.005))
    for ann in fig.layout.annotations:
        ann.font.size = 12
        ann.font.color = tok["text_secondary"]
    # plotly_layout 里的 xaxis/yaxis 只作用于第一个子图，分面图必须逐轴刷一遍，
    # 否则第 2..N 格会退回 plotly 默认网格色（深色模式下尤其明显）。
    for key in [k for k in fig.layout if k.startswith(("xaxis", "yaxis"))]:
        fig.layout[key].update(showgrid=True, gridcolor=tok["grid"], gridwidth=1,
                               zeroline=False, showline=False, ticks="",
                               showticklabels=False,
                               tickfont=dict(color=tok["muted"], size=11))
    return fig


def build_simple_animation(df: pd.DataFrame, mode: str, tr: RetrievalTrace,
                           height: int = 620) -> go.Figure:
    """检索过程动画（实现版）：静态语料 + 5 个定向更新的高亮 trace。"""
    tok = t()
    smap = P.series_map(mode)
    fig = go.Figure()

    for name in P.CATEGORY_NAMES:
        sub = df[df["category"] == name]
        fig.add_trace(go.Scattergl(
            x=sub["x"], y=sub["y"], mode="markers",
            name=f"{name} · {len(sub)} 块",
            marker=dict(size=5, color=smap[name], opacity=0.3),
            customdata=_corpus_customdata(sub), hovertemplate=CORPUS_HOVER))

    qx, qy = tr.query_xy

    # 每个 trace 都带上完整样式后再放进 Frame。Frame 会整体替换 trace 的属性，
    # 只写 x/y 的空 trace 会让该帧退回 plotly 默认色（#636efa）——那就凭空多出
    # 一个不在调色板里的色相，动画一播就露馅。
    def q_trace(x, y):
        return go.Scatter(x=x, y=y, mode="markers", name="① 问题向量落点",
                          marker=dict(size=18, symbol="star", color=tok["highlight"],
                                      line=dict(width=2, color=tok["surface"])))

    def l_trace(x, y):
        return go.Scatter(x=x, y=y, mode="lines", name="④ 问题→最近邻连线",
                          line=dict(width=2, color=tok["highlight"]), opacity=0.45)

    def hit_trace(hits, symbol, label):
        cd = np.stack([[h["company"], h["page_start"], h["section"], h["rank"],
                        round(h["score"], 3)] for h in hits], axis=-1) if hits \
            else np.empty((0, 5))
        return go.Scatter(
            x=[h["xy"][0] for h in hits], y=[h["xy"][1] for h in hits],
            mode="markers", name=label,
            marker=dict(size=11, symbol=symbol, color="rgba(0,0,0,0)",
                        line=dict(width=2, color=tok["highlight"])),
            customdata=cd,
            hovertemplate=("<b>%{customdata[0]}</b> · 第 %{customdata[1]} 页<br>"
                           "%{customdata[2]} · 通道排名 #%{customdata[3]} · 分数 %{customdata[4]}"
                           "<extra></extra>"))

    def f_trace(x, y, text, colors):
        return go.Scatter(x=x, y=y, mode="markers+text",
                          name="⑤ 融合后送入模型的 Top-N",
                          text=text, textposition="top center",
                          marker=dict(size=15, symbol="diamond", color=colors,
                                      line=dict(width=2, color=tok["surface"])),
                          textfont=dict(size=12, color=tok["text_primary"]))

    E_Q, E_V = q_trace([], []), hit_trace([], "circle", "② 向量通道 Top-K（圆点）")
    E_B = hit_trace([], "square", "③ BM25 通道 Top-K（方块）")
    E_L, E_F = l_trace([], []), f_trace([], [], [], [])

    for tr_static in (E_Q, E_V, E_B, E_L, E_F):      # 语料占 0..2，高亮从 3 开始
        fig.add_trace(tr_static)
    idx_q, idx_v, idx_b, idx_l, idx_f = 3, 4, 5, 6, 7

    # 全局坐标按 `_row` 取（= chunks.jsonl 行序，build_index 保证向量矩阵同序）。
    # 这里**不能**用 .iloc —— 传入的 df 可能已被公司筛选，位置索引会整体错位。
    xy = {int(r): (float(x), float(y))
          for r, x, y in zip(df["row"], df["x"], df["y"])}
    fused_pts = [(g, xy[g.idx]) for g in tr.fused if g.idx in xy]

    fq = q_trace([qx], [qy])
    fv = hit_trace(tr.vector_hits, "circle", "② 向量通道 Top-K（圆点）")
    fb = hit_trace(tr.bm25_hits, "square", "③ BM25 通道 Top-K（方块）")
    xs, ys = [], []
    for _, (px, py) in fused_pts:            # None 断开折线，否则连成一片蛛网
        xs += [qx, px, None]
        ys += [qy, py, None]
    fl = l_trace(xs, ys)
    ff = f_trace([pt[0] for _, pt in fused_pts], [pt[1] for _, pt in fused_pts],
                 [str(i + 1) for i in range(len(fused_pts))],
                 [smap[P.category_of(g.chunk["sw_l2"])] for g, _ in fused_pts])

    steps = [
        (0, "① 原始语料：%d 个文本块在 PCA 平面上的分布" % len(df), E_Q, E_V, E_B, E_L, E_F),
        (1, "② 问题被编码成 1024 维向量，投影落图", fq, E_V, E_B, E_L, E_F),
        (2, "③ 向量通道：余弦相似度召回 Top-%d（圆点）" % tr.top_k_vector, fq, fv, E_B, E_L, E_F),
        (3, "④ BM25 通道：jieba 倒排召回 Top-%d（方块）" % tr.top_k_bm25, fq, fv, fb, E_L, E_F),
        (4, "⑤ 两路结果落到同一张图上，位置互补", fq, fv, fb, fl, E_F),
        (5, "⑥ RRF 融合后送入 DeepSeek 的 Top-%d（菱形+序号）" % len(tr.fused), fq, fv, fb, fl, ff),
    ]

    frames = []
    for step, label, a, b, c, d, e in steps:
        frames.append(go.Frame(
            name=label, data=[a, b, c, d, e], traces=[idx_q, idx_v, idx_b, idx_l, idx_f]))

    fig.frames = frames
    fig.update_layout(**P.plotly_layout(mode, height=height))
    fig.update_layout(
        margin=dict(l=40, r=20, t=104, b=40),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=1.14, xanchor="left",
            showactive=False, bgcolor=tok["surface"], bordercolor=tok["border"],
            font=dict(color=tok["text_primary"], size=12),
            buttons=[
                dict(label="▶ 播放检索过程", method="animate",
                     args=[None, dict(frame=dict(duration=900, redraw=True),
                                      transition=dict(duration=250),
                                      fromcurrent=False)]),
                dict(label="⏭ 跳到末帧", method="animate",
                     args=[[frames[-1].name], dict(mode="immediate",
                                                   frame=dict(duration=0, redraw=True))]),
            ])],
        sliders=[dict(
            active=0, x=0.28, len=0.72, y=1.14, pad=dict(t=0, b=0),
            currentvalue=dict(prefix="步骤：", font=dict(size=12, color=tok["text_secondary"])),
            steps=[dict(label=f"{i+1}", method="animate",
                        args=[[f.name], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True))])
                   for i, f in enumerate(frames)])],
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


# ==========================================================================
# 侧边栏：全局筛选与参数（一处筛选，全应用生效）
# ==========================================================================
def sidebar(df: pd.DataFrame, manifest: dict) -> dict:
    sb = st.sidebar
    sb.title("📊 半年报知识库")
    sb.caption("申万一级行业 · 食品饮料 · 10 家公司")

    theme = sb.radio("图表配色", ["浅色", "深色"], horizontal=True, key="theme_pick")
    st.session_state["theme"] = "dark" if theme == "深色" else "light"
    sb.caption("深色取自同一色阶的深色档位（独立校验，不是把浅色自动反转）；"
               "页面外壳保持浅色，避免与 Streamlit 自身主题冲突。")

    sb.divider()
    sb.subheader("筛选")
    all_companies = sorted(df["company"].unique())
    picked = sb.multiselect("公司范围", all_companies, default=all_companies,
                            help="筛选后，问答检索与星图都只在该范围内进行")
    if not picked:
        picked = all_companies
    focus = sb.selectbox("星图聚焦公司", ["（不聚焦）"] + picked)

    sb.divider()
    sb.subheader("检索参数")
    kv = sb.slider("向量通道 Top-K", 5, 50, config.TOP_K_VECTOR, 1)
    kb = sb.slider("BM25 通道 Top-K", 5, 50, config.TOP_K_BM25, 1)
    kf = sb.slider("最终送模型 Top-N", 2, 12, config.TOP_K_FINAL, 1)
    rrf = sb.slider("RRF 常数 k", 10, 100, config.RRF_K, 5,
                    help="score = Σ 1/(k + rank)，k 越大越平滑两路差异")

    sb.divider()
    sb.subheader("索引状态")
    if manifest:
        sb.metric("文本块", f"{manifest.get('n_chunks', 0):,}")
        sb.caption(
            f"向量模型：`{manifest.get('embed_model') or '未构建'}`\n\n"
            f"维度 {manifest.get('embed_dim', 0)} · 编码字段 `{manifest.get('embed_field')}`\n\n"
            f"BM25 词表 {manifest.get('bm25', {}).get('vocab', 0):,} · "
            f"k1={manifest.get('bm25', {}).get('k1')} b={manifest.get('bm25', {}).get('b')}"
            f"（索引字段 `{manifest.get('bm25', {}).get('field', 'text')}`）\n\n"
            f"构建于 {manifest.get('built_at', '—')}")
    else:
        sb.warning("尚未构建索引")

    client = DeepSeekClient()
    if client.available:
        sb.success("DeepSeek API Key 已就绪")
    else:
        sb.warning("未配置 DEEPSEEK_API_KEY\n\n检索功能可用，生成部分将跳过。")

    return dict(companies=picked, focus=None if focus.startswith("（") else focus,
                kv=kv, kb=kb, kf=kf, rrf=rrf)


# ==========================================================================
# 面板三：检索动态过程
# ==========================================================================
def render_pipeline(tr: RetrievalTrace) -> None:
    st.markdown("#### 问题 → 向量 → 降维落图 → 邻居召回 → 融合 → 生成")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**① 问题分词（BM25 通道的输入）**")
        st.code("  ".join(tr.bm25_tokens) or "（无有效词元）", language=None)
        st.caption("jieba + 行业自定义词典。BM25 靠这些词精确命中专有名词、年份与财务指标。")
    with c2:
        qv = np.asarray(tr.query_vector, dtype="float32")
        st.markdown("**② 问题向量（向量通道的输入）**")
        st.code(f"维度 = {qv.size}    L2 范数 = {np.linalg.norm(qv):.4f}\n"
                f"前 8 维 = [{', '.join(f'{v:+.4f}' for v in qv[:8])}, ...]", language=None)
        st.caption(f"编码前会套上指令模板：`{config.EMBED_QUERY_INSTRUCTION}`（非对称检索）")

    st.markdown("**③ 降维落图**")
    st.markdown(
        f"问题向量先与查询无关地被投到建库时拟合的 PCA 子空间："
        f"`z = (q - mean) @ components.T` → 取前两维 → **({tr.query_xy[0]:+.3f}, {tr.query_xy[1]:+.3f})**"
        f"。语料 2717 个块用的是**同一套** `mean/components`，"
        f"所以问题落点与召回邻居在同一张图上是**可比**的——这是散点图能用来解释检索的前提。")

    st.markdown("**④ 两通道召回 + RRF 融合（真实算式）**")
    rk = tr.rrf_k
    rows = []
    for i, r in enumerate(tr.fused, start=1):
        vr, br = r.vector_rank, r.bm25_rank
        v_part = 1.0 / (rk + vr) if vr else 0.0
        b_part = 1.0 / (rk + br) if br else 0.0
        rows.append({
            "送入序号": i,
            "公司": r.chunk["company"],
            "章节": r.chunk["section"],
            "页码": r.chunk["page_start"] if r.chunk["page_start"] == r.chunk["page_end"]
                    else f"{r.chunk['page_start']}-{r.chunk['page_end']}",
            "命中来源": r.source,
            "向量排名": f"#{vr}" if vr else "—",
            "1/(k+rank_v)": f"{v_part:.5f}" if vr else "—",
            "BM25 排名": f"#{br}" if br else "—",
            "1/(k+rank_b)": f"{b_part:.5f}" if br else "—",
            "RRF 总分": f"{r.rrf_score:.5f}",
            "余弦": f"{r.vector_score:.4f}" if r.vector_score is not None else "—",
            "BM25 分": f"{r.bm25_score:.3f}" if r.bm25_score is not None else "—",
        })
    stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)
    st.caption(
        "只融合**排名**、不融合原始分数：余弦相似度（0~1 有界）与 BM25 分数（无上界、随语料变化）"
        "尺度不可比，直接加权会隐性放大大分数量纲的那一路。"
        f"RRF 用 1/(k+rank) 把两路压到同一尺度，k={rk}。"
        "「双路命中」的块得分天然更高——这正是混合检索优于单通道的地方。")

    both = sum(1 for r in tr.fused if r.vector_rank and r.bm25_rank)
    st.markdown(
        f"最终 Top-{len(tr.fused)} 中，**{both}** 个块被两路同时命中。"
        f"向量通道召回 {len(tr.vector_hits)} 个、BM25 通道召回 {len(tr.bm25_hits)} 个。")

    if tr.balanced:
        dist = "、".join(f"{c} {n} 块" for c, n in tr.company_counts.items())
        pool = "、".join(f"{c} {n} 块" for c, n in tr.pool_counts.items())
        st.info(
            "**本轮启用了跨公司配额轮转**（问题涉及多家公司）。\n\n"
            f"候选池分布（配额前）：{pool}\n\n"
            f"最终配额：{dist}\n\n"
            "为什么需要它：对比题的提问里同时出现好几家公司名，"
            "RRF 分数会被「命中词最多的那一家」吃掉——实测有一轮 Top-8 全落在同一家公司，"
            "经 Top-N 收口后其余公司一块都进不来，对比题在**检索阶段**就已经退化成单公司题。"
            "配额轮转先按公司分桶、桶内按 RRF 降序，再轮转取（每家先取第 1 名，再取第 2 名…），"
            "保证每家都有名额。\n\n"
            "**注意上表的「送入序号」不再严格按 RRF 总分单调递减**——"
            "`RRF 总分` 列仍是未改动的真实分数，改的只是入围顺序。"
            f"每家最多 {config.PER_COMPANY_K} 块（`config.PER_COMPANY_K`）。")

        if tr.per_company_channel:
            st.info(
                "**本轮两路召回也按公司分别取材**"
                f"（每家公司每个通道各取 Top-{config.PER_COMPANY_CHANNEL_K}，再并集重排）。\n\n"
                "只做上面的融合层轮转是不够的：`TOP_K_VECTOR` / `TOP_K_BM25` 是在融合"
                "**之前**按全池截断的，而且公司之间不加区分。实测有一道四公司对比题里，"
                "其中一家向量排到第 21 名、BM25 排到第 36 名，两路 top-20 一个都没进——"
                "**候选池里根本没有它的块**，融合层再怎么轮转也变不出来。\n\n"
                "对比「候选池分布」与「最终配额」两行即可定位病灶："
                "池子里就没有某家 → 检索层问题；池子里有却没选中 → 融合层问题。")




# ==========================================================================
# 面板一：问答
# ==========================================================================
def render_sources(answer: str, chunks: list[dict]) -> None:
    cited = parse_citations(answer)     # 与 evaluate.py 同一套解析，避免两边判分打架
    if not cited:
        st.info("本次回答未使用任何角标——若内容看起来具体，请警惕无据生成。")
        return
    invalid = [c for c in cited if c > len(chunks)]
    if invalid:
        st.error(f"回答引用了不存在的角标 {invalid}（本次只提供了 {len(chunks)} 个文本块）")
    st.markdown("#### 出处（严格溯源）")
    for i in cited:
        if i > len(chunks):
            continue
        ch = chunks[i - 1]
        page = page_label(ch["page_start"], ch["page_end"])
        with st.expander(f"【{i}】{ch['company']} · {ch['section']} · 第 {page} 页  "
                         f"（{ch['sw_l2']}）"):
            st.caption(f"来源文件：{ch['doc']}　|　chunk_id：{ch['chunk_id']}　|　"
                       f"{ch['char_len']} 字")
            st.text(ch["text"])


def run_qa(question: str, cfg: dict, retriever: HybridRetriever,
           df: pd.DataFrame) -> None:
    mode = st.session_state.get("theme", "light")

    with st.status("双通道检索中 ...", expanded=True) as status:
        st.write("① jieba 分词 + 指令模板套用")
        tr = retriever.search_trace(question, top_k_final=cfg["kf"],
                                    top_k_vector=cfg["kv"], top_k_bm25=cfg["kb"],
                                    rrf_k=cfg["rrf"], companies=cfg["companies"])
        st.write(f"② 向量通道召回 {len(tr.vector_hits)} 块 · "
                 f"BM25 通道召回 {len(tr.bm25_hits)} 块")
        st.write(f"③ RRF 融合，取 Top-{len(tr.fused)}")
        status.update(label=f"检索完成：融合后 Top-{len(tr.fused)}", state="complete")

    st.session_state["last_trace"] = tr

    stretch(st.plotly_chart, build_simple_animation(df, mode, tr), key="qa_anim")

    chunks = [r.chunk for r in tr.fused]
    client = DeepSeekClient()
    if not client.available:
        st.warning("未配置 DEEPSEEK_API_KEY，已跳过生成环节。下面是检索到的原文：")
        for i, ch in enumerate(chunks, 1):
            with st.expander(f"【{i}】{ch['company']} · {ch['section']} · 第 {ch['page_start']} 页"):
                st.text(ch["text"])
        return

    st.markdown("#### 回答")
    try:
        answer = st.write_stream(
            client.stream(build_messages(question, chunks),
                          temperature=config.ANSWER_TEMPERATURE,
                          max_tokens=config.ANSWER_MAX_TOKENS))
    except DeepSeekError as exc:
        st.error(f"调用 DeepSeek 失败：{exc}")
        return
    st.session_state["last_answer"] = answer
    render_sources(answer, chunks)

    with st.expander("查看送入模型的完整 Prompt（含全部上下文原文）"):
        msgs = build_messages(question, chunks)
        st.caption(f"system 提示词 {len(msgs[0]['content'])} 字 · "
                   f"user 消息 {len(msgs[1]['content'])} 字")
        st.code(msgs[0]["content"], language=None)
        st.code(msgs[1]["content"], language=None)


# ==========================================================================
# 主流程
# ==========================================================================
def main() -> None:
    st.session_state.setdefault("theme", "light")

    if not index_ready():
        return
    try:
        retriever = get_retriever()
    except IndexNotBuilt as exc:
        st.error(f"**索引无法载入**\n\n{exc}")
        st.code("python build_index.py", language="bash")
        return

    df = load_corpus(str(config.CHUNK_FILE), str(config.PCA_FILE))
    manifest = load_manifest()
    cfg = sidebar(df, manifest)
    mode = st.session_state["theme"]
    tok = t()

    scope = df[df["company"].isin(cfg["companies"])]
    st.markdown(
        f"<div style='color:{tok['text_secondary']};font-size:13px'>"
        f"当前范围 <b>{len(scope):,}</b> / {len(df):,} 个文本块 · "
        f"{len(cfg['companies'])} 家公司 · 索引 {manifest.get('n_chunks', 0):,} 块</div>",
        unsafe_allow_html=True)

    tab_qa, tab_map, tab_pipe, tab_about = st.tabs(
        ["💬 问答", "🗺️ 文本块星图", "🎬 检索动态过程", "📐 方法与口径"])

    # ---------------------------------------------------------------- 问答
    with tab_qa:
        examples = [
            "贵州茅台2026年上半年营业收入和归母净利润是多少？",
            "对比贵州茅台、五粮液、泸州老窖的毛利率和渠道库存策略差异",
            "伊利股份和青岛啤酒的成本端压力主要体现在哪里？",
            "多家白酒公司2026年上半年的经销商数量和回款情况对比",
        ]
        cols = st.columns(len(examples))
        for c, q in zip(cols, examples):
            if stretch(c.button, q[:14] + "…", help=q):
                st.session_state["question"] = q

        with st.form("qa_form"):
            q = st.text_area("提问", value=st.session_state.get("question", ""),
                             height=80, label_visibility="collapsed",
                             placeholder="例：对比贵州茅台、五粮液、泸州老窖的毛利率差异")
            submitted = st.form_submit_button("提问", type="primary")
        if submitted and q.strip():
            st.session_state["question"] = q.strip()

        question = st.session_state.get("question", "").strip()
        if question and (submitted or st.session_state.get("last_trace") is None):
            run_qa(question, cfg, retriever, scope)
        elif not question:
            st.info("输入问题，或点上面的示例直接开跑。检索结果会带公司 / 章节 / 页码角标。")

    # ---------------------------------------------------------------- 星图
    with tab_map:
        left, right = st.columns([3, 1])
        with right:
            view = st.radio("视图", ["单图 · 大类着色", "小倍数 · 每家公司一格"],
                            help="公司身份超过 3 类时不新增色相，改用分面承载")
            if view.startswith("单图") and cfg["focus"]:
                st.caption(f"已聚焦 **{cfg['focus']}**，其余公司压为灰色底噪")
        with left:
            if view.startswith("单图"):
                fig = build_explorer(scope, mode,
                                     "company" if cfg["focus"] else "category",
                                     focus_company=cfg["focus"])
            else:
                fig = build_facets(scope, mode)
            event = stretch(st.plotly_chart, fig, on_select="rerun",
                            selection_mode="points",
                            key=f"map_{view}_{cfg['focus']}")

        picked_chunks: list[dict] = []
        try:
            points = (event or {}).get("selection", {}).get("points", [])
        except AttributeError:
            points = getattr(getattr(event, "selection", None), "points", []) or []
        for p in points:
            cd = p.get("customdata")
            if cd:
                picked_chunks.append({"chunk_id": cd[0], "company": cd[1],
                                      "page": cd[2], "section": cd[3]})

        st.markdown("#### 点击散点查看原文")
        if picked_chunks:
            for pc in picked_chunks[:5]:
                row = df[df["chunk_id"] == pc["chunk_id"]]
                if row.empty:
                    continue
                r = row.iloc[0]
                st.markdown(
                    f"**{r['company']}** · {r['section']} · 第 {r['page']} 页 · "
                    f"{r['sw_l2']} · {r['char_len']} 字")
                st.text(r["text"][:1500])
                st.divider()
        else:
            st.caption("在图上点任意一个点，这里会显示它对应的原文、公司、章节与页码。")

        with st.expander(f"明细表（{len(scope):,} 个文本块，图的等价可读版本）"):
            show = scope[["company", "sw_l2", "category", "section", "page",
                          "char_len", "chunk_id"]].copy()
            show["原文摘要"] = scope["text"].str.slice(0, 60)
            show.columns = ["公司", "申万二级", "大类", "章节", "页码", "字数",
                            "chunk_id", "原文摘要"]
            stretch(st.dataframe, show, hide_index=True, height=380)
            st.download_button("下载明细 CSV",
                               show.to_csv(index=False).encode("utf-8-sig"),
                               file_name="chunks_scope.csv", mime="text/csv")

    # ------------------------------------------------------------- 检索过程
    with tab_pipe:
        tr = st.session_state.get("last_trace")
        if tr is None:
            st.info("先去「问答」页提一个问题，这里会展示那次检索的完整动态过程。")
        else:
            st.markdown(f"**本次问题：** {tr.query}")
            render_pipeline(tr)
            st.divider()
            stretch(st.plotly_chart, build_simple_animation(scope, mode, tr),
                    key="pipe_anim")
            st.caption("点图左上角的「▶ 播放检索过程」，或用下方步骤条逐帧查看。")

    # ------------------------------------------------------------- 方法与口径
    with tab_about:
        st.markdown(f"""
#### 数据口径
* **行业**：申万一级行业「{config.SW_INDUSTRY}」，{len(config.COMPANIES)} 家代表公司，
  共 {manifest.get('n_chunks', 0):,} 个文本块。
* **报告期**：以 `data/manifest.json` 记录的实际取数结果为准（本库为 2026 年半年度报告）。
* **来源**：巨潮资讯网（cninfo.com.cn），证监会指定信息披露平台，沪深统一接口。
* **切块**：目标 {config.CHUNK_SIZE} 字 / 重合 {config.CHUNK_OVERLAP} 字，
  切点优先落在句末标点，允许跨页并记录精确页码区间。

#### 双通道检索
| 通道 | 输入的字段 | 机制 | 擅长的查询 |
|---|---|---|---|
| 向量（Qwen3-Embedding-0.6B） | `tagged_text`（带标签） | 语义近邻，跨表述 | "白酒公司怎么应对渠道压力" |
| BM25（jieba 倒排） | `text`（纯正文） | 词面精确匹配 | "2026 年上半年 归母净利润"、"合同负债" |
| RRF 融合 | — | 1/(k+rank) 排名倒数求和 | 二者互补，双路命中的块排到最前 |
| 跨公司配额轮转 | — | 按公司分桶后轮转选取 | 多公司对比题（保证每家都有名额） |
| 通道层按公司召回 | — | 每家公司每通道各取 Top-N 再并集 | 多公司对比题（保证每家的块进得了候选池） |

两路的**输入本身就不一样**：向量通道连标签一起编码（标签里的公司名、章节名是语义锚点），
BM25 只索引正文（标签里的"页码:88""块序号:12"是高频数字噪声，会把倒排词表冲淡）。
公司/章节维度的过滤由检索白名单硬过滤完成，不依赖词面命中。

**跨公司配额**只在问题点名多家公司时生效，而且分**两层**，缺一不可：

* **通道层**（`config.PER_COMPANY_CHANNEL_K`）：两个通道各自按公司分别取 Top-N，
  再并集、按分数全局重排。只做融合层的轮转不够，因为 `TOP_K_VECTOR` / `TOP_K_BM25`
  是在融合**之前**按全池截断的，公司之间不加区分——某家公司可能两路都排在
  top-k 之外，压根进不了候选池，融合层再轮转也变不出来。
* **融合层**（`config.BALANCE_MULTI_COMPANY` + `PER_COMPANY_K`）：先按公司分桶、
  桶内按 RRF 降序，再轮转取用，每家最多 `PER_COMPANY_K` 块。它不修改 RRF 打分，
  只改最终入围顺序，所以"送入序号"不再严格单调，界面上有显式标注。

不这么做，多家公司的对比题在检索阶段就会被单公司垄断（实测 Top-8 全落在一家），
后面的生成再强也救不回来。两层都开时目标公司覆盖率实测由 91% 升到 100%。

#### 可视化设计说明
* 散点图是 **all-pairs** 形态（平面上任意两点都可能相邻），默认色板在 all-pairs
  判据下**只有前 3 个分类槽位达标**（已用校验脚本实测：浅色最差配对
  CVD ΔE 9.2 / 常视觉 ΔE 24.0，深色 9.4 / 20.9）。因此身份维度**只用 3 个分类色**
  （白酒 / 啤酒 / 大众品），第 4 类身份改用**小倍数分面**承载，不生成新色相、不循环色槽。
* 浅色模式下 aqua 对表面色仅 2.74:1（低于 3:1），按补偿规则每个图表都配了
  **明细表视图**，颜色从不单独承载信息。
* 查询向量落点用**墨色星标**而非系列色——它不是第 4 个类别。
* 两通道的召回点用**形状**区分（圆=向量，方=BM25，菱形=融合后），
  不再额外消耗色相；2px 表面色描边保证重叠处仍可读。

#### 已知限制
* pdfplumber 对扫描件无效；复杂合并单元格的还原会有错位（影响表格可读性，不影响召回与溯源）。
* PCA 两维仅解释部分方差，散点图的邻接关系**不等于**原始空间的相似度排名——
  它是解释工具，不是判据；判据始终是 1024 维上的余弦相似度。
* LLM 生成仍可能出错。系统对策是强制角标 + 出处卡片，让每句话都能被人工回原文核对。
""")
        st.caption("本知识库仅供课程研究与技术演示，报告原文版权归各上市公司及交易所所有。")


main()
