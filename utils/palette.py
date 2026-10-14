# -*- coding: utf-8 -*-
"""阶段五可视化配色令牌（已用 validate_palette.js 校验，勿随手改）。

配色不是"挑好看"，而是按身份/量级/极性各司其职，并跑脚本验证。
本文件是这套方法的参数实例：色板、表面色、墨色都集中在这里，改主题只动这一处。

关键约束（为什么只有 3 个分类色）
--------------------------------
文本块散点图属于 **all-pairs 形态**——平面上任意两点都可能挨在一起，
所以必须用 all-pairs 判据校验，而不是只校验相邻槽位。
默认调色板的 8 个分类槽在 all-pairs 下**只有前 3 个能通过**：

    light: 最差配对 CVD ΔE 9.2、常视觉 ΔE 24.0   -> 全通过
    dark : 最差配对 CVD ΔE 9.4、常视觉 ΔE 20.9   -> 全通过

第 4 槽（黄）会与第 2 槽（橙）同时出现，该配对在 all-pairs 下不达标。
因此本系统**硬性只用 3 个分类色**：把 10 家公司按申万二级行业折成 3 大类
（白酒 / 啤酒 / 大众品），公司级的身份交给"小倍数分面 + 交互 + 明细表"承载，
绝不生成第 4 个色相、也不循环使用色槽。

light 模式下 aqua 对表面色只有 2.74:1（低于 3:1），按"补偿规则"必须提供
可读的替代通道——本应用每个图表都配了明细表视图，满足该要求。
"""
from __future__ import annotations

LIGHT = {
    "surface": "#fcfcfb",          # 图表表面
    "page": "#f9f9f7",             # 页面底色
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",            # 轴/次要标签
    "grid": "#e1e0d9",             # 网格发丝线
    "axis": "#c3c2b7",
    "border": "rgba(11,11,11,0.10)",
    # 分类槽 1..3（固定顺序，永不循环）
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    "corpus_muted": "#d8d7d0",     # 语料背景点：非分类色，是"底噪"
    "highlight": "#0b0b0b",        # 查询向量落点：用墨色而非系列色
}

DARK = {
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "text_primary": "#f0efec",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "border": "rgba(255,255,255,0.10)",
    "series": ["#3987e5", "#d95926", "#199e70"],
    "corpus_muted": "#3a3a37",
    "highlight": "#f0efec",
}

# 三大类：键为展示名，值是该类包含的申万二级行业，槽位顺序固定
CATEGORIES = [
    ("白酒", ("白酒Ⅱ",)),
    ("啤酒", ("非白酒",)),
    ("大众品（乳品·调味）", ("饮料乳品", "调味发酵品Ⅱ")),
]

CATEGORY_NAMES = [name for name, _ in CATEGORIES]
CATEGORY_OF_L2 = {l2: name for name, l2s in CATEGORIES for l2 in l2s}


def tokens(mode: str = "light") -> dict:
    return DARK if mode == "dark" else LIGHT


def category_of(sw_l2: str) -> str:
    """申万二级行业 -> 3 大类展示名。"""
    return CATEGORY_OF_L2.get(sw_l2, CATEGORY_NAMES[-1])


def series_map(mode: str = "light") -> dict[str, str]:
    """大类名 -> 色值。固定顺序分配，任何筛选都不改变已有映射。"""
    tok = tokens(mode)
    return {name: tok["series"][i] for i, (name, _) in enumerate(CATEGORIES)}


def plotly_layout(mode: str = "light", height: int = 560, title: str = "") -> dict:
    """统一的图表外壳：发丝线网格、收敛的坐标轴、无边框留白。"""
    tok = tokens(mode)
    return dict(
        title=dict(text=title, font=dict(size=15, color=tok["text_primary"],
                                        family='system-ui, -apple-system, "Segoe UI", sans-serif'),
                   x=0.01, xanchor="left"),
        height=height,
        paper_bgcolor=tok["surface"],
        plot_bgcolor=tok["surface"],
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  size=12, color=tok["text_secondary"]),
        margin=dict(l=48, r=24, t=56 if title else 24, b=44),
        hoverlabel=dict(bgcolor=tok["surface"], bordercolor=tok["border"],
                        font=dict(color=tok["text_primary"], size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=tok["text_secondary"], size=12),
                    itemsizing="constant"),
        xaxis=dict(showgrid=True, gridcolor=tok["grid"], gridwidth=1, zeroline=False,
                   showline=False, ticks="", title=dict(text="", font=dict(size=11)),
                   tickfont=dict(color=tok["muted"], size=11)),
        yaxis=dict(showgrid=True, gridcolor=tok["grid"], gridwidth=1, zeroline=False,
                   showline=False, ticks="", title=dict(text="", font=dict(size=11)),
                   tickfont=dict(color=tok["muted"], size=11)),
    )
