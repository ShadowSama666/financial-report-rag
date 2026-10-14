# -*- coding: utf-8 -*-
"""全局配置：路径、行业口径、公司清单、解析/切块/检索参数。

阶段一的产物之一。所有脚本共用本文件，避免口径散落在各处。
"""
from __future__ import annotations

import os
from pathlib import Path

# .env 必须在读取下面任何"可被环境变量覆盖"的配置之前加载好。
# 各脚本都是第一行 `import config`，而 load_env() 一般在 main() 里才调用，
# 只依赖调用顺序会**静默**读不到 .env 里的覆盖值（EMBED_MODEL 就踩过这个坑：
# 文档说可以填本地模型路径，实际填了不生效）。这里自己先补一次，与顺序无关。
# 已存在的真实环境变量优先级更高（override=False）。
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except Exception:       # dotenv 没装 / .env 不存在都不该影响 import
    pass

# --------------------------------------------------------------------------
# 1. 路径
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DATA_DIR  = ROOT / "data"
PDF_DIR   = DATA_DIR / "pdfs"    # 阶段二：半年度报告 PDF 原文（.gitignore）
PAGE_DIR  = DATA_DIR / "pages"   # 阶段三：逐页解析缓存（.gitignore）
CHUNK_DIR = DATA_DIR / "chunks"  # 阶段三：带标签文本块
INDEX_DIR = ROOT / "index"       # 阶段四：向量 / BM25 索引（.gitignore）
EVAL_DIR  = ROOT / "eval"        # 阶段六：测试题与评测记录

for _d in (PDF_DIR, PAGE_DIR, CHUNK_DIR, INDEX_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MANIFEST_FILE = DATA_DIR / "manifest.json"    # 下载清单（入库，记录本次取数口径）
CHUNK_FILE    = CHUNK_DIR / "chunks.jsonl"    # 最终知识库
STATS_FILE    = CHUNK_DIR / "chunks_stats.csv"

# --------------------------------------------------------------------------
# 2. 行业口径：申万一级行业「食品饮料」10 家代表公司
# --------------------------------------------------------------------------
SW_INDUSTRY = "食品饮料"

COMPANIES = [
    {"code": "600519", "name": "贵州茅台", "exchange": "SH", "sw_l2": "白酒Ⅱ"},
    {"code": "000858", "name": "五粮液",   "exchange": "SZ", "sw_l2": "白酒Ⅱ"},
    {"code": "600887", "name": "伊利股份", "exchange": "SH", "sw_l2": "饮料乳品"},
    {"code": "600809", "name": "山西汾酒", "exchange": "SH", "sw_l2": "白酒Ⅱ"},
    {"code": "002304", "name": "洋河股份", "exchange": "SZ", "sw_l2": "白酒Ⅱ"},
    {"code": "000568", "name": "泸州老窖", "exchange": "SZ", "sw_l2": "白酒Ⅱ"},
    {"code": "603288", "name": "海天味业", "exchange": "SH", "sw_l2": "调味发酵品Ⅱ"},
    {"code": "600600", "name": "青岛啤酒", "exchange": "SH", "sw_l2": "非白酒"},
    {"code": "000596", "name": "古井贡酒", "exchange": "SZ", "sw_l2": "白酒Ⅱ"},
    {"code": "603369", "name": "今世缘",   "exchange": "SH", "sw_l2": "白酒Ⅱ"},
]
COMPANY_BY_CODE = {c["code"]: c for c in COMPANIES}
COMPANY_BY_NAME = {c["name"]: c for c in COMPANIES}

# --------------------------------------------------------------------------
# 3. 下载参数（阶段二）
# --------------------------------------------------------------------------
REPORT_CATEGORY = "category_bndbg_szsh"   # 巨潮公告分类：半年报
TARGET_YEAR = None            # None = 每家自动取最新一期半年报；填 2025 则锁定 2025H1
SEARCH_WINDOW = "{year}-06-01~{year}-12-31"   # 半年报法定披露窗口（7-8 月为主）

REQUEST_INTERVAL = 1.2        # 请求间隔（秒），做文明爬虫，避免被限流
REQUEST_TIMEOUT = 30
MAX_RETRY = 3

TITLE_MUST_HAVE = ("半年度报告",)
TITLE_MUST_NOT_HAVE = ("摘要", "英文", "English", "更正", "修正", "补充", "取消",
                       "说明", "已取消", "（更新）", "公告")

# --------------------------------------------------------------------------
# 4. 解析与切块参数（阶段三）
# --------------------------------------------------------------------------
PDF_ENGINE = "pdfplumber"     # pdfplumber（表格还原好、较慢）| pypdf（快、表格差）
EXTRACT_TABLES = True         # 是否做表格结构还原

CHUNK_SIZE = 800              # 目标字数
CHUNK_OVERLAP = 120           # 相邻块重合字数
MIN_CHUNK_SIZE = 200          # 短于此长度的块丢弃（文档末块除外）

DEDUP_HEADER_RATIO = 0.5      # 出现在 >50% 页面的短行 → 判定为页眉/页脚，剔除
DEDUP_HEADER_MIN_PAGES = 5    # 且至少出现 5 页，避免误杀
HEADER_FOOTER_LINES = 3       # 只在页面首/末 N 行内做页码行检测

TOC_DOTS_RE = r"[.．·…]{4,}"  # 目录点线，用于识别并丢弃目录块

# --------------------------------------------------------------------------
# 5. 双通道索引参数（阶段四）
# --------------------------------------------------------------------------
EMBED_MODEL = os.environ.get("EMBED_MODEL") or "Qwen/Qwen3-Embedding-0.6B"
EMBED_BATCH = int(os.environ.get("EMBED_BATCH") or 16)
EMBED_MAX_LEN = 1024       # 800 字中文 ≈ 600~800 token，留足余量避免截断块尾
USE_TAGGED_TEXT_FOR_EMBED = True   # 连同【公司|行业|章节|页码】标签一起编码，显著提升召回

# BM25 通道走**正文**，不带标签。理由：
#   1) 标签里的「页码:88」「块序号:12」是高频数字词，进倒排只会污染词表；
#   2) 公司名会出现在该公司每个块的标签里，词频拉满反而稀释 IDF 的判别力；
#   3) 公司/章节维度的"过滤"交给检索时的 allowed 白名单硬过滤，不靠词面命中。
# 于是两个通道的差异不只在算法（语义 vs 词面），连输入都不同——这正是
# 阶段五「检索动态过程」里两路召回结果位置互补的原因。
USE_TAGGED_TEXT_FOR_BM25 = False

# Qwen3-Embedding 官方建议：文档端裸编码，查询端加 instruct 前缀（非对称检索）
EMBED_QUERY_INSTRUCTION = ("给定一个中文上市公司半年报分析问题，"
                           "检索能够回答该问题的原文段落")
EMBED_QUERY_TEMPLATE = "Instruct: {instruction}\nQuery: {query}"

PCA_DIMS = 50              # 降维到 50 维做检索可视化底图（前两维即散点图坐标）

BM25_K1 = 1.5
BM25_B = 0.75

# --------------------------------------------------------------------------
# 6. 检索与生成参数（阶段五）
# --------------------------------------------------------------------------
TOP_K_VECTOR = 20
TOP_K_BM25 = 20
RRF_K = 60                 # RRF 融合常数
TOP_K_FINAL = 6            # 最终交给 LLM 的文本块数

# 跨公司**配额选取**：单一问题点多家公司时，RRF 分数天然会向"命中词更多的那一家"
# 倾斜。实测 Q09（茅台/五粮液/老窖/汾酒）用 BM25 取 Top8 全部落在泸州老窖，
# 再经 Top-6 收口后其它三家一块都进不来 —— 对比题变成单公司题。
# 打开后只改"最终选哪几块"，不改 RRF 打分与排名：先按公司分桶，桶内按 RRF 降序，
# 再轮转取（每家先取第 1 名，再取第 2 名…），从而保证每家公司都有配额。
BALANCE_MULTI_COMPANY = True
PER_COMPANY_K = 3          # 轮转时每家公司最多占几块（0 表示不限制）

# 但**只做融合层配额还不够**：TOP_K_VECTOR / TOP_K_BM25 是在"融合之前"按全池
# 截断的，而且是公司无关的。实测 Q09（茅台/五粮液/老窖/汾酒）里山西汾酒向量排到
# 第 21 名、BM25 排到第 36 名，两路 top-20 一个都没进 —— 候选池里根本没有汾酒的块，
# 融合层再怎么轮转也变不出来（轮转只能重排已有候选，不能凭空造候选）。
# 而汾酒的报表里"合同负债"确实出现在 10 个块里，属于纯粹的检索口径问题。
# 打开后：多公司问题改为**按公司分别取 Top-N 再并集重排**，
# 从候选池层面保证每家公司都有块进来。
PER_COMPANY_CHANNEL_K = 5  # 多公司问题时，每家公司每个通道各取几块进候选池
                           # 设为 0 可单独关掉（只保留下面的融合层轮转），用于 A/B 对照

ANSWER_TEMPERATURE = 0.2
ANSWER_MAX_TOKENS = 2048

# 索引文件（index/ 整体被 .gitignore 排除）
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNK_IDS_FILE = INDEX_DIR / "chunk_ids.json"
BM25_FILE = INDEX_DIR / "bm25.pkl"
PCA_FILE = INDEX_DIR / "pca.npz"
INDEX_MANIFEST = INDEX_DIR / "index_manifest.json"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
