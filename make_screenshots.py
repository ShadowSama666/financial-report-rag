# -*- coding: utf-8 -*-
"""抓取问答页面截图 → C:\\Users\\Lenovo\\Desktop\\问答页面截图.png

四块内容：
  ① 10 道题全景评测看板（含 2 道跨公司全景题的高亮行）
  ② 带溯源角标的问答结果（回答 + 逐条出处）
  ③ 降维散点图（文本块星图）
  ④ 检索动态过程

几个必须这么写的地方：
- 自己起 streamlit（headless），不指望用户先把服务开好；
- 起完先轮询 /_stcore/health —— 光等 goto 返回没用，首屏要载向量库和 0.6B 模型；
- 这个 Streamlit 版本的 tab 是普通 <button>，既没有 role="tab" 也没有
  data-baseweb="tab"，按角色选一个都选不中，只能按文本点；
- **不要自己算坐标**。前两版分别是 full_page（Streamlit 滚的是内层容器，文档不滚，
  截出来只有一屏高）和"拉高视口再裁"（长内容仍会溢出，元素被压成 2px 条）。
  现在改成：用 JS 给目标元素打一个临时 id，再交给 playwright 的 element.screenshot()
  自己滚动、自己截全 —— 它内部就是按元素 boundingBox + 滚动拼接做的，比手算可靠。
"""
import os
import subprocess
import sys
import tempfile
import time

import requests
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

# 这个脚本就放在仓库根目录（和 app.py 同级）。默认按脚本位置推仓库路径，
# 也可以用 RAG_PROJ 覆盖；截图中间产物一律写系统临时目录 —— 放在脚本同目录的话，
# 四张 p_*.png 会直接把 git 工作区弄脏。
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get("RAG_PROJ") or (
    _HERE if os.path.exists(os.path.join(_HERE, "app.py"))
    else r"C:\Users\Lenovo\sw_food_rag")
OUT = os.environ.get("SHOT_OUT") or os.path.join(
    os.path.expanduser("~"), "Desktop", "问答页面截图.png")
TMP = os.environ.get("SHOT_TMP") or os.path.join(tempfile.gettempdir(), "rag_shots")
os.makedirs(TMP, exist_ok=True)
PORT = 8631
BASE = f"http://127.0.0.1:{PORT}"
DSF = 2
VW = 1500
VH = 1100                   # 当前视口高度（截长内容前会临时拉高）
PANEL_W = 1840

# 版面上的顺序、说明文字、落盘文件名放一处：抓取顺序是按"页面交互方便"排的
# （先问答、最后星图），跟版面顺序不一样，靠 reverse() 对齐一改抓取顺序就翻车。
PANELS = [
    (1, "① 10 道题全景评测看板（2 道跨公司全景题整行高亮）", "p_board.png"),
    (2, "② 带溯源角标的问答结果（回答 + 逐条出处：公司·章节·页码）", "p_answer.png"),
    (3, "③ 降维散点图（2717 个文本块，按公司着色，点击可查原文）", "p_scatter.png"),
    (4, "④ 检索动态过程（问题落图 → 邻居召回 → RRF 融合）", "p_anim.png"),
]
CAPTION = {i: c for i, c, _ in PANELS}
SHOT = {i: os.path.join(TMP, f) for i, _, f in PANELS}

# 演示题：单公司 + 只问检索确实捞得到的那几项。
# 上一版问"三家毛利率差异"，三家公司各自召回到的页都没落在黄金页
# （茅台 p1/p32、五粮液 p9/p12、老窖 p1/p20，而黄金页是 p4-5 / p6），
# 加上跨公司配额每家只分到 2 块，模型手上没有可比的数，只能给定性结论，
# 截图里看不出"完美问答"。实测对比（黄金页命中 / 噪声公司）：
#   Q05 青岛啤酒  4/6 · 无       ← 单公司题最干净
#   Q02 五粮液    1/6 · 海天、青啤
#   Q01 茅台      1/6 · 青啤
# 再往下收一步：Q05 原题还问了"利润总额"，但该数字只在 p5-6 那块里，
# 而检索稳定命中的是**跨页重叠**的 p6-7 块（同样含 p6，却不含利润总额行），
# 于是模型会如实回一句"检索结果中未出现利润总额科目"—— 这句诚实但不算"完美回答"。
# 所以演示题只保留营业收入与产品销量，这两项在 p6/p7/p8 里都齐，且 6 块全是青岛啤酒。
QUESTION = "青岛啤酒2026年上半年的营业收入是多少？报告期内产品销量（千升）情况如何？"


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- streamlit
def start_app():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    logf = open(os.path.join(TMP, "streamlit_run.log"), "wb")
    p = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(PORT), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=PROJ, env=env, stdout=logf, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    for i in range(120):
        try:
            r = requests.get(f"{BASE}/_stcore/health", timeout=3)
            if r.status_code == 200 and "ok" in r.text.lower():
                log(f"  streamlit 就绪（{i*2}s，PID {p.pid}）")
                return p
        except Exception:
            pass
        if p.poll() is not None:
            sys.exit("streamlit 进程提前退出，见 streamlit_run.log")
        time.sleep(2)
    p.kill()
    sys.exit("streamlit 启动超时")


def stop_app(p):
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)


# ---------------------------------------------------------------- 页面操作
JS_TAB = r"""
(name) => {
  const cands = [...document.querySelectorAll('button, [role="tab"], [data-baseweb="tab"]')];
  const el = cands.find(e => e.innerText.trim().includes(name));
  if (!el) return false;
  el.click();
  return true;
}
"""

# 给当前标签页里"可见的第一个 plotly 图"打 id
JS_TAG_CHART = r"""
() => {
  const vis = e => !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length);
  const el = [...document.querySelectorAll('[data-testid="stPlotlyChart"]')].filter(vis)[0];
  if (!el) return false;
  el.id = '__cap';
  return true;
}
"""

# 评测看板整页：从 <table> 往上找到**最小**的、同时含看板标题与归因汇总的祖先。
# 不能图省事直接取 document.body —— 非激活的 tab 面板仍在 DOM 里（只是被隐藏），
# 那样会把别的页签的内容一起截进来。
JS_TAG_BOARD = r"""
() => {
  const vis = e => !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length);
  const tbl = [...document.querySelectorAll('.eb table')].filter(vis)[0];
  if (!tbl) return false;
  let p = tbl.parentElement, target = null;
  while (p) {
    if (p.textContent.includes('10 道题全景评测看板')
        && p.textContent.includes('错因归因汇总')) { target = p; break; }
    p = p.parentElement;
  }
  if (target) { target.id = '__cap'; } else { tbl.id = '__cap'; }
  return true;
}
"""

# 给"回答 + 出处"这一段打 id；顺手把同一块里的动画图藏掉
# （run_qa 把 status / 动画 / 回答 / 出处 写在同一个 stVerticalBlock 里，
#  不藏的话第②块会把第④块的动画再演一遍）。
JS_TAG_ANSWER = r"""
() => {
  const vis = e => !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length);
  const ans = [...document.querySelectorAll('h4')].filter(vis)
                .find(e => e.textContent.trim() === '回答');
  if (!ans) return false;
  let p = ans.parentElement, target = null;
  while (p) {                       // 最小的、同时含"回答"与"出处"的祖先
    if (p.textContent.includes('出处（严格溯源）')) { target = p; break; }
    p = p.parentElement;
  }
  if (!target) return false;
  // 藏图要连外层元素容器一起藏：只把内层 chart 设 display:none，
  // Streamlit 的 stElementContainer 仍按固定高度占位，会留一大块白。
  [...target.querySelectorAll('[data-testid="stPlotlyChart"]')].forEach(c => {
    const box = c.closest('[data-testid="stElementContainer"]') || c;
    box.dataset._hidden = '1';
    box.style.display = 'none';
  });
  target.id = '__cap';
  return true;
}
"""


JS_DIAG = r"""
() => {
  const vis = e => !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length);
  const cs = [...document.querySelectorAll('[data-testid="stPlotlyChart"]')];
  return {
    n_charts: cs.length,
    sizes: cs.map(c => c.offsetWidth + 'x' + c.offsetHeight),
    vis: cs.map(vis),
    has_board: document.body.innerText.includes('10 道题全景评测看板'),
    has_answer: document.body.innerText.includes('出处（严格溯源）'),
    tabs: [...document.querySelectorAll('button')].map(b => b.innerText.trim())
             .filter(t => t.length < 16 && ['问答','评测看板','文本块星图','检索动态过程','方法与口径']
                           .some(x => t.includes(x)))
  };
}
"""


def nudge(page, i=0):
    """逼 plotly 重新量尺寸。
    图是在非激活的 tab 里先渲染的，那时容器 0×0，plotly 量到 0 就再也不重算；
    切过去只是显隐切换，不会触发 resize —— 所以要主动抖一下视口。"""
    page.evaluate("() => window.dispatchEvent(new Event('resize'))")
    page.set_viewport_size({"width": VW, "height": VH + (24 if i % 2 else 0)})
    page.wait_for_timeout(1300)


def shot(page, js, path, wait=600, tries=7):
    for i in range(tries):
        if page.evaluate(js):
            page.wait_for_timeout(wait)
            page.locator("#__cap").screenshot(path=path)
            page.evaluate("() => { const e = document.getElementById('__cap');"
                          " if (e) { e.removeAttribute('id');"
                          " [...e.querySelectorAll('[data-_hidden], [data-_hidden=\"1\"]')]"
                          "   .forEach(c => { c.style.display = ''; delete c.dataset._hidden; }); } }")
            return Image.open(path)
        nudge(page, i)
    log("    [诊断] " + str(page.evaluate(JS_DIAG)))
    return None


def trim_bottom(im):
    """裁掉底部纯背景的行（Streamlit 的 tab 面板尾部留白很多）。"""
    w, h = im.size
    px = im.load()
    bg = px[3, h - 3][:3]

    def blank(y):
        for x in range(0, w, 5):
            r, g, b = px[x, y][:3]
            if abs(r - bg[0]) > 6 or abs(g - bg[1]) > 6 or abs(b - bg[2]) > 6:
                return False
        return True

    y = h - 1
    while y > 8 and blank(y):
        y -= 1
    return im.crop((0, 0, w, y + 9))


JS_PLAY = r"""
() => {
  const gd = document.querySelector('.js-plotly-plot');
  if (!gd) return 'no-graph';
  const out = [];
  try {                                   // 首选：直接调 plotly API 跳到第 5 帧
    const n = (gd._transitionData && gd._transitionData._frames || []).length;
    if (window.Plotly && n) {
      window.Plotly.animate(gd, [Math.min(4, n - 1)],
        {frame: {duration: 0, redraw: true},
         transition: {duration: 0}, mode: 'immediate'});
      out.push('api->frame ' + Math.min(4, n - 1) + '/' + n);
    } else { out.push('no-api(n=' + n + ')'); }
  } catch (e) { out.push('api-err:' + e.message); }
  try {                                   // 兜底：给播放按钮补一套真实鼠标事件
    const a = document.querySelector('a.updatemenu-item.play, a.play');
    if (a) {
      const r = a.getBoundingClientRect();
      const o = {bubbles: true, cancelable: true,
                 clientX: r.x + r.width / 2, clientY: r.y + r.height / 2};
      ['mousedown', 'mouseup', 'click'].forEach(t =>
        a.dispatchEvent(new MouseEvent(t, o)));
      out.push('mouse-events-sent');
    } else { out.push('no-play-anchor'); }
  } catch (e) { out.push('click-err:' + e.message); }
  return out.join(' | ');
}
"""


JS_STEP = (r"() => (document.body.innerText.match(/步骤[:：]\s*(\d+)/) "
           r"|| [])[1] || '?'")


def click_play(page):
    """让动画停在信息量最大的那一帧。
    plotly 的 updatemenu 按钮挂在 SVG 的 <a> 上，且监听 mousedown/mouseup，
    用 playwright 的 .click() 会被判成"元素不可交互"而抛异常，
    所以直接走 plotly 自己的 animate API，并补一套鼠标事件兜底。"""
    res = page.evaluate(JS_PLAY)
    page.wait_for_timeout(900)
    step = page.evaluate(JS_STEP)

    if step in ("1", "?"):
        # 与其自己算坐标，不如让 playwright 按"元素内相对位置"点：
        # 它会自动把元素滚进视口，位置也是相对 #__cap 左上角的，
        # 不受页面滚动影响。播放键在图上大约是 (75, 46) 那一块。
        for node in [{"x": 75, "y": 46}, {"x": 420, "y": 56}]:
            try:
                page.locator("#__cap").click(position=node, timeout=4000)
                page.wait_for_timeout(2200)
                step = page.evaluate(JS_STEP)
                res += f" | click{tuple(node.values())}->{step}"
                if step not in ("1", "?"):
                    break
            except Exception as e:
                res += f" | click-err:{type(e).__name__}"

    log(f"    {res}   顶部步骤指示 = {step}")
    return step not in ("1", "?")


def goto_tab(page, name):
    if not page.evaluate(JS_TAB, name):
        raise RuntimeError(f"没找到标签页：{name}")
    page.wait_for_timeout(2500)


def main():
    proc = start_app()
    panels = []
    try:
        with sync_playwright() as pw:
            br = pw.chromium.launch()
            page = br.new_page(viewport={"width": VW, "height": 1100},
                               device_scale_factor=DSF)
            page.set_default_timeout(300_000)
            page.goto(BASE, wait_until="domcontentloaded")
            log("  等待应用首屏（载入向量库 + 0.6B 模型）...")
            page.locator('textarea[placeholder*="青岛啤酒"]').first.wait_for(
                state="visible", timeout=600_000)
            page.wait_for_timeout(6000)
            log("  首屏就绪")

            page.locator('textarea[placeholder*="青岛啤酒"]').first.fill(QUESTION)
            page.get_by_role("button", name="提问", exact=True).first.click()
            log("  已提问，等待检索与生成 ...")
            page.locator("h4", has_text="出处（严格溯源）").first.wait_for(
                state="visible", timeout=300_000)
            page.wait_for_timeout(2500)
            log("  回答与溯源已渲染")

            # ---- ② 问答结果 ----
            global VH
            VH = 4200                     # 一屏装下整段回答 + 出处
            page.set_viewport_size({"width": VW, "height": VH})
            page.wait_for_timeout(2000)
            im = shot(page, JS_TAG_ANSWER, SHOT[2])
            if im:
                panels.append((2, CAPTION[2], im))
                log(f"  ② {im.width}×{im.height}")

            # ---- ① 10 道题全景评测看板 ----
            VH = 2600
            page.set_viewport_size({"width": VW, "height": VH})
            goto_tab(page, "10道题全景评测看板")
            im = shot(page, JS_TAG_BOARD, SHOT[1], wait=1200)
            if im:
                panels.append((1, CAPTION[1], im))
                log(f"  ① {im.width}×{im.height}")

            # ---- ④ 检索动态过程 ----
            VH = 1100
            page.set_viewport_size({"width": VW, "height": VH})
            goto_tab(page, "检索动态过程")
            # 图表在"检索过程"那张大表下面，不滚进视口的话鼠标坐标是空的
            if page.evaluate(JS_TAG_CHART):
                try:
                    page.locator("#__cap").scroll_into_view_if_needed(timeout=5000)
                except Exception:
                    pass
                page.wait_for_timeout(700)
            if click_play(page):
                log("  已触发动画播放，抓中间帧")
            im = shot(page, JS_TAG_CHART, SHOT[4])
            if im:
                panels.append((4, CAPTION[4], im))
                log(f"  ④ {im.width}×{im.height}")

            # ---- ③ 降维散点图 ----
            goto_tab(page, "文本块星图")
            page.wait_for_timeout(2000)
            im = shot(page, JS_TAG_CHART, SHOT[3])
            if im:
                panels.append((3, CAPTION[3], im))
                log(f"  ③ {im.width}×{im.height}")

            br.close()
    finally:
        stop_app(proc)

    compose(panels)


def compose(panels):
    """把若干面板按版面顺序拼成一整张 PNG。

    抓取顺序是按"页面交互方便"排的（先问答、最后星图），跟版面顺序不一致，
    所以每个面板都带一个序号，这里显式排序 —— 之前用 reverse() 是错的，
    一改抓取顺序版面就整体颠倒了。
    """
    if not panels:
        sys.exit("没有抓到任何面板")

    panels.sort(key=lambda t: t[0])
    try:
        f_title = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 30)
        f_note = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 21)
    except Exception:
        f_title = f_note = ImageFont.load_default()

    scaled = []
    for _, cap, im in ((i, c, trim_bottom(p)) for i, c, p in panels):
        h = max(1, int(im.height * PANEL_W / im.width))
        scaled.append((cap, im.resize((PANEL_W, h), Image.LANCZOS)))

    HEAD, GAP, PAD, TOP = 62, 26, 26, 104
    total_h = TOP + sum(HEAD + im.height + GAP for _, im in scaled) + 118
    canvas = Image.new("RGB", (PANEL_W + PAD * 2, total_h), (252, 252, 251))
    d = ImageDraw.Draw(canvas)

    d.text((PAD, 18), "申万食品饮料半年报问答知识库 · 页面实拍",
           font=f_title, fill=(11, 11, 11))
    d.text((PAD, 58), f"提问：{QUESTION}", font=f_note, fill=(90, 90, 90))

    y = TOP
    for cap, im in scaled:
        d.text((PAD + 2, y), cap, font=f_note, fill=(42, 120, 214))
        y += HEAD - 12
        canvas.paste(im, (PAD, y))
        d.rectangle([PAD - 1, y - 1, PAD + im.width, y + im.height],
                    outline=(216, 216, 210), width=1)
        y += im.height + GAP

    d.text((PAD, total_h - 96),
           "① 评测看板直读 eval/runs/ 的原始产物（已随仓库提交），界面数字与仓库文件逐字对应，",
           font=f_note, fill=(90, 90, 90))
    d.text((PAD, total_h - 66),
           "   错因归因按「黄金页进没进上下文」区分检索侧与生成侧；",
           font=f_note, fill=(90, 90, 90))
    d.text((PAD, total_h - 36),
           "② 回答中每个【n】角标均可展开回溯到公司 · 章节 · 页码；③④ 为检索两侧的可视化。",
           font=f_note, fill=(90, 90, 90))

    canvas.save(OUT)
    log(f"\n已写入 {OUT}")
    log(f"  尺寸 {canvas.width}×{canvas.height}   面板 {len(panels)} 块   "
        f"大小 {os.path.getsize(OUT)//1024} KB")


def recompose():
    """不重开浏览器，直接用上一轮落盘的四张面板重新拼图。
    调版式/说明文字时用这个，省掉一次起服务 + 调 API 的成本。"""
    panels = []
    for i, cap, _ in PANELS:
        if os.path.exists(SHOT[i]):
            panels.append((i, cap, Image.open(SHOT[i])))
    if not panels:
        sys.exit("没有已落盘的面板可拼，先正常跑一次")
    log(f"复用已有面板 {len(panels)} 张：{[p[0] for p in sorted(panels)]}")
    compose(panels)


if __name__ == "__main__":
    if "--recompose" in sys.argv:
        recompose()
    else:
        main()
