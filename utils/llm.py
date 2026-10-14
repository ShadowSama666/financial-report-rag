# -*- coding: utf-8 -*-
"""DeepSeek 生成模型客户端（OpenAI 兼容协议，用 requests 直连，不引入额外 SDK）。

API Key 只从环境变量 DEEPSEEK_API_KEY 读取，绝不写进代码或配置文件。
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterator, Sequence

import requests

import config
from .common import get_logger, load_env

# 角标形如【1】【2】。前端渲染与评测判分**必须用同一套解析**，
# 否则界面上显示"角标齐全"、评测表里却判"角标不合规"，两边互相打架。
# 之前 app.py 与 evaluate.py 各抄了一份正则，现统一到这里。
CITE_RE = re.compile(r"【(\d+)】")


def parse_citations(answer: str) -> list[int]:
    """抽出答案里引用的角标序号（去重升序，1-based，对应 chunks 列表下标+1）。"""
    return sorted({int(m) for m in CITE_RE.findall(answer)})

logger = get_logger("llm")


class DeepSeekError(RuntimeError):
    pass


class DeepSeekClient:
    """极简 DeepSeek 客户端。支持流式与非流式对话。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout: int = 120) -> None:
        load_env()
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL")
                         or config.DEEPSEEK_BASE_URL).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or config.DEEPSEEK_MODEL
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise DeepSeekError(
                "未配置 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入真实 Key。")
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    # ---------------------------------------------------------------- 非流式
    def chat(self, messages: Sequence[dict], temperature: float = 0.2,
             max_tokens: int = 2048) -> str:
        payload = {"model": self.model, "messages": list(messages),
                   "temperature": temperature, "max_tokens": max_tokens,
                   "stream": False}
        try:
            resp = requests.post(f"{self.base_url}/chat/completions",
                                 headers=self._headers(), json=payload,
                                 timeout=self.timeout)
        except requests.RequestException as exc:
            raise DeepSeekError(f"请求 DeepSeek 失败：{exc}") from exc
        if resp.status_code != 200:
            raise DeepSeekError(f"DeepSeek 返回 {resp.status_code}：{resp.text[:300]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise DeepSeekError(f"响应结构异常：{json.dumps(data)[:300]}") from exc

    # ------------------------------------------------------------------ 流式
    def stream(self, messages: Sequence[dict], temperature: float = 0.2,
               max_tokens: int = 2048) -> Iterator[str]:
        """逐段 yield 生成内容，供 Streamlit st.write_stream 使用。"""
        payload = {"model": self.model, "messages": list(messages),
                   "temperature": temperature, "max_tokens": max_tokens,
                   "stream": True}
        try:
            resp = requests.post(f"{self.base_url}/chat/completions",
                                 headers=self._headers(), json=payload,
                                 timeout=self.timeout, stream=True)
        except requests.RequestException as exc:
            raise DeepSeekError(f"请求 DeepSeek 失败：{exc}") from exc
        if resp.status_code != 200:
            raise DeepSeekError(f"DeepSeek 返回 {resp.status_code}：{resp.text[:300]}")

        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("data: "):
                raw = raw[6:]
            if raw.strip() == "[DONE]":
                break
            try:
                delta = json.loads(raw)["choices"][0].get("delta", {})
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            piece = delta.get("content")
            if piece:
                yield piece


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """你是申万食品饮料行业的资深卖方分析师，基于给定的半年报原文片段回答问题。

【铁律】
1. 只依据 <检索结果> 中的原文作答，不得使用任何外部知识、不得凭记忆补数字。
2. 每一处结论后面必须紧跟出处角标，格式为【序号】，序号对应 <检索结果> 的编号。
   例：报告期内公司营业收入同比增长10.5%【1】。
3. 如果检索结果不足以回答，就直接说"检索到的原文不足以回答该问题"，
   并说明缺什么，绝不允许编造数字。
4. 涉及金额时保留原文口径与单位（元 / 万元 / 亿元），不擅自换算。
5. 跨公司对比题：必须先分列各公司原文数据，再给出对比结论，每项数据都带角标。
6. 用简体中文、结构化输出（小标题 + 要点），不要客套话。"""


def build_context(chunks: Sequence[dict]) -> str:
    """把召回的文本块拼成带编号的上下文，编号即溯源角标。"""
    blocks = []
    for i, ch in enumerate(chunks, start=1):
        page = (f"{ch['page_start']}" if ch["page_start"] == ch["page_end"]
                else f"{ch['page_start']}-{ch['page_end']}")
        blocks.append(
            f"[{i}] 公司：{ch['company']} | 章节：{ch['section']} | 页码：{page}\n{ch['text']}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, chunks: Sequence[dict]) -> list[dict]:
    user = (f"<检索结果>\n{build_context(chunks)}\n</检索结果>\n\n"
            f"<问题>\n{question}\n</问题>\n\n"
            f"请严格依据检索结果回答，并在每个结论后标注【序号】角标。")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def build_eval_messages(question: str, answer: str, key_points: Sequence[str],
                        ref: str = "") -> list[dict]:
    """评测用：让模型当判官，逐条判定关键点是否被答案正确覆盖。"""
    points = "\n".join(f"{i}. {p}" for i, p in enumerate(key_points, start=1))
    user = f"""你是严格的事实核查员。请判断下面这份回答是否覆盖了全部关键点。

<问题>
{question}
</问题>

<参考答案要点>
{points}
</参考答案要点>
{f'<参考答案>{chr(10)}{ref}{chr(10)}</参考答案>' if ref else ''}

<待判定回答>
{answer}
</待判定回答>

只输出 JSON，不要任何其他文字：
{{
  "key_points": [{{"point": "关键点原文", "covered": true/false, "reason": "简短理由"}}],
  "citation_ok": true/false,
  "hallucination": true/false,
  "verdict": "答对" | "部分答对" | "答错",
  "failure_reason": "若未答对，从下列选项选一个：检索漏召 / 召回噪声干扰 / 生成幻觉 / 跨公司聚合失败 / 数值口径错误 / 其他；答对则填 无"
}}"""
    return [{"role": "system", "content": "你是严谨的金融事实核查员，只输出合法 JSON。"},
            {"role": "user", "content": user}]
