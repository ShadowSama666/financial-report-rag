# -*- coding: utf-8 -*-
"""公共小工具：Windows 控制台中文编码、日志、环境变量、文件大小格式化。"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_LOG_INITED = False


def setup_console() -> None:
    """Windows 下把 stdout/stderr 切到 UTF-8，避免中文 + 进度条报 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    global _LOG_INITED
    logger = logging.getLogger(name)
    if not _LOG_INITED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logging.basicConfig(level=level, handlers=[handler])
        _LOG_INITED = True
    logger.setLevel(level)
    return logger


def load_env(dotenv_path: Path | str | None = None) -> None:
    """加载 .env（若安装 python-dotenv）。已存在的环境变量不被覆盖。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def slugify_filename(text: str) -> str:
    """把公司名/标题转成安全的文件名片段（保留中文与字母数字）。"""
    bad = '<>:"/\\|?*\n\r\t'
    return "".join(ch for ch in text if ch not in bad).strip()


def page_label(page_start: int, page_end: int) -> str:
    """页码显示口径：同页显示「26」，跨页显示「26-29」。

    散点图悬浮、出处角标、评测 recall_detail.csv 都读这一个函数——
    三处各写一遍的话，很容易出现界面写「26-29」而评测表写「26」的错位。
    """
    if page_start == page_end:
        return str(page_start)
    return f"{page_start}-{page_end}"
