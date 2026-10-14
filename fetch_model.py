# -*- coding: utf-8 -*-
"""把 Qwen3-Embedding-0.6B 权重抓到本地目录（断点续传 + 自动重试）。

为什么不用 `snapshot_download()`
-------------------------------
国内镜像站（hf-mirror.com）传 1.2GB 大文件时，huggingface_hub 的默认读超时是
**10 秒**，慢网段必然触发 `The read operation timed out` / `SSL: UNEXPECTED_EOF`。
它虽然会 "Trying to resume download..."，但重试间隔很短、超时没变，
实测会卡在某个整数分块边界（本项目卡在 480 MiB）反复失败、不再前进。

本脚本自己管下载循环：**读超时放宽到 300 秒**、失败后退避重试、每次从
已落盘的字节数续传（HTTP Range），所以慢速抖动只会拖长时间，不会前功尽弃。
它还能直接接管 huggingface_hub 留下的 `.incomplete` 半成品，省掉重复流量。

用法
----
    python fetch_model.py                     # 默认下到 models/Qwen3-Embedding-0.6B
    python fetch_model.py --repo Qwen/Qwen3-Embedding-0.6B --out models/xxx
    python fetch_model.py --seed-from-hf-cache   # 自动接管 HF 缓存里的半成品

完成后把 .env 里的 EMBED_MODEL 指向该目录（绝对路径即可），
build_index.py / app.py / evaluate.py 都会用本地权重，不再联网。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

from utils.common import setup_console

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
CHUNK = 1024 * 256
# 读超时给足：慢网段下 10 秒必挂；连接超时保持短，快速失败才能尽快重试
TIMEOUT = (30, 300)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_files(repo: str) -> list[dict]:
    url = f"{ENDPOINT}/api/models/{repo}?blobs=true"
    with requests.get(url, timeout=TIMEOUT) as r:
        r.raise_for_status()
        data = r.json()
    out = []
    for f in data.get("siblings", []):
        lfs = f.get("lfs") or {}
        out.append({
            "name": f["rfilename"],
            "size": int(f.get("size") or lfs.get("size") or 0),
            "sha256": lfs.get("sha256"),
        })
    return out


def file_url(repo: str, name: str) -> str:
    return f"{ENDPOINT}/{repo}/resolve/main/{name}"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def sha256_of(path: Path, budget: float = 180.0) -> str | None:
    """算 sha256；超预算就放弃校验（大文件在慢盘上可能要几分钟）。"""
    h = hashlib.sha256()
    t0 = time.time()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1024 * 1024)
            if not b:
                break
            h.update(b)
            if time.time() - t0 > budget:
                return None
    return h.hexdigest()


def fetch(repo: str, name: str, dest: Path, size: int, sha256: str | None,
          max_attempts: int = 60) -> bool:
    """断点续传下载单个文件。返回是否成功。"""
    dest.parent.mkdir(parents=True, exist_ok=True)   # 1_Pooling/config.json 这类带子目录
    if dest.exists() and size and dest.stat().st_size == size:
        log(f"已完整，跳过：{name} ({human(size)})")
        return True

    url = file_url(repo, name)
    attempt = 0
    last_report = 0.0
    while attempt < max_attempts:
        have = dest.stat().st_size if dest.exists() else 0
        if size and have > size:          # 续传拼接出错，重来
            log(f"本地文件比目标大（{have} > {size}），删除重下：{name}")
            dest.unlink()
            have = 0
        if size and have == size:
            break

        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True,
                              timeout=TIMEOUT) as r:
                if r.status_code == 416:      # 已下完
                    break
                if have and r.status_code != 206:
                    # 服务端不支持 Range，只能从头来
                    log("服务端未返回 206，放弃已下部分，重新开始。")
                    have = 0
                r.raise_for_status()
                mode = "ab" if have else "wb"
                with open(dest, mode) as fh:
                    done = have
                    for block in r.iter_content(CHUNK):
                        if not block:
                            continue
                        fh.write(block)
                        done += len(block)
                        now = time.time()
                        if now - last_report > 10:
                            last_report = now
                            pct = f"{done / size:6.1%}" if size else "  ?  "
                            log(f"  {name}  {pct}  {human(done)} / {human(size) or '?'}")
        except (FileNotFoundError, PermissionError, IsADirectoryError) as exc:
            # 本地错误（路径/权限）重试多少次都一样，立刻报错，别浪费时间在退避上
            log(f"× {name} 本地写入失败：{exc}")
            return False
        except Exception as exc:
            attempt += 1
            have_now = dest.stat().st_size if dest.exists() else 0
            wait = min(5 * attempt, 45)
            log(f"  第 {attempt} 次中断（{type(exc).__name__}: {str(exc)[:70]}）"
                f"，已落盘 {human(have_now)}，{wait}s 后续传")
            time.sleep(wait)
            continue

        # 一次连接正常结束，检查是否真的下完了
        if size and dest.stat().st_size != size:
            attempt += 1
            log(f"  连接结束但只到 {human(dest.stat().st_size)} / {human(size)}，续传")
            continue
        break

    got = dest.stat().st_size if dest.exists() else 0
    if size and got != size:
        log(f"× {name} 未完成：{human(got)} / {human(size)}")
        return False

    if sha256:
        log(f"  校验 sha256（{name}，{human(got)}，稍候）…")
        got_hash = sha256_of(dest)
        if got_hash is None:
            log(f"  ! 校验超时跳过（文件大小已正确：{human(got)}）")
        elif got_hash != sha256:
            log(f"× {name} sha256 不匹配，删除重下")
            dest.unlink()
            return False
        else:
            log(f"  ✓ {name} sha256 校验通过")

    log(f"✓ {name} 完成（{human(got)}）")
    return True


def seed_from_hf_cache(dest: Path, sha_prefix: str) -> bool:
    """接管 huggingface_hub 留下的 .incomplete 半成品，省掉重复流量。"""
    hub = Path(os.environ.get("HF_HOME") or
               Path.home() / ".cache" / "huggingface") / "hub"
    if not hub.exists():
        return False
    cands = [p for p in hub.rglob(f"{sha_prefix}*.incomplete") if p.is_file()]
    if not cands:
        return False
    src = max(cands, key=lambda p: p.stat().st_size)
    dst_size = dest.stat().st_size if dest.exists() else 0
    if src.stat().st_size <= dst_size:
        log(f"HF 缓存的半成品不比本地大（{human(src.stat().st_size)}），不接管")
        return False
    log(f"接管 HF 缓存半成品 {src.name}（{human(src.stat().st_size)}）→ {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, open(dest, "wb") as fout:
        while True:
            b = fin.read(1024 * 1024)
            if not b:
                break
            fout.write(b)
    src.unlink()          # 已搬走，删掉避免继续占地方
    return True


def main() -> int:
    setup_console()          # Windows 控制台默认 GBK，不切 UTF-8 会死在 ✓/× 上
    ap = argparse.ArgumentParser(description="断点续传下载 embedding 模型权重")
    ap.add_argument("--repo", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--out", type=Path, default=None,
                    help="本地目录（默认 models/<模型名>）")
    ap.add_argument("--seed-from-hf-cache", action="store_true",
                    help="自动接管 huggingface_hub 的半成品 .incomplete")
    args = ap.parse_args()

    out = args.out or Path("models") / args.repo.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    log(f"镜像端点 {ENDPOINT}")
    log(f"目标目录 {out.resolve()}")

    try:
        files = list_files(args.repo)
    except Exception as exc:
        log(f"× 无法读取文件清单（{exc}）；检查网络或 HF_ENDPOINT")
        return 1

    total = sum(f["size"] for f in files)
    log(f"共 {len(files)} 个文件，合计 {human(total)}")

    # 大文件若本地只有半成品，先接管 HF 缓存里更大的那份
    if args.seed_from_hf_cache:
        for f in files:
            if f["sha256"] and (f["size"] or 0) > 50 * 1024 * 1024:
                seed_from_hf_cache(out / f["name"], f["sha256"][:24])

    failed = []
    for i, f in enumerate(files, 1):
        log(f"[{i}/{len(files)}] {f['name']}  {human(f['size'])}")
        if not fetch(args.repo, f["name"], out / f["name"], f["size"], f["sha256"]):
            failed.append(f["name"])

    print()
    if failed:
        log(f"× 有 {len(failed)} 个文件失败：{failed}")
        log("  重跑本脚本即可续传（已完成的会跳过）：python fetch_model.py --seed-from-hf-cache")
        return 1

    log("全部完成。请在 .env 中设置：")
    print(f"    EMBED_MODEL={out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
