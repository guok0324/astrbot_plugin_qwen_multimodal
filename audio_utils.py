"""音频/图片临时文件与字节工具。

为什么需要单独一层：
* OneBot 协议端（NapCat / Lagrange）在收到语音后**不是立刻**读文件，而是排队做
  SILK 转码。如果插件发完就删，会随机出现「语音发送失败」「播放无声」。
  所以统一延迟删除（默认 60 秒），并把删除任务集中管理，插件卸载时能取消。
* 不同模型返回的音频封装格式并不稳定（文档写 mp3，实际常回 wav）。这里靠魔术字节
  判断真实格式并纠正扩展名，避免协议端因为 ``.mp3`` 里装着 WAV 而拒收。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

from astrbot.api import logger

AUDIO_EXTS = {".mp3", ".wav", ".pcm", ".silk", ".amr", ".ogg", ".m4a", ".flac"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


# ---------------------------------------------------------------------------
# 目录与清理
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def purge_dir(path: Path, suffixes: Optional[Set[str]] = None) -> int:
    """清空目录里的历史临时文件，返回删除数量。"""
    removed = 0
    path = Path(path)
    if not path.exists():
        return 0
    for item in path.iterdir():
        try:
            if item.is_file() and (suffixes is None or item.suffix.lower() in suffixes):
                item.unlink()
                removed += 1
        except OSError as exc:  # 文件被协议端占用时忽略即可
            logger.debug(f"[qwen_multimodal] 清理临时文件失败 {item}: {exc}")
    return removed


async def cleanup_file_later(path: Path, delay: float = 60.0) -> None:
    """延迟删除文件（给 NapCat 留出转码时间）。"""
    try:
        await asyncio.sleep(max(1.0, delay))
        if Path(path).exists():
            Path(path).unlink()
    except asyncio.CancelledError:
        raise
    except OSError as exc:
        logger.debug(f"[qwen_multimodal] 延迟删除失败 {path}: {exc}")


@dataclass
class TempFileManager:
    """管理 temp_audio / temp_image 两个目录与延迟删除任务。"""

    base_dir: Path
    ttl_seconds: float = 60.0
    audio_dir: Path = field(init=False)
    image_dir: Path = field(init=False)
    _tasks: Set[asyncio.Task] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.audio_dir = ensure_dir(self.base_dir / "temp_audio")
        self.image_dir = ensure_dir(self.base_dir / "temp_image")

    # ------------------------------------------------------------------ 分配
    def new_audio_path(self, ext: str = ".mp3") -> Path:
        ext = ext if ext.startswith(".") else f".{ext}"
        return self.audio_dir / f"{uuid.uuid4().hex}{ext.lower()}"

    def new_image_path(self, ext: str = ".png") -> Path:
        ext = ext if ext.startswith(".") else f".{ext}"
        return self.image_dir / f"{uuid.uuid4().hex}{ext.lower()}"

    # ------------------------------------------------------------------ 清理
    def purge(self) -> None:
        audio = purge_dir(self.audio_dir, AUDIO_EXTS)
        image = purge_dir(self.image_dir, IMAGE_EXTS)
        if audio or image:
            logger.info(
                f"[qwen_multimodal] 已清理历史临时文件：音频 {audio} 个，图片 {image} 个。"
            )

    def schedule_cleanup(self, path: Path, delay: Optional[float] = None) -> asyncio.Task:
        task = asyncio.create_task(
            cleanup_file_later(Path(path), self.ttl_seconds if delay is None else delay)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.purge()


# ---------------------------------------------------------------------------
# 音频字节处理
# ---------------------------------------------------------------------------


def sniff_audio_ext(data: bytes) -> Optional[str]:
    """根据魔术字节判断音频封装格式。"""
    if not data:
        return None
    head = data[:12]
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return ".wav"
    if head[:4] == b"OggS":
        return ".ogg"
    if head[:4] == b"fLaC":
        return ".flac"
    if head[:3] == b"ID3":
        return ".mp3"
    if head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa"):
        return ".mp3"
    if head[:4] == b"#!AMR":
        return ".amr"
    if head[:4] == b"\x02#!S":
        return ".silk"
    if head[4:8] == b"ftyp":
        return ".m4a"
    if b"LAME" in data[:512] or b"Info" in data[:64]:
        return ".mp3"
    return None


def ext_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    clean = url.split("?")[0].split("#")[0]
    suffix = Path(clean).suffix.lower()
    return suffix if suffix in AUDIO_EXTS | IMAGE_EXTS else None


def decode_base64_audio(value: str) -> bytes:
    """解码 Base64 音频；兼容 data URI 前缀与 URL-safe 变体。"""
    if not value:
        return b""
    raw = value.strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    raw = raw.replace("\n", "").replace("\r", "").replace(" ", "")
    try:
        return base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        try:
            return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
        except (binascii.Error, ValueError):
            return b""


def write_bytes_atomic(path: Path, data: bytes) -> Path:
    """先写临时文件再原子替换，避免发送线程读到半个文件。"""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))
    return path


def normalize_audio_file(path: Path, data: bytes, prefer_ext: str = ".mp3") -> Path:
    """按真实字节修正扩展名；若已有同名错误扩展名文件则重命名。"""
    actual = sniff_audio_ext(data)
    desired = actual or (prefer_ext if prefer_ext.startswith(".") else f".{prefer_ext}")
    if path.suffix.lower() != desired:
        target = path.with_suffix(desired)
        try:
            if path.exists():
                os.replace(str(path), str(target))
            return target
        except OSError:
            return path
    return path


def file_info(path: Path) -> str:
    """给日志用的简短描述。"""
    try:
        size = Path(path).stat().st_size
    except OSError:
        size = 0
    return f"{Path(path).name} ({size / 1024:.1f} KB)"


def is_valid_media_file(path: Path, min_bytes: int = 512) -> bool:
    try:
        return Path(path).is_file() and Path(path).stat().st_size >= min_bytes
    except OSError:
        return False


def iso_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha1_short(text: str, length: int = 8) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]
