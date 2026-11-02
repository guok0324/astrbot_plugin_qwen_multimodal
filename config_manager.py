"""配置读取与运行时状态持久化。

本模块承担两件事：

1. ``PluginConfig``：把 AstrBot 的嵌套配置字典包装成带默认值、带类型转换的只读视图。
   AstrBot 的 ``_conf_schema.json`` 在升级时**只会补全缺失项，不会删除多余项**，
   而且用户手改配置时可能把数字写成字符串，因此这里统一做兜底与强制类型转换。
2. ``StateStore``：把“锁定音色档案 / 会话级模式 / 上次发语音时间”持久化到
   ``data/qwen_multimodal_state.json``，读写全部走 asyncio 锁 + 原子替换，避免
   群聊并发时写坏文件。
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from astrbot.api import logger

PLUGIN_NAME = "astrbot_plugin_qwen_multimodal"

_MISSING = object()


async def _to_thread(func, *args):
    """asyncio.to_thread 的 3.8 兼容封装（AstrBot 一般运行于 3.10+）。"""
    to_thread = getattr(asyncio, "to_thread", None)
    if to_thread is not None:
        return await to_thread(func, *args)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args))


def resolve_data_dir() -> Path:
    """解析插件数据目录。

    优先使用 AstrBot 提供的 ``StarTools.get_data_dir()``；不同版本签名不同
    （旧版需要插件名，新版不需要），因此逐个尝试。全部失败时退回到
    AstrBot 数据根目录下的 ``plugin_data/<插件名>``，再失败则用插件自身目录下的
    ``data``，保证插件在极端环境下也能启动而不是直接崩溃。
    """
    try:
        from astrbot.api.star import StarTools  # type: ignore

        # 新版 AstrBot 的 get_data_dir() 会通过调用栈猜测插件名，而我们的调用
        # 发生在 config_manager 里，猜测必然失败；所以显式传插件名优先。
        for call in (
            lambda: StarTools.get_data_dir(PLUGIN_NAME),
            lambda: StarTools.get_data_dir(),
        ):
            try:
                value = call()
            except TypeError:
                continue
            except Exception:  # noqa: BLE001 - 任何异常都只代表该方式不可用
                continue
            if value:
                return Path(str(value))
    except Exception:  # noqa: BLE001
        pass

    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path  # type: ignore

        return Path(str(get_astrbot_data_path())) / "plugin_data" / PLUGIN_NAME
    except Exception:  # noqa: BLE001
        pass

    return Path(__file__).resolve().parent / "data"


def deep_get(data: Any, dotted: str, default: Any = None) -> Any:
    """按 ``a.b.c`` 路径读取嵌套字典。"""
    current = data
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    if current is None:
        return default
    return current


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """返回 base 被 override 覆盖后的新字典（递归）。"""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# 与 _conf_schema.json 保持一致的默认值。Schema 已经提供默认值，这里再写一份是为了
# 在“配置文件缺失 / 插件被热重载 / 用户删除了某个分组”时依然可以工作。
DEFAULT_CONFIG: Dict[str, Any] = {
    "dashscope_api_key": "",
    "tts_settings": {
        "enable_tts": True,
        "tts_model": "qwen-audio-3.1-tts-plus",
        "tts_api_endpoint": (
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
            "multimodal-generation/generation"
        ),
        "trigger_mode": "emotion_auto",
        "emotion_threshold": 6,
        "base_probability": 0.15,
        "max_tts_chars": 120,
        "min_tts_chars": 2,
        "cooldown_seconds": 15,
        "dual_send_text": False,
        "audio_format": "mp3",
        "max_tts_chunks": 3,
    },
    "persona_voice_settings": {
        "auto_match_persona": True,
        "manual_override_voice": "",
        "allow_dynamic_prosody": True,
        "auto_recalibrate_on_persona_change": False,
    },
    "image_settings": {
        "enable_image_gen": True,
        "image_model": "qwen-image-2.1-pro",
        "image_trigger_mode": "auto_and_command",
        "default_size": "1024*1024",
        "negative_prompt": (
            "lowres, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality, "
            "watermark, blurry"
        ),
        "poll_timeout_seconds": 60,
        "max_images_per_reply": 2,
    },
    "group_control": {
        "enable_in_private": True,
        "group_mode": "all",
        "group_list": [],
    },
    "advanced": {
        "request_timeout_seconds": 120,
        "temp_file_ttl_seconds": 60,
        "sample_rate": 24000,
    },
}

VALID_TRIGGER_MODES = ("emotion_auto", "hybrid_prob", "always", "keyword_only", "off")


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "y", "是", "开")
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return default
    return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return float(value)
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


class PluginConfig:
    """AstrBot 配置的安全只读包装。"""

    def __init__(self, raw: Optional[Dict[str, Any]] = None) -> None:
        self.raw: Dict[str, Any] = raw if isinstance(raw, dict) else {}
        self._merged = deep_merge(DEFAULT_CONFIG, self.raw)

    # ------------------------------------------------------------------ 基础读取
    def get(self, dotted: str, default: Any = None) -> Any:
        value = deep_get(self.raw, dotted, _MISSING)
        if value is _MISSING or value is None:
            value = deep_get(self._merged, dotted, default)
        return default if value is None else value

    def s(self, dotted: str, default: str = "") -> str:
        value = self.get(dotted, default)
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    def i(self, dotted: str, default: int = 0) -> int:
        return _as_int(self.get(dotted, default), default)

    def f(self, dotted: str, default: float = 0.0) -> float:
        return _as_float(self.get(dotted, default), default)

    def b(self, dotted: str, default: bool = False) -> bool:
        return _as_bool(self.get(dotted, default), default)

    def lst(self, dotted: str, default: Optional[List[Any]] = None) -> List[Any]:
        value = self.get(dotted, default if default is not None else [])
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [item.strip() for item in value.replace("，", ",").split(",") if item.strip()]
        return list(default or [])

    # ------------------------------------------------------------------ 校验
    def validate(self) -> List[str]:
        """返回配置问题列表（仅用于启动日志提醒，不阻塞加载）。"""
        issues: List[str] = []
        if not self.api_key:
            issues.append("未配置 dashscope_api_key，语音与生图功能将不可用。")
        if self.trigger_mode not in VALID_TRIGGER_MODES:
            issues.append(
                f"tts_settings.trigger_mode={self.trigger_mode!r} 非法，已回退 emotion_auto。"
            )
        if not 1 <= self.emotion_threshold <= 10:
            issues.append("tts_settings.emotion_threshold 超出 1-10，已自动钳制。")
        if self.max_tts_chars <= self.min_tts_chars:
            issues.append("max_tts_chars 必须大于 min_tts_chars，已自动修正。")
        if self.group_mode not in ("all", "whitelist", "blacklist"):
            issues.append("group_control.group_mode 非法，已按 all 处理。")
        return issues

    # ------------------------------------------------------------------ 常用快捷属性
    @property
    def api_key(self) -> str:
        return self.s("dashscope_api_key", "")

    @property
    def enable_tts(self) -> bool:
        return self.b("tts_settings.enable_tts", True)

    @property
    def enable_image(self) -> bool:
        return self.b("image_settings.enable_image_gen", True)

    @property
    def tts_model(self) -> str:
        return self.s("tts_settings.tts_model", "qwen-audio-3.1-tts-plus")

    @property
    def tts_endpoint(self) -> str:
        return self.s(
            "tts_settings.tts_api_endpoint",
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
            "multimodal-generation/generation",
        )

    @property
    def image_model(self) -> str:
        return self.s("image_settings.image_model", "qwen-image-2.1-pro")

    @property
    def image_negative_prompt(self) -> str:
        return self.s("image_settings.negative_prompt", "")

    @property
    def trigger_mode(self) -> str:
        mode = self.s("tts_settings.trigger_mode", "emotion_auto")
        return mode if mode in VALID_TRIGGER_MODES else "emotion_auto"

    @property
    def image_trigger_mode(self) -> str:
        mode = self.s("image_settings.image_trigger_mode", "auto_and_command")
        return mode if mode in ("auto_and_command", "command_only") else "auto_and_command"

    @property
    def emotion_threshold(self) -> int:
        return max(1, min(10, self.i("tts_settings.emotion_threshold", 6)))

    @property
    def base_probability(self) -> float:
        return max(0.0, min(1.0, self.f("tts_settings.base_probability", 0.15)))

    @property
    def min_tts_chars(self) -> int:
        return max(1, self.i("tts_settings.min_tts_chars", 2))

    @property
    def max_tts_chars(self) -> int:
        value = self.i("tts_settings.max_tts_chars", 120)
        return value if value > self.min_tts_chars else self.min_tts_chars + 10

    @property
    def cooldown_seconds(self) -> int:
        return max(0, self.i("tts_settings.cooldown_seconds", 15))

    @property
    def dual_send_text(self) -> bool:
        return self.b("tts_settings.dual_send_text", False)

    @property
    def audio_format(self) -> str:
        fmt = self.s("tts_settings.audio_format", "mp3").lower()
        return fmt if fmt in ("mp3", "wav", "pcm") else "mp3"

    @property
    def max_tts_chunks(self) -> int:
        return max(1, min(10, self.i("tts_settings.max_tts_chunks", 3)))

    @property
    def sample_rate(self) -> int:
        return max(8000, self.i("advanced.sample_rate", 24000))

    @property
    def request_timeout(self) -> int:
        return max(10, self.i("advanced.request_timeout_seconds", 120))

    @property
    def temp_file_ttl(self) -> int:
        return max(5, self.i("advanced.temp_file_ttl_seconds", 60))

    @property
    def poll_timeout(self) -> int:
        return max(10, self.i("image_settings.poll_timeout_seconds", 60))

    @property
    def default_size(self) -> str:
        size = self.s("image_settings.default_size", "1024*1024")
        return size if size in ("1024*1024", "720*1280", "1280*720") else "1024*1024"

    @property
    def max_images_per_reply(self) -> int:
        return max(1, min(5, self.i("image_settings.max_images_per_reply", 2)))

    @property
    def auto_match_persona(self) -> bool:
        return self.b("persona_voice_settings.auto_match_persona", True)

    @property
    def manual_voice(self) -> str:
        return self.s("persona_voice_settings.manual_override_voice", "")

    @property
    def allow_dynamic_prosody(self) -> bool:
        return self.b("persona_voice_settings.allow_dynamic_prosody", True)

    @property
    def auto_recalibrate(self) -> bool:
        return self.b("persona_voice_settings.auto_recalibrate_on_persona_change", False)

    @property
    def group_mode(self) -> str:
        mode = self.s("group_control.group_mode", "all")
        return mode if mode in ("all", "whitelist", "blacklist") else "all"

    @property
    def group_list(self) -> List[str]:
        return [str(item).strip() for item in self.lst("group_control.group_list", [])]

    def allow_session(self, is_group: bool, group_id: str = "") -> bool:
        """判断当前会话是否允许触发语音/生图。"""
        if is_group:
            mode = self.group_mode
            if mode == "whitelist":
                return str(group_id) in self.group_list
            if mode == "blacklist":
                return str(group_id) not in self.group_list
            return True
        return self.b("group_control.enable_private", self.b("group_control.enable_in_private", True))


# ---------------------------------------------------------------------------
# 状态持久化
# ---------------------------------------------------------------------------

DEFAULT_STATE: Dict[str, Any] = {
    "schema_version": 1,
    "locked_voice_profile": None,
    "session_overrides": {},
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class StateStore:
    """``qwen_multimodal_state.json`` 的异步安全读写封装。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: Dict[str, Any] = copy.deepcopy(DEFAULT_STATE)
        self._loaded = False
        self._lock: Optional[asyncio.Lock] = None

    # ------------------------------------------------------------------ 基础
    @property
    def lock(self) -> asyncio.Lock:
        """延迟创建锁，避免在事件循环之外实例化。"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def load(self, force: bool = False) -> Dict[str, Any]:
        if self._loaded and not force:
            return self._data
        async with self.lock:
            if self._loaded and not force:
                return self._data
            self._data = await self._read_from_disk()
            self._loaded = True
        return self._data

    async def _read_from_disk(self) -> Dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            return copy.deepcopy(DEFAULT_STATE)
        try:
            raw = await _to_thread(self.path.read_text, "utf-8")
            data = json.loads(raw) if raw.strip() else {}
        except (OSError, ValueError) as exc:
            logger.warning(
                f"[qwen_multimodal] 状态文件解析失败，将备份并使用默认状态: {exc}"
            )
            try:
                backup = self.path.with_suffix(".json.corrupt")
                await _to_thread(self.path.replace, backup)
            except OSError:
                pass
            return copy.deepcopy(DEFAULT_STATE)

        merged = deep_merge(DEFAULT_STATE, data if isinstance(data, dict) else {})
        if not isinstance(merged.get("session_overrides"), dict):
            merged["session_overrides"] = {}
        return merged

    async def save(self) -> None:
        async with self.lock:
            await self._write_to_disk()

    async def _write_to_disk(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._data, ensure_ascii=False, indent=2)
        tmp_path = self.path.with_suffix(f".tmp.{os.getpid()}")
        try:
            await _to_thread(tmp_path.write_text, payload, "utf-8")
            await _to_thread(os.replace, str(tmp_path), str(self.path))
        except OSError as exc:
            logger.error(f"[qwen_multimodal] 状态文件写入失败: {exc}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------ 音色档案
    def locked_profile(self) -> Optional[Dict[str, Any]]:
        profile = self._data.get("locked_voice_profile")
        return profile if isinstance(profile, dict) and profile.get("voice_id") else None

    async def lock_voice(self, profile: Dict[str, Any]) -> None:
        async with self.lock:
            record = dict(profile)
            record.setdefault("locked_at", _now_iso())
            self._data["locked_voice_profile"] = record
            await self._write_to_disk()

    async def reset_voice(self) -> None:
        async with self.lock:
            self._data["locked_voice_profile"] = None
            await self._write_to_disk()

    # ------------------------------------------------------------------ 会话
    def session(self, key: str) -> Dict[str, Any]:
        overrides = self._data.get("session_overrides")
        if not isinstance(overrides, dict):
            overrides = {}
            self._data["session_overrides"] = overrides
        data = overrides.get(key)
        if not isinstance(data, dict):
            data = {}
            overrides[key] = data
        return data

    def session_mode(self, key: str) -> Optional[str]:
        mode = self.session(key).get("tts_mode")
        return mode if mode in VALID_TRIGGER_MODES else None

    async def set_session_mode(self, key: str, mode: str) -> None:
        async with self.lock:
            self.session(key)["tts_mode"] = mode
            self.session(key)["updated_at"] = _now_iso()
            await self._write_to_disk()

    def last_speech_ts(self, key: str) -> float:
        value = self.session(key).get("last_tts_timestamp")
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def in_cooldown(self, key: str, cooldown_seconds: int) -> bool:
        if cooldown_seconds <= 0:
            return False
        last = self.last_speech_ts(key)
        if last <= 0:
            return False
        return (time.time() - last) < cooldown_seconds

    async def record_speech(self, key: str) -> None:
        async with self.lock:
            session = self.session(key)
            session["last_tts_timestamp"] = time.time()
            session["tts_count"] = _as_int(session.get("tts_count"), 0) + 1
            await self._write_to_disk()

    def session_tts_count(self, key: str) -> int:
        return _as_int(self.session(key).get("tts_count"), 0)
