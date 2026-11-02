"""音色目录、人设解析与「AI 自主选音色」引擎。

设计要点
--------
* 千问 TTS 的音色 ID 是一串英文（Cherry / Ethan / longxiaochun ...），而 AstrBot 的
  人设是一段中文 Prompt。普通插件让用户手填音色，这里改成：**让 AI 读自己的人设，
  自己挑一个最像自己的声音**，并把挑选理由（第一人称心路历程）一起存下来。
* 音色挑选只在首次发生时调用一次 LLM，随后以 ``persona_hash`` 永久锁定；人设没变就
  永远不会再花这笔钱。人设改了且用户开启了自动重校准，才会重新“觉醒”。
* 任何一步失败（没有 Provider、LLM 超时、返回不是 JSON、音色 ID 不在目录里）都必须
  优雅降级：关键词正则匹配 → Cherry 兜底，绝不因为选音色失败而阻断聊天。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from astrbot.api import logger

try:  # 包内相对导入；插件被以单文件方式加载时退回绝对导入
    from .config_manager import PluginConfig, StateStore
except ImportError:  # pragma: no cover
    from config_manager import PluginConfig, StateStore  # type: ignore


async def _maybe_await(value: Any) -> Any:
    """兼容不同 AstrBot 版本：同一个方法既可能是同步的，也可能是协程。"""
    if inspect.isawaitable(value):
        return await value
    return value


# ---------------------------------------------------------------------------
# 音色目录
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceEntry:
    """一条音色档案。``voice_id`` 为百炼 Qwen-TTS 的默认音色 ID。"""

    voice_id: str
    name_cn: str
    alias: str
    gender: str
    tags: Tuple[str, ...]
    base_speed: float
    base_pitch: float
    description: str
    alt_ids: Tuple[str, ...] = field(default_factory=tuple)

    def all_ids(self) -> List[str]:
        """所有可用于查找本音色的标识：ID、专属别名、备用 ID。"""
        return [self.voice_id, self.alias, *self.alt_ids]

    def label(self) -> str:
        return f"{self.name_cn}（{self.voice_id}）"


VOICE_CATALOG: Tuple[VoiceEntry, ...] = (
    VoiceEntry(
        voice_id="Cherry",
        name_cn="芊悦",
        alias="longxiaochun",
        gender="女",
        tags=("温柔", "甜美", "邻家", "治愈", "女友", "软萌"),
        base_speed=1.0,
        base_pitch=1.0,
        description="温柔甜美的邻家少女，声线干净治愈，适合日常闲聊与安慰人。",
    ),
    VoiceEntry(
        voice_id="Serena",
        name_cn="苏瑶",
        alias="longxiaoxia",
        gender="女",
        tags=("元气", "活泼", "傲娇", "古灵精怪", "二次元", "话痨"),
        base_speed=1.08,
        base_pitch=1.05,
        description="元气满满的少女音，语速偏快，情绪外放，适合活泼傲娇的人设。",
    ),
    VoiceEntry(
        voice_id="Chelsie",
        name_cn="千雪",
        alias="longwan",
        gender="女",
        tags=("清冷", "御姐", "知性", "高冷", "毒舌", "大姐姐"),
        base_speed=0.95,
        base_pitch=0.95,
        description="清冷知性的御姐音，语速从容，尾音克制，适合高冷毒舌的大姐姐。",
    ),
    VoiceEntry(
        voice_id="Momo",
        name_cn="茉茉",
        alias="longyue",
        gender="女",
        tags=("俏皮", "可爱", "撒娇", "粘人", "小恶魔"),
        base_speed=1.05,
        base_pitch=1.08,
        description="俏皮软糯的少女音，尾音上挑带笑，适合撒娇粘人的小恶魔人设。",
    ),
    VoiceEntry(
        voice_id="Ethan",
        name_cn="晨煦",
        alias="longanyang",
        gender="男",
        tags=("阳光", "少年", "温暖", "学长", "清爽"),
        base_speed=1.02,
        base_pitch=1.0,
        description="阳光清爽的少年音，明亮干净，适合温柔学长或元气少年。",
    ),
    VoiceEntry(
        voice_id="Vincent",
        name_cn="文森",
        alias="longshuo",
        gender="男",
        tags=("低音炮", "成熟", "稳重", "霸道总裁", "磁性", "叔系"),
        base_speed=0.92,
        base_pitch=0.90,
        description="低沉磁性的成熟男声，语速偏慢，适合霸道总裁或沉稳叔系角色。",
    ),
)

# 中文别名（龙小淳 / 龙小夏 / 龙婉 / 龙悦 / 龙安洋 / 龙硕）→ 默认音色 ID
_CN_DISPLAY_ALIASES: Dict[str, str] = {
    "芊悦": "龙小淳",
    "苏瑶": "龙小夏",
    "千雪": "龙婉",
    "茉茉": "龙悦",
    "晨煦": "龙安洋",
    "文森": "龙硕",
}

_CN_ALIASES: Dict[str, str] = {}
for _entry in VOICE_CATALOG:
    _CN_ALIASES[_entry.voice_id.lower()] = _entry.voice_id
    _CN_ALIASES[_entry.alias] = _entry.voice_id
    _CN_ALIASES[_entry.name_cn] = _entry.voice_id
    _CN_ALIASES[_CN_DISPLAY_ALIASES[_entry.name_cn]] = _entry.voice_id

VOICE_INDEX: Dict[str, VoiceEntry] = {}
for _entry in VOICE_CATALOG:
    for _alias in _entry.all_ids():
        VOICE_INDEX[_alias.lower()] = _entry
    VOICE_INDEX[_entry.name_cn] = _entry
    VOICE_INDEX[_CN_DISPLAY_ALIASES[_entry.name_cn]] = _entry


def get_voice_entry(voice_id: str) -> Optional[VoiceEntry]:
    """按 ID / 别名 / 中文名查找音色。找不到返回 None（代表用户自定义音色）。"""
    if not voice_id:
        return None
    key = str(voice_id).strip()
    if key.lower() in VOICE_INDEX:
        return VOICE_INDEX[key.lower()]
    if key in VOICE_INDEX:
        return VOICE_INDEX[key]
    return None


def canonical_voice_id(voice_id: str) -> Tuple[str, Optional[VoiceEntry]]:
    """把任意用户输入规范成可提交给百炼的 voice 参数。"""
    entry = get_voice_entry(voice_id)
    if entry is not None:
        return entry.voice_id, entry
    return str(voice_id).strip(), None


# ---------------------------------------------------------------------------
# 情绪韵律预设
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmotionPreset:
    speed_mult: float
    pitch_mult: float
    instruct: str
    keywords: Tuple[str, ...] = ()


# 键名即 ``<tts emotion="...">`` 中允许出现的情绪类型
EMOTION_PRESETS: Dict[str, EmotionPreset] = {
    "happy": EmotionPreset(
        1.10, 1.06, "语气轻快愉悦，声音带笑，句尾自然上扬", ("开心", "高兴", "快乐", "愉快")
    ),
    "excited": EmotionPreset(
        1.18,
        1.12,
        "语气非常激动急促，带有强烈的情绪起伏",
        ("激动", "兴奋", "期待", "惊喜"),
    ),
    "angry": EmotionPreset(
        1.18, 1.12, "语气非常激动急促，带有强烈的情绪起伏", ("生气", "愤怒", "火大", "恼火")
    ),
    "coquettish": EmotionPreset(
        0.92, 1.08, "语气软糯甜美，带着撒娇和害羞的轻微气音，尾音拖长", ("撒娇", "黏人", "甜腻")
    ),
    "shy": EmotionPreset(
        0.92, 1.08, "语气软糯甜美，带着撒娇和害羞的轻微气音，尾音拖长", ("害羞", "腼腆", "脸红")
    ),
    "sad": EmotionPreset(
        0.85, 0.90, "语气低落委屈，语速缓慢，略带哭腔和叹息感", ("难过", "伤心", "失落", "沮丧")
    ),
    "aggrieved": EmotionPreset(
        0.85, 0.90, "语气低落委屈，语速缓慢，略带哭腔和叹息感", ("委屈", "可怜", "想哭")
    ),
    "sleepy": EmotionPreset(
        0.80, 0.92, "声音轻柔慵懒，像刚睡醒或在耳边轻声细语", ("困", "累", "疲惫", "懒")
    ),
    "whisper": EmotionPreset(
        0.80, 0.92, "声音轻柔慵懒，像刚睡醒或在耳边轻声细语", ("耳语", "悄悄", "小声")
    ),
    "sarcastic": EmotionPreset(
        1.08, 1.05, "语气傲娇不屑，重音明显，带着哼声", ("嘲讽", "阴阳怪气", "冷笑")
    ),
    "tsundere": EmotionPreset(
        1.08, 1.05, "语气傲娇不屑，重音明显，带着哼声", ("傲娇", "嘴硬", "口是心非")
    ),
    "shocked": EmotionPreset(
        1.12, 1.10, "语气惊讶错愕，语调先扬后顿，带有明显的不可置信", ("震惊", "惊讶", "吃惊")
    ),
    "neutral": EmotionPreset(1.0, 1.0, "", ("平静", "普通", "日常")),
}

# 中文情绪词 → 英文标签
_EMOTION_ALIASES: Dict[str, str] = {}
for _name, _preset in EMOTION_PRESETS.items():
    _EMOTION_ALIASES[_name] = _name
    for _kw in _preset.keywords:
        _EMOTION_ALIASES[_kw] = _name

_EMOTION_ALIASES.update(
    {
        "开心": "happy",
        "高兴": "happy",
        "激动": "excited",
        "兴奋": "excited",
        "生气": "angry",
        "愤怒": "angry",
        "撒娇": "coquettish",
        "害羞": "shy",
        "难过": "sad",
        "委屈": "aggrieved",
        "困": "sleepy",
        "耳语": "whisper",
        "嘲讽": "sarcastic",
        "傲娇": "tsundere",
        "震惊": "shocked",
        "平淡": "neutral",
    }
)


def normalize_emotion(value: Any) -> str:
    """把任意情绪写法（中文、大小写、带空格）规范成预设键名。"""
    if not value:
        return "neutral"
    key = str(value).strip().strip("\"'").lower()
    if key in _EMOTION_ALIASES:
        return _EMOTION_ALIASES[key]
    for alias, name in _EMOTION_ALIASES.items():
        if alias in key:
            return name
    return "neutral"


@dataclass
class Prosody:
    """最终提交给 TTS 的韵律参数。"""

    speed: float
    pitch: float
    instruct: str
    emotion: str = "neutral"
    intensity: int = 5


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# 音色档案
# ---------------------------------------------------------------------------


@dataclass
class VoiceProfile:
    voice_id: str
    voice_name: str
    base_speed: float
    base_pitch: float
    characteristic_style: str
    reason: str
    persona_name: str = ""
    persona_hash: str = ""
    source: str = "fallback"
    locked_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "voice_name": self.voice_name,
            "base_speed": round(self.base_speed, 3),
            "base_pitch": round(self.base_pitch, 3),
            "characteristic_style": self.characteristic_style,
            "reason": self.reason,
            "persona_name": self.persona_name,
            "persona_hash": self.persona_hash,
            "source": self.source,
            "locked_at": self.locked_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceProfile":
        entry = get_voice_entry(str(data.get("voice_id", "")))
        return cls(
            voice_id=str(data.get("voice_id", "") or "Cherry"),
            voice_name=str(data.get("voice_name") or (entry.name_cn if entry else "自定义音色")),
            base_speed=clamp(_to_float(data.get("base_speed"), 1.0), 0.5, 2.0),
            base_pitch=clamp(_to_float(data.get("base_pitch"), 1.0), 0.5, 2.0),
            characteristic_style=str(data.get("characteristic_style", "") or ""),
            reason=str(data.get("reason", "") or ""),
            persona_name=str(data.get("persona_name", "") or ""),
            persona_hash=str(data.get("persona_hash", "") or ""),
            source=str(data.get("source", "locked") or "locked"),
            locked_at=str(data.get("locked_at", "") or ""),
        )


def _to_float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 人设读取
# ---------------------------------------------------------------------------


PERSONA_KEYWORD_MAP: Tuple[Tuple[str, str], ...] = (
    (r"傲娇|毒舌|御姐|高冷|清冷|大姐姐|女王", "Chelsie"),
    (r"撒娇|粘人|黏人|软萌|小恶魔|俏皮|可爱|妹妹", "Momo"),
    (r"元气|活泼|话痨|二次元|古灵精怪|少女", "Serena"),
    (r"少年|学长|阳光|清爽|正太", "Ethan"),
    (r"总裁|叔|成熟|磁性|低音|霸道|沉稳", "Vincent"),
    (r"温柔|治愈|邻家|女友|贴心|知心", "Cherry"),
)


class VoicePersonaEngine:
    """负责「人设 → 音色档案」的解析、锁定与韵律计算。"""

    def __init__(self, context: Any, config: PluginConfig, store: StateStore) -> None:
        self.context = context
        self.config = config
        self.store = store
        self._profile: Optional[VoiceProfile] = None
        self._calibrating = False

    # ------------------------------------------------------------------ 目录
    @staticmethod
    def catalog() -> List[VoiceEntry]:
        return list(VOICE_CATALOG)

    # ------------------------------------------------------------------ 人设
    async def _resolve_persona(self, event: Any = None) -> Tuple[str, str]:
        """尽力从 AstrBot 各处取出当前人设的 (名称, Prompt)。

        不同 AstrBot 版本 / 不同 Provider 的 API 差异很大，这里逐层尝试，
        任何一层拿到内容就立即返回，全部失败则给一段通用人设。
        """
        umo = getattr(event, "unified_msg_origin", None)

        # 1) 会话绑定的 persona（最准确）
        persona_id = None
        try:
            conv_mgr = getattr(self.context, "conversation_manager", None)
            if conv_mgr is not None and umo:
                cid = await conv_mgr.get_curr_conversation_id(umo)
                if cid:
                    conv = await conv_mgr.get_conversation(umo, cid)
                    persona_id = getattr(conv, "persona_id", None)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[qwen_multimodal] 读取会话 persona 失败: {exc}")

        persona_mgr = getattr(self.context, "persona_manager", None)
        if persona_mgr is not None:
            # 2) 指定 persona_id（AstrBot 里 get_persona 是协程）
            if persona_id:
                try:
                    persona = await _maybe_await(persona_mgr.get_persona(persona_id))
                    name, prompt = self._persona_to_text(persona)
                    if prompt:
                        return name, prompt
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"[qwen_multimodal] get_persona 失败: {exc}")
            # 3) 会话默认 persona
            try:
                if umo:
                    persona = await _maybe_await(
                        persona_mgr.get_default_persona_v3(umo)
                    )
                    name, prompt = self._persona_to_text(persona)
                    if prompt:
                        return name, prompt
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[qwen_multimodal] get_default_persona_v3 失败: {exc}")

            # 4) 全量列表里挑第一个非空人设（通常是用户自定义的那个）
            try:
                all_personas = await _maybe_await(persona_mgr.get_all_personas())
                for persona in all_personas or []:
                    name, prompt = self._persona_to_text(persona)
                    if prompt:
                        return name, prompt
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[qwen_multimodal] get_all_personas 失败: {exc}")

        # 5) Provider 层缓存的 curr_personality
        try:
            provider_mgr = getattr(self.context, "provider_manager", None)
            curr = getattr(provider_mgr, "curr_personality", None) if provider_mgr else None
            name, prompt = self._persona_to_text(curr)
            if prompt:
                return name, prompt
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[qwen_multimodal] 读取 curr_personality 失败: {exc}")

        # 6) 会话历史里的 system prompt 兜底
        try:
            conv_mgr = getattr(self.context, "conversation_manager", None)
            if conv_mgr is not None and umo:
                cid = await conv_mgr.get_curr_conversation_id(umo)
                conv = await conv_mgr.get_conversation(umo, cid) if cid else None
                history = getattr(conv, "history", None) or []
                if isinstance(history, str):
                    try:
                        history = json.loads(history)
                    except (TypeError, ValueError):
                        history = []
                for item in history if isinstance(history, list) else []:
                    if isinstance(item, dict) and item.get("role") == "system":
                        content = str(item.get("content", "") or "").strip()
                        if content:
                            return "", content
        except Exception:  # noqa: BLE001
            pass

        return "", ""

    @staticmethod
    def _persona_to_text(persona: Any) -> Tuple[str, str]:
        if persona is None:
            return "", ""
        if isinstance(persona, dict):
            name = str(persona.get("name") or persona.get("persona_id") or "")
            prompt = persona.get("prompt") or persona.get("system_prompt") or ""
            if isinstance(prompt, list):
                prompt = "\n".join(str(part) for part in prompt)
            return name, str(prompt or "").strip()
        name = str(getattr(persona, "name", "") or getattr(persona, "persona_id", "") or "")
        prompt = getattr(persona, "system_prompt", "") or getattr(persona, "prompt", "")
        if isinstance(prompt, list):
            prompt = "\n".join(str(part) for part in prompt)
        return name, str(prompt or "").strip()

    @staticmethod
    def persona_fingerprint(name: str, prompt: str) -> str:
        normalized = re.sub(r"\s+", " ", f"{name}\n{prompt}".strip()).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------ 锁定与校准
    async def get_profile(self, event: Any = None) -> VoiceProfile:
        """返回当前生效的音色档案（可能来自配置、状态文件或现场校准）。"""
        manual = self.config.manual_voice
        if manual:
            voice_id, entry = canonical_voice_id(manual)
            profile = VoiceProfile(
                voice_id=voice_id,
                voice_name=entry.name_cn if entry else manual,
                base_speed=entry.base_speed if entry else 1.0,
                base_pitch=entry.base_pitch if entry else 1.0,
                characteristic_style=entry.description if entry else "用户手动指定",
                reason="由管理员在插件配置中手动指定，覆盖 AI 自主选择。",
                source="manual",
            )
            self._profile = profile
            return profile

        await self.store.load()
        locked = self.store.locked_profile()
        if locked:
            profile = VoiceProfile.from_dict(locked)
            self._profile = profile
            if self.config.auto_recalibrate and profile.persona_hash:
                name, prompt = await self._resolve_persona(event)
                if prompt and self.persona_fingerprint(name, prompt) != profile.persona_hash:
                    logger.info("[qwen_multimodal] 检测到人设变更，开始重新校准音色。")
                    return await self.calibrate(event)
            return profile

        return await self.calibrate(event)

    async def calibrate(self, event: Any = None, force: bool = False) -> VoiceProfile:
        """让 AI 阅读人设并选择一个音色，成功后持久化锁定。"""
        if self._calibrating and not force:
            # 已有校准在跑，等一下复用结果，避免并发打爆 LLM
            for _ in range(60):
                await asyncio.sleep(0.5)
                if not self._calibrating:
                    break
            if self._profile is not None:
                return self._profile

        self._calibrating = True
        try:
            name, prompt = await self._resolve_persona(event)
            persona_hash = self.persona_fingerprint(name, prompt) if prompt else ""
            profile = await self._ask_llm_for_voice(event, name, prompt, persona_hash)
            if profile is None:
                profile = self._fallback_profile(name, prompt, persona_hash)
                logger.warning(
                    "[qwen_multimodal] AI 自主选音色失败，已按人设关键词回退到 %s。",
                    profile.voice_id,
                )
            await self.store.lock_voice(profile.to_dict())
            self._profile = profile
            logger.info(
                "[qwen_multimodal] 音色已锁定: %s (%s) speed=%.2f pitch=%.2f",
                profile.voice_name,
                profile.voice_id,
                profile.base_speed,
                profile.base_pitch,
            )
            return profile
        finally:
            self._calibrating = False

    async def _ask_llm_for_voice(
        self, event: Any, persona_name: str, persona_prompt: str, persona_hash: str
    ) -> Optional[VoiceProfile]:
        if not persona_prompt:
            return None
        provider_id = None
        try:
            provider_id = await self.context.get_current_chat_provider_id(
                umo=getattr(event, "unified_msg_origin", "")
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[qwen_multimodal] 获取 Provider 失败: {exc}")
        if not provider_id:
            logger.warning("[qwen_multimodal] 没有可用的对话 Provider，无法让 AI 选音色。")
            return None

        prompt = self._build_calibration_prompt(persona_name, persona_prompt)
        try:
            response = await asyncio.wait_for(
                self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt),
                timeout=min(60, self.config.request_timeout),
            )
        except asyncio.TimeoutError:
            logger.warning("[qwen_multimodal] 音色校准 LLM 调用超时。")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[qwen_multimodal] 音色校准 LLM 调用失败: {exc}")
            return None

        text = str(getattr(response, "completion_text", "") or "").strip()
        data = parse_json_block(text)
        if not isinstance(data, dict):
            logger.warning("[qwen_multimodal] 音色校准返回无法解析为 JSON，将使用关键词兜底。")
            return None

        raw_voice = str(data.get("voice_id") or "").strip()
        voice_id, entry = canonical_voice_id(raw_voice)
        if not voice_id:
            return None
        if entry is not None:
            # 目录内音色：以目录的基准值为准，允许模型在 ±0.1 内微调
            base_speed = clamp(
                _to_float(data.get("base_speed"), entry.base_speed), 0.85, 1.2
            )
            base_pitch = clamp(
                _to_float(data.get("base_pitch"), entry.base_pitch), 0.85, 1.2
            )
            name_cn = entry.name_cn
            style = str(data.get("characteristic_style") or entry.description).strip()
        else:
            base_speed = clamp(_to_float(data.get("base_speed"), 1.0), 0.85, 1.2)
            base_pitch = clamp(_to_float(data.get("base_pitch"), 1.0), 0.85, 1.2)
            name_cn = str(data.get("voice_name") or raw_voice)
            style = str(data.get("characteristic_style") or "自定义音色").strip()

        reason = str(data.get("reason") or "").strip()
        if not reason:
            reason = f"我觉得这个声音最像{hpersona(persona_name)}，所以选它。"
        return VoiceProfile(
            voice_id=voice_id,
            voice_name=name_cn,
            base_speed=base_speed,
            base_pitch=base_pitch,
            characteristic_style=style,
            reason=reason,
            persona_name=persona_name,
            persona_hash=persona_hash,
            source="llm",
        )

    @staticmethod
    def _build_calibration_prompt(persona_name: str, persona_prompt: str) -> str:
        catalog_lines = []
        for entry in VOICE_CATALOG:
            catalog_lines.append(
                "- {vid}｜{cn}｜{gender}｜{tags}｜{desc}".format(
                    vid=entry.voice_id,
                    cn=entry.name_cn,
                    gender=entry.gender,
                    tags="/".join(entry.tags),
                    desc=entry.description,
                )
            )
        catalog_text = "\n".join(catalog_lines)
        safe_persona = persona_prompt[:3000]
        return (
            "你现在要做一件事：读完下面这段角色设定，从音色库里挑出「最像你自己」的那一个声音。\n"
            "不要客套，不要解释任务，不要输出 Markdown 代码块，只输出一个 JSON 对象。\n\n"
            f"【你的角色设定】\n{safe_persona}\n\n"
            f"【可选音色库】\n{catalog_text}\n\n"
            "【选择要求】\n"
            "1. 只选一个最贴合角色性别、年龄感、性格与说话节奏的音色，不要为了好听而选不匹配的。\n"
            "2. voice_id 必须严格使用音色库里给出的英文 ID；若角色设定明显不适合库中任何音色，"
            "也可以写一个百炼上真实存在的自定义音色 ID。\n"
            "3. base_speed 与 base_pitch 是「这个角色平时的基准语速/音调」，范围 0.85-1.2，"
            "情绪激动时的变化由插件负责，不要在这里夸张。\n"
            "4. reason 必须用第一人称，像在跟用户解释「我为什么长这样一把嗓子」，"
            "40-80 字，要具体到性格细节，不要写空话。\n"
            "5. characteristic_style 用 8-20 字概括这个声音的特质。\n\n"
            "【输出格式（严格 JSON，无注释无多余文字）】\n"
            "{\n"
            '  "voice_id": "Cherry",\n'
            '  "voice_name": "芊悦",\n'
            '  "base_speed": 1.0,\n'
            '  "base_pitch": 1.0,\n'
            '  "characteristic_style": "温柔治愈的邻家少女音",\n'
            '  "reason": "我说话的时候习惯把尾音放软，因为……"\n'
            "}"
        )

    # ------------------------------------------------------------------ 兜底
    def _fallback_profile(
        self, persona_name: str, persona_prompt: str, persona_hash: str
    ) -> VoiceProfile:
        text = f"{persona_name} {persona_prompt}".strip()
        chosen: Optional[VoiceEntry] = None
        for pattern, voice_id in PERSONA_KEYWORD_MAP:
            if re.search(pattern, text):
                chosen = get_voice_entry(voice_id)
                if chosen is not None:
                    break
        if chosen is None:
            chosen = get_voice_entry("Cherry")
        assert chosen is not None
        speed = chosen.base_speed + random.uniform(-0.02, 0.02)
        pitch = chosen.base_pitch + random.uniform(-0.02, 0.02)
        return VoiceProfile(
            voice_id=chosen.voice_id,
            voice_name=chosen.name_cn,
            base_speed=clamp(round(speed, 3), 0.85, 1.2),
            base_pitch=clamp(round(pitch, 3), 0.85, 1.2),
            characteristic_style=chosen.description,
            reason=(
                f"我看过自己的人设后，觉得{chosen.name_cn}这把嗓子最贴合我现在的性格——"
                f"{chosen.description}"
            ),
            persona_name=persona_name,
            persona_hash=persona_hash,
            source="fallback",
        )

    # ------------------------------------------------------------------ 韵律
    def compute_prosody(
        self,
        profile: VoiceProfile,
        emotion: str,
        intensity: int = 5,
        speed_delta: float = 0.0,
        pitch_delta: float = 0.0,
        instruct: str = "",
    ) -> Prosody:
        """把「音色基准 + 情绪预设 + AI 微调」合成为最终韵律。"""
        emo = normalize_emotion(emotion)
        preset = EMOTION_PRESETS.get(emo, EMOTION_PRESETS["neutral"])
        intensity = max(1, min(10, int(intensity or 5)))

        if not self.config.allow_dynamic_prosody:
            return Prosody(
                speed=clamp(profile.base_speed, 0.5, 2.0),
                pitch=clamp(profile.base_pitch, 0.5, 2.0),
                instruct=instruct.strip() or preset.instruct,
                emotion=emo,
                intensity=intensity,
            )

        # 强度会轻微放大预设幅度：强度 5 时就是预设原值，10 时额外 +30%
        scale = 1.0 + max(0, intensity - 5) * 0.06
        speed_mult = 1.0 + (preset.speed_mult - 1.0) * scale
        pitch_mult = 1.0 + (preset.pitch_mult - 1.0) * scale

        speed = profile.base_speed * speed_mult + clamp(speed_delta, -0.3, 0.4)
        pitch = profile.base_pitch * pitch_mult + clamp(pitch_delta, -0.2, 0.3)

        final_instruct = instruct.strip() or preset.instruct
        if preset.instruct and instruct.strip() and instruct.strip() != preset.instruct:
            final_instruct = f"{preset.instruct}；{instruct.strip()}"
        return Prosody(
            speed=clamp(round(speed, 3), 0.5, 2.0),
            pitch=clamp(round(pitch, 3), 0.5, 2.0),
            instruct=final_instruct,
            emotion=emo,
            intensity=intensity,
        )

    # ------------------------------------------------------------------ 展示
    def format_profile_text(self, profile: VoiceProfile) -> str:
        entry = get_voice_entry(profile.voice_id)
        gender = entry.gender if entry else "未知"
        tags = "／".join(entry.tags) if entry else "自定义"
        return (
            "🎙️ 我的声音档案\n"
            f"━━━━━━━━━━━━\n"
            f"音色：{profile.voice_name}（{profile.voice_id}）\n"
            f"性别：{gender}\n"
            f"特质：{tags}\n"
            f"基准语速：{profile.base_speed:.2f}　基准音调：{profile.base_pitch:.2f}\n"
            f"风格概括：{profile.characteristic_style or '—'}\n"
            f"来源：{_source_text(profile.source)}\n"
            f"━━━━━━━━━━━━\n"
            f"我当时是这么想的：\n{profile.reason or '（还没有留下心路历程）'}"
        )

    @staticmethod
    def format_catalog_text() -> str:
        lines = ["🎧 千问音色图鉴", "━━━━━━━━━━━━"]
        for index, entry in enumerate(VOICE_CATALOG, start=1):
            lines.append(
                f"{index}. {entry.name_cn}（{entry.voice_id}）· {entry.gender}\n"
                f"   特质：{'／'.join(entry.tags)}\n"
                f"   基准：语速 {entry.base_speed:.2f} / 音调 {entry.base_pitch:.2f}\n"
                f"   {entry.description}"
            )
        lines.append("━━━━━━━━━━━━")
        lines.append("提示：/qwen_voice_set <音色ID> [语速] [语调] 可以手动锁定音色。")
        return "\n".join(lines)


def hpersona(name: str) -> str:
    """小工具：人设名为空时用「我自己」代替，避免句子读起来别扭。"""
    return f"「{name}」" if name else "我自己"


def _source_text(source: str) -> str:
    return {
        "llm": "AI 阅读人设后自主选择",
        "manual": "管理员手动指定",
        "fallback": "关键词兜底（LLM 不可用）",
        "locked": "已锁定的人设档案",
    }.get(source, source or "未知")


def parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    """从可能带 ```json 围栏或前后废话的文本中抠出第一个 JSON 对象。"""
    if not text:
        return None
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.S)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except ValueError:
        pass
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : index + 1]
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict):
                            return data
                    except ValueError:
                        break
        start = cleaned.find("{", start + 1)
    return None
