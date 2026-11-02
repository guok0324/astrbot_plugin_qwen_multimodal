"""解析 LLM 回复里的隐式多模态标签。

LLM 被 system prompt 要求输出这样的结构::

    好的呀，那我给你唱一小段～
    <tts emotion="coquettish" intensity="7" speed_delta="0.05"
         pitch_delta="0.08" instruct="语气软糯，尾音上挑">人家才不是想你了呢，只是刚好路过~</tts>
    对了，你要的图我画好啦：
    <img_gen prompt="雨夜霓虹下的和风街道，水彩" size="720*1280" />

本模块只做纯文本解析，不依赖 AstrBot，方便离线单测。解析结果是一串**有序**片段，
因为“先说话、再补一句文字、再发图”的顺序对拟人感很重要。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# 正则
# ---------------------------------------------------------------------------

_FLAGS = re.IGNORECASE | re.DOTALL

TTS_TAG_RE = re.compile(r"<tts\b([^>]*)>(.*?)</\s*tts\s*>", _FLAGS)
TTS_OPEN_RE = re.compile(r"<tts\b[^>]*>", _FLAGS)
TTS_CLOSE_RE = re.compile(r"</\s*tts\s*>", _FLAGS)
IMG_TAG_RE = re.compile(r"<img_gen\b([^>]*?)(?:/\s*>|>\s*(?:</\s*img_gen\s*>)?)", _FLAGS)
IMG_ANY_RE = re.compile(r"<img_gen\b[^>]*?/?>", _FLAGS)

ATTR_RE = re.compile(
    r"([A-Za-z_][\w-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|“([^”]*)”|‘([^’]*)’|([^\s\"'>]+))"
)

CODE_FENCE_RE = re.compile(r"```")
URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"'）)】]+"
    r"|(?:[a-zA-Z0-9-]+\.)+(?:com|cn|net|org|io|dev|app|xyz|top|me|cc)(?:/[^\s<>\"'）)】]*)?",
    re.IGNORECASE,
)

# 括号动作：中英文圆括号、方头括号、书名号式动作、星号包裹
_ACTION_PATTERNS = [
    re.compile(r"（[^（）]{0,80}）"),
    re.compile(r"\([^()]{0,80}\)"),
    re.compile(r"【[^【】]{0,80}】"),
    re.compile(r"\[[^\[\]]{0,80}\]"),
    re.compile(r"\*[^*\n]{1,80}\*"),
    re.compile(r"＊[^＊\n]{1,80}＊"),
]

_MARKDOWN_PATTERNS = [
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"`[^`\n]*`"),
    re.compile(r"\*\*([^*\n]+)\*\*"),
    re.compile(r"__([^_\n]+)__"),
    re.compile(r"~~([^~\n]+)~~"),
    re.compile(r"(?m)^\s{0,3}#{1,6}\s*"),
    re.compile(r"(?m)^\s*[-*+]\s+"),
    re.compile(r"(?m)^\s*>\s?"),
    re.compile(r"(?m)^\s*\|.*\|\s*$"),
    # 兜底：成对标记被括号清洗破坏后残留的孤立符号
    re.compile(r"\*+"),
    re.compile(r"(?<![0-9A-Za-z])_+(?![0-9A-Za-z])"),
]

# Emoji / 颜文字：保留「哈哈」「哼」「唔」等拟声词
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U0001F000-\U0001F2FF"
    "\u2600-\u27BF"
    "\uFE0F\u200D"
    "\u2190-\u21FF"
    "]",
    re.UNICODE,
)
_KAOMOJI_RE = re.compile(
    r"[（(][^（）()\n]{0,20}?(?:[ω´`・∀≧≦°▽◇△▲∇][^（）()\n]{0,20}?)[）)]"
)
_TILDE_RE = re.compile(r"[~～]{2,}")

_WS_RE = re.compile(r"[ \t\u3000]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

EMOTION_TYPE_ALIASES = {
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
    "平静": "neutral",
}

VALID_EMOTIONS = (
    "happy",
    "excited",
    "angry",
    "coquettish",
    "shy",
    "sad",
    "aggrieved",
    "shocked",
    "sarcastic",
    "sleepy",
    "whisper",
    "tsundere",
    "neutral",
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class TTSChunk:
    text: str
    emotion: str = "neutral"
    intensity: int = 5
    speed_delta: float = 0.0
    pitch_delta: float = 0.0
    instruct: str = ""
    raw: str = ""

    def __repr__(self) -> str:  # 便于调试
        return (
            f"TTSChunk(text={self.text!r}, emotion={self.emotion!r}, "
            f"intensity={self.intensity}, ds={self.speed_delta}, dp={self.pitch_delta})"
        )


@dataclass
class ImagePrompt:
    prompt: str
    size: str = ""
    raw: str = ""


@dataclass
class TextPart:
    text: str

    @property
    def is_code(self) -> bool:
        return "```" in self.text or "`" in self.text


@dataclass
class ParseResult:
    parts: List[Union[TextPart, TTSChunk, ImagePrompt]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------ 便捷视图
    @property
    def tts_chunks(self) -> List[TTSChunk]:
        return [part for part in self.parts if isinstance(part, TTSChunk)]

    @property
    def images(self) -> List[ImagePrompt]:
        return [part for part in self.parts if isinstance(part, ImagePrompt)]

    @property
    def texts(self) -> List[TextPart]:
        return [part for part in self.parts if isinstance(part, TextPart)]

    @property
    def plain_text(self) -> str:
        """剥离所有标签后的纯文本（用于「不允许发语音」的会话）。"""
        pieces: List[str] = []
        for part in self.parts:
            if isinstance(part, TextPart):
                pieces.append(part.text)
            elif isinstance(part, TTSChunk):
                pieces.append(part.text)
        return "\n".join(piece for piece in pieces if piece.strip()).strip()

    @property
    def text_only(self) -> str:
        """仅文本片段的拼接（不含 <tts> 台词）。"""
        return "\n".join(part.text for part in self.texts if part.text.strip()).strip()

    @property
    def has_code(self) -> bool:
        return any("```" in part.text for part in self.texts)

    @property
    def has_tts(self) -> bool:
        return bool(self.tts_chunks)

    @property
    def has_image(self) -> bool:
        return bool(self.images)

    @property
    def is_empty(self) -> bool:
        return not self.parts or (not self.plain_text and not self.images)

    def total_tts_chars(self) -> int:
        return sum(len(chunk.text) for chunk in self.tts_chunks)

    def total_chars(self) -> int:
        return len(self.plain_text)


# ---------------------------------------------------------------------------
# 属性解析
# ---------------------------------------------------------------------------


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("＝", "=")
        .replace("\u00a0", " ")
    )


def parse_attributes(raw: str) -> Dict[str, str]:
    """解析标签属性，容忍单/双/全角引号与不带引号的值。"""
    if not raw:
        return {}
    normalized = _normalize_quotes(raw)
    attrs: Dict[str, str] = {}
    for match in ATTR_RE.finditer(normalized):
        key = match.group(1).strip().lower()
        value = next((group for group in match.groups()[1:] if group is not None), "")
        attrs[key] = value.strip()
    return attrs


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _parse_int(value: Any, default: int = 5) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def normalize_emotion(value: Any) -> str:
    """情绪值规范化：中文 → 英文标签；未知值回退 neutral。"""
    if value is None:
        return "neutral"
    key = str(value).strip().strip("\"'").lower()
    if not key:
        return "neutral"
    if key in VALID_EMOTIONS:
        return key
    if key in EMOTION_TYPE_ALIASES:
        return EMOTION_TYPE_ALIASES[key]
    for alias, name in EMOTION_TYPE_ALIASES.items():
        if alias in key:
            return name
    return "neutral"


def _clean_instruct(value: str) -> str:
    text = _normalize_quotes(value or "").strip()
    text = text.strip("\"'")
    return text[:200]


# ---------------------------------------------------------------------------
# 标签构造
# ---------------------------------------------------------------------------


def build_tts_tag(chunk: TTSChunk) -> str:
    """把 TTSChunk 还原成标签（用于失败回退时放回原文）。"""
    attrs = [f'emotion="{chunk.emotion}"', f'intensity="{chunk.intensity}"']
    if abs(chunk.speed_delta) > 1e-6:
        attrs.append(f'speed_delta="{chunk.speed_delta:+.2f}"')
    if abs(chunk.pitch_delta) > 1e-6:
        attrs.append(f'pitch_delta="{chunk.pitch_delta:+.2f}"')
    if chunk.instruct:
        safe = chunk.instruct.replace('"', "＂")
        attrs.append(f'instruct="{safe}"')
    return f"<tts {' '.join(attrs)}>{chunk.text}</tts>"


# ---------------------------------------------------------------------------
# 主解析
# ---------------------------------------------------------------------------


def parse(text: str, max_chunks: int = 3, max_images: int = 2) -> ParseResult:
    """把一段 LLM 原始回复解析成有序片段序列。

    容错策略：
    * 标签未闭合时，把 ``<tts ...>`` 之后的内容当作普通文本，绝不吞掉用户内容；
    * 属性缺省全部有默认值；
    * 空的 ``<tts></tts>`` 与空 ``<img_gen />`` 直接丢弃并记录 warning。
    """
    result = ParseResult()
    if not text:
        return result

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _normalize_quotes(text)

    matches: List[Tuple[int, int, str, Any]] = []

    for match in TTS_TAG_RE.finditer(text):
        attrs = parse_attributes(match.group(1))
        body = match.group(2).strip()
        if not body:
            result.warnings.append("丢弃空 <tts> 标签")
            continue
        chunk = TTSChunk(
            text=body,
            emotion=normalize_emotion(attrs.get("emotion")),
            intensity=max(1, min(10, _parse_int(attrs.get("intensity"), 5))),
            speed_delta=max(-0.3, min(0.4, _parse_float(attrs.get("speed_delta"), 0.0))),
            pitch_delta=max(-0.2, min(0.3, _parse_float(attrs.get("pitch_delta"), 0.0))),
            instruct=_clean_instruct(attrs.get("instruct", "")),
            raw=match.group(0),
        )
        matches.append((match.start(), match.end(), "tts", chunk))

    for match in IMG_TAG_RE.finditer(text):
        attrs = parse_attributes(match.group(1))
        prompt = (attrs.get("prompt") or "").strip()
        if not prompt:
            result.warnings.append("丢弃空 <img_gen> 标签")
            continue
        size = (attrs.get("size") or "").strip()
        if size and size not in ("1024*1024", "720*1280", "1280*720"):
            size = ""
        matches.append((match.start(), match.end(), "img", ImagePrompt(prompt=prompt, size=size, raw=match.group(0))))

    matches.sort(key=lambda item: item[0])

    # 处理未闭合的 <tts>（出现在所有完整标签之外的情况）
    leftovers = TTS_OPEN_RE.sub("", TTS_CLOSE_RE.sub("", text))
    if TTS_OPEN_RE.search(text):
        cleaned_full = TTS_TAG_RE.sub("", text)
        if TTS_OPEN_RE.search(cleaned_full):
            result.warnings.append("检测到未闭合的 <tts> 标签，已按普通文本处理")

    cursor = 0
    for start, end, kind, payload in matches:
        if start < cursor:  # 重叠（嵌套/畸形），跳过
            continue
        if start > cursor:
            _append_text(result, text[cursor:start])
        if kind == "tts":
            result.parts.append(payload)
        else:
            result.parts.append(payload)
        cursor = end

    if cursor < len(text):
        tail = text[cursor:]
        # 未闭合的开标签会留在 tail 里，直接剥掉标记但保留文字
        tail = TTS_OPEN_RE.sub("", tail)
        tail = IMG_ANY_RE.sub("", tail)
        _append_text(result, tail)

    # 限额
    chunks = result.tts_chunks
    if len(chunks) > max_chunks:
        result.warnings.append(f"语音段数 {len(chunks)} 超过上限 {max_chunks}，多余部分转文字")
        _demote_excess_tts(result, max_chunks)
    images = result.images
    if len(images) > max_images:
        result.warnings.append(f"图片数 {len(images)} 超过上限 {max_images}，多余部分忽略")
        _drop_excess_images(result, max_images)

    _merge_adjacent_text(result)
    _trim(result)
    return result


def _append_text(result: ParseResult, raw: str) -> None:
    cleaned = raw.strip("\n")
    if not cleaned.strip():
        return
    result.parts.append(TextPart(text=cleaned.strip()))


def _demote_excess_tts(result: ParseResult, keep: int) -> None:
    seen = 0
    new_parts: List[Union[TextPart, TTSChunk, ImagePrompt]] = []
    for part in result.parts:
        if isinstance(part, TTSChunk):
            seen += 1
            if seen > keep:
                new_parts.append(TextPart(text=part.text))
                continue
        new_parts.append(part)
    result.parts = new_parts


def _drop_excess_images(result: ParseResult, keep: int) -> None:
    seen = 0
    new_parts: List[Union[TextPart, TTSChunk, ImagePrompt]] = []
    for part in result.parts:
        if isinstance(part, ImagePrompt):
            seen += 1
            if seen > keep:
                continue
        new_parts.append(part)
    result.parts = new_parts


def _merge_adjacent_text(result: ParseResult) -> None:
    merged: List[Union[TextPart, TTSChunk, ImagePrompt]] = []
    for part in result.parts:
        if isinstance(part, TextPart) and merged and isinstance(merged[-1], TextPart):
            merged[-1] = TextPart(text=f"{merged[-1].text}\n{part.text}".strip())
        else:
            merged.append(part)
    result.parts = merged


def _trim(result: ParseResult) -> None:
    result.parts = [
        part for part in result.parts if not (isinstance(part, TextPart) and not part.text.strip())
    ]


# ---------------------------------------------------------------------------
# 剥离与清洗
# ---------------------------------------------------------------------------


def strip_tags(text: str) -> str:
    """去掉标签但保留台词，用于“本会话不允许发语音”时的降级。"""
    if not text:
        return ""

    def _tts_repl(match: re.Match) -> str:
        return match.group(2)

    cleaned = TTS_TAG_RE.sub(_tts_repl, text)
    cleaned = IMG_ANY_RE.sub("", cleaned)
    cleaned = TTS_OPEN_RE.sub("", cleaned)
    cleaned = TTS_CLOSE_RE.sub("", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def clean_tts_text(text: str) -> str:
    """六步语音文本清洗：动作 → Markdown → URL → 颜文字 → 波浪号 → 空白。"""
    if not text:
        return ""
    cleaned = text

    # 1) 去掉括号里的动作描写
    for pattern in _ACTION_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    # 2) 去掉 Markdown 标记
    for pattern in _MARKDOWN_PATTERNS:
        if pattern.groups:
            cleaned = pattern.sub(lambda m: m.group(1), cleaned)
        else:
            cleaned = pattern.sub("", cleaned)

    # 3) 去掉 URL
    cleaned = URL_RE.sub("", cleaned)

    # 4) 去掉颜文字与 Emoji
    cleaned = _EMOJI_RE.sub("", cleaned)
    cleaned = _KAOMOJI_RE.sub("", cleaned)

    # 5) 连续波浪号压成一个
    cleaned = _TILDE_RE.sub("~", cleaned)

    # 6) 收尾：去掉多余空白与首尾停顿标点
    cleaned = cleaned.replace("\u200b", "").replace("\ufeff", "")
    cleaned = _WS_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
    cleaned = re.sub(r"^[\s，,、。.~～\-—…]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def count_tts_chars(text: str) -> int:
    """按“可读字符”计数（去掉空白）。"""
    return len(re.sub(r"\s+", "", text or ""))


def looks_like_voice_request(message: str) -> bool:
    """用户是否明确要求“发语音/说话”。"""
    if not message:
        return False
    keywords = (
        "语音",
        "听你",
        "听一下",
        "说一句",
        "说句话",
        "念一下",
        "念给我",
        "叫声",
        "叫一声",
        "唱一",
        "唱首",
        "唱几句",
        "发条语音",
        "用声音",
        "开口说",
        "说话给我听",
        "voice",
        "audio message",
    )
    lowered = message.lower()
    return any(keyword in message or keyword in lowered for keyword in keywords)


def contains_tts_tag(text: str) -> bool:
    return bool(TTS_TAG_RE.search(text or ""))


def iter_tts_texts(text: str) -> Iterable[str]:
    for match in TTS_TAG_RE.finditer(text or ""):
        body = match.group(2).strip()
        if body:
            yield body
