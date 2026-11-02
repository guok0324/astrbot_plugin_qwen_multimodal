"""阿里云百炼（DashScope）客户端：语音合成 + 异步图像生成。

协议要点（已用真实 Key 验证）
------------------------------
TTS（qwen3-tts-flash 等）::

    POST {endpoint}
    Authorization: Bearer <key>
    {"model": "...", "input": {"text": "...", "voice": "Cherry"},
     "parameters": {"format": "mp3", "sample_rate": 24000,
                    "rate": 1.0, "pitch": 1.0, "instruct": "..."}}
    -> {"output": {"audio": {"url": "https://...", "data": "<base64 可选>"}}}

生图（万相 / 千问图像，异步任务）::

    POST .../text2image/image-synthesis   (X-DashScope-Async: enable)
    -> {"output": {"task_id": "...", "task_status": "PENDING"}}
    GET  .../tasks/{task_id}  -> task_status: PENDING/RUNNING/SUCCEEDED/FAILED
    SUCCEEDED 时取 output.results[0].url

模型名的现实情况：手册里写的 ``qwen-audio-3.1-tts-plus`` / ``qwen-image-2.1-pro``
在部分账号上并不存在（会返回 ``Model not exist`` 或 ``url error``）。因此这里实现了
**模型候选链**：先用用户配置的模型，失败后依次尝试该账号上真实可用的同族模型，并把
最终可用的模型缓存下来，避免每次都触发失败请求。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from astrbot.api import logger

try:  # 包内导入兼容
    from .audio_utils import (
        decode_base64_audio,
        ensure_dir,
        ext_from_url,
        normalize_audio_file,
        sniff_audio_ext,
        write_bytes_atomic,
    )
    from .config_manager import PluginConfig
except ImportError:  # pragma: no cover
    from audio_utils import (  # type: ignore
        decode_base64_audio,
        ensure_dir,
        ext_from_url,
        normalize_audio_file,
        sniff_audio_ext,
        write_bytes_atomic,
    )
    from config_manager import PluginConfig  # type: ignore

DEFAULT_TTS_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
    "multimodal-generation/generation"
)
IMAGE_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
    "text2image/image-synthesis"
)
TASK_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
SPEECH_REST_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
    "text2speech/speech-synthesis"
)

# 账号上真实可用的同族模型（按优先级）。配置里的模型永远排在最前面。
TTS_MODEL_CHAIN: Tuple[str, ...] = ("qwen3-tts-flash", "qwen-tts")
IMAGE_MODEL_CHAIN: Tuple[str, ...] = (
    "qwen-image-2.1-pro",
    "qwen-image-plus",
    "wanx2.1-t2i-turbo",
    "wanx-v1",
)

MODEL_MISSING_HINTS = (
    "model not exist",
    "url error",
    "model_not_found",
    "invalid model",
)
INSPECTION_HINTS = ("datainspectionfailed", "data_inspection_failed", "inappropriate content")


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class DashScopeError(Exception):
    """百炼接口错误基类。"""

    def __init__(self, message: str, status: int = 0, code: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code

    def __str__(self) -> str:
        return f"[{self.status}/{self.code}] {self.message}" if self.code else self.message


class ModelNotFoundError(DashScopeError):
    """模型不存在或该账号未开通。"""


class ContentInspectionError(DashScopeError):
    """内容安全审核未通过（图片/文本违规）。"""


class AuthError(DashScopeError):
    """API Key 无效或权限不足。"""


class RateLimitError(DashScopeError):
    """触发限流。"""


def _classify_error(status: int, code: str, message: str) -> DashScopeError:
    blob = f"{code} {message}".lower()
    if status in (401, 403) or "invalidapikey" in blob or "unauthorized" in blob:
        return AuthError(message or "API Key 无效或已过期", status, code)
    if status == 429 or "throttl" in blob or "rate limit" in blob:
        return RateLimitError(message or "请求过于频繁，已被限流", status, code)
    if any(hint in blob for hint in INSPECTION_HINTS):
        return ContentInspectionError(message or "内容审核未通过", status, code)
    if status == 404 or any(hint in blob for hint in MODEL_MISSING_HINTS):
        return ModelNotFoundError(message or "模型不存在", status, code)
    return DashScopeError(message or "百炼接口返回错误", status, code)


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------


@dataclass
class SpeechResult:
    path: Path
    model: str
    voice: str
    duration_bytes: int


@dataclass
class ImageResult:
    path: Path
    model: str
    prompt: str
    actual_prompt: str = ""


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


class DashScopeClient:
    """轻量异步客户端；一个插件实例复用一个 aiohttp 连接池。"""

    def __init__(self, config: PluginConfig, temp_audio_dir: Path, temp_image_dir: Path) -> None:
        self.config = config
        self.temp_audio_dir = ensure_dir(Path(temp_audio_dir))
        self.temp_image_dir = ensure_dir(Path(temp_image_dir))
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._tts_model: Optional[str] = None
        self._image_model: Optional[str] = None
        self._unavailable_tts_models: set[str] = set()
        self._unavailable_image_models: set[str] = set()

    # ------------------------------------------------------------------ HTTP
    @property
    def api_key(self) -> str:
        return self.config.api_key

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            return self._session
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(
                    total=self.config.request_timeout, connect=20, sock_read=180
                )
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers={"User-Agent": "AstrBot-qwen-multimodal/1.0"},
                )
        return self._session

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None and not session.closed:
            try:
                await session.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[qwen_multimodal] 关闭 HTTP 会话异常: {exc}")

    async def _post_json(
        self, url: str, payload: Dict[str, Any], extra_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise AuthError("未配置 DashScope API Key", 401, "MissingApiKey")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        session = await self._get_session()
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                text = await response.text()
                status = response.status
        except asyncio.TimeoutError as exc:
            raise DashScopeError("请求百炼接口超时") from exc
        except aiohttp.ClientError as exc:
            raise DashScopeError(f"网络请求失败: {exc}") from exc

        data = _loads(text)
        if status >= 400 or (isinstance(data, dict) and data.get("code")):
            code = str((data or {}).get("code", "")) if isinstance(data, dict) else ""
            message = str((data or {}).get("message", text[:300])) if isinstance(data, dict) else text[:300]
            raise _classify_error(status, code, message)
        if not isinstance(data, dict):
            raise DashScopeError(f"接口返回非 JSON 内容: {text[:200]}", status)
        return data

    async def _get_json(self, url: str, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise AuthError("未配置 DashScope API Key", 401, "MissingApiKey")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if extra_headers:
            headers.update(extra_headers)
        session = await self._get_session()
        try:
            async with session.get(url, headers=headers) as response:
                text = await response.text()
                status = response.status
        except asyncio.TimeoutError as exc:
            raise DashScopeError("查询百炼任务超时") from exc
        except aiohttp.ClientError as exc:
            raise DashScopeError(f"网络请求失败: {exc}") from exc
        data = _loads(text)
        if status >= 400:
            code = str((data or {}).get("code", "")) if isinstance(data, dict) else ""
            message = str((data or {}).get("message", text[:300])) if isinstance(data, dict) else text[:300]
            raise _classify_error(status, code, message)
        if not isinstance(data, dict):
            raise DashScopeError(f"任务接口返回非 JSON: {text[:200]}", status)
        return data

    async def _download(self, url: str, dest: Path) -> Path:
        if not url:
            raise DashScopeError("下载地址为空")
        session = await self._get_session()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=180)) as response:
                if response.status >= 400:
                    raise DashScopeError(f"下载媒体失败: HTTP {response.status}", response.status)
                payload = await response.read()
        except asyncio.TimeoutError as exc:
            raise DashScopeError("下载媒体超时") from exc
        except aiohttp.ClientError as exc:
            raise DashScopeError(f"下载媒体失败: {exc}") from exc
        if not payload:
            raise DashScopeError("下载到的媒体内容为空")
        return write_bytes_atomic(dest, payload)

    # ------------------------------------------------------------------ TTS
    def _tts_candidates(self) -> List[str]:
        configured = self.config.tts_model
        chain = [configured] if configured else []
        chain.extend(model for model in TTS_MODEL_CHAIN if model not in chain)
        if self._tts_model:
            chain = [self._tts_model] + [m for m in chain if m != self._tts_model]
        seen: List[str] = []
        for model in chain:
            if model not in seen and model not in self._unavailable_tts_models:
                seen.append(model)
        return seen

    async def synthesize_speech(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        pitch: float = 1.0,
        instruct: str = "",
        audio_format: Optional[str] = None,
        model: Optional[str] = None,
    ) -> SpeechResult:
        """合成语音并落盘，返回本地文件。失败抛 DashScopeError 子类。"""
        if not text.strip():
            raise DashScopeError("待合成文本为空")
        fmt = (audio_format or self.config.audio_format or "mp3").lower()
        candidates = [model] if model else self._tts_candidates()
        last_error: Optional[DashScopeError] = None
        tried_alt = False

        for candidate in candidates:
            try:
                result = await self._tts_once(candidate, text, voice, speed, pitch, instruct, fmt)
                self._tts_model = candidate
                return SpeechResult(
                    path=result,
                    model=candidate,
                    voice=voice,
                    duration_bytes=result.stat().st_size,
                )
            except ContentInspectionError:
                raise
            except AuthError:
                raise
            except ModelNotFoundError as exc:
                last_error = exc
                self._unavailable_tts_models.add(candidate)
                logger.info(
                    "[qwen_multimodal] TTS 模型 %s 不可用（%s），尝试下一个候选模型。",
                    candidate,
                    exc.message[:80],
                )
                continue
            except DashScopeError as exc:
                last_error = exc
                logger.warning(
                    "[qwen_multimodal] TTS 调用失败 model=%s: %s", candidate, exc
                )
                continue

        # 多模态端点全部失败后，再尝试一次传统语音合成 REST 协议（cosyvoice 系）
        if not tried_alt:
            tried_alt = True
            try:
                path = await self._tts_via_rest(text, voice, speed, pitch, fmt)
                self._tts_model = "cosyvoice-v2"
                return SpeechResult(
                    path=path,
                    model="cosyvoice-v2",
                    voice=voice,
                    duration_bytes=path.stat().st_size,
                )
            except DashScopeError as exc:
                last_error = exc
                logger.warning(f"[qwen_multimodal] 备用 TTS 协议同样失败: {exc}")

        raise last_error or DashScopeError("语音合成失败")

    async def _tts_once(
        self,
        model: str,
        text: str,
        voice: str,
        speed: float,
        pitch: float,
        instruct: str,
        fmt: str,
    ) -> Path:
        parameters: Dict[str, Any] = {
            "voice": voice,
            "format": fmt,
            "sample_rate": self.config.sample_rate,
            "rate": round(max(0.5, min(2.0, speed)), 3),
            "pitch": round(max(0.5, min(2.0, pitch)), 3),
        }
        if instruct.strip():
            parameters["instruct"] = instruct.strip()[:200]

        payload = {
            "model": model,
            "input": {"text": text.strip(), "voice": voice},
            "parameters": parameters,
        }
        endpoint = self.config.tts_endpoint or DEFAULT_TTS_ENDPOINT
        data = await self._post_json(endpoint, payload)

        output = data.get("output") or {}
        audio = output.get("audio") if isinstance(output, dict) else None
        if not isinstance(audio, dict):
            # 少数协议把音频直接放在 output 下
            audio = output if isinstance(output, dict) and ("url" in output or "data" in output) else {}

        dest = self.temp_audio_dir / f"{_uuid()}.{fmt}"
        url = str(audio.get("url") or "").strip()
        b64 = str(audio.get("data") or "").strip()
        if b64:
            raw = decode_base64_audio(b64)
            if raw:
                write_bytes_atomic(dest, raw)
                return normalize_audio_file(dest, raw, f".{fmt}")
        if url:
            # 先按 URL 的扩展名占位，下载后再按魔术字节纠正
            ext = ext_from_url(url) or f".{fmt}"
            dest = dest.with_suffix(ext)
            await self._download(url, dest)
            payload_bytes = dest.read_bytes()[:4096]
            return normalize_audio_file(dest, payload_bytes, f".{fmt}")

        raise DashScopeError("TTS 响应中既没有 audio.url 也没有 audio.data")

    async def _tts_via_rest(
        self, text: str, voice: str, speed: float, pitch: float, fmt: str
    ) -> Path:
        """传统语音合成协议（异步任务）。仅在多模态协议失败时调用一次。"""
        rest_voice = voice if voice.startswith("long") else "longxiaochun_v2"
        payload = {
            "model": "cosyvoice-v2",
            "input": {"text": text.strip(), "voice": rest_voice},
            "parameters": {
                "format": fmt,
                "sample_rate": self.config.sample_rate,
                "rate": round(max(0.5, min(2.0, speed)), 3),
                "pitch": round(max(0.5, min(2.0, pitch)), 3),
            },
        }
        data = await self._post_json(
            SPEECH_REST_ENDPOINT, payload, {"X-DashScope-Async": "enable"}
        )
        output = data.get("output") or {}
        task_id = str(output.get("task_id") or "").strip()
        if not task_id:
            audio = output.get("audio") if isinstance(output.get("audio"), dict) else {}
            url = str(audio.get("url") or output.get("url") or "")
            if url:
                dest = self.temp_audio_dir / f"{_uuid()}.{fmt}"
                await self._download(url, dest)
                return dest
            raise DashScopeError("备用语音协议未返回 task_id")

        deadline = time.monotonic() + self.config.poll_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(2.0)
            info = await self._get_json(TASK_ENDPOINT.format(task_id=task_id))
            status = str((info.get("output") or {}).get("task_status", "")).upper()
            if status == "SUCCEEDED":
                output = info.get("output") or {}
                audio = output.get("audio") if isinstance(output.get("audio"), dict) else {}
                url = str(audio.get("url") or output.get("audio_url") or "")
                if not url:
                    results = output.get("results") or []
                    if results and isinstance(results, list):
                        url = str(results[0].get("url") or "")
                if not url:
                    raise DashScopeError("备用语音任务成功但没有音频地址")
                dest = self.temp_audio_dir / f"{_uuid()}.{fmt}"
                await self._download(url, dest)
                return dest
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                message = str((info.get("output") or {}).get("message", "任务失败"))
                raise DashScopeError(f"备用语音任务失败: {message}")
        raise DashScopeError("备用语音任务轮询超时")

    # ------------------------------------------------------------------ 生图
    def _image_candidates(self) -> List[str]:
        configured = self.config.image_model
        chain = [configured] if configured else []
        chain.extend(model for model in IMAGE_MODEL_CHAIN if model not in chain)
        if self._image_model:
            chain = [self._image_model] + [m for m in chain if m != self._image_model]
        seen: List[str] = []
        for model in chain:
            if model not in seen and model not in self._unavailable_image_models:
                seen.append(model)
        return seen

    async def generate_image(
        self,
        prompt: str,
        size: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ImageResult:
        if not prompt.strip():
            raise DashScopeError("图片提示词为空")
        size = size or self.config.default_size
        if size not in ("1024*1024", "720*1280", "1280*720"):
            size = self.config.default_size
        negative = (
            negative_prompt if negative_prompt is not None else self.config.image_negative_prompt
        ).strip()

        candidates = [model] if model else self._image_candidates()
        last_error: Optional[DashScopeError] = None
        for candidate in candidates:
            try:
                result = await self._image_once(candidate, prompt, size, negative)
                self._image_model = candidate
                return result
            except ContentInspectionError:
                raise
            except AuthError:
                raise
            except ModelNotFoundError as exc:
                last_error = exc
                self._unavailable_image_models.add(candidate)
                logger.info(
                    "[qwen_multimodal] 生图模型 %s 不可用（%s），尝试下一个候选模型。",
                    candidate,
                    exc.message[:80],
                )
                continue
            except DashScopeError as exc:
                last_error = exc
                logger.warning(
                    "[qwen_multimodal] 生图失败 model=%s: %s", candidate, exc
                )
                continue
        raise last_error or DashScopeError("图像生成失败")

    async def _image_once(
        self, model: str, prompt: str, size: str, negative: str
    ) -> ImageResult:
        payload: Dict[str, Any] = {
            "model": model,
            "input": {"prompt": prompt.strip()},
            "parameters": {
                "size": size,
                "n": 1,
                "prompt_extend": model.startswith("wanx"),
                "watermark": False,
            },
        }
        if negative:
            payload["input"]["negative_prompt"] = negative

        data = await self._post_json(IMAGE_ENDPOINT, payload, {"X-DashScope-Async": "enable"})
        output = data.get("output") or {}
        task_id = str(output.get("task_id") or "").strip()

        if not task_id:
            # 少数模型同步返回结果
            url = _extract_image_url(output)
            if url:
                return await self._finish_image(url, model, prompt, "")
            raise DashScopeError("生图接口未返回 task_id")

        deadline = time.monotonic() + self.config.poll_timeout
        poll_interval = 2.0
        actual_prompt = ""
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval)
            info = await self._get_json(TASK_ENDPOINT.format(task_id=task_id))
            out = info.get("output") or {}
            status = str(out.get("task_status", "")).upper()
            if status == "SUCCEEDED":
                results = out.get("results") or []
                url = _extract_image_url(out)
                if results and isinstance(results, list):
                    actual_prompt = str(results[0].get("actual_prompt", "") or "")
                if not url:
                    raise DashScopeError("生图任务成功但没有返回图片地址")
                return await self._finish_image(url, model, prompt, actual_prompt)
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                message = str(out.get("message") or "任务失败")
                code = str(out.get("code") or "")
                if any(hint in f"{code} {message}".lower() for hint in INSPECTION_HINTS):
                    raise ContentInspectionError(message, 400, code or "DataInspectionFailed")
                raise DashScopeError(f"生图任务失败: {message}")
            # PENDING / RUNNING 继续等
            poll_interval = min(3.0, poll_interval + 0.2)

        raise DashScopeError(f"生图轮询超时（{self.config.poll_timeout}s）")

    async def _finish_image(
        self, url: str, model: str, prompt: str, actual_prompt: str
    ) -> ImageResult:
        ext = ext_from_url(url) or ".png"
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        dest = self.temp_image_dir / f"{_uuid()}{ext}"
        await self._download(url, dest)
        head = dest.read_bytes()[:12]
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            if dest.suffix.lower() not in (".png",):
                dest = dest.with_suffix(".png")
        elif head[:2] == b"\xff\xd8":
            if dest.suffix.lower() not in (".jpg", ".jpeg"):
                target = dest.with_suffix(".jpg")
                try:
                    dest.replace(target)
                    dest = target
                except OSError:
                    pass
        if not dest.exists() or dest.stat().st_size < 512:
            raise DashScopeError("生成的图片文件异常（过小或不存在）")
        return ImageResult(path=dest, model=model, prompt=prompt, actual_prompt=actual_prompt)

    # ------------------------------------------------------------------ 杂项
    async def probe_tts_models(self, voices: Optional[List[str]] = None) -> Dict[str, Any]:
        """调试用：逐个尝试模型候选，返回可用性报告（不写临时文件）。"""
        report: Dict[str, Any] = {"configured": self.config.tts_model, "models": {}}
        for model in [self.config.tts_model, *TTS_MODEL_CHAIN]:
            if model in report["models"]:
                continue
            try:
                await self._tts_once(
                    model, "测试", (voices or ["Cherry"])[0], 1.0, 1.0, "", "mp3"
                )
                report["models"][model] = "ok"
            except DashScopeError as exc:
                report["models"][model] = f"{type(exc).__name__}: {exc.message[:60]}"
            except Exception as exc:  # noqa: BLE001
                report["models"][model] = f"{type(exc).__name__}: {exc}"
        return report


def _loads(text: str) -> Any:
    import json

    try:
        return json.loads(text) if text.strip() else {}
    except ValueError:
        return {}


def _extract_image_url(output: Dict[str, Any]) -> str:
    results = output.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            url = first.get("url") or first.get("b64_image")
            if url:
                return str(url)
    for key in ("url", "image_url", "output_url"):
        value = output.get(key)
        if value:
            return str(value)
    return ""


def _uuid() -> str:
    import uuid as _uuid_module

    return _uuid_module.uuid4().hex
