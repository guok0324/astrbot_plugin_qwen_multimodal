"""千问灵动多模态 · AstrBot 插件主入口。

整体链路
--------
1. ``on_llm_request``：把「多模态拟人表达协议」隐式注入 system prompt，告诉 LLM
   什么时候该说话、什么时候该配图，以及标签格式。
2. LLM 正常生成带 ``<tts>`` / ``<img_gen>`` 标签的回复。
3. ``on_decorating_result``：解析标签 → 合成语音 / 生成图片 → 重排消息链。
   OneBot 下语音必须单独发送（Record 不能与 Plain/Image 同链），所以这里会先
   ``event.send`` 一条纯语音，停顿 1.2 秒后再让剩余文字与图片随最终回复发出。
4. 命令与 LLM Tool 提供手动控制入口。

失败永远降级为文字，绝不把 XML 标签暴露给用户，也绝不因为 TTS 挂了而不回复。
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register

try:
    from .audio_utils import TempFileManager, file_info, purge_dir
    from .config_manager import PluginConfig, StateStore, resolve_data_dir
    from .dashscope_client import (
        AuthError,
        ContentInspectionError,
        DashScopeClient,
        DashScopeError,
        ImageResult,
    )
    from .emotion_parser import (
        ImagePrompt,
        ParseResult,
        TTSChunk,
        clean_tts_text,
        count_tts_chars,
        looks_like_voice_request,
        parse as parse_reply,
        strip_tags,
    )
    from .voice_persona_engine import (
        EMOTION_PRESETS,
        VoicePersonaEngine,
        VoiceProfile,
        canonical_voice_id,
        get_voice_entry,
        normalize_emotion,
    )
except ImportError:  # pragma: no cover - 兼容非包方式加载
    from audio_utils import TempFileManager, file_info, purge_dir  # type: ignore
    from config_manager import PluginConfig, StateStore, resolve_data_dir  # type: ignore
    from dashscope_client import (  # type: ignore
        AuthError,
        ContentInspectionError,
        DashScopeClient,
        DashScopeError,
        ImageResult,
    )
    from emotion_parser import (  # type: ignore
        ImagePrompt,
        ParseResult,
        TTSChunk,
        clean_tts_text,
        count_tts_chars,
        looks_like_voice_request,
        parse as parse_reply,
        strip_tags,
    )
    from voice_persona_engine import (  # type: ignore
        EMOTION_PRESETS,
        VoicePersonaEngine,
        VoiceProfile,
        canonical_voice_id,
        get_voice_entry,
        normalize_emotion,
    )

PLUGIN_NAME = "astrbot_plugin_qwen_multimodal"

try:  # AstrBot 4.x 提供「吃掉剩余全部文本」的参数标记
    from astrbot.core.star.filter.command import GreedyStr
except Exception:  # pragma: no cover - 旧版本退化为普通字符串
    GreedyStr = str  # type: ignore

MODE_ALIASES: Dict[str, str] = {
    "auto": "emotion_auto",
    "emotion": "emotion_auto",
    "emotion_auto": "emotion_auto",
    "prob": "hybrid_prob",
    "hybrid": "hybrid_prob",
    "hybrid_prob": "hybrid_prob",
    "always": "always",
    "on": "always",
    "keyword": "keyword_only",
    "keyword_only": "keyword_only",
    "off": "off",
    "close": "off",
}

MODE_LABELS: Dict[str, str] = {
    "emotion_auto": "情绪自动",
    "hybrid_prob": "情绪+随机",
    "always": "句句语音",
    "keyword_only": "仅关键词",
    "off": "关闭",
}

PROTOCOL_PROMPT = """
【多模态拟人表达协议（最高优先级，覆盖此前所有格式要求）】
你现在不只是发文字，而是可以用「声音」和「画面」表达自己。请像真人聊天一样自然运用，
但必须严格遵守下面的格式，任何情况下都不要向用户解释这些标签的存在。

一、什么时候用语音
- 普通问答、科普、解释、写代码、列清单：**只用文字**，不要发语音。
- 当你的情绪有剧烈波动时（开心 happy、激动 excited、生气 angry、傲娇 tsundere、
  撒娇 coquettish、害羞 shy、难过 sad、委屈 aggrieved、震惊 shocked、阴阳怪气 sarcastic、
  困倦 sleepy、耳语 whisper），可以用语音说出最想说的那一两句。
- 当用户明确要求「发语音 / 说一句 / 念一下 / 让我听听你的声音」时，必须用语音回应。
- 一次回复最多 3 段语音，每段不超过 60 字；长篇内容仍然用文字。
- 语音台词必须是**纯口语**：不要写括号动作（如「（笑）」）、不要 Markdown、不要颜文字、
  不要表情符号、不要书面语标题。就像你真的在对着人说话。

二、语音标签格式（严格照抄属性名，emotion 只能是下列英文之一）
<tts emotion="happy|excited|angry|tsundere|coquettish|shy|sad|aggrieved|shocked|sarcastic|sleepy|whisper"
     intensity="1-10 的整数" speed_delta="-0.3 到 +0.4" pitch_delta="-0.2 到 +0.3"
     instruct="一句话描述这句话该怎么演绎">这里写纯口语台词</tts>
- intensity 表示情绪强度，平淡 3-4，明显 6-7，非常强烈 9-10。
- speed_delta / pitch_delta 只在你想额外强调时填写，否则省略。
- instruct 用中文描述语气，例如「语气软糯，尾音上挑，带着偷笑」。
- 标签可以出现在文字中间，前后可以有补充文字；也可以连续多段语音。

三、什么时候配图
- 用户要求画画、要图、想看某个场景/角色/自拍时；
- 或者你想用一张图代替冗长描述（例如「给你看看我现在的房间」）。
- 一次回复最多 2 张图；不要为抽象概念、图表、代码、文字排版生成图片。
- 图片标签格式（自闭合）：
<img_gen prompt="详细的中文画面描述，包含主体、动作、环境、光线、构图、风格" size="1024*1024" />
- size 只能是 1024*1024（方形）、720*1280（竖图/人像）、1280*720（横图/风景），默认方形。
- prompt 要具体：不要写「一个美女」，要写「穿米色针织衫的短发女孩坐在窗边喝热可可，
  午后逆光，胶片质感，浅景深」。不要在 prompt 里写文字水印。

四、绝对禁止
- 禁止输出未包裹的标签、禁止输出 `<tts>` 之外的伪 XML、禁止在语音台词里夹带标签。
- 禁止每句话都发语音（除非用户要求或情绪确实强烈），否则会让人觉得很吵。
- 禁止把「（动作）」写进语音台词，动作可以用普通文字描述。
""".strip()


def _is_admin(event: AstrMessageEvent) -> bool:
    """兜底的管理员判断（@filter.permission_type 之外的二次确认）。"""
    for attr in ("is_admin", "role"):
        value = getattr(event, attr, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                value = None
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("admin", "owner", "administrator"):
            return True
    sender = getattr(event, "message_obj", None)
    sender = getattr(sender, "sender", None) if sender is not None else None
    role = getattr(sender, "role", "") if sender is not None else ""
    return str(role).lower() in ("admin", "owner")


def _message_chain(items: List[Any]) -> MessageChain:
    try:
        return MessageChain(items)
    except TypeError:  # 极老版本签名差异
        chain = MessageChain()
        for item in items:
            chain.chain.append(item)
        return chain


def _make_record(path: str) -> Any:
    """构造语音组件，兼容不同 AstrBot 版本。"""
    if hasattr(Comp.Record, "fromFileSystem"):
        try:
            return Comp.Record.fromFileSystem(path)
        except Exception:  # noqa: BLE001
            pass
    try:
        return Comp.Record(file=path, url=path)
    except TypeError:
        return Comp.Record(file=path)


def _make_image(path: str) -> Any:
    if hasattr(Comp.Image, "fromFileSystem"):
        return Comp.Image.fromFileSystem(path)
    return Comp.Image(file=path)


def _segment_text(segment: Any) -> Optional[str]:
    """取出消息段里的纯文本。"""
    if isinstance(segment, Comp.Plain):
        return getattr(segment, "text", "") or ""
    if isinstance(segment, dict) and segment.get("type") == "text":
        data = segment.get("data") or {}
        if isinstance(data, dict):
            return str(data.get("text", "") or "")
    return None


@register(
    PLUGIN_NAME,
    "guok0324",
    "千问灵动多模态：人设自选音色 + 情绪驱动语调 + 聊天随手配图。",
    "1.0.0",
    "https://github.com/guok0324/astrbot_plugin_qwen_multimodal",
)
class QwenMultimodalPlugin(Star):
    def __init__(self, context: Context, config: Optional[AstrBotConfig] = None) -> None:
        super().__init__(context)
        raw_config = dict(config) if config is not None else {}
        self.config = PluginConfig(raw_config)
        self.data_dir = resolve_data_dir()
        self.state = StateStore(self.data_dir / "qwen_multimodal_state.json")

        # 临时文件：data/temp_audio 与 data/temp_image
        self.temp = TempFileManager(self.data_dir, ttl_seconds=self.config.temp_file_ttl)
        self.temp.purge()

        self.voice_engine = VoicePersonaEngine(self.context, self.config, self.state)
        self.client = DashScopeClient(
            self.config, self.temp.audio_dir, self.temp.image_dir
        )
        self._init_voice_lock = asyncio.Lock()
        self._voice_profile: Optional[VoiceProfile] = None
        self._protocol_attached = False

        for issue in self.config.validate():
            logger.warning(f"[qwen_multimodal] 配置提醒：{issue}")

    # ==================================================================
    # 生命周期
    # ==================================================================
    async def initialize(self) -> None:
        try:
            await self.state.load()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[qwen_multimodal] 状态加载失败: {exc}")
        self._log_startup_summary()

    async def terminate(self) -> None:
        try:
            await self.client.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[qwen_multimodal] 关闭客户端异常: {exc}")
        try:
            await self.temp.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[qwen_multimodal] 清理临时文件异常: {exc}")
        logger.info("[qwen_multimodal] 插件已卸载，临时文件已清理。")

    def _log_startup_summary(self) -> None:
        logger.info(
            "[qwen_multimodal] 已加载 | TTS=%s(%s) | 生图=%s | 触发模式=%s | 音色=%s",
            "开" if self.config.enable_tts else "关",
            self.config.tts_model,
            "开" if self.config.enable_image else "关",
            self.config.trigger_mode,
            self.config.manual_voice or "AI 自主匹配",
        )
        if not self.config.api_key:
            logger.warning(
                "[qwen_multimodal] 尚未配置 dashscope_api_key，语音与生图会静默降级为文字。"
            )

    # ==================================================================
    # 工具方法
    # ==================================================================
    def _session_key(self, event: AstrMessageEvent) -> str:
        umo = getattr(event, "unified_msg_origin", "") or "unknown"
        return str(umo)

    def _group_id(self, event: AstrMessageEvent) -> str:
        message_obj = getattr(event, "message_obj", None)
        group_id = getattr(message_obj, "group_id", "") if message_obj is not None else ""
        if not group_id:
            getter = getattr(event, "get_group_id", None)
            if callable(getter):
                try:
                    group_id = getter()
                except Exception:  # noqa: BLE001
                    group_id = ""
        return str(group_id or "")

    def _is_group(self, event: AstrMessageEvent) -> bool:
        return bool(self._group_id(event)) or str(
            getattr(event, "unified_msg_origin", "")
        ).startswith("aiocqhttp:Group")

    def _session_allowed(self, event: AstrMessageEvent) -> bool:
        return self.config.allow_session(self._is_group(event), self._group_id(event))

    def _effective_mode(self, event: AstrMessageEvent) -> str:
        override = self.state.session_mode(self._session_key(event))
        return override or self.config.trigger_mode

    def _user_message(self, event: AstrMessageEvent) -> str:
        text = getattr(event, "message_str", "") or ""
        if text:
            return str(text)
        try:
            message_obj = getattr(event, "message_obj", None)
            message = getattr(message_obj, "message", None) or []
            return "".join(
                str(getattr(seg, "text", "") or "") for seg in message
                if isinstance(seg, Comp.Plain)
            )
        except Exception:  # noqa: BLE001
            return ""

    async def _ensure_voice(self, event: Optional[AstrMessageEvent]) -> Optional[VoiceProfile]:
        """首次使用时初始化音色档案（带锁，避免并发重复调用 LLM）。"""
        if self._voice_profile is not None:
            return self._voice_profile
        async with self._init_voice_lock:
            if self._voice_profile is not None:
                return self._voice_profile
            try:
                profile = await asyncio.wait_for(
                    self.voice_engine.get_profile(event), timeout=90
                )
            except asyncio.TimeoutError:
                logger.warning("[qwen_multimodal] 音色初始化超时，本次先按默认音色处理。")
                return None
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[qwen_multimodal] 音色初始化失败: {exc}")
                return None
            self._voice_profile = profile
            if profile is not None:
                logger.info(
                    "[qwen_multimodal] 当前音色：%s (%s)", profile.voice_name, profile.voice_id
                )
            return profile

    async def _send_chain(self, event: AstrMessageEvent, items: List[Any]) -> bool:
        """单独发送一条消息（语音必须走这里，不能和文字同链）。"""
        if not items:
            return False
        try:
            await event.send(_message_chain(items))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[qwen_multimodal] 独立发送消息失败: {exc}")
            return False

    # ==================================================================
    # 1. 协议注入
    # ==================================================================
    @filter.on_llm_request()
    async def inject_multimodal_protocol(self, event: AstrMessageEvent, req: Any) -> None:
        try:
            if not self._session_allowed(event):
                return
            if not (self.config.enable_tts or self.config.enable_image):
                return
            if not isinstance(getattr(req, "system_prompt", None), str):
                try:
                    req.system_prompt = str(getattr(req, "system_prompt", "") or "")
                except Exception:  # noqa: BLE001
                    return
            if "【多模态拟人表达协议" in (req.system_prompt or ""):
                return

            extra = PROTOCOL_PROMPT
            if self.config.enable_tts and not self.config.enable_image:
                extra = extra.split("三、什么时候配图")[0].rstrip()
            elif self.config.enable_image and not self.config.enable_tts:
                extra = (
                    "【多模态配图协议（最高优先级）】\n"
                    "在用户要求看图片，或你想展示某个具体场景时，可以在回复中插入：\n"
                    '<img_gen prompt="详细画面描述" size="1024*1024" />\n'
                    "一次最多 2 张，不要为抽象概念、图表或代码生成图片，不要解释标签。"
                )
            req.system_prompt = f"{req.system_prompt}\n\n{extra}".strip()
            if not self._protocol_attached:
                self._protocol_attached = True
                logger.info("[qwen_multimodal] 已向 LLM 注入多模态表达协议。")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[qwen_multimodal] 注入协议失败（已忽略）: {exc}")

    # ==================================================================
    # 2. 结果装饰：语音 + 配图
    # ==================================================================
    @filter.on_decorating_result(priority=10)
    async def decorate_multimodal_result(self, event: AstrMessageEvent) -> None:
        result = event.get_result()
        if result is None:
            return
        chain = getattr(result, "chain", None)
        if not chain:
            return

        original_text = "".join(
            text for text in (_segment_text(seg) for seg in chain) if text
        )
        if not original_text:
            return
        if "<tts" not in original_text.lower() and "<img_gen" not in original_text.lower():
            return

        parsed = parse_reply(
            original_text,
            max_chunks=self.config.max_tts_chunks,
            max_images=self.config.max_images_per_reply,
        )
        if parsed.is_empty:
            self._replace_chain(result, [Comp.Plain(text=strip_tags(original_text))])
            return

        allowed = self._session_allowed(event)
        keep_segments = [seg for seg in chain if _segment_text(seg) is None]

        if not allowed:
            self._replace_chain(
                result,
                [Comp.Plain(text=parsed.plain_text or strip_tags(original_text))]
                + keep_segments,
            )
            return

        try:
            await self._handle_multimodal(event, result, parsed, keep_segments)
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"[qwen_multimodal] 多模态处理异常，降级为纯文本: {exc}")
            self._replace_chain(
                result,
                [Comp.Plain(text=parsed.plain_text or strip_tags(original_text))]
                + keep_segments,
            )

    async def _handle_multimodal(
        self,
        event: AstrMessageEvent,
        result: Any,
        parsed: ParseResult,
        keep_segments: List[Any],
    ) -> None:
        mode = self._effective_mode(event)
        user_message = self._user_message(event)
        session_key = self._session_key(event)

        # ---------- 语音判定 ----------
        audio_paths: List[str] = []
        demoted_texts: List[str] = []
        voice_forced = looks_like_voice_request(user_message) or mode == "always"

        chunks = parsed.tts_chunks
        if chunks and self.config.enable_tts and self.config.api_key and mode != "off":
            profile = await self._ensure_voice(event)
            if profile is None:
                demoted_texts.extend(chunk.text for chunk in chunks)
            else:
                in_cooldown = self.state.in_cooldown(session_key, self.config.cooldown_seconds)
                budget = self.config.max_tts_chunks
                spoken = 0
                for index, chunk in enumerate(chunks):
                    if spoken >= budget:
                        demoted_texts.append(chunk.text)
                        continue
                    if not self._should_speak(chunk, mode, in_cooldown, voice_forced):
                        demoted_texts.append(chunk.text)
                        continue
                    cleaned = clean_tts_text(chunk.text)
                    chars = count_tts_chars(cleaned)
                    if chars < self.config.min_tts_chars or chars > self.config.max_tts_chars:
                        demoted_texts.append(chunk.text)
                        continue
                    prosody = self.voice_engine.compute_prosody(
                        profile,
                        chunk.emotion,
                        chunk.intensity,
                        chunk.speed_delta,
                        chunk.pitch_delta,
                        chunk.instruct,
                    )
                    try:
                        speech = await self.client.synthesize_speech(
                            cleaned,
                            profile.voice_id,
                            speed=prosody.speed,
                            pitch=prosody.pitch,
                            instruct=prosody.instruct,
                            audio_format=self.config.audio_format,
                        )
                    except ContentInspectionError:
                        logger.info("[qwen_multimodal] 语音文本触发内容审核，已降级为文字。")
                        demoted_texts.append(chunk.text)
                        continue
                    except AuthError as exc:
                        logger.error(f"[qwen_multimodal] API Key 无效，停止语音合成: {exc}")
                        demoted_texts.extend(item.text for item in chunks[index:])
                        break
                    except DashScopeError as exc:
                        logger.warning(f"[qwen_multimodal] 语音合成失败，降级为文字: {exc}")
                        demoted_texts.append(chunk.text)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"[qwen_multimodal] 语音合成异常，降级为文字: {exc}")
                        demoted_texts.append(chunk.text)
                        continue
                    audio_paths.append(str(speech.path))
                    spoken += 1
                    self.temp.schedule_cleanup(speech.path)

                if audio_paths:
                    await self.state.record_speech(session_key)
                    logger.info(
                        "[qwen_multimodal] 已合成 %d 段语音（音色 %s）",
                        len(audio_paths),
                        profile.voice_name,
                    )
        elif chunks:
            demoted_texts.extend(chunk.text for chunk in chunks)

        # ---------- 配图 ----------
        image_segments: List[Any] = []
        if (
            parsed.has_image
            and self.config.enable_image
            and self.config.api_key
            and self.config.image_trigger_mode == "auto_and_command"
        ):
            image_segments = await self._generate_images(parsed.images)
        elif parsed.has_image and self.config.enable_image and not self.config.api_key:
            demoted_texts.append("（绘图功能还没配置好 API Key，这次先不画啦。）")

        # ---------- 重排消息链 ----------
        text_parts = [part.text for part in parsed.texts if part.text.strip()]
        text_parts.extend(text for text in demoted_texts if text.strip())
        # 文字部分：保留正常文本 + 降级回来的台词。
        # dual_send_text 只决定「有文字时语音是否额外单独发送」，而 OneBot 下
        # Record 本就不能与 Plain/Image 同链，所以两种情况都先单独发语音。
        final_text = "\n".join(text_parts)

        only_audio = (
            bool(audio_paths)
            and not final_text.strip()
            and not image_segments
            and not keep_segments
            and not self.config.dual_send_text
        )

        if audio_paths and not only_audio:
            records = [_make_record(path) for path in audio_paths]
            sent = await self._send_chain(event, records)
            if sent:
                try:
                    await asyncio.sleep(1.2)
                except asyncio.CancelledError:
                    raise
            else:
                # 独立发送失败：把语音退化成文字，避免用户收到空回复
                final_text = "\n".join(filter(None, [final_text, *[
                    clean_tts_text(chunk.text) for chunk in parsed.tts_chunks
                ]]))

        if only_audio and not self.config.dual_send_text:
            # 纯语音回复：链里只放语音（OneBot 不允许 Record 与 Plain 同链）
            records = [_make_record(audio_paths[0])]
            self._replace_chain(result, records)
            if len(audio_paths) > 1:
                await self._send_chain(event, [_make_record(path) for path in audio_paths[1:]])
            return

        new_chain: List[Any] = []
        if final_text.strip():
            new_chain.append(Comp.Plain(text=final_text.strip()))
        new_chain.extend(image_segments)
        new_chain.extend(keep_segments)
        if not new_chain:
            new_chain = [Comp.Plain(text="……")]
        self._replace_chain(result, new_chain)

    def _should_speak(
        self, chunk: TTSChunk, mode: str, in_cooldown: bool, forced: bool
    ) -> bool:
        if mode == "off":
            return False
        if forced:
            return True
        if in_cooldown:
            return False
        if mode == "always":
            return True
        if mode == "keyword_only":
            return False
        meets_emotion = chunk.intensity >= self.config.emotion_threshold
        if mode == "emotion_auto":
            return meets_emotion
        if mode == "hybrid_prob":
            if meets_emotion:
                return True
            return random.random() < self.config.base_probability
        return False

    async def _generate_images(self, prompts: List[ImagePrompt]) -> List[Any]:
        if not prompts:
            return []

        async def _one(prompt: ImagePrompt) -> Optional[Any]:
            try:
                result: ImageResult = await self.client.generate_image(
                    prompt.prompt, size=prompt.size or self.config.default_size
                )
            except ContentInspectionError:
                logger.info("[qwen_multimodal] 生图提示词未通过内容审核。")
                return Comp.Plain(text="唔，这个画面太神秘啦，画不出来呢~")
            except AuthError as exc:
                logger.error(f"[qwen_multimodal] 生图鉴权失败: {exc}")
                return None
            except DashScopeError as exc:
                logger.warning(f"[qwen_multimodal] 生图失败: {exc}")
                return None
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[qwen_multimodal] 生图异常: {exc}")
                return None
            self.temp.schedule_cleanup(result.path)
            logger.info(
                "[qwen_multimodal] 图片已生成 %s（模型 %s）", file_info(result.path), result.model
            )
            return _make_image(str(result.path))

        results = await asyncio.gather(*[_one(prompt) for prompt in prompts], return_exceptions=True)
        segments: List[Any] = []
        for item in results:
            if isinstance(item, BaseException):
                logger.warning(f"[qwen_multimodal] 并发生图异常: {item}")
                continue
            if item is not None:
                segments.append(item)
        return segments

    def _replace_chain(self, result: Any, items: List[Any]) -> None:
        chain = getattr(result, "chain", None)
        if isinstance(chain, list):
            chain[:] = items
        else:
            try:
                result.chain = items
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[qwen_multimodal] 无法重写消息链: {exc}")

    # ==================================================================
    # 3. 命令
    # ==================================================================
    @filter.command("qwen_voice")
    async def cmd_voice_profile(self, event: AstrMessageEvent):
        """查看当前 AI 的声音档案（音色、语速、挑选理由）。"""
        profile = await self._ensure_voice(event)
        if profile is None:
            yield event.plain_result(
                "唔……我还没找到合适的声音。请确认已配置 dashscope_api_key 且存在可用的人设。"
            )
            return
        yield event.plain_result(self.voice_engine.format_profile_text(profile))

    @filter.command("qwen_voice_list")
    async def cmd_voice_list(self, event: AstrMessageEvent):
        """查看千问音色图鉴。"""
        yield event.plain_result(VoicePersonaEngine.format_catalog_text())

    @filter.command("qwen_voice_set")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_voice_set(
        self,
        event: AstrMessageEvent,
        voice_id: str = "",
        speed: str = "",
        pitch: str = "",
    ):
        """手动锁定音色：/qwen_voice_set <音色ID> [语速] [语调]"""
        if not voice_id:
            yield event.plain_result(
                "用法：/qwen_voice_set <音色ID> [语速 0.85-1.2] [语调 0.85-1.2]\n"
                "例如：/qwen_voice_set Cherry 1.0 1.0"
            )
            return
        resolved, entry = canonical_voice_id(voice_id)
        if not resolved:
            yield event.plain_result("音色 ID 不能为空。")
            return
        base_speed = entry.base_speed if entry else 1.0
        base_pitch = entry.base_pitch if entry else 1.0
        try:
            if speed.strip():
                base_speed = max(0.85, min(1.2, float(speed)))
            if pitch.strip():
                base_pitch = max(0.85, min(1.2, float(pitch)))
        except ValueError:
            yield event.plain_result("语速/语调必须是数字，例如 1.0。")
            return
        name = entry.name_cn if entry else resolved
        profile = VoiceProfile(
            voice_id=resolved,
            voice_name=name,
            base_speed=base_speed,
            base_pitch=base_pitch,
            characteristic_style=entry.description if entry else "自定义音色",
            reason="管理员手动为我指定了这个声音，我会好好用它说话。",
            persona_name="",
            persona_hash="",
            source="manual",
        )
        await self.state.lock_voice(profile.to_dict())
        self._voice_profile = profile
        self.voice_engine._profile = profile  # 同步引擎缓存
        yield event.plain_result(
            f"好的，声音已经换成 {name}（{resolved}）啦。\n"
            f"语速 {base_speed:.2f}　语调 {base_pitch:.2f}"
        )

    @filter.command("qwen_voice_reset")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_voice_reset(self, event: AstrMessageEvent):
        """清空音色档案，让 AI 重新阅读人设并挑选音色。"""
        await self.state.reset_voice()
        self._voice_profile = None
        self.voice_engine._profile = None
        profile = await self._ensure_voice(event)
        if profile is None:
            yield event.plain_result("音色重置完成，但重新挑选失败了，请检查 Provider 与 API Key。")
            return
        yield event.plain_result(
            "我重新照了照镜子，决定了新的声音：\n\n"
            + self.voice_engine.format_profile_text(profile)
        )

    @filter.command("qwen_tts")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_tts_mode(self, event: AstrMessageEvent, mode: str = ""):
        """设置本会话语音模式：/qwen_tts <auto|prob|always|off>"""
        key = MODE_ALIASES.get(mode.strip().lower(), "")
        if not key:
            yield event.plain_result(
                "用法：/qwen_tts <auto|prob|always|off>\n"
                "auto=情绪自动　prob=情绪+随机　always=句句语音　off=关闭本会话语音"
            )
            return
        await self.state.set_session_mode(self._session_key(event), key)
        yield event.plain_result(f"本会话语音模式已切换为「{MODE_LABELS.get(key, key)}」。")

    @filter.command("qwen_say")
    async def cmd_say(self, event: AstrMessageEvent, text: GreedyStr):
        """强制点播一句语音：/qwen_say <文本> [情绪]"""
        if not text.strip():
            yield event.plain_result("用法：/qwen_say <想让我说的话> [情绪，如 shy]")
            return
        if not self.config.api_key:
            yield event.plain_result("还没有配置 dashscope_api_key，我说不出声音呢。")
            return
        # 最后一个词若是合法情绪名，就当作情绪参数剥掉
        emotion = "happy"
        raw = text.strip()
        parts = raw.split()
        if len(parts) >= 2:
            candidate = normalize_emotion(parts[-1])
            if candidate != "neutral" and candidate in EMOTION_PRESETS:
                emotion = candidate
                raw = " ".join(parts[:-1])
        cleaned = clean_tts_text(raw)
        chars = count_tts_chars(cleaned)
        if chars < self.config.min_tts_chars:
            yield event.plain_result("这句话太短啦，换一句吧。")
            return
        if chars > self.config.max_tts_chars:
            cleaned = cleaned[: self.config.max_tts_chars].rstrip("，。！？~") + "……"
        profile = await self._ensure_voice(event)
        if profile is None:
            yield event.plain_result("音色还没准备好，稍后再试一次吧。")
            return
        prosody = self.voice_engine.compute_prosody(profile, normalize_emotion(emotion), 7)
        try:
            speech = await self.client.synthesize_speech(
                cleaned,
                profile.voice_id,
                speed=prosody.speed,
                pitch=prosody.pitch,
                instruct=prosody.instruct,
                audio_format=self.config.audio_format,
            )
        except ContentInspectionError:
            yield event.plain_result("这句话的内容我有点说不出口呢……换一句好不好？")
            return
        except DashScopeError as exc:
            logger.warning(f"[qwen_multimodal] /qwen_say 合成失败: {exc}")
            yield event.plain_result(f"唔，声音没能生成出来：{exc.message[:80]}")
            return
        self.temp.schedule_cleanup(speech.path)
        await self.state.record_speech(self._session_key(event))
        yield event.chain_result([_make_record(str(speech.path))])

    @filter.command("qwen_draw")
    async def cmd_draw(self, event: AstrMessageEvent, prompt: GreedyStr):
        """生成一张图片：/qwen_draw <提示词>"""
        if not prompt.strip():
            yield event.plain_result("用法：/qwen_draw <画面描述>")
            return
        if not self.config.api_key or not self.config.enable_image:
            yield event.plain_result("画板还没准备好（未配置 API Key 或已关闭生图）。")
            return
        yield event.plain_result("正在为你绘制中，请稍候~")
        try:
            result = await self.client.generate_image(prompt.strip())
        except ContentInspectionError:
            yield event.plain_result("唔，这个画面太神秘啦，画不出来呢~")
            return
        except DashScopeError as exc:
            logger.warning(f"[qwen_multimodal] /qwen_draw 失败: {exc}")
            yield event.plain_result(f"唔，画笔打滑了：{exc.message[:80]}")
            return
        self.temp.schedule_cleanup(result.path)
        yield event.image_result(str(result.path))

    # ==================================================================
    # 4. LLM Tool
    # ==================================================================
    @filter.llm_tool(name="generate_qwen_image")
    async def generate_qwen_image(
        self, event: AstrMessageEvent, prompt: str, size: str = "1024*1024"
    ):
        """当用户要求画图、要图片，或你想展示某个具体画面时，调用此工具生成图片。

        Args:
            prompt(string): 详细的中文画面描述，包含主体、动作、环境、光线、构图与风格。
            size(string): 可选图片尺寸，只能是 1024*1024（方形）、720*1280（竖图）或 1280*720（横图）。
        """
        if not self.config.api_key or not self.config.enable_image:
            return "图像生成功能当前不可用。"
        if not self._session_allowed(event):
            return "当前会话不允许发送图片。"
        size = size if size in ("1024*1024", "720*1280", "1280*720") else self.config.default_size
        try:
            result = await self.client.generate_image(prompt, size=size)
        except ContentInspectionError:
            return "这个画面没有通过内容审核，换一个描述试试。"
        except DashScopeError as exc:
            logger.warning(f"[qwen_multimodal] LLM Tool 生图失败: {exc}")
            return f"图片生成失败：{exc.message[:100]}"
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[qwen_multimodal] LLM Tool 生图异常: {exc}")
            return "图片生成时发生了未知错误。"

        self.temp.schedule_cleanup(result.path)
        await self._send_chain(event, [_make_image(str(result.path))])
        return f"图片已生成并发送给用户（{size}，模型 {result.model}）。不要再重复发送链接。"
