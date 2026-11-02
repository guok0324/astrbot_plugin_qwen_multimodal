# 千问灵动多模态 · astrbot_plugin_qwen_multimodal

给 AstrBot 的 LLM 补上「声音」和「画笔」。

语言模型本身只会输出文字。这个插件在 LLM 收到请求时悄悄塞进一份**隐式表达协议**，让模型在情绪到位时自己决定：
这句话该「说」出来（`<tts>`），还是该「画」出来（`<img_gen>`）。插件再把这些标签还原成真正的语音和图片，
整个过程对用户完全透明——不会有一句 XML 漏到聊天里。

- 语音：阿里云百炼 Qwen-TTS，支持情绪驱动的语速/音高/语气演绎
- 音色：首次使用由 AI 阅读当前人格设定，自己挑一个声音并说明理由，之后永久锁定
- 生图：百炼文生图（异步任务 + 轮询），支持自动触发与 `/qwen_draw` 手动触发
- 兼容：OneBot v11（NapCat / LLGO / Lagrange）、Telegram、飞书

---

## 一、安装

1. 把整个 `astrbot_plugin_qwen_multimodal` 目录放进 AstrBot 的 `data/plugins/` 下。
2. 重启 AstrBot，或在 WebUI「插件」页点击重载。
3. 打开插件配置，填入 **DashScope API Key**（阿里云百炼控制台 → API-KEY 管理）。
4. 保存后即可在聊天中生效。

依赖（`requirements.txt`，AstrBot 通常已自带）：

```bash
pip install aiohttp aiofiles pydantic
```

> 需要 AstrBot `>= 4.5.7`。插件使用 `data/plugin_data/astrbot_plugin_qwen_multimodal/` 存放状态与临时文件。

---

## 二、它是怎么工作的

```
用户消息 ──► LLM（system_prompt 里被注入表达协议）
                │
                ▼
        "呜…你终于回来了（小声）<tts emotion="aggrieved" intensity="7">我还以为你今天不来了呢</tts>"
                │
                ▼
        on_decorating_result 钩子
           ├─ 解析标签 → TTSChunk / 文本 / ImagePrompt
           ├─ 情绪 + 音色档案 → 语速、音高、instruct
           ├─ 调 TTS → 本地音频 → 单独发送 Record
           └─ 调生图 → 下载 PNG → 追加 Image
                │
                ▼
        用户听到语音、看到图片，文字部分照常显示
```

三个关键设计：

1. **协议注入只加常量文本**。`on_llm_request` 追加的是一段固定协议，不含时间、用户名等易变内容，
   避免破坏 AstrBot 的 prompt 缓存。
2. **失败一律无感降级**。合成失败、审核拦截、Key 失效、超时——最差情况是把 `<tts>` 里的台词
   还原成普通文字发出去，用户永远不会看到标签，也不会收到空回复。
3. **OneBot 兼容优先**。`Record` 绝不与 `Plain`/`Image` 放在同一条消息链里：先单独 `event.send`
   语音，`sleep 1.2s` 等 NapCat 完成 SILK 转码，再发送文字与图片。

---

## 三、模型标签协议

LLM 被要求按下面的格式输出（标签对用户不可见）：

```text
<tts emotion="shy" intensity="7" speed_delta="-0.05" pitch_delta="0.1"
     instruct="语气软糯，尾音微微上扬">你、你刚才说什么呀……</tts>
<img_gen prompt="黄昏的电车车厢，窗边坐着穿校服的少女，逆光，胶片质感" size="1280*720" />
```

| 属性 | 取值 | 说明 |
| --- | --- | --- |
| `emotion` | happy / excited / angry / coquettish / shy / sad / aggrieved / sleepy / whisper / sarcastic / tsundere / shocked / neutral | 决定情绪韵律预设 |
| `intensity` | 1–10 整数 | 情绪强度，越高越夸张；同时用于触发判定 |
| `speed_delta` | -0.3 ~ +0.4 | 在音色基准语速上叠加 |
| `pitch_delta` | -0.2 ~ +0.3 | 在音色基准音高上叠加 |
| `instruct` | 任意短句 | 直接传给 TTS 的语气演绎描述，留空则用预设 |

最终语速 = `clamp(音色基准 + 情绪倍数修正 + speed_delta, 0.5, 2.0)`，音高同理。

---

## 四、音色自适配

首次触发语音时，插件会：

1. 读取当前会话的人格（Persona）设定；
2. 用一次独立的 LLM 请求问它：「你觉得自己适合什么声音？」；
3. 模型返回 JSON：`voice_id`、`base_speed`、`base_pitch`、`characteristic_style`、`reason`；
4. 计算人格指纹（sha256 前 16 位）并把档案写进 `qwen_multimodal_state.json`，**永久锁定**；
5. 人格被修改后（指纹变化）自动重新挑选——可在配置里关掉。

内置 6 个自荐音色（同时支持百炼任意自定义音色 ID）：

| 音色 ID | 中文名 | 气质 |
| --- | --- | --- |
| Cherry | 芊悦 | 温柔甜美，邻家治愈 |
| Serena | 苏瑶 | 元气活泼，古灵精怪 |
| Chelsie | 千雪 | 清冷御姐，知性毒舌 |
| Momo | 茉茉 | 俏皮可爱，撒娇粘人 |
| Ethan | 晨煦 | 阳光少年，清爽温暖 |
| Vincent | 文森 | 低音炮，成熟磁性 |

LLM 挑音色失败时，会按人格关键词正则兜底（如「傲娇」→ Serena，「御姐」→ Chelsie）。

---

## 五、指令

| 指令 | 权限 | 说明 |
| --- | --- | --- |
| `/qwen_voice` | 所有人 | 查看当前音色档案，包括 AI 挑选这个声音的理由 |
| `/qwen_voice_list` | 所有人 | 音色图鉴 |
| `/qwen_voice_set <音色ID> [语速] [语调]` | 管理员 | 手动锁定音色，例如 `/qwen_voice_set Chelsie 0.95 0.9` |
| `/qwen_voice_reset` | 管理员 | 清空档案，让 AI 重新读人格挑声音，并把心路历程发出来 |
| `/qwen_tts <auto\|prob\|always\|off>` | 管理员 | 设置本会话的语音触发模式 |
| `/qwen_say <文本> [情绪]` | 所有人 | 强制点播一句，例如 `/qwen_say 晚安呀 shy` |
| `/qwen_draw <提示词>` | 所有人 | 手动生图，先回「正在为你绘制中，请稍候~」 |

同时注册了 LLM 工具 `generate_qwen_image(prompt, size)`，模型可以自己决定调用它画图。

---

## 六、配置要点

### 语音触发模式（`tts_settings.trigger_mode`）

| 模式 | 行为 |
| --- | --- |
| `emotion_auto` | 默认。`intensity >= emotion_threshold` 且不在冷却中才发语音 |
| `hybrid_prob` | 情绪到位必发；否则按 `base_probability` 随机发 |
| `always` | 只要有 `<tts>` 就发，无视冷却 |
| `keyword_only` | 只在用户说出「语音 / 说一句 / 念一下 / 唱…」时发 |
| `off` | 完全关闭语音，标签自动剥成文字 |

无论哪种模式，用户明确索要语音、或模式为 `always` 时都会**无视冷却**立即发声。

### 其他常用项

- `max_tts_chars` / `min_tts_chars`：过长过短的台词会放弃合成、回退文字
- `cooldown_seconds`：同一会话两次语音之间的最小间隔
- `dual_send_text`：语音之外是否保留文字版（默认关闭，纯语音回复更自然）
- `image_trigger_mode`：`auto_and_command` 允许模型自主配图；`command_only` 只响应 `/qwen_draw`
- `group_control`：私聊开关 + 群聊白/黑名单，被禁止的会话会剥掉所有标签只发文字

---

## 七、模型兼容策略

百炼的模型名会随账号权限变化。插件采用**候选链 + 成功缓存**：

- TTS：配置的模型 → `qwen3-tts-flash` → `qwen-tts`，遇到「模型不存在 / url error」自动试下一个；
  全部失败后再尝试一次 CosyVoice 异步协议。
- 生图：`qwen-image-2.1-pro` → `qwen-image-plus` → `wanx2.1-t2i-turbo` → `wanx-v1`。

失败的模型会被记入黑名单，本次运行期间不再重试，因此第一次慢一点，之后直接命中可用模型。

---

## 八、目录结构

```
astrbot_plugin_qwen_multimodal/
├── main.py                 # 插件入口：协议注入、结果装饰、指令、LLM 工具
├── config_manager.py       # 配置读取（带类型强制转换）+ 状态持久化
├── voice_persona_engine.py # 音色目录、情绪预设、人格→音色校准
├── emotion_parser.py       # <tts>/<img_gen> 解析、六步文本清洗
├── dashscope_client.py     # TTS / 生图 API 客户端（含降级与重试）
├── audio_utils.py          # 临时文件管理、音频格式嗅探
├── metadata.yaml
├── _conf_schema.json
├── requirements.txt
└── README.md
```

运行时数据：

```
data/plugin_data/astrbot_plugin_qwen_multimodal/
├── qwen_multimodal_state.json   # 锁定的音色档案 + 会话覆盖
├── temp_audio/                  # 合成音频，发送后 60s 清理
└── temp_image/                  # 生成图片，发送后 60s 清理
```

---

## 九、常见问题

**发了语音但听不到 / 变成文件**
不同协议端对音频格式支持不同。可在配置里把 `audio_format` 换成 `wav` 试试；NapCat 需要正确的
ffmpeg/SILK 支持。

**一直不说话**
检查：Key 是否填写、`enable_tts` 是否开启、`trigger_mode` 是否为 `off`、是否处于冷却期，
以及模型的 `intensity` 是否够高（可临时用 `/qwen_tts always` 验证链路）。

**报 400 DataInspectionFailed**
台词或提示词触发了内容审核。语音会静默降级为文字，生图会回复一句可爱的话术，不会中断对话。

**人格换了但音色没变**
`auto_recalibrate_on_persona_change` 关闭时会保持原音色；执行 `/qwen_voice_reset` 可强制重选。

**临时文件越来越多**
正常情况下发送后 60 秒自动删除；插件启动与卸载时也会清空历史残留。若仍有堆积，检查进程是否有写权限。

---

## 十、开发与测试

```bash
python -m py_compile astrbot_plugin_qwen_multimodal/*.py   # 语法检查
python tests/run_tests.py                                  # 32 个离线用例（伪造 astrbot 包）
python tests/live_check.py                                 # 真实 Key 联调（读取 tests/.env.local）
```

`tests/` 目录不属于插件产物，发布时可删除。

---

## License

AGPL-3.0-or-later，与 AstrBot 保持一致。
