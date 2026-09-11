# Android 客户端（规划中，后续开发）

独立的安卓 App 形式用户端。**不引入任何新的服务端接口**——App 与网页形式
（`client/web/`）消费同一套 REST API，服务器无需为 App 做任何改动。

## 与服务端的契约

### 音频格式约定

App 端录音建议直接产出 **16kHz / 单声道 / PCM16 WAV**（与网页端浏览器内
转换后的格式一致，也是模型音频前端的原生格式）：

- Android 上用 `AudioRecord`（采样率 16000、`CHANNEL_IN_MONO`、
  `ENCODING_PCM_16BIT`）边录边写 WAV，体积约 1.1MB/分钟；
- 或录 AAC/M4A 后用 `MediaCodec`/`MediaExtractor` 解码转 PCM 再封装 WAV；
- 兜底：服务器也接受 mp3/m4a/ogg/webm/flac 等格式（见
  `server/storage.py` 的扩展名白名单），但模型端解码依赖 PyAV，
  WAV 契约最稳妥。

### API 调用流程

```
1. POST /api/health                      — 启动时连通性检查
2. POST /api/auth/register | /login      — 拿 token，存 EncryptedSharedPreferences
3. POST /api/tasks?filename=x.wav&duration=6.0
   Header: Authorization: Bearer <token>
   Body:   音频二进制（流式上传，建议 OkHttp RequestBody + 进度回调）
4. GET  /api/tasks/{id}                  — 轮询状态（pending/processing/refining/ready/failed）
5. ready 后取 transcript.segments + summary 渲染；
   GET /api/tasks/{id}/audio 可回放原始录音
```

完整接口定义见 `meeting_service/README.md` 的 API 概览。

### App 端功能清单（建议）

- 前台服务录音（`FOREGROUND_SERVICE_MICROPHONE`，Android 14+ 需声明）；
- 录音中断续传 / 失败重试（服务端上传是原子的，失败即整个重传）；
- 任务列表 + 状态轮询（WorkManager 周期任务或 FCM 推送，二选一）；
- 分段结果展示 + 说话人筛选 + 导出 SRT 分享；
- RECORD_AUDIO 运行时权限引导。

## 服务端已为 App 准备好的点

- CORS 全开放（`allow_origins=["*"]`），App 调试期可直接连开发服务器；
- 鉴权全部走 `Authorization: Bearer`，无 Cookie 依赖；
- `/api/health` 健康检查已就绪；
- 上传无 multipart 依赖，原始 body 直传，OkHttp/Retrofit 都很简单。
