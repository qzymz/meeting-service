# meeting-service — 三端会议实录服务（自托管）

基于 [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
模型的会议/对话转写系统：**带说话人分离的转写 + 时间戳 + 可选 LLM 纪要**，
按「用户端 → 公网服务器 → 模型端」三端架构设计，用户端与模型端均无需公网 IP。

```
用户端（两种形式，均无公网IP）        公网服务器（有公网IP）                模型端（GPU，无公网IP）
┌──────────────────┐   上传录音    ┌──────────────────────┐   轮询认领   ┌──────────────────┐
│ 形式1: 网页        │ ───────────▶ │ 任务队列（SQLite）     │ ◀────────── │ worker 循环       │
│  （服务器直接挂载） │              │ 音频对象存储（磁盘）    │   下载音频   │ MOSS 0.9B 转写    │
│ 形式2: 安卓App     │ ◀─────────── │ LLM 纪要精炼（可选）   │ ──────────▶ │ +说话人分离       │
│  （后续开发）      │   拉取结果    └──────────────────────┘   上传结果    └──────────────────┘
└──────────────────┘
```

- **服务器零 AI 计算、零 torch 依赖**：1核 1G VPS 即可，只做 HTTP 中转 + 状态机 + 落盘；
- 用户端网页由服务器直接挂载在 `/`，浏览器打开即用（录音自动转 16kHz WAV 上传）；
- 任务状态机 `pending → processing → refining → ready`，原子认领 + 租约超时回收；
- LLM 纪要可选：配置任意 OpenAI 兼容接口即自动生成会议纪要，失败不影响转写。
- **超长音频自动切片**：超过 `--chunk-seconds`（默认 25 分钟）的录音在静音边界
  自动切分，逐段转写后时间戳偏移拼接；跨段说话人可选声纹对齐（装
  `resemblyzer`+`webrtcvad-wheels` 即启用，未安装时跨段说话人给全新编号，宁可
  拆开不误合并）。

## 目录结构

```
├── run_server.py        # 公网服务器入口
├── server/              # FastAPI 应用：API + 状态机 + 精炼
│   ├── app.py           #   路由（/api 用户端、/worker 模型端）
│   ├── db.py            #   SQLite 任务状态机（原子认领、租约）
│   ├── auth.py          #   用户 PBKDF2 密码 + worker key
│   ├── storage.py       #   音频文件落盘
│   └── refine.py        #   说话人统计 + 可选 LLM 纪要
├── worker/main.py       # 模型端：轮询→下载→GPU转写→心跳→上传
├── client/
│   ├── web/index.html   # 用户端形式1：网页（服务器挂载在 /）
│   └── android/         # 用户端形式2：安卓App（规划，见其 README）
├── deploy/              # 公网服务器一键部署（PM2 托管）
│   ├── deploy.sh        #   一键：install/update/status/logs/restart/stop
│   ├── ecosystem.config.js  # PM2 进程定义（读 env/meeting.env）
│   ├── meeting.env.example  # 配置模板（worker key、端口、LLM 等）
│   ├── server-requirements.txt  # 服务端最小依赖（无 torch）
│   └── nginx-meeting.conf.example  # HTTPS 反代示例
└── test_e2e.py          # 端到端测试（假worker，无需GPU）
```

## 部署

### 1. 公网服务器（Linux VPS）

```bash
git clone https://github.com/qzymz/meeting-service.git && cd meeting-service
npm install -g pm2                       # 未装 pm2 时先装
bash deploy/deploy.sh                    # 一键：建venv→装依赖→生成配置→PM2启动
```

首次运行会打印随机生成的 worker key（也在 `deploy/env/meeting.env`），改配置后
`bash deploy/deploy.sh restart` 生效。开机自启执行一次 `pm2 save && pm2 startup`。
HTTPS 反代示例见 `deploy/nginx-meeting.conf.example`。

**手机/平板录音必须 HTTPS**（浏览器只在安全上下文开放麦克风，明文 HTTP 下
`navigator.mediaDevices` 为 undefined）。免备案方案：DuckDNS 免费子域名 +
8443 端口 + DNS 验证签 Let's Encrypt 证书，完整步骤见
`deploy/nginx-meeting-https.example`；临时测试可用其中的 cloudflared 快速隧道。

手动前台运行（调试用）：`python run_server.py --host 0.0.0.0 --port 8000`。

### 2. 模型端（GPU 机器，无需公网 IP）

worker 需要 MOSS-Transcribe-Diarize 模型包与本仓库代码在**同一个 Python 环境**：

```bash
# 2.1 安装模型环境（一次性，约 3GB）
git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git
cd MOSS-Transcribe-Diarize
uv venv .venv --python 3.12
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
uv pip install -e ".[torch-runtime]" --torch-backend=auto

# 2.2 （可选）启用跨切片说话人声纹对齐
pip install resemblyzer webrtcvad-wheels

# 2.3 拉取本仓库并启动 worker（保持在同一 venv 中）
git clone https://github.com/qzymz/meeting-service.git
cd meeting-service
python -m worker.main \
    --server http://<服务器地址>:8000 \
    --worker-key <deploy.sh 打印的key> \
    --model OpenMOSS-Team/MOSS-Transcribe-Diarize
```

首次推理会从 HuggingFace 下载模型（约 1.8GB，国内可加
`HF_ENDPOINT=https://hf-mirror.com`）。之后常驻轮询即可，多 GPU 机器可起多个
worker 并行消费。

### 3. 用户端

浏览器打开 `http://<服务器地址>:8000/` → 注册 → 录音或选择音频/视频文件 →
等待「已完成」→ 查看说话人分段 / 说话人统计 / AI纪要 → 导出 SRT / JSON。

## 端到端测试

```bash
pip install -r requirements.txt
python test_e2e.py          # 端到端（假worker，无需GPU）
python test_split.py        # 切片/说话人对齐单元测试（仅需 numpy）
```

启动真实服务器子进程 + 假 worker，覆盖 24 项检查：注册登录、上传鉴权、原子认领、
音频下载一致性、结果入库、说话人统计、越权访问拒绝、租约超时回收、失败上报。

## 配置项（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MTD_DATA_DIR` | `data` | 数据目录（SQLite + 音频文件） |
| `MTD_WORKER_KEY` | 自动生成 | 模型端认领任务的 API key |
| `MTD_LEASE_SECONDS` | `3600` | 任务租约时长，超时未完成自动回收 |
| `MTD_MAX_UPLOAD_MB` | `500` | 单个音频上传上限 |
| `MTD_HOST` / `MTD_PORT` | `127.0.0.1` / `8000` | 监听地址（PM2 部署时用） |
| `MTD_LLM_BASE_URL` | 空 | OpenAI 兼容 API 地址 |
| `MTD_LLM_API_KEY` | 空 | LLM API key |
| `MTD_LLM_MODEL` | 空 | 模型名，如 `gpt-4o-mini` / `glm-4.7` |

LLM 三项全填才生成 AI 纪要；LLM 失败不影响转写结果返回。

worker 端参数（`python -m worker.main --help` 查看全部）：`--chunk-seconds`
（超长音频切分阈值，默认 1500 秒）、`--split-search-window`（切点在静音区
搜索的范围，默认 90 秒）、`--align-threshold`（跨片说话人声纹匹配阈值，
默认 0.72）、`--download-attempts`（下载重试次数）。

## API 概览

用户端（`Authorization: Bearer <token>`）：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 连通性检查 |
| POST | `/api/auth/register` / `/api/auth/login` | `{username, password}` → `{token}` |
| POST | `/api/tasks?filename=x.wav&duration=6.0` | 音频二进制体 → `{task_id}` |
| GET | `/api/tasks` / `/api/tasks/{id}` | 任务列表 / 详情（ready 时含 transcript+summary） |
| GET | `/api/tasks/{id}/audio` | 下载原始音频 |

模型端（`X-Worker-Key: <key>`）：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/worker/claim?worker_id=w1` | 原子认领一个任务（无任务返回 `{"task": null}`） |
| GET | `/worker/tasks/{id}/audio` | 下载音频 |
| POST | `/worker/tasks/{id}/heartbeat` | 续租 |
| POST | `/worker/tasks/{id}/result` | `{text, segments, duration_sec, worker_meta}` |
| POST | `/worker/tasks/{id}/failure` | `{error}` |

## 生产环境注意事项

- 用反向代理（Nginx/Caddy）终结 **HTTPS**，会议录音是高敏数据；
- `MTD_WORKER_KEY` 泄露等于开放所有用户录音下载，务必强随机并保密；
- 磁盘按 16kHz WAV ≈ 115MB/小时增长，规划定期清理或改对象存储直传；
- 部署在 frp 等内网穿透隧道后，隧道可能回收空闲 keep-alive 连接导致偶发断连：
  worker 轮询已内置异常重试，不受影响；脚本化访问建议加 `Connection: close` 或重试；
- 多模型端天然并行（认领原子化）；SQLite 单机足够，多服务器时再换 PostgreSQL。

## License

Apache-2.0（沿用上游 MOSS-Transcribe-Diarize）
