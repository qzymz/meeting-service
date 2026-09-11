#!/usr/bin/env bash
# meeting-service 公网服务器一键部署（PM2 托管）
#
#   ./deploy.sh            # 安装依赖 + 配置 + PM2 启动（幂等，可重复执行）
#   ./deploy.sh status     # PM2 进程状态
#   ./deploy.sh logs       # 跟随日志
#   ./deploy.sh restart    # 重启（修改 env/meeting.env 后执行）
#   ./deploy.sh stop       # 停止
#   ./deploy.sh update     # 拉取代码 + 升级依赖 + 平滑重启
#
# 首次运行会：创建 .venv（仅轻量服务端依赖，无 torch）→ 生成 env/meeting.env
#（含随机 worker key）→ pm2 startOrReload。之后 `pm2 save` + `pm2 startup`
# 即可实现开机自启（见脚本末尾提示）。
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEPLOY_DIR/.." && pwd)"
ENV_DIR="$DEPLOY_DIR/env"
ENV_FILE="$ENV_DIR/meeting.env"
VENV_DIR="$REPO_ROOT/.venv"
APP_NAME="meeting-server"

say()  { printf '\033[1;36m[deploy]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[deploy][FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- checks
# 逐个候选验证可真正执行且版本达标（跳过 Windows Store 的 python3 占位 stub）
pick_python() {
  local cand
  for cand in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
      PYTHON="$(command -v "$cand")"
      return 0
    fi
  done
  return 1
}

check_env() {
  pick_python || fail "未找到可用的 Python >= 3.10（Windows 下注意 Store 版 python3 是无效占位符）"

  command -v node >/dev/null 2>&1 || fail "未找到 node（PM2 依赖），请先安装 Node.js 16+"
  if ! command -v pm2 >/dev/null 2>&1; then
    say "未找到 pm2，尝试 npm install -g pm2 ..."
    npm install -g pm2 || fail "pm2 安装失败，可手动执行: npm install -g pm2"
  fi
}

# ---------------------------------------------------------------- venv
ensure_venv() {
  local VENV_PY
  if [[ -f "$VENV_DIR/bin/python" ]]; then
    VENV_PY="$VENV_DIR/bin/python"
  elif [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
    VENV_PY="$VENV_DIR/Scripts/python.exe"
  else
    say "创建虚拟环境 $VENV_DIR ..."
    if ! "$PYTHON" -m venv "$VENV_DIR"; then
      fail "venv 创建失败。Ubuntu/Debian 常见原因：缺少 python3-venv，先 apt install python3.12-venv 再重跑"
    fi
    if [[ -f "$VENV_DIR/bin/python" ]]; then
      VENV_PY="$VENV_DIR/bin/python"
    else
      VENV_PY="$VENV_DIR/Scripts/python.exe"
    fi
  fi
  say "安装服务端依赖（fastapi/uvicorn/requests，无 torch）..."
  if "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    "$VENV_PY" -m pip install -q -r "$DEPLOY_DIR/server-requirements.txt"
  elif command -v uv >/dev/null 2>&1; then
    # uv 创建的 venv 不带 pip，直接用 uv 装等价
    uv pip install -q --python "$VENV_PY" -r "$DEPLOY_DIR/server-requirements.txt"
  else
    "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1
    "$VENV_PY" -m pip install -q -r "$DEPLOY_DIR/server-requirements.txt"
  fi
}

# ---------------------------------------------------------------- config
ensure_config() {
  mkdir -p "$ENV_DIR"
  if [[ ! -f "$ENV_FILE" ]]; then
    say "初始化配置 $ENV_FILE ..."
    sed "s#^MTD_DATA_DIR=.*#MTD_DATA_DIR=$ENV_DIR/data#" \
      "$DEPLOY_DIR/meeting.env.example" > "$ENV_FILE"
  fi
  # 占位符 auto → 随机 worker key（仅首次）
  if grep -q '^MTD_WORKER_KEY=auto' "$ENV_FILE"; then
    KEY="$("$PYTHON" -c 'import secrets; print("wsk_" + secrets.token_hex(24))')"
    # 兼容 macOS sed 与 GNU sed：用 python 写回，避免 -i 差异
    "$PYTHON" - "$ENV_FILE" "$KEY" <<'PYEOF'
import sys
path, key = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
open(path, "w", encoding="utf-8").write(
    text.replace("MTD_WORKER_KEY=auto", f"MTD_WORKER_KEY={key}")
)
PYEOF
    say "已生成随机 worker key：$KEY"
    say "（模型端启动参数 --worker-key 请使用此值）"
  fi
  DATA_DIR="$("$PYTHON" - "$ENV_FILE" <<'PYEOF'
import sys
for line in open(sys.argv[1], encoding="utf-8"):
    if line.startswith("MTD_DATA_DIR="):
        print(line.split("=", 1)[1].strip())
        break
PYEOF
)"
  mkdir -p "$DATA_DIR"
}

# ---------------------------------------------------------------- actions
cmd_install() {
  check_env
  ensure_venv
  ensure_config
  say "PM2 启动/重载 $APP_NAME ..."
  pm2 startOrReload "$DEPLOY_DIR/ecosystem.config.js"
  sleep 2
  pm2 status "$APP_NAME"
  say "完成。常用命令：deploy.sh {status|logs|restart|stop}"
  say "开机自启（按提示执行一次）：pm2 save && pm2 startup"
}

cmd_update() {
  check_env
  say "git pull ..."
  git -C "$REPO_ROOT" pull --ff-only || fail "git pull 失败（有本地改动？）"
  ensure_venv
  ensure_config
  pm2 startOrReload "$DEPLOY_DIR/ecosystem.config.js" --update-env
  say "已更新并重载。"
}

case "${1:-install}" in
  install)  cmd_install ;;
  update)   cmd_update ;;
  status)   pm2 status "$APP_NAME" ;;
  logs)     pm2 logs "$APP_NAME" ;;
  restart)  pm2 startOrReload "$DEPLOY_DIR/ecosystem.config.js" ;;  # startOrReload 会重读 env 文件（pm2 restart --update-env 不会）
  stop)     pm2 stop "$APP_NAME" ;;
  *) echo "用法: $0 [install|update|status|logs|restart|stop]" >&2; exit 1 ;;
esac
