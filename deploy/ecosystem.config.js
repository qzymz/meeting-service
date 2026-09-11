// PM2 process definition for the meeting-service public server.
//
//   pm2 start   deploy/ecosystem.config.js   # 首次启动
//   pm2 restart meeting-server                               # 改 env 后重启
//   pm2 logs    meeting-server
//
// Configuration comes from env/meeting.env (see meeting.env.example);
// deploy.sh wraps all of this as a one-click flow.
const fs = require("fs");
const path = require("path");

const REPO_ROOT = path.resolve(__dirname, "..");
const ENV_FILE = path.join(__dirname, "env", "meeting.env");

function loadEnv(file) {
  const env = {};
  if (!fs.existsSync(file)) return env;
  for (const line of fs.readFileSync(file, "utf8").split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
    if (m) env[m[1]] = m[2];
  }
  return env;
}

const env = loadEnv(ENV_FILE);
const pyUnix = path.join(REPO_ROOT, ".venv", "bin", "python");
const pyWin = path.join(REPO_ROOT, ".venv", "Scripts", "python.exe");

module.exports = {
  apps: [
    {
      name: "meeting-server",
      script: path.join(REPO_ROOT, "run_server.py"),
      interpreter: fs.existsSync(pyUnix) ? pyUnix : pyWin,
      cwd: REPO_ROOT,
      env: env,
      autorestart: true,
      max_restarts: 20,
      restart_delay: 3000,
      max_memory_restart: "500M",
      error_file: path.join(__dirname, "env", "meeting-server.error.log"),
      out_file: path.join(__dirname, "env", "meeting-server.out.log"),
      time: true,
    },
  ],
};
