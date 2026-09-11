"""End-to-end test for meeting_service.

Launches the real server as a subprocess and exercises the full state
machine over HTTP with a fake worker (no GPU required):

  register -> upload -> worker claim -> download -> submit result -> ready
  plus: auth failures, lease expiry reclaim, explicit failure path.

Run from the repo root:

    .venv/Scripts/python meeting_service/test_e2e.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parent
PORT = 8912
BASE = f"http://127.0.0.1:{PORT}"
WORKER_KEY = "test-worker-key-123"
DATA_DIR = Path(tempfile.mkdtemp(prefix="mtd_e2e_"))

SERVER = None
PASSED = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASSED
    if not cond:
        print(f"FAIL: {name} {detail}")
        stop_server()
        sys.exit(1)
    PASSED += 1
    print(f"  ok: {name}")


def start_server():
    global SERVER
    env = dict(os.environ, MTD_LEASE_SECONDS="2")
    SERVER = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "run_server.py"),
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "--data-dir", str(DATA_DIR),
            "--worker-key", WORKER_KEY,
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            if requests.get(f"{BASE}/", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    print("server failed to start")
    stop_server()
    sys.exit(1)


def stop_server():
    if SERVER:
        SERVER.terminate()
        try:
            SERVER.wait(timeout=5)
        except subprocess.TimeoutExpired:
            SERVER.kill()


def make_wav(seconds: float = 3.0) -> bytes:
    sr = 16000
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wav = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


WKEY = {"X-Worker-Key": WORKER_KEY}


def main():
    print("== meeting_service e2e ==")
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    start_server()
    try:
        run_tests()
    finally:
        stop_server()
        shutil.rmtree(DATA_DIR, ignore_errors=True)
    print(f"\nALL {PASSED} CHECKS PASSED")


def run_tests():
    # -- auth ---------------------------------------------------------
    print("[auth]")
    user = f"u{uuid.uuid4().hex[:8]}"
    r = requests.post(f"{BASE}/api/auth/register", json={"username": user, "password": "secret123"})
    check("register", r.status_code == 200 and r.json().get("token"))
    token = r.json()["token"]
    AUTH = {"Authorization": f"Bearer {token}"}

    r = requests.post(f"{BASE}/api/auth/login", json={"username": user, "password": "wrong"})
    check("login rejects bad password", r.status_code == 401)

    r = requests.post(f"{BASE}/api/auth/register", json={"username": user, "password": "secret123"})
    check("duplicate register rejected", r.status_code == 409)

    # -- upload auth ----------------------------------------------------
    wav = make_wav()
    r = requests.post(f"{BASE}/api/tasks", params={"filename": "t.wav"}, data=wav)
    check("upload without token rejected", r.status_code == 401)

    # -- main flow ------------------------------------------------------
    print("[upload -> claim -> result -> ready]")
    r = requests.post(
        f"{BASE}/api/tasks",
        params={"filename": "meeting.wav", "duration": "3.0"},
        data=wav, headers=AUTH,
    )
    check("upload accepted", r.status_code == 200, r.text)
    task_id = r.json()["task_id"]

    r = requests.get(f"{BASE}/api/tasks/{task_id}", headers=AUTH).json()
    check("status pending", r["status"] == "pending")

    r = requests.post(f"{BASE}/worker/claim")
    check("claim without key rejected", r.status_code == 401)
    r = requests.post(f"{BASE}/worker/claim", params={"worker_id": "w1"}, headers={"X-Worker-Key": "nope"})
    check("claim with wrong key rejected", r.status_code == 401)

    r = requests.post(f"{BASE}/worker/claim", params={"worker_id": "w1"}, headers=WKEY).json()
    task = r["task"]
    check("claim returns our task", task and task["id"] == task_id, json.dumps(r))

    audio = requests.get(f"{BASE}{task['download_url']}", headers=WKEY)
    check("worker downloads identical audio", audio.status_code == 200 and audio.content == wav)

    r = requests.get(f"{BASE}/worker/tasks/{task_id}/audio")
    check("download without key rejected", r.status_code == 401)

    segments = [
        {"start": 0.5, "end": 2.0, "speaker": "S01", "text": "大家好，我们开始会议。"},
        {"start": 2.2, "end": 3.0, "speaker": "S02", "text": "好的。"},
    ]
    text = "[0.5][S01]大家好，我们开始会议。[2.0][2.2][S02]好的。[3.0]"
    r = requests.post(
        f"{BASE}/worker/tasks/{task_id}/result",
        headers=WKEY,
        json={"text": text, "segments": segments, "duration_sec": 3.0, "worker_meta": {"model": "fake"}},
    )
    check("result accepted", r.status_code == 200 and r.json()["status"] == "ready", r.text)

    r = requests.post(
        f"{BASE}/worker/tasks/{task_id}/result",
        headers=WKEY, json={"text": "dup", "segments": []},
    )
    check("duplicate result rejected", r.status_code == 409)

    r = requests.get(f"{BASE}/api/tasks/{task_id}", headers=AUTH).json()
    check("task ready", r["status"] == "ready")
    check("transcript stored", r["transcript"]["segments"] == segments)
    stats = r["summary"]["stats"]
    check("speaker stats", stats["speaker_count"] == 2 and stats["speech_sec"] == 2.3, json.dumps(stats))

    # audio playback for owner
    r = requests.get(f"{BASE}/api/tasks/{task_id}/audio", headers=AUTH)
    check("owner downloads audio", r.status_code == 200 and r.content == wav)

    # another user cannot see the task
    other = requests.post(
        f"{BASE}/api/auth/register", json={"username": user + "b", "password": "secret123"}
    ).json()
    r = requests.get(
        f"{BASE}/api/tasks/{task_id}",
        headers={"Authorization": f"Bearer {other['token']}"},
    )
    check("cross-user access forbidden", r.status_code == 403)

    # -- lease expiry ---------------------------------------------------
    print("[lease expiry reclaim]")
    r = requests.post(f"{BASE}/api/tasks", params={"filename": "stuck.wav"}, data=make_wav(1), headers=AUTH)
    task2 = r.json()["task_id"]
    requests.post(f"{BASE}/worker/claim", params={"worker_id": "dead-worker"}, headers=WKEY)
    time.sleep(3.0)  # lease_seconds=2 in this test env, with margin
    r = requests.post(f"{BASE}/worker/claim", params={"worker_id": "w2"}, headers=WKEY).json()
    reclaimed = r["task"]
    check("expired task reclaimed", reclaimed and reclaimed["id"] == task2, json.dumps(r))
    check("attempts incremented", reclaimed["attempts"] == 2)

    r = requests.post(
        f"{BASE}/worker/tasks/{task2}/result",
        headers=WKEY, json={"text": "[0.1][S01]hi[0.9]",
                            "segments": [{"start": 0.1, "end": 0.9, "speaker": "S01", "text": "hi"}]},
    )
    check("reclaimed task finishes", r.status_code == 200)

    # -- explicit failure -------------------------------------------------
    print("[failure path]")
    r = requests.post(f"{BASE}/api/tasks", params={"filename": "bad.wav"}, data=make_wav(1), headers=AUTH)
    task3 = r.json()["task_id"]
    requests.post(f"{BASE}/worker/claim", params={"worker_id": "w3"}, headers=WKEY)
    r = requests.post(f"{BASE}/worker/tasks/{task3}/failure", headers=WKEY, json={"error": "OOM"})
    check("failure reported", r.status_code == 200)
    r = requests.get(f"{BASE}/api/tasks/{task3}", headers=AUTH).json()
    check("task failed with error", r["status"] == "failed" and "OOM" in r["error"])

    # -- task list --------------------------------------------------------
    r = requests.get(f"{BASE}/api/tasks", headers=AUTH).json()
    check("list shows 3 tasks", len(r["tasks"]) == 3)

    # -- cleanup / deletion -----------------------------------------------
    print("[deletion]")
    other_tok = requests.post(
        f"{BASE}/api/auth/register", json={"username": user + "c", "password": "secret123"}
    ).json()["token"]
    r = requests.delete(
        f"{BASE}/api/tasks/{task3}", headers={"Authorization": f"Bearer {other_tok}"}
    )
    check("cross-user delete forbidden", r.status_code == 403)

    r = requests.delete(f"{BASE}/api/tasks/{task3}", headers=AUTH)
    check("delete own task", r.status_code == 200 and r.json()["deleted"] == task3)
    r = requests.get(f"{BASE}/api/tasks/{task3}", headers=AUTH)
    check("deleted task gone", r.status_code == 404)

    r = requests.delete(f"{BASE}/api/tasks?scope=finished", headers=AUTH)
    check("bulk delete finished", r.status_code == 200 and r.json()["deleted"] == 2, r.text)
    r = requests.get(f"{BASE}/api/tasks", headers=AUTH).json()
    check("no ready/failed remain", all(t["status"] not in ("ready", "failed") for t in r["tasks"]))

    r = requests.delete(f"{BASE}/api/tasks?scope=bogus", headers=AUTH)
    check("bad scope rejected", r.status_code == 422)


if __name__ == "__main__":
    main()
