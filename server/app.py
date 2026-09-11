"""Public server: client API, worker API, and the web client."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import auth
from .db import Database, PENDING, PROCESSING, REFINING, utcnow
from .refine import refine
from .storage import ALLOWED_AUDIO_EXTS, Storage

CLIENT_INDEX = Path(__file__).resolve().parent.parent / "client" / "web" / "index.html"


class Settings:
    def __init__(self) -> None:
        self.data_dir = Path(os.environ.get("MTD_DATA_DIR", "data")).resolve()
        self.worker_key = os.environ.get("MTD_WORKER_KEY", "").strip()
        self.lease_seconds = int(os.environ.get("MTD_LEASE_SECONDS", "3600"))
        self.max_upload_bytes = int(os.environ.get("MTD_MAX_UPLOAD_MB", "500")) * 1024 * 1024
        self.worker_key_file = self.data_dir / ".worker_key"


class RegisterBody(BaseModel):
    username: str
    password: str


class ResultBody(BaseModel):
    text: str
    segments: list[dict]
    duration_sec: float | None = None
    worker_meta: dict | None = None


class FailureBody(BaseModel):
    error: str


def create_app() -> FastAPI:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if not settings.worker_key:
        # Generate once and persist so restarts keep existing workers valid.
        if settings.worker_key_file.exists():
            settings.worker_key = settings.worker_key_file.read_text().strip()
        else:
            settings.worker_key = auth.new_worker_key()
            settings.worker_key_file.write_text(settings.worker_key)

    db = Database(settings.data_dir / "app.db")
    storage = Storage(settings.data_dir)
    print(f"[meeting-service] data dir: {settings.data_dir}")
    print(f"[meeting-service] worker key: {settings.worker_key}")

    app = FastAPI(title="meeting-service", docs_url="/api/docs")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------- helpers

    def current_user(request: Request) -> dict:
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            raise HTTPException(401, "Missing bearer token")
        row = db.get_user_by_token(header[7:].strip())
        if row is None:
            raise HTTPException(401, "Invalid token")
        return {"id": row["id"], "username": row["username"]}

    def require_worker(x_worker_key: str = Header(default="")) -> None:
        if not settings.worker_key or x_worker_key != settings.worker_key:
            raise HTTPException(401, "Invalid worker key")

    def task_or_404(task_id: int) -> dict:
        row = db.get_task(task_id)
        if row is None:
            raise HTTPException(404, "Task not found")
        return dict(row)

    def task_to_json(row: dict, include_result: bool = True) -> dict:
        payload = {
            "id": row["id"],
            "status": row["status"],
            "filename": row["original_filename"],
            "size_bytes": row["size_bytes"],
            "duration_sec": row["duration_sec"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "attempts": row["attempts"],
            "error": row["error"],
        }
        if include_result and row["status"] == "ready":
            payload["transcript"] = {
                "text": row["transcript_text"],
                "segments": json.loads(row["segments_json"] or "[]"),
            }
            payload["summary"] = json.loads(row["summary_json"] or "null")
        return payload

    # ----------------------------------------------------------- web client

    @app.get("/")
    def index():
        return FileResponse(CLIENT_INDEX)

    @app.get("/api/health")
    def health():
        return {"ok": True, "service": "meeting-service"}

    # ----------------------------------------------------------- auth API

    @app.post("/api/auth/register")
    def register(body: RegisterBody):
        if len(body.username) < 2 or len(body.password) < 6:
            raise HTTPException(400, "username >= 2 chars, password >= 6 chars")
        if db.get_user_by_username(body.username) is not None:
            raise HTTPException(409, "Username already exists")
        token = auth.new_token()
        db.create_user(body.username, auth.hash_password(body.password), token)
        return {"username": body.username, "token": token}

    @app.post("/api/auth/login")
    def login(body: RegisterBody):
        row = db.get_user_by_username(body.username)
        if row is None or not auth.verify_password(body.password, row["password_hash"]):
            raise HTTPException(401, "Invalid credentials")
        return {"username": row["username"], "token": row["token"]}

    # ----------------------------------------------------------- client API

    @app.post("/api/tasks")
    async def upload_audio(
        request: Request,
        filename: str = Query(..., description="original file name"),
        duration: float | None = Query(None, description="audio duration in seconds"),
        user: dict = Depends(current_user),
    ):
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_AUDIO_EXTS:
            raise HTTPException(400, f"Unsupported file type '{ext}'")
        task_id = db.create_task(user["id"], filename, ext, 0, duration)
        path = storage.audio_path(task_id, ext)
        size = 0
        with open(path, "wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    fh.close()
                    path.unlink(missing_ok=True)
                    raise HTTPException(413, "Audio too large")
                fh.write(chunk)
        if size == 0:
            path.unlink(missing_ok=True)
            raise HTTPException(400, "Empty upload")
        db.update_size(task_id, size)
        return {"task_id": task_id, "status": PENDING, "size_bytes": size}

    @app.get("/api/tasks")
    def list_my_tasks(user: dict = Depends(current_user)):
        return {"tasks": [task_to_json(dict(r), include_result=False) for r in db.list_tasks(user["id"])]}

    @app.delete("/api/tasks/{task_id}")
    def delete_my_task(task_id: int, user: dict = Depends(current_user)):
        row = task_or_404(task_id)
        if row["user_id"] != user["id"]:
            raise HTTPException(403, "Not your task")
        if not db.delete_task(task_id, user["id"]):
            raise HTTPException(404, "Task not found")
        storage.audio_path(row["id"], row["audio_ext"]).unlink(missing_ok=True)
        return {"deleted": task_id}

    @app.delete("/api/tasks")
    def delete_bulk(scope: str = Query("finished", pattern="^(finished|all)$"), user: dict = Depends(current_user)):
        rows = db.list_tasks(user["id"], limit=100000)
        if scope == "finished":
            rows = [r for r in rows if r["status"] in ("ready", "failed")]
            deleted = db.delete_finished(user["id"])
        else:
            deleted = db.delete_all(user["id"])
        for r in rows:
            storage.audio_path(r["id"], r["audio_ext"]).unlink(missing_ok=True)
        return {"deleted": deleted}

    @app.get("/api/tasks/{task_id}")
    def get_my_task(task_id: int, user: dict = Depends(current_user)):
        row = task_or_404(task_id)
        if row["user_id"] != user["id"]:
            raise HTTPException(403, "Not your task")
        return task_to_json(row)

    @app.get("/api/tasks/{task_id}/audio")
    def get_my_audio(task_id: int, user: dict = Depends(current_user)):
        row = task_or_404(task_id)
        if row["user_id"] != user["id"]:
            raise HTTPException(403, "Not your task")
        path = storage.audio_path(row["id"], row["audio_ext"])
        if not path.exists():
            raise HTTPException(404, "Audio missing")
        return FileResponse(path, media_type="application/octet-stream", filename=row["original_filename"])

    # ----------------------------------------------------------- worker API

    @app.post("/worker/claim")
    def claim_task(worker_id: str = Query("worker-1"), _: None = Depends(require_worker)):
        row = db.claim_task(worker_id, settings.lease_seconds)
        if row is None:
            return JSONResponse({"task": None})
        task = dict(row)
        return {
            "task": {
                "id": task["id"],
                "filename": task["original_filename"],
                "audio_ext": task["audio_ext"],
                "size_bytes": task["size_bytes"],
                "duration_sec": task["duration_sec"],
                "attempts": task["attempts"],
                "download_url": f"/worker/tasks/{task['id']}/audio",
            }
        }

    @app.get("/worker/tasks/{task_id}/audio")
    def download_audio(task_id: int, _: None = Depends(require_worker)):
        row = task_or_404(task_id)
        path = storage.audio_path(row["id"], row["audio_ext"])
        if not path.exists():
            raise HTTPException(404, "Audio missing")
        return FileResponse(path, media_type="application/octet-stream", filename=row["original_filename"])

    @app.post("/worker/tasks/{task_id}/heartbeat")
    def heartbeat(task_id: int, worker_id: str = Query("worker-1"), _: None = Depends(require_worker)):
        if not db.renew_lease(task_id, worker_id, settings.lease_seconds):
            raise HTTPException(409, "Task not claimable by this worker")
        return {"ok": True}

    @app.post("/worker/tasks/{task_id}/result")
    def submit_result(task_id: int, body: ResultBody, _: None = Depends(require_worker)):
        row = task_or_404(task_id)
        if row["status"] != PROCESSING:
            raise HTTPException(409, f"Task in status '{row['status']}', expected '{PROCESSING}'")
        if not db.store_result(task_id, body.text, body.segments, body.duration_sec, body.worker_meta):
            raise HTTPException(409, "Failed to store result")

        # LLM refinement runs in the background so the worker's HTTP call
        # returns immediately — slow LLMs must not look like failed uploads.
        def _finish():
            try:
                summary = refine(body.text, body.segments)
            except Exception as exc:  # transcript is already safe; never lose it
                summary = {"stats": None, "summary_text": None, "llm_model": None, "llm_error": str(exc)}
            db.mark_ready(task_id, summary)

        threading.Thread(target=_finish, daemon=True, name=f"refine-{task_id}").start()
        return {"status": REFINING}

    @app.post("/worker/tasks/{task_id}/failure")
    def submit_failure(task_id: int, body: FailureBody, _: None = Depends(require_worker)):
        if not db.mark_failed(task_id, body.error):
            raise HTTPException(409, "Task already finished")
        return {"status": "failed"}

    @app.post("/worker/tasks/{task_id}/requeue")
    def requeue_failed(task_id: int, _: None = Depends(require_worker)):
        """Put a failed task back into the queue (audio is kept on the server)."""
        if not db.requeue(task_id):
            row = task_or_404(task_id)
            raise HTTPException(409, f"Task in status '{row['status']}', only failed tasks can be requeued")
        return {"status": PENDING}

    return app


app = create_app()
