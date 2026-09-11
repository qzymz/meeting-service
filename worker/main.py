"""Model-end worker: polls the public server, downloads audio, transcribes
with MOSS-Transcribe-Diarize, and uploads the diarized transcript.

Runs behind NAT: it only makes outbound HTTP connections.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

import requests

try:
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        dtype_from_name,
        generate_transcription,
        resolve_device,
    )
except ImportError as exc:  # standalone repo: model package is an external dep
    raise SystemExit(
        "This worker needs the MOSS-Transcribe-Diarize model package in the same\n"
        "Python environment. Install it first (GPU machine):\n"
        "  git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git\n"
        "  uv venv .venv --python 3.12 && source .venv/bin/activate\n"
        "  uv pip install -e ./MOSS-Transcribe-Diarize\"[torch-runtime]\" --torch-backend=auto\n"
        "Then run this worker with that environment's python."
    ) from exc

LOGGER = logging.getLogger("meeting-worker")


# ------------------------------------------------------------------ model

def load_model(model_path: str, device, dtype):
    """Load model with the repo's attention-backend priority (fa2 > sdpa > eager)."""
    import importlib.util
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    if device.type == "cuda" and importlib.util.find_spec("flash_attn") is not None:
        candidates = ("flash_attention_2", "sdpa", "eager")
    else:
        candidates = ("sdpa", "eager")

    model = None
    for attn in candidates:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype="auto",
                attn_implementation=attn,
            )
            break
        except Exception:
            if attn == candidates[-1]:
                raise
            LOGGER.warning("attn_implementation=%s failed, trying next", attn, exc_info=True)
    model = model.to(dtype=dtype).to(device).eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


# ------------------------------------------------------------------ worker

class Worker:
    def __init__(self, args):
        self.server = args.server.rstrip("/")
        self.key = args.worker_key
        self.worker_id = args.worker_id
        self.poll_interval = args.poll_interval
        self.lease_seconds = args.lease_seconds
        self.download_attempts = args.download_attempts
        self.once = args.once
        self.model_id = args.model
        self.device_name = args.device
        self.dtype_name = args.dtype
        self.max_new_tokens = args.max_new_tokens
        self.prompt = args.prompt
        self.work_dir = Path(args.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._processor = None

    # ------------------------------------------------------------- helpers

    def _headers(self):
        return {"X-Worker-Key": self.key}

    def _claim(self):
        resp = requests.post(
            f"{self.server}/worker/claim",
            params={"worker_id": self.worker_id},
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("task")

    def _download(self, task) -> Path:
        """Download with retries — the public path drops long transfers occasionally."""
        path = self.work_dir / f"task_{task['id']}{task['audio_ext']}"
        part = path.with_suffix(path.suffix + ".part")
        attempts = self.download_attempts
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                headers = dict(self._headers(), **{"Connection": "close"})
                with requests.get(
                    f"{self.server}{task['download_url']}",
                    headers=headers, stream=True, timeout=(15, 180),
                ) as resp:
                    resp.raise_for_status()
                    with open(part, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                part.replace(path)  # only complete files get the final name
                return path
            except Exception as exc:
                last_exc = exc
                LOGGER.warning(
                    "task %s download attempt %d/%d failed: %r",
                    task["id"], attempt, attempts, exc,
                )
                part.unlink(missing_ok=True)
                if attempt < attempts:
                    time.sleep(min(30, 2 ** attempt))
        raise last_exc  # type: ignore[misc]

    def _heartbeat_loop(self, task_id: int, stop: threading.Event):
        while not stop.wait(self.lease_seconds / 3):
            try:
                requests.post(
                    f"{self.server}/worker/tasks/{task_id}/heartbeat",
                    params={"worker_id": self.worker_id},
                    headers=self._headers(),
                    timeout=15,
                ).raise_for_status()
            except Exception:
                LOGGER.warning("heartbeat for task %s failed", task_id, exc_info=True)

    def _fail(self, task_id: int, error: str):
        try:
            requests.post(
                f"{self.server}/worker/tasks/{task_id}/failure",
                headers=self._headers(),
                json={"error": error},
                timeout=30,
            ).raise_for_status()
        except Exception:
            LOGGER.error("failed to report failure for task %s", task_id, exc_info=True)

    def _ensure_model(self):
        if self._model is None:
            import torch

            device = resolve_device(self.device_name)
            dtype = dtype_from_name(self.dtype_name) if device.type == "cuda" else torch.float32
            LOGGER.info("loading model %s on %s (%s)...", self.model_id, device, dtype)
            start = time.time()
            self._model, self._processor = load_model(self.model_id, device, dtype)
            self._device, self._dtype = device, dtype
            LOGGER.info("model ready in %.1fs", time.time() - start)
        return self._model, self._processor, self._device, self._dtype

    # ------------------------------------------------------------- pipeline

    def transcribe(self, audio_path: Path, duration_hint: float | None):
        model, processor, device, dtype = self._ensure_model()
        if self.max_new_tokens:
            max_new_tokens = self.max_new_tokens
        else:
            # ~18 tokens per speech second keeps long meetings within budget.
            duration = duration_hint or self._probe_duration(audio_path) or 300.0
            max_new_tokens = max(2048, min(65536, int(duration * 18)))

        messages = build_transcription_messages(str(audio_path), prompt=self.prompt)
        start = time.time()
        result = generate_transcription(
            model,
            processor,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            device=device,
            dtype=dtype,
        )
        segments = [
            {"start": seg.start, "end": seg.end, "speaker": seg.speaker, "text": seg.text}
            for seg in parse_transcript(result["text"])
        ]
        meta = {
            "model": self.model_id,
            "device": str(device),
            "generated_tokens": result["generated_tokens"],
            "elapsed_sec": round(time.time() - start, 2),
            "max_new_tokens": max_new_tokens,
        }
        LOGGER.info(
            "task done: %d segments, %d tokens in %.1fs",
            len(segments), result["generated_tokens"], time.time() - start,
        )
        return result["text"], segments, meta

    @staticmethod
    def _probe_duration(path: Path) -> float | None:
        try:
            import soundfile as sf

            return sf.info(str(path)).duration
        except Exception:
            pass
        try:  # non-PCM containers (m4a/mp3/mov/...) — decode header via PyAV
            import av

            with av.open(str(path)) as container:
                return float(container.duration) / 1e6 if container.duration else None
        except Exception:
            return None

    def process_task(self, task):
        task_id = task["id"]
        audio_path = None
        stop = threading.Event()
        hb = threading.Thread(target=self._heartbeat_loop, args=(task_id, stop), daemon=True)
        hb.start()
        try:
            LOGGER.info("task %d: downloading %s (%d bytes)...", task_id, task["filename"], task["size_bytes"])
            audio_path = self._download(task)
            text, segments, meta = self.transcribe(audio_path, task.get("duration_sec"))
            duration = task.get("duration_sec") or self._probe_duration(audio_path)
            resp = requests.post(
                f"{self.server}/worker/tasks/{task_id}/result",
                headers=self._headers(),
                json={
                    "text": text,
                    "segments": segments,
                    "duration_sec": duration,
                    "worker_meta": meta,
                },
                timeout=60,
            )
            resp.raise_for_status()
            LOGGER.info("task %d: result accepted", task_id)
        except Exception as exc:
            LOGGER.exception("task %d failed", task_id)
            self._fail(task_id, f"{type(exc).__name__}: {exc}")
        finally:
            stop.set()
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)

    def run(self):
        LOGGER.info("worker %s polling %s every %ds", self.worker_id, self.server, self.poll_interval)
        while True:
            try:
                task = self._claim()
                if task:
                    self.process_task(task)
                    if self.once:
                        return
                else:
                    if self.once:
                        LOGGER.info("no pending task (--once)")
                        return
                    time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                LOGGER.info("bye")
                return
            except Exception:
                LOGGER.exception("poll failed; retrying")
                time.sleep(self.poll_interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description="MOSS meeting-service model worker")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="public server base URL")
    parser.add_argument("--worker-key", default="", help="worker API key (or MTD_WORKER_KEY env)")
    parser.add_argument("--worker-id", default=f"worker-{int(time.time()) % 100000}")
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--lease-seconds", type=int, default=3600)
    parser.add_argument("--download-attempts", type=int, default=4, help="retries for audio download over flaky links")
    parser.add_argument("--model", default="OpenMOSS-Team/MOSS-Transcribe-Diarize")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 = auto from audio duration")
    parser.add_argument("--prompt", default="", help="custom transcription prompt (empty = default)")
    parser.add_argument("--work-dir", default="worker_tmp")
    parser.add_argument("--once", action="store_true", help="process a single task then exit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    import os

    args.worker_key = args.worker_key or os.environ.get("MTD_WORKER_KEY", "")
    if not args.worker_key:
        parser.error("worker key required: --worker-key or MTD_WORKER_KEY")

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    Worker(args).run()


if __name__ == "__main__":
    main()
