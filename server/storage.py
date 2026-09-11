"""Audio file storage on local disk."""

from __future__ import annotations

from pathlib import Path

ALLOWED_AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac",
    ".webm", ".mp4", ".mkv", ".mov",
}


class Storage:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.audio_dir = self.root / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def audio_path(self, task_id: int, ext: str) -> Path:
        return self.audio_dir / f"{task_id}{ext}"
