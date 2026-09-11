"""Unit tests for the long-audio chunking helpers (no GPU/model needed).

Run from the repo root:

    python meeting_service/test_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from worker.audio_split import find_split_points, split_wave
from worker.speaker_align import SpeakerAligner

PASSED = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASSED
    if not cond:
        print(f"FAIL: {name} {detail}")
        sys.exit(1)
    PASSED += 1
    print(f"  ok: {name}")


def make_wave(sr=16000, total_sec=90.0, speech_sec=10.0, silence_sec=10.0):
    """10s tone / 10s near-silence alternating pattern (cuts at 20s multiples
    land inside silence given a small search window)."""
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, speech_sec, int(sr * speech_sec)))
    silence = (0.001 * np.random.randn(int(sr * silence_sec))).astype(np.float32)
    piece = np.concatenate([tone.astype(np.float32), silence])
    reps = int(np.ceil(total_sec / (speech_sec + silence_sec)))
    return np.tile(piece, reps)[: int(sr * total_sec)]


def test_split_points():
    print("[find_split_points]")
    sr = 16000
    wave = make_wave(sr=sr)
    pts = find_split_points(wave, sr, chunk_seconds=20.0, search_window=4.0)
    check("four cuts for 90s/20s", len(pts) == 4, str(len(pts)))  # at 20/40/60/80s
    # cuts must land in silence: RMS of the 1s around each cut should be tiny
    for p in pts:
        around = wave[p - sr // 2: p + sr // 2]
        rms = float(np.sqrt(np.mean(around ** 2)))
        check(f"cut at {p / sr:.1f}s in silence (rms={rms:.4f})", rms < 0.02)


def test_split_wave():
    print("[split_wave]")
    sr = 16000
    wave = make_wave(sr=sr)
    parts = split_wave(wave, sr, chunk_seconds=20.0, search_window=4.0)
    check("piece count = cuts+1", len(parts) == 5)
    total = sum(len(p) for p, _ in parts)
    check("no samples lost", total == len(wave))
    offsets = [off for _, off in parts]
    check("offsets increasing from zero", offsets == sorted(offsets) and offsets[0] == 0.0)


def test_short_audio_not_split():
    wave = make_wave(total_sec=15.0)
    check("short audio no cuts", find_split_points(wave, 16000, chunk_seconds=20.0) == [])


def test_aligner():
    print("[SpeakerAligner]")
    rng = np.random.default_rng(7)
    emb_a1 = rng.normal(size=256)
    emb_a2 = emb_a1 + rng.normal(scale=0.02, size=256)   # same voice, chunk 2
    emb_b = rng.normal(size=256)                          # different voice
    aligner = SpeakerAligner(threshold=0.72)

    m1 = aligner.add_chunk({"S01": emb_a1, "S02": emb_b})
    check("chunk1 keeps order", m1 == {"S01": "S01", "S02": "S02"}, str(m1))

    m2 = aligner.add_chunk({"S02": emb_a2})  # chunk2 labeled the A-voice as S02
    check("cross-chunk voice matched", m2 == {"S02": "S01"}, str(m2))

    m3 = aligner.add_chunk({"S01": rng.normal(size=256)})  # unknown voice
    check("unknown voice gets fresh label", m3 == {"S01": "S03"}, str(m3))


if __name__ == "__main__":
    test_split_points()
    test_split_wave()
    test_short_audio_not_split()
    test_aligner()
    print(f"\nALL {PASSED} CHECKS PASSED")
