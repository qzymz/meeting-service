"""Silence-aware splitting of long audio for chunked transcription.

The model handles long audio in one pass, but context and VRAM bound the
practical single-pass duration. For longer recordings we cut at the
quietest spots near each chunk boundary (so words are not sliced) and
transcribe chunk by chunk, offsetting timestamps afterwards.
"""

from __future__ import annotations

import numpy as np


def find_split_points(
    wave: np.ndarray,
    sr: int,
    chunk_seconds: float,
    search_window: float = 90.0,
    unit_seconds: float = 1.0,
) -> list[int]:
    """Return sample indices to cut at, one per chunk_seconds of audio.

    Each cut is snapped to the quietest ``unit_seconds`` window within
    +/- ``search_window`` of the nominal boundary, so splits land in
    pauses instead of mid-word.
    """
    total = len(wave)
    if total <= chunk_seconds * sr:
        return []
    win = max(1, int(sr * unit_seconds))
    points: list[int] = []
    target = chunk_seconds
    while target * sr < total:
        center = int(target * sr)
        lo = max(0, center - int(search_window * sr))
        hi = min(total - win, center + int(search_window * sr))
        if hi <= lo:
            points.append(center)
        else:
            seg = wave[lo:hi]
            n_win = len(seg) // win
            energy = np.sqrt(np.mean(seg[: n_win * win].reshape(n_win, win) ** 2, axis=1))
            quiet = lo + int(np.argmin(energy)) * win + win // 2
            points.append(quiet)
        target += chunk_seconds
    return points


def split_wave(
    wave: np.ndarray,
    sr: int,
    chunk_seconds: float,
    search_window: float = 90.0,
) -> list[tuple[np.ndarray, float]]:
    """Split into (chunk_wave, offset_seconds) pieces at quiet boundaries."""
    cuts = [0] + find_split_points(wave, sr, chunk_seconds, search_window) + [len(wave)]
    return [(wave[a:b], a / sr) for a, b in zip(cuts, cuts[1:]) if b > a]
