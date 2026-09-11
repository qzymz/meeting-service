"""Post-transcription refinement: built-in speaker statistics plus an
optional LLM summary through any OpenAI-compatible chat-completions API.
"""

from __future__ import annotations

import os


def speaker_stats(segments: list[dict]) -> dict:
    per_speaker: dict[str, dict] = {}
    total = 0.0
    for seg in segments:
        dur = max(0.0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
        total += dur
        entry = per_speaker.setdefault(
            seg.get("speaker", "?"),
            {"speaker": seg.get("speaker", "?"), "segments": 0, "speech_sec": 0.0},
        )
        entry["segments"] += 1
        entry["speech_sec"] += dur
    speakers = sorted(per_speaker.values(), key=lambda s: -s["speech_sec"])
    for entry in speakers:
        entry["speech_sec"] = round(entry["speech_sec"], 2)
        entry["share"] = round(entry["speech_sec"] / total, 3) if total > 0 else 0.0
    return {
        "speech_sec": round(total, 2),
        "speaker_count": len(speakers),
        "speakers": speakers,
    }


def _llm_summary(text: str, base_url: str, api_key: str, model: str) -> str:
    import requests

    prompt = (
        "以下是一场会议/对话的带说话人标签的转写记录。请用中文输出："
        "1) 会议纪要（要点列表）；2) 关键决定；3) 待办事项（如有）。\n\n" + text
    )
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def refine(transcript_text: str, segments: list[dict]) -> dict:
    summary = {"stats": speaker_stats(segments), "summary_text": None, "llm_model": None}
    base_url = os.environ.get("MTD_LLM_BASE_URL", "").strip()
    api_key = os.environ.get("MTD_LLM_API_KEY", "").strip()
    model = os.environ.get("MTD_LLM_MODEL", "").strip()
    if base_url and api_key and model and transcript_text.strip():
        try:
            summary["summary_text"] = _llm_summary(transcript_text, base_url, api_key, model)
            summary["llm_model"] = model
        except Exception as exc:  # LLM is best-effort; transcript stays available
            summary["llm_error"] = str(exc)
    return summary
