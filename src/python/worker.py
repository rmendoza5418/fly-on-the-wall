"""
worker.py — Long-lived transcription worker process

Spawned ONCE by the Electron main process at app startup. Keeps the Whisper
model warm in memory so transcription starts immediately after each recording
rather than waiting 3–5 seconds for model load.

Accepts jobs via stdin (newline-delimited JSON commands).
Emits events via stdout (newline-delimited JSON events).

--- STDIN COMMANDS ---

Start a transcription job:
    {"cmd": "transcribe", "job_id": "abc", "audio_file": "/path/to.wav",
     "language": "en", "enable_diarization": false, "hf_token": null}

Ping / health check:
    {"cmd": "ping"}

Graceful shutdown:
    {"cmd": "quit"}

--- STDOUT EVENTS ---

    {"event": "ready", "device": "cuda|cpu", "model": "small"}
    {"event": "pong"}
    {"event": "job_started",   "job_id": "abc"}
    {"event": "job_progress",  "job_id": "abc", "pct": 45}
    {"event": "job_complete",  "job_id": "abc", "transcript_file": "...",
                                "summary_file": "...", "action_items_file": "...",
                                "word_count": 312, "action_item_count": 4,
                                "duration_secs": 3.2, "segments_file": "..."}
    {"event": "job_error",     "job_id": "abc", "message": "..."}
    {"event": "warning",       "message": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from actions import extract_action_items, write_action_items
from history import log_session

SUMMARY_SENTENCES = 8
DEFAULT_MODEL = "small"
MODELS_DIR = Path.home() / ".fotw" / "models"


# ------------------------------------------------------------------ #
# IPC helpers
# ------------------------------------------------------------------ #

def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


def read_command() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        emit({"event": "warning", "message": f"Unparseable command: {line.strip()}"})
        return {}


# ------------------------------------------------------------------ #
# Model loading
# ------------------------------------------------------------------ #

def load_model(model_size: str) -> tuple[WhisperModel, str]:
    """Load Whisper model; try CUDA first, fall back to CPU with int8."""
    try:
        import torch
        if torch.cuda.is_available():
            model = WhisperModel(model_size, device="cuda", compute_type="float16",
                                 download_root=str(MODELS_DIR))
            return model, "cuda"
    except Exception:
        pass

    model = WhisperModel(model_size, device="cpu", compute_type="int8",
                         download_root=str(MODELS_DIR))
    return model, "cpu"


# ------------------------------------------------------------------ #
# Transcription
# ------------------------------------------------------------------ #

def transcribe(
    model: WhisperModel,
    audio_path: Path,
    language: str,
    job_id: str,
) -> tuple[str, list[dict]]:
    """
    Transcribe audio file. Returns (full_text, segments).
    Segments include timestamps for navigation and diarization.
    """
    segments_gen, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=False,
    )

    total = info.duration
    parts = []
    raw_segments = []

    for seg in segments_gen:
        parts.append(seg.text.strip())
        raw_segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        pct = int((seg.end / total) * 100) if total > 0 else 0
        emit({"event": "job_progress", "job_id": job_id, "pct": min(pct, 99)})

    return " ".join(parts), raw_segments


def build_timestamped_transcript(segments: list[dict]) -> str:
    """Build a transcript string with [MM:SS] markers per segment."""
    lines = []
    for seg in segments:
        m, s = divmod(int(seg["start"]), 60)
        ts = f"[{m:02d}:{s:02d}]"
        lines.append(f"{ts} {seg['text']}")
    return "\n".join(lines)


def summarize(text: str) -> str:
    if not text.strip():
        return "No speech detected."
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    sentences = summarizer(parser.document, SUMMARY_SENTENCES)
    if not sentences:
        return text[:500]
    return "\n• " + "\n• ".join(str(s) for s in sentences)


# ------------------------------------------------------------------ #
# File output
# ------------------------------------------------------------------ #

def write_outputs(
    audio_path: Path,
    full_text: str,
    segments: list[dict],
    summary: str,
    action_items: list[str],
) -> dict:
    stem = audio_path.stem
    out_dir = audio_path.parent
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Timestamped transcript
    transcript_text = build_timestamped_transcript(segments)
    transcript_path = out_dir / f"{stem}_transcript.txt"
    transcript_path.write_text(
        f"Fly on the Wall — Transcript\nRecorded: {ts}\n{'='*50}\n\n{transcript_text}\n",
        encoding="utf-8",
    )

    # Summary
    summary_path = out_dir / f"{stem}_summary.txt"
    summary_path.write_text(
        f"Fly on the Wall — Summary\nRecorded: {ts}\n{'='*50}\n\nKey Points:\n{summary}\n",
        encoding="utf-8",
    )

    # Raw segments JSON (for diarization or later re-processing)
    segments_path = out_dir / f"{stem}_segments.json"
    segments_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")

    # Action items
    action_path = write_action_items(audio_path, action_items)

    return {
        "transcript_file": str(transcript_path),
        "summary_file": str(summary_path),
        "segments_file": str(segments_path),
        "action_items_file": str(action_path) if action_path else None,
    }


# ------------------------------------------------------------------ #
# Job handler
# ------------------------------------------------------------------ #

def handle_transcribe(model: WhisperModel, cmd: dict) -> None:
    job_id = cmd.get("job_id", "unknown")
    audio_file = cmd.get("audio_file")
    language = cmd.get("language", "en")
    enable_diarization = cmd.get("enable_diarization", False)
    hf_token = cmd.get("hf_token")

    if not audio_file:
        emit({"event": "job_error", "job_id": job_id, "message": "No audio_file provided"})
        return

    audio_path = Path(audio_file)
    if not audio_path.exists():
        emit({"event": "job_error", "job_id": job_id, "message": f"File not found: {audio_file}"})
        return

    emit({"event": "job_started", "job_id": job_id})
    t0 = time.monotonic()

    try:
        full_text, segments = transcribe(model, audio_path, language, job_id)

        # Optional speaker diarization
        if enable_diarization:
            try:
                from diarize import diarize, format_diarized_transcript
                annotated = diarize(audio_path, segments, hf_token=hf_token)
                diarized_text = format_diarized_transcript(annotated)
                # Replace segments with diarized version for output
                transcript_text = diarized_text
                # Keep raw text for action item extraction
            except ImportError:
                emit({"event": "warning", "message": "pyannote.audio not installed — skipping diarization"})
            except Exception as e:
                emit({"event": "warning", "message": f"Diarization failed: {e} — continuing without"})

        action_items = extract_action_items(full_text)
        summary = summarize(full_text)
        paths = write_outputs(audio_path, full_text, segments, summary, action_items)

        elapsed = time.monotonic() - t0
        word_count = len(full_text.split())

        # Log to history DB
        try:
            log_session(
                recorded_at=datetime.now(),
                duration_secs=0,  # recorder emits this separately
                audio_path=str(audio_path),
                transcript_path=paths["transcript_file"],
                summary_path=paths["summary_file"],
                action_items_path=paths.get("action_items_file"),
                word_count=word_count,
                action_item_count=len(action_items),
                device_label=None,
                model_used=model.model_size_or_path if hasattr(model, "model_size_or_path") else "unknown",
            )
        except Exception as e:
            emit({"event": "warning", "message": f"History log failed: {e}"})

        emit({
            "event": "job_complete",
            "job_id": job_id,
            **paths,
            "word_count": word_count,
            "action_item_count": len(action_items),
            "processing_secs": round(elapsed, 2),
        })

    except Exception as exc:
        emit({"event": "job_error", "job_id": job_id, "message": str(exc)})


# ------------------------------------------------------------------ #
# Main loop
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    model, device = load_model(args.model)
    emit({"event": "ready", "device": device, "model": args.model})

    while True:
        cmd = read_command()
        if cmd is None:
            # stdin closed — parent process exited
            break

        action = cmd.get("cmd")

        if action == "transcribe":
            handle_transcribe(model, cmd)
        elif action == "ping":
            emit({"event": "pong"})
        elif action == "quit":
            break
        elif action:
            emit({"event": "warning", "message": f"Unknown command: {action}"})


if __name__ == "__main__":
    main()
