"""
transcriber.py — Transcription and summarization process

Invoked by the Electron main process after recording stops.
Loads the Whisper model, transcribes the WAV file, generates a summary,
and writes both to disk. All processing is local — no network calls.

Usage:
    python transcriber.py --audio /tmp/fotw_20240115_143022.wav [--model small]

IPC protocol (stdout, newline-delimited JSON):
  {"event": "transcription_started", "audio_file": "..."}
  {"event": "transcription_progress", "pct": 45}
  {"event": "transcription_complete", "transcript_file": "...", "summary_file": "..."}
  {"event": "error", "message": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# faster-whisper is a CTranslate2-optimized Whisper implementation —
# ~4x faster than openai-whisper with identical output quality.
from faster_whisper import WhisperModel
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "en"
SUMMARY_SENTENCES = 8
DEFAULT_MODEL = "small"          # options: tiny, base, small, medium, large-v3
MODELS_DIR = Path.home() / ".fotw" / "models"


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


def load_model(model_size: str) -> WhisperModel:
    """
    Load Whisper model with CPU or GPU compute type.
    Uses int8 quantization on CPU for a ~2x speed boost with minimal quality loss.
    """
    emit({"event": "model_loading", "model": model_size})
    try:
        # Try GPU first (CUDA), fall back to CPU
        model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16",
            download_root=str(MODELS_DIR),
        )
        emit({"event": "model_loaded", "device": "cuda"})
    except Exception:
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root=str(MODELS_DIR),
        )
        emit({"event": "model_loaded", "device": "cpu"})
    return model


def transcribe(model: WhisperModel, audio_path: Path) -> str:
    """
    Transcribe audio file. Returns full transcript text.
    Streams segments so we can emit progress events.
    """
    emit({"event": "transcription_started", "audio_file": str(audio_path)})

    segments, info = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,             # Voice Activity Detection — skips silence
        vad_parameters={"min_silence_duration_ms": 500},
    )

    total_duration = info.duration
    transcript_parts = []

    for segment in segments:
        transcript_parts.append(segment.text.strip())
        pct = int((segment.end / total_duration) * 100) if total_duration > 0 else 0
        emit({"event": "transcription_progress", "pct": min(pct, 99)})

    return " ".join(transcript_parts)


def summarize(text: str, n_sentences: int = SUMMARY_SENTENCES) -> str:
    """
    Generate an extractive summary using LSA (Latent Semantic Analysis).
    Runs entirely offline — no API calls.
    """
    if not text.strip():
        return "No speech detected in recording."

    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    sentences = summarizer(parser.document, n_sentences)
    return "\n• " + "\n• ".join(str(s) for s in sentences)


def write_outputs(audio_path: Path, transcript: str, summary: str) -> tuple[Path, Path]:
    """Write transcript and summary to the same directory as the audio file."""
    stem = audio_path.stem
    out_dir = audio_path.parent

    transcript_path = out_dir / f"{stem}_transcript.txt"
    summary_path = out_dir / f"{stem}_summary.txt"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    transcript_path.write_text(
        f"Fly on the Wall — Transcript\nRecorded: {ts}\n{'='*50}\n\n{transcript}\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        f"Fly on the Wall — Summary\nRecorded: {ts}\n{'='*50}\n\nKey Points:\n{summary}\n",
        encoding="utf-8",
    )

    return transcript_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to WAV file to transcribe")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model size")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        emit({"event": "error", "message": f"Audio file not found: {audio_path}"})
        sys.exit(1)

    try:
        model = load_model(args.model)
        transcript = transcribe(model, audio_path)
        summary = summarize(transcript)
        transcript_path, summary_path = write_outputs(audio_path, transcript, summary)

        emit({
            "event": "transcription_complete",
            "transcript_file": str(transcript_path),
            "summary_file": str(summary_path),
            "word_count": len(transcript.split()),
        })

    except Exception as exc:
        emit({"event": "error", "message": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
