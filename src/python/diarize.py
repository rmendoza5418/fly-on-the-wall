"""
diarize.py — Optional speaker diarization using pyannote.audio.

Assigns speaker labels to Whisper transcript segments so the output
reads as "Speaker 1: ..." rather than an unlabeled wall of text.

Requires:
    pip install pyannote.audio
    A free Hugging Face token with access to:
      - pyannote/speaker-diarization-3.1
      - pyannote/segmentation-3.0

Set the token via:
    export HF_TOKEN=hf_your_token_here
    or pass --hf-token to the CLI

Usage (standalone):
    python diarize.py --audio /path/to/recording.wav [--hf-token hf_...]

Returns annotated transcript as JSON to stdout:
    [{"speaker": "SPEAKER_00", "start": 0.0, "end": 4.2, "text": "..."}, ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def is_available() -> bool:
    """Check whether pyannote.audio is installed."""
    try:
        import pyannote.audio  # noqa: F401
        return True
    except ImportError:
        return False


def diarize(
    audio_path: Path,
    segments: list[dict],  # [{"start": float, "end": float, "text": str}]
    hf_token: str | None = None,
    num_speakers: int | None = None,
) -> list[dict]:
    """
    Run speaker diarization on the audio file and merge results with
    Whisper transcript segments.

    Args:
        audio_path: Path to WAV file
        segments: Whisper transcript segments (start/end/text)
        hf_token: Hugging Face token for pyannote model access
        num_speakers: Optional hint for number of speakers (improves accuracy)

    Returns:
        List of dicts with speaker label added: {"speaker": ..., "start": ..., "end": ..., "text": ...}
    """
    if not is_available():
        raise ImportError(
            "pyannote.audio is not installed. Run: pip install pyannote.audio\n"
            "Then accept the model license at: https://hf.co/pyannote/speaker-diarization-3.1"
        )

    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "Hugging Face token required for speaker diarization. "
            "Set HF_TOKEN environment variable or pass --hf-token."
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    # Run diarization
    diarization_kwargs = {}
    if num_speakers:
        diarization_kwargs["num_speakers"] = num_speakers

    diarization = pipeline(str(audio_path), **diarization_kwargs)

    # Build a speaker lookup: for each time point, which speaker is active?
    def speaker_at(t: float) -> str:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= t <= turn.end:
                return speaker
        return "UNKNOWN"

    # Assign speaker to each Whisper segment using the segment midpoint
    annotated = []
    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2
        annotated.append({
            "speaker": speaker_at(mid),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        })

    return annotated


def format_diarized_transcript(annotated_segments: list[dict]) -> str:
    """
    Format annotated segments into a readable transcript with speaker labels.

    Groups consecutive segments from the same speaker to avoid excessive
    label repetition.

    Output:
        SPEAKER_00 [00:00 – 00:14]
        Hello everyone, welcome to the meeting.

        SPEAKER_01 [00:15 – 00:32]
        Thanks for having us. Let's get started.
    """
    if not annotated_segments:
        return ""

    lines = []
    current_speaker = None
    current_start = None
    current_texts = []
    current_end = None

    def flush():
        if current_speaker and current_texts:
            start_fmt = _fmt_time(current_start)
            end_fmt = _fmt_time(current_end)
            lines.append(f"\n{current_speaker} [{start_fmt} – {end_fmt}]")
            lines.append(" ".join(current_texts))

    for seg in annotated_segments:
        if seg["speaker"] != current_speaker:
            flush()
            current_speaker = seg["speaker"]
            current_start = seg["start"]
            current_texts = [seg["text"].strip()]
        else:
            current_texts.append(seg["text"].strip())
        current_end = seg["end"]

    flush()
    return "\n".join(lines).strip()


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "00:00"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Speaker diarization for Fly on the Wall")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--segments", required=True, help="JSON file with Whisper segments")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    args = parser.parse_args()

    segments = json.loads(Path(args.segments).read_text())

    try:
        annotated = diarize(
            Path(args.audio),
            segments,
            hf_token=args.hf_token,
            num_speakers=args.num_speakers,
        )
        transcript = format_diarized_transcript(annotated)
        print(json.dumps({"status": "ok", "transcript": transcript, "segments": annotated}))
    except Exception as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
