"""
recorder.py — Audio capture process

Spawned as a child process by the Electron main process via IPC.
Captures system/microphone audio to a temp WAV file, then signals
the transcriber when recording stops.

IPC protocol (stdout, newline-delimited JSON):
  {"event": "recording_started", "file": "/tmp/fotw_<ts>.wav"}
  {"event": "recording_stopped", "file": "/tmp/fotw_<ts>.wav", "duration_secs": 42.1}
  {"event": "error", "message": "..."}
"""

from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16_000        # Whisper works best at 16kHz
CHANNELS = 1                # Mono sufficient for speech
DTYPE = "float32"
CHUNK_SECS = 0.5            # Buffer flush interval
TEMP_DIR = Path(tempfile.gettempdir())


def emit(event: dict) -> None:
    """Write a JSON event to stdout for the Electron main process to consume."""
    print(json.dumps(event), flush=True)


class AudioRecorder:
    def __init__(self):
        self.output_path: Path | None = None
        self.writer: sf.SoundFile | None = None
        self.start_time: float | None = None
        self._running = False

    def start(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = TEMP_DIR / f"fotw_{ts}.wav"
        self.writer = sf.SoundFile(
            self.output_path,
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            format="WAV",
            subtype="PCM_16",
        )
        self.start_time = time.time()
        self._running = True
        return self.output_path

    def stop(self) -> float:
        self._running = False
        duration = time.time() - (self.start_time or time.time())
        if self.writer:
            self.writer.close()
            self.writer = None
        return duration

    def callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            emit({"event": "warning", "message": str(status)})
        if self._running and self.writer:
            self.writer.write(indata.copy())


def main() -> None:
    recorder = AudioRecorder()

    def handle_stop(signum, frame):
        if recorder._running:
            duration = recorder.stop()
            emit({
                "event": "recording_stopped",
                "file": str(recorder.output_path),
                "duration_secs": round(duration, 2),
            })
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    try:
        output_path = recorder.start()
        emit({"event": "recording_started", "file": str(output_path)})

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=int(SAMPLE_RATE * CHUNK_SECS),
            callback=recorder.callback,
        ):
            # Block until signal received
            while recorder._running:
                time.sleep(0.1)

    except Exception as exc:
        emit({"event": "error", "message": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
