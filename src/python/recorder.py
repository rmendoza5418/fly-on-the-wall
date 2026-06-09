"""
recorder.py — Audio capture process

Spawned as a child process by the Electron main process via IPC.
Captures microphone audio to a WAV file in the configured output directory,
then signals the worker when recording stops.

IPC protocol (stdout, newline-delimited JSON):
  {"event": "recording_started", "file": "..."}
  {"event": "recording_stopped", "file": "...", "duration_secs": 42.1}
  {"event": "level", "rms": 0.03}          ← audio level, ~10x/sec
  {"event": "warning", "message": "..."}
  {"event": "error", "message": "..."}

CLI args:
  --output-dir   DIR     Where to write WAV files (default: ~/Documents/FlyOnTheWall)
  --max-minutes  N       Auto-stop after N minutes (default: 180)
  --device       INDEX   sounddevice input device index (default: system default)
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "float32"
CHUNK_SECS = 0.5
LEVEL_INTERVAL = 0.1     # emit audio level event every N seconds
DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "FlyOnTheWall"
DEFAULT_MAX_MINUTES = 180


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


class AudioRecorder:
    def __init__(self, output_dir: Path, max_minutes: int, device: int | None):
        self.output_dir = output_dir
        self.max_seconds = max_minutes * 60
        self.device = device
        self.output_path: Path | None = None
        self._writer: sf.SoundFile | None = None
        self._start_time: float | None = None
        self._running = False
        self._lock = threading.Lock()
        self._last_level_emit = 0.0
        self._rms = 0.0

    def start(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = self.output_dir / f"fotw_{ts}.wav"
        self._writer = sf.SoundFile(
            self.output_path,
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            format="WAV",
            subtype="PCM_16",
        )
        self._start_time = time.monotonic()
        self._running = True
        return self.output_path

    def stop(self) -> float:
        with self._lock:
            self._running = False
            duration = time.monotonic() - (self._start_time or time.monotonic())
            if self._writer:
                self._writer.close()
                self._writer = None
        return duration

    def elapsed(self) -> float:
        return time.monotonic() - (self._start_time or time.monotonic())

    def callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            emit({"event": "warning", "message": str(status)})

        with self._lock:
            if not self._running or self._writer is None:
                return
            self._writer.write(indata.copy())

        # Compute RMS for level metering
        self._rms = float(np.sqrt(np.mean(indata ** 2)))
        now = time.monotonic()
        if now - self._last_level_emit >= LEVEL_INTERVAL:
            self._last_level_emit = now
            emit({"event": "level", "rms": round(self._rms, 4)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-minutes", type=int, default=DEFAULT_MAX_MINUTES)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    recorder = AudioRecorder(
        output_dir=Path(args.output_dir),
        max_minutes=args.max_minutes,
        device=args.device,
    )

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

        kwargs = dict(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=int(SAMPLE_RATE * CHUNK_SECS),
            callback=recorder.callback,
        )
        if recorder.device is not None:
            kwargs["device"] = recorder.device

        with sd.InputStream(**kwargs):
            while recorder._running:
                time.sleep(0.1)
                # Auto-stop guard
                if recorder.elapsed() >= recorder.max_seconds:
                    emit({"event": "warning", "message": f"Max recording duration reached ({args.max_minutes} min). Stopping."})
                    handle_stop(None, None)

    except Exception as exc:
        # Clean up partial file on error
        if recorder.output_path and recorder.output_path.exists():
            try:
                recorder.output_path.unlink()
            except OSError:
                pass
        emit({"event": "error", "message": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
