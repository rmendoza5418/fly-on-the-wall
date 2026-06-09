/**
 * main.js — Electron main process
 *
 * Manages the application lifecycle, system tray icon, and child process
 * communication with the Python recorder and transcriber.
 *
 * Architecture:
 *   Electron (this file) ←stdout IPC→ Python recorder.py
 *   Electron (this file) ←stdout IPC→ Python transcriber.py
 *
 * The app runs tray-only (no dock icon on macOS, no taskbar on Windows)
 * to stay unobtrusive during meetings.
 */

const { app, Tray, Menu, Notification, nativeImage, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const readline = require("readline");

// ------------------------------------------------------------------ //
// Config
// ------------------------------------------------------------------ //

const PYTHON_BIN = process.env.FOTW_PYTHON || "python3";
const RECORDER_SCRIPT = path.join(__dirname, "../python/recorder.py");
const TRANSCRIBER_SCRIPT = path.join(__dirname, "../python/transcriber.py");
const WHISPER_MODEL = process.env.FOTW_MODEL || "small";
const ICON_DIR = path.join(__dirname, "../../assets");

// ------------------------------------------------------------------ //
// State
// ------------------------------------------------------------------ //

/** @type {Tray | null} */
let tray = null;

/** @type {import("child_process").ChildProcess | null} */
let recorderProcess = null;

/** @type {string | null} */
let currentAudioFile = null;

let isRecording = false;

// ------------------------------------------------------------------ //
// App setup — tray-only, no dock/taskbar
// ------------------------------------------------------------------ //

app.whenReady().then(() => {
  // macOS: hide from dock
  if (process.platform === "darwin") {
    app.dock.hide();
  }

  setupTray();
});

app.on("window-all-closed", (e) => {
  // Prevent default quit when all windows close — we're a tray app
  e.preventDefault();
});

// ------------------------------------------------------------------ //
// Tray
// ------------------------------------------------------------------ //

function setupTray() {
  const iconPath = path.join(
    ICON_DIR,
    isRecording ? "icon-recording.png" : "icon-idle.png"
  );
  const icon = nativeImage.createFromPath(iconPath);
  tray = new Tray(icon.resize({ width: 16, height: 16 }));
  tray.setToolTip("Fly on the Wall");
  updateTrayMenu();
}

function updateTrayMenu() {
  const menu = Menu.buildFromTemplate([
    {
      label: isRecording ? "⏹  Stop Recording" : "⏺  Start Recording",
      click: isRecording ? stopRecording : startRecording,
    },
    { type: "separator" },
    {
      label: "Open Last Summary...",
      enabled: !!currentAudioFile,
      click: openLastSummary,
    },
    { type: "separator" },
    { label: "Quit", click: () => app.exit(0) },
  ]);

  tray.setContextMenu(menu);

  const iconName = isRecording ? "icon-recording.png" : "icon-idle.png";
  const iconPath = path.join(ICON_DIR, iconName);
  if (fs.existsSync(iconPath)) {
    tray.setImage(nativeImage.createFromPath(iconPath).resize({ width: 16, height: 16 }));
  }
}

// ------------------------------------------------------------------ //
// Recording lifecycle
// ------------------------------------------------------------------ //

function startRecording() {
  if (isRecording) return;

  recorderProcess = spawn(PYTHON_BIN, [RECORDER_SCRIPT], {
    stdio: ["ignore", "pipe", "pipe"],
  });

  // Parse newline-delimited JSON from Python stdout
  const rl = readline.createInterface({ input: recorderProcess.stdout });
  rl.on("line", (line) => {
    try {
      handleRecorderEvent(JSON.parse(line));
    } catch (_) {
      console.warn("Unparseable recorder output:", line);
    }
  });

  recorderProcess.stderr.on("data", (data) => {
    console.error("[recorder stderr]", data.toString());
  });

  recorderProcess.on("exit", (code) => {
    if (code !== 0 && isRecording) {
      showNotification("Recording Error", "The recorder stopped unexpectedly.");
      isRecording = false;
      updateTrayMenu();
    }
  });
}

function stopRecording() {
  if (!recorderProcess || !isRecording) return;
  recorderProcess.kill("SIGTERM");
}

/** @param {{ event: string, file?: string, duration_secs?: number, message?: string }} evt */
function handleRecorderEvent(evt) {
  switch (evt.event) {
    case "recording_started":
      isRecording = true;
      currentAudioFile = evt.file;
      updateTrayMenu();
      showNotification("Recording started", "Fly on the Wall is listening.");
      break;

    case "recording_stopped":
      isRecording = false;
      updateTrayMenu();
      showNotification(
        "Recording stopped",
        `${Math.round(evt.duration_secs || 0)}s captured. Transcribing...`
      );
      if (evt.file) {
        runTranscriber(evt.file);
      }
      break;

    case "error":
      showNotification("Recorder error", evt.message || "Unknown error");
      isRecording = false;
      updateTrayMenu();
      break;
  }
}

// ------------------------------------------------------------------ //
// Transcription
// ------------------------------------------------------------------ //

function runTranscriber(audioFile) {
  const proc = spawn(
    PYTHON_BIN,
    [TRANSCRIBER_SCRIPT, "--audio", audioFile, "--model", WHISPER_MODEL],
    { stdio: ["ignore", "pipe", "pipe"] }
  );

  const rl = readline.createInterface({ input: proc.stdout });
  rl.on("line", (line) => {
    try {
      handleTranscriberEvent(JSON.parse(line));
    } catch (_) {
      console.warn("Unparseable transcriber output:", line);
    }
  });

  proc.stderr.on("data", (data) => {
    console.error("[transcriber stderr]", data.toString());
  });
}

/** @param {{ event: string, pct?: number, summary_file?: string, word_count?: number, message?: string }} evt */
function handleTranscriberEvent(evt) {
  switch (evt.event) {
    case "transcription_progress":
      tray.setToolTip(`Transcribing… ${evt.pct || 0}%`);
      break;

    case "transcription_complete":
      tray.setToolTip("Fly on the Wall");
      showNotification(
        "Transcription complete",
        `${evt.word_count || 0} words — click to open summary`,
        evt.summary_file
      );
      break;

    case "error":
      showNotification("Transcription error", evt.message || "Unknown error");
      break;
  }
}

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function showNotification(title, body, filePath) {
  const n = new Notification({ title, body });
  if (filePath) {
    n.on("click", () => shell.openPath(filePath));
  }
  n.show();
}

function openLastSummary() {
  if (!currentAudioFile) return;
  const stem = path.basename(currentAudioFile, ".wav");
  const summaryPath = path.join(path.dirname(currentAudioFile), `${stem}_summary.txt`);
  if (fs.existsSync(summaryPath)) {
    shell.openPath(summaryPath);
  } else {
    showNotification("Not ready", "Summary is still being generated.");
  }
}
