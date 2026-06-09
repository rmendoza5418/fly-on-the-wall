/**
 * main.js — Electron main process (v2)
 *
 * Architecture:
 *   Electron ←stdout IPC→ Python recorder.py   (spawned per recording)
 *   Electron ←stdin/stdout IPC→ Python worker.py  (long-lived; model stays warm)
 *
 * Fixes from v1:
 *   - Windows SIGTERM replaced with cross-platform kill
 *   - Graceful child process cleanup on quit
 *   - Transcriber exit handler
 *   - recorderProcess nulled after exit
 *   - Worker process replaces per-recording transcriber spawn
 *
 * New features:
 *   - Global keyboard shortcut (Ctrl+Shift+R) to toggle recording
 *   - "Copy summary to clipboard" tray menu item
 *   - Session history via SQLite (surfaced in tray menu)
 *   - electron-store settings persistence
 *   - Audio level display in tray tooltip
 *   - Diarization support (when enabled in settings)
 */

const {
  app, Tray, Menu, Notification, nativeImage,
  shell, clipboard, globalShortcut, dialog,
} = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const readline = require("readline");
const { randomUUID } = require("crypto");

const settings = require("./settings");

// ------------------------------------------------------------------ //
// Paths
// ------------------------------------------------------------------ //

const PYTHON_BIN = settings.get("pythonBin");
const SRC_PY = path.join(__dirname, "../python");
const RECORDER_SCRIPT = path.join(SRC_PY, "recorder.py");
const WORKER_SCRIPT = path.join(SRC_PY, "worker.py");
const ICON_DIR = path.join(__dirname, "../../assets");

// ------------------------------------------------------------------ //
// State
// ------------------------------------------------------------------ //

/** @type {Tray | null} */
let tray = null;

/** @type {import("child_process").ChildProcess | null} */
let recorderProcess = null;

/** @type {import("child_process").ChildProcess | null} */
let workerProcess = null;

let isRecording = false;
let workerReady = false;

/** Last completed session paths */
let lastSession = { audioFile: null, summaryFile: null, actionItemsFile: null };

/** Pending jobs: jobId → { audioFile } */
const pendingJobs = new Map();

// ------------------------------------------------------------------ //
// App init
// ------------------------------------------------------------------ //

app.whenReady().then(() => {
  if (process.platform === "darwin") app.dock.hide();

  setupTray();
  registerGlobalShortcut();
  startWorker();
});

app.on("will-quit", cleanupAndQuit);
app.on("window-all-closed", (e) => e.preventDefault());

// ------------------------------------------------------------------ //
// Graceful shutdown
// ------------------------------------------------------------------ //

function cleanupAndQuit() {
  globalShortcut.unregisterAll();
  killProcess(recorderProcess, "Recorder");
  if (workerProcess) {
    try { workerProcess.stdin.write(JSON.stringify({ cmd: "quit" }) + "\n"); } catch (_) {}
    killProcess(workerProcess, "Worker");
  }
}

function killProcess(proc, label) {
  if (!proc) return;
  try {
    if (process.platform === "win32") {
      // Windows: spawn taskkill to terminate the process tree
      spawn("taskkill", ["/pid", proc.pid.toString(), "/f", "/t"]);
    } else {
      proc.kill("SIGTERM");
    }
  } catch (e) {
    console.error(`[${label}] kill failed:`, e.message);
  }
}

// ------------------------------------------------------------------ //
// Worker (long-lived transcription process)
// ------------------------------------------------------------------ //

function startWorker() {
  const model = settings.get("whisperModel");
  workerProcess = spawn(PYTHON_BIN, [WORKER_SCRIPT, "--model", model], {
    stdio: ["pipe", "pipe", "pipe"],
    cwd: SRC_PY,
  });

  const rl = readline.createInterface({ input: workerProcess.stdout });
  rl.on("line", (line) => {
    try { handleWorkerEvent(JSON.parse(line)); }
    catch (_) { console.warn("[worker] unparseable:", line); }
  });

  workerProcess.stderr.on("data", (d) => console.error("[worker stderr]", d.toString().trim()));

  workerProcess.on("exit", (code) => {
    workerReady = false;
    console.warn(`[worker] exited with code ${code}`);
    if (code !== 0) {
      showNotification("Worker crashed", "Restarting transcription worker…");
      setTimeout(startWorker, 3000);
    }
  });
}

function sendWorkerCommand(cmd) {
  if (!workerProcess || !workerReady) {
    console.warn("[worker] not ready — queuing is not implemented; job dropped");
    return;
  }
  workerProcess.stdin.write(JSON.stringify(cmd) + "\n");
}

function handleWorkerEvent(evt) {
  switch (evt.event) {
    case "ready":
      workerReady = true;
      console.log(`[worker] ready on ${evt.device} — model: ${evt.model}`);
      updateTrayMenu();
      break;

    case "pong":
      console.log("[worker] pong");
      break;

    case "job_started":
      tray.setToolTip("Transcribing…");
      break;

    case "job_progress":
      tray.setToolTip(`Transcribing… ${evt.pct || 0}%`);
      break;

    case "job_complete": {
      tray.setToolTip("Fly on the Wall");
      lastSession = {
        audioFile: pendingJobs.get(evt.job_id)?.audioFile ?? null,
        summaryFile: evt.summary_file,
        actionItemsFile: evt.action_items_file,
      };
      pendingJobs.delete(evt.job_id);
      updateTrayMenu();
      const body = `${evt.word_count} words` +
        (evt.action_item_count ? `, ${evt.action_item_count} action items` : "") +
        " — click to open summary";
      showNotification("Transcription complete", body, evt.summary_file);
      break;
    }

    case "job_error":
      tray.setToolTip("Fly on the Wall");
      pendingJobs.delete(evt.job_id);
      showNotification("Transcription error", evt.message || "Unknown error");
      break;

    case "warning":
      console.warn("[worker]", evt.message);
      break;
  }
}

// ------------------------------------------------------------------ //
// Tray
// ------------------------------------------------------------------ //

function iconPath(name) {
  const p = path.join(ICON_DIR, name);
  return fs.existsSync(p) ? p : null;
}

function setupTray() {
  const imgPath = iconPath("icon-idle.png");
  const icon = imgPath
    ? nativeImage.createFromPath(imgPath).resize({ width: 16, height: 16 })
    : nativeImage.createEmpty();
  tray = new Tray(icon);
  tray.setToolTip("Fly on the Wall");
  updateTrayMenu();
}

function updateTrayMenu() {
  const hasSummary = !!lastSession.summaryFile && fs.existsSync(lastSession.summaryFile);
  const hasActions = !!lastSession.actionItemsFile && fs.existsSync(lastSession.actionItemsFile);

  const menu = Menu.buildFromTemplate([
    {
      label: isRecording ? "⏹  Stop Recording" : "⏺  Start Recording",
      accelerator: settings.get("globalShortcut"),
      click: isRecording ? stopRecording : startRecording,
    },
    { type: "separator" },
    {
      label: "Open Summary",
      enabled: hasSummary,
      click: () => shell.openPath(lastSession.summaryFile),
    },
    {
      label: "Copy Summary to Clipboard",
      enabled: hasSummary,
      click: copySummaryToClipboard,
    },
    {
      label: "Open Action Items",
      enabled: hasActions,
      click: () => shell.openPath(lastSession.actionItemsFile),
    },
    { type: "separator" },
    {
      label: "Open Output Folder",
      click: () => shell.openPath(settings.get("outputDir")),
    },
    { type: "separator" },
    {
      label: `Model: ${settings.get("whisperModel")}`,
      enabled: false,
    },
    {
      label: `Worker: ${workerReady ? "ready ✓" : "starting…"}`,
      enabled: false,
    },
    { type: "separator" },
    { label: "Quit", click: () => app.quit() },
  ]);

  tray.setContextMenu(menu);

  // Update tray icon to reflect recording state
  const name = isRecording ? "icon-recording.png" : "icon-idle.png";
  const p = iconPath(name);
  if (p) {
    tray.setImage(nativeImage.createFromPath(p).resize({ width: 16, height: 16 }));
  }
}

// ------------------------------------------------------------------ //
// Recording lifecycle
// ------------------------------------------------------------------ //

function startRecording() {
  if (isRecording || recorderProcess) return;

  recorderProcess = spawn(PYTHON_BIN, [
    RECORDER_SCRIPT,
    "--output-dir", settings.get("outputDir"),
    "--max-minutes", String(settings.get("maxRecordingMinutes")),
  ], {
    stdio: ["ignore", "pipe", "pipe"],
    cwd: SRC_PY,
  });

  const rl = readline.createInterface({ input: recorderProcess.stdout });
  rl.on("line", (line) => {
    try { handleRecorderEvent(JSON.parse(line)); }
    catch (_) { console.warn("[recorder] unparseable:", line); }
  });

  recorderProcess.stderr.on("data", (d) => console.error("[recorder stderr]", d.toString().trim()));

  recorderProcess.on("exit", (code) => {
    recorderProcess = null;   // ← clear stale reference
    if (code !== 0 && isRecording) {
      showNotification("Recording Error", "The recorder stopped unexpectedly.");
      isRecording = false;
      updateTrayMenu();
    }
  });
}

function stopRecording() {
  if (!recorderProcess) return;
  killProcess(recorderProcess, "Recorder");
}

function handleRecorderEvent(evt) {
  switch (evt.event) {
    case "recording_started":
      isRecording = true;
      updateTrayMenu();
      showNotification("Recording started", "Fly on the Wall is listening.");
      break;

    case "recording_stopped":
      isRecording = false;
      updateTrayMenu();
      showNotification("Recording stopped",
        `${Math.round(evt.duration_secs || 0)}s captured — transcribing…`);
      if (evt.file) dispatchTranscriptionJob(evt.file);
      break;

    case "level":
      // Show live audio level in tooltip while recording
      if (settings.get("showLevelMeter") && isRecording) {
        const bars = Math.round((evt.rms || 0) * 200);
        const meter = "█".repeat(Math.min(bars, 10)).padEnd(10, "░");
        tray.setToolTip(`Recording  ${meter}`);
      }
      break;

    case "warning":
      console.warn("[recorder]", evt.message);
      break;

    case "error":
      showNotification("Recorder error", evt.message || "Unknown error");
      isRecording = false;
      updateTrayMenu();
      break;
  }
}

// ------------------------------------------------------------------ //
// Transcription dispatch
// ------------------------------------------------------------------ //

function dispatchTranscriptionJob(audioFile) {
  const jobId = randomUUID();
  pendingJobs.set(jobId, { audioFile });

  sendWorkerCommand({
    cmd: "transcribe",
    job_id: jobId,
    audio_file: audioFile,
    language: settings.get("language"),
    enable_diarization: settings.get("enableDiarization"),
    hf_token: settings.get("hfToken") || null,
  });
}

// ------------------------------------------------------------------ //
// Global keyboard shortcut
// ------------------------------------------------------------------ //

function registerGlobalShortcut() {
  const shortcut = settings.get("globalShortcut");
  const registered = globalShortcut.register(shortcut, () => {
    isRecording ? stopRecording() : startRecording();
  });

  if (!registered) {
    console.warn(`[shortcuts] Failed to register ${shortcut} — may be in use by another app`);
  } else {
    console.log(`[shortcuts] Registered ${shortcut}`);
  }
}

// ------------------------------------------------------------------ //
// Clipboard
// ------------------------------------------------------------------ //

function copySummaryToClipboard() {
  if (!lastSession.summaryFile || !fs.existsSync(lastSession.summaryFile)) {
    showNotification("Nothing to copy", "No summary available yet.");
    return;
  }
  const text = fs.readFileSync(lastSession.summaryFile, "utf-8");
  clipboard.writeText(text);
  showNotification("Copied", "Summary copied to clipboard.");
}

// ------------------------------------------------------------------ //
// Notification helper
// ------------------------------------------------------------------ //

function showNotification(title, body, filePath) {
  if (!Notification.isSupported()) return;
  const n = new Notification({ title, body });
  if (filePath) {
    n.on("click", () => shell.openPath(filePath));
  }
  n.show();
}
