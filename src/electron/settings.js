/**
 * settings.js — Persistent user preferences via electron-store.
 *
 * Wraps electron-store with typed defaults so the rest of the app
 * can call settings.get("outputDir") without worrying about first-run state.
 */

const Store = require("electron-store");
const os = require("os");
const path = require("path");

const DEFAULT_OUTPUT_DIR = path.join(os.homedir(), "Documents", "FlyOnTheWall");

const schema = {
  outputDir: {
    type: "string",
    default: DEFAULT_OUTPUT_DIR,
  },
  whisperModel: {
    type: "string",
    enum: ["tiny", "base", "small", "medium", "large-v3"],
    default: "small",
  },
  language: {
    type: "string",
    default: "en",
  },
  maxRecordingMinutes: {
    type: "number",
    minimum: 1,
    maximum: 480,
    default: 180,
  },
  enableDiarization: {
    type: "boolean",
    default: false,
  },
  hfToken: {
    type: "string",
    default: "",
  },
  globalShortcut: {
    type: "string",
    default: "CommandOrControl+Shift+R",
  },
  pythonBin: {
    type: "string",
    default: "python3",
  },
  showLevelMeter: {
    type: "boolean",
    default: true,
  },
  ollamaEnabled: {
    type: "boolean",
    default: false,
  },
  ollamaModel: {
    type: "string",
    default: "llama3.2",
  },
  ollamaUrl: {
    type: "string",
    default: "http://localhost:11434",
  },
};

const store = new Store({ schema, name: "fotw-settings" });

module.exports = store;
