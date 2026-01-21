# Stealth Meeting Assistant (Privacy-Focused)

A local-first, AI-powered desktop utility designed to securely record, transcribe, and summarize meetings without relying on cloud APIs.

## 🎯 Project Overview
This project demonstrates the integration of **Electron** (frontend/GUI) with **Python** (AI/Backend) to create a seamless desktop application. The core philosophy is **Privacy by Design**—all audio processing, transcription, and summarization happens entirely on the user's device.

## ✨ Key Features
- **Local AI Engine**: Utilizes `faster-whisper` for identifying speech and `sumy` for NLP-based summarization.
- **Unobtrusive UI**: Designed as a "tray-only" application using Electron's `Tray` API to minimize screen real estate usage.
- **Cross-Platform**: Compatible with macOS (including Apple Silicon optimization), Windows, and Linux.
- **Auto-Summarization**: Automatically generates bullet-point summaries of meetings immediately after recording stops.

## 🛠 Tech Stack
| Component | Technology | Role |
|-----------|------------|------|
| **Frontend** | Electron, Node.js | System tray management, OS integration, Notifications |
| **AI/Backend** | Python 3.9+ | Audio processing, Transcription pipeline |
| **Speech Model** | Faster-Whisper | Optimized implementation of OpenAI's Whisper model |
| **NLP** | NLTK, Sumy | LSA (Latent Semantic Analysis) for text summarization |
| **Audio** | SoundDevice, SoundFile | High-fidelity audio capture |

## 📐 Architecture
The application follows a **multi-process architecture**:

1.  **Main Process (Node.js)**: Handles the application lifecycle, system tray events, and native OS notifications.
2.  **Child Process (Python)**:
    - **Recorder**: Spawns a dedicated process to capture audio streams directly to disk.
    - **Transcriber**: Triggered post-recording; loads the Whisper model to generate text and summary files.
3.  **IPC**: Communication occurs via standard I/O streams, ensuring robust error handling and process management.

## 🔒 Privacy & Security
- **No Cloud Data**: Zero audio or text data leaves the machine.
- **Offline Capable**: Fully functional without an internet connection (after initial model download).
- **Stealth Mode**: Configured with `LSUIElement` on macOS to run as a background agent without cluttering the Dock.

## 🚀 Future Roadmap
- [ ] Speaker Diarization (identifying distinct speakers)
- [ ] Real-time transcription streaming
- [ ] Searchable history database (SQLite/Vector DB)

---
*Note: This repository contains the portfolio overview. Source code is private.*
