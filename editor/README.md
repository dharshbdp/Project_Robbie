# Robbie Studio — Intelligent Desktop Companion & Code Editor

**Robbie Studio** is the official desktop environment and code editor for **Project Robbie**, built with **Tauri v2** and a high-performance **Obsidian Glass Vanilla CSS/JS** frontend.

It connects directly to Robbie's trained **25M Conversational Transformer**, **Computer Awareness (Hardware Telemetry)**, **Persistent Memory Storage**, and **Autonomous Web Research Engine**.

---

## Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   Robbie Studio Frontend (HTML / CSS / JS)              │
│   • Interactive Code Scratchpad (Python Execution & Ask Robbie)        │
│   • Multi-Turn Dialogue Companion (Robbie-25M Stage 2 Model)           │
│   • Live System Telemetry Dashboard (CPU, RAM, GPU, Disks, Power)      │
│   • Memory Inspector (Persistent Profile & Session Working Memory)     │
│   • Autonomous Multi-Source Web Research Engine                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ window.__TAURI__.core.invoke()
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Tauri v2 Rust Backend (src-tauri)                   │
│   • Native Window Management (Glassmorphism & Desktop OS Shell)        │
│   • IPC Dispatcher (`ask_robbie`, `get_system_telemetry`, etc.)       │
│   • Subprocess Bridge to Project Robbie Virtual Environment            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ JSON IPC (editor/bridge.py)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Robbie Core Python Engine                         │
│   • ChatEngine (Robbie-25M Causal Transformer, BPE Tokenizer)          │
│   • ComputerAwareness (HardwareMonitor, ProcessMonitor, Workspace)     │
│   • PersistentMemory (Disk-backed JSON storage at robbie_memory.json)  │
│   • ResearchAssistant (UnifiedSearchEngine, Scraper, Synthesizer)      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```text
editor/
├── bridge.py                # Python JSON-IPC bridge between Tauri & Robbie Core
├── package.json             # NPM package manifest with @tauri-apps/cli
├── README.md                # Desktop Editor documentation
├── src/                     # Modern Vanilla Frontend
│   ├── index.html           # Main Studio UI layout (Editor, Chat, Split, System)
│   ├── main.js              # Application logic & Tauri invoke handlers
│   └── styles.css           # Obsidian Glass design system (pure Vanilla CSS)
└── src-tauri/               # Tauri v2 Rust Backend
    ├── Cargo.toml           # Rust manifest & dependencies
    ├── tauri.conf.json      # Window geometry, capabilities & application metadata
    └── src/
        ├── lib.rs           # Tauri IPC commands & Python bridge spawner
        └── main.rs          # Desktop executable entrypoint
```

---

## Getting Started

### 1. Prerequisites
- **Node.js**: v20+ (`node -v`, `npm.cmd -v`)
- **Rust toolchain**: 1.78+ (`cargo --version`, `rustc --version`)
- **Python**: Project Robbie virtual environment (`.venv`) with dependencies installed.

### 2. Launch Development Mode

#### Option A: Quick Double-Click / CMD
Double-click [`launch_editor.bat`](file:///z:/Project_Robbie/Project_Robbie/launch_editor.bat) in the project root.

#### Option B: Windows PowerShell
If Windows blocks script execution (`npm.ps1 cannot be loaded`), either:
1. Run with `.cmd`:
```powershell
cd editor
npm.cmd run tauri dev
```
2. Or enable script execution once in PowerShell (Administrator):
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
npm run tauri dev
```

### 3. Build Production Executable
```bash
npm run tauri build
```
The compiled `.exe` installer will be generated under `editor/src-tauri/target/release/`.

---

## Key Features

1. **Split-Screen Pair Programming**: Code on the left in the interactive scratchpad, chat with Robbie-25M on the right.
2. **In-Editor Code Execution**: Click **"Run Code"** to execute Python scripts in real time and view output in the built-in terminal drawer.
3. **One-Click Code Explanation**: Click **"Ask Robbie"** on any code snippet to have Robbie analyze the logic.
4. **Live System Telemetry**: Monitor hardware stats (CPU cores, RAM usage, Quadro GPU VRAM, Battery, Disks, and Top Processes).
5. **Grounded Web Research**: Run autonomous queries across Wikipedia and DuckDuckGo with verified citations and multi-source rankings.
6. **Dual Mode Support**: Works inside Tauri desktop windows or in standard browsers with automatic mock fallbacks during development.
