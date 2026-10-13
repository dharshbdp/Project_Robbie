// ============================================================================
// Robbie Studio — V2 Structural Redesign Architecture
// Fully Integrated: Chat-First Core, Dedicated Research, Interactive System,
// 3-Region Developer Workspace, and Segmented Memory Manager.
// Preserves: Robbie-25M Neural Model, Real Telemetry, and Read-Only Safety.
// ============================================================================

// Check for Tauri IPC bridge
const isTauri = typeof window !== 'undefined' && window.__TAURI__?.core?.invoke !== undefined;
const invoke = isTauri ? window.__TAURI__.core.invoke : mockInvoke;

// ============================================================================
// 1. Monochromatic SVG Vector Icon Library (Lucide-Style)
// ============================================================================
const ICONS = {
  browser: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>`,
  folder: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>`,
  taskmgr: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>`,
  terminal: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>`,
  photos: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>`,
  notepad: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>`,
  calc: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><rect x="4" y="2" width="16" height="20" rx="2"/><line x1="8" y1="6" x2="16" y2="6"/><line x1="8" y1="10" x2="8.01" y2="10"/><line x1="12" y1="10" x2="12.01" y2="10"/><line x1="16" y1="10" x2="16.01" y2="10"/><line x1="8" y1="14" x2="8.01" y2="14"/><line x1="12" y1="14" x2="12.01" y2="14"/><line x1="16" y1="14" x2="16.01" y2="14"/><line x1="8" y1="18" x2="8.01" y2="18"/><line x1="12" y1="18" x2="12.01" y2="18"/><line x1="16" y1="18" x2="16.01" y2="18"/></svg>`,
  settings: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`,
  code: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`,
  generic: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>`,
  file: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`,
  copy: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`,
  check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`,
  trash: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>`,
  external: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>`,
};

const DEFAULT_SPEED_DIAL = [
  { id: 'chrome', name: 'Google Chrome', target: 'chrome', sub: 'Browser', icon: 'browser' },
  { id: 'explorer', name: 'File Explorer', target: 'explorer', sub: 'File System', icon: 'folder' },
  { id: 'taskmgr', name: 'Task Manager', target: 'taskmgr', sub: 'Process Monitor', icon: 'taskmgr' },
  { id: 'terminal', name: 'Terminal', target: 'wt', sub: 'PowerShell / WT', icon: 'terminal' },
  { id: 'photos', name: 'Photos', target: 'photos', sub: 'Media Viewer', icon: 'photos' },
  { id: 'notepad', name: 'Notepad', target: 'notepad', sub: 'Text Editor', icon: 'notepad' },
  { id: 'calc', name: 'Calculator', target: 'calc', sub: 'Utility', icon: 'calc' },
  { id: 'settings', name: 'Settings', target: 'settings', sub: 'System Config', icon: 'settings' },
];

const STARTER_CODE = `# Project Robbie — Sandboxed Python Scratchpad (Read-Only Safety Enforced)
import sys
import torch
from robbie.system.diagnostics import SystemDiagnostics

# Inspect hardware performance programmatically
report = SystemDiagnostics.diagnose_cpu()
print("Robbie Architecture: Verified")
print("Processor Headline:", report.headline)
print("Primary Bottleneck:", report.primary_bottleneck)
print("Diagnostic Findings Count:", len(report.findings))
`;

// ============================================================================
// 2. Application State & Storage Management
// ============================================================================
const state = {
  theme: localStorage.getItem('robbie-theme') || 'dark',
  activeView: 'chat',
  sidebarCollapsed: false,
  chatHistoryDrawerOpen: false,
  chatMessages: [],
  isGenerating: false,
  researchReports: [],
  activeReport: null,
  telemetry: null,
  computerSubtab: 'overview',
  diagnosticReport: null,
  speedDialApps: loadSpeedDialFromStorage(),
  explorerPath: '.',
  activePreviewFile: null,
  workspaceActiveFile: 'scratchpad.py',
  workspaceAskDrawerOpen: false,
  memory: { profile: {}, notes: {}, episodes: [], session: {} },
  memoryActiveTab: 'profile',
  memorySearchFilter: '',
  telemetryInterval: parseInt(localStorage.getItem('robbie-telemetry-interval') || '5000', 10),
  telemetryTimer: null,
};

function loadSpeedDialFromStorage() {
  try {
    const raw = localStorage.getItem('robbie-speed-dial');
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch (e) {
    console.warn('[SpeedDial] Storage read error:', e);
  }
  return [...DEFAULT_SPEED_DIAL];
}

function saveSpeedDialToStorage(apps) {
  try {
    localStorage.setItem('robbie-speed-dial', JSON.stringify(apps));
  } catch (e) {
    console.warn('[SpeedDial] Storage write error:', e);
  }
}

// ============================================================================
// 3. Mock Fallback for Browser Preview & Standalone Testing
// ============================================================================
async function mockInvoke(cmd, args = {}) {
  console.log(`[Mock IPC] ${cmd}`, args);

  if (cmd === 'greet') {
    return `Hello, ${args.name || 'Developer'}! Robbie Studio is online.`;
  }

  if (cmd === 'ask_robbie') {
    const prompt = (args.prompt || '').trim();
    const q = prompt.toLowerCase();

    if (q.includes('cpu') || q.includes('fan') || q.includes('load') || q.includes('usage') || q.includes('slow')) {
      const rawDiag = await mockInvoke('run_diagnostic', { query: 'cpu' });
      const diag = JSON.parse(rawDiag);
      return JSON.stringify({
        reply: `### CPU Load Diagnostic\n\nI checked your live telemetry counters. Processor utilization is nominal across 12 logical cores (~24% load).\n\nTop active tasks include \`Antigravity IDE.exe\` (1.3 GB) and background processes.\n\n*Action:* No severe bottleneck detected. Close background video streams if you need maximum responsiveness.`,
        tool_action: 'diagnostic',
        tool_data: diag,
        slots: { user_name: "Developer", current_topic: "hardware_cpu" },
        turns: state.chatMessages.length + 1,
        model: "Robbie-25M (Stage 2 Conversational)"
      });
    }

    if (q.includes('fps') || q.includes('game') || q.includes('stutter') || q.includes('lag') || q.includes('gpu')) {
      const rawDiag = await mockInvoke('run_diagnostic', { query: 'fps' });
      const diag = JSON.parse(rawDiag);
      return JSON.stringify({
        reply: `### Frame Rate & GPU Telemetry Analysis\n\nI evaluated your Quadro RTX 3000 graphics subsystem.\n\n* Dedicated VRAM headroom is healthy: 5.9 GB free out of 6.0 GB total.\n* Power state: AC Connected (100% battery, high-performance plan).\n\n*Recommendation:* Background browser tabs and IDE processes are active. Ensure hardware acceleration is turned off in background apps for optimal game frame pacing.`,
        tool_action: 'diagnostic',
        tool_data: diag,
        slots: { user_name: "Developer", current_topic: "hardware_gpu" },
        turns: state.chatMessages.length + 1,
        model: "Robbie-25M (Stage 2 Conversational)"
      });
    }

    if (q.startsWith('research ') || q.startsWith('search ')) {
      const topic = prompt.replace(/^(research|search)\s+/i, '').trim();
      const rawRes = await mockInvoke('run_research', { topic });
      const res = JSON.parse(rawRes);
      return JSON.stringify({
        reply: `I conducted autonomous multi-source research on **${res.topic}**.\n\n${res.summary}\n\nKey finding: ${res.findings[0]}`,
        tool_action: 'research',
        tool_data: res,
        slots: { user_name: "Developer", current_topic: "research" },
        turns: state.chatMessages.length + 1,
        model: "Robbie-25M (Stage 2 Conversational)"
      });
    }

    if (q.startsWith('open ') || q.startsWith('launch ')) {
      const app = prompt.replace(/^(open|launch)\s+/i, '').trim();
      await mockInvoke('launch_app', { app });
      return JSON.stringify({
        reply: `Sent Windows start command for **${app}**. Application process initialized.`,
        tool_action: 'launch',
        tool_data: { app },
        slots: { user_name: "Developer", last_action: `launched_${app}` },
        turns: state.chatMessages.length + 1,
        model: "Robbie-25M (Stage 2 Conversational)"
      });
    }

    return JSON.stringify({
      reply: `I received: "${prompt}". Robbie-25M is online with full local computer awareness, deep research, and diagnostics! How can I assist your engineering session?`,
      slots: { user_name: "Developer" },
      turns: state.chatMessages.length + 1,
      model: "Robbie-25M (Stage 2 Conversational)"
    });
  }

  if (cmd === 'get_system_telemetry') {
    return JSON.stringify({
      cpu: { processor: "Intel64 Family 6 Model 165 (AMD64)", logical_cores: 12, cpu_usage_percent: 24.5 },
      memory: { total_gb: 31.73, available_gb: 18.20, used_gb: 13.53, percent_used: 42.5 },
      gpu: { cuda_available: true, device_name: "Quadro RTX 3000", vram_total_gb: 6.0, vram_free_gb: 5.9 },
      battery: { percent: 100, is_charging: true, power_source: "AC Power" },
      disks: [
        { drive: "C:", total_gb: 175.1, free_gb: 50.6, percent_used: 71.1 },
        { drive: "Z:", total_gb: 300.8, free_gb: 170.8, percent_used: 43.2 }
      ],
      processes: [
        { pid: 1352, name: "Antigravity IDE.exe", memory_mb: 1352.0 },
        { pid: 1193, name: "python.exe", memory_mb: 1193.0 },
        { pid: 462, name: "MsMpEng.exe", memory_mb: 462.0 },
        { pid: 9792, name: "explorer.exe", memory_mb: 430.7 },
        { pid: 12036, name: "msedgewebview2.exe", memory_mb: 430.0 }
      ]
    });
  }

  if (cmd === 'get_memory_profile' || cmd === 'manage_memory') {
    return JSON.stringify({
      profile: { user_name: "Developer", user_role: "Lead Engineer", preferred_editor: "Robbie Studio" },
      notes: { project_goal: "Train custom 25M Transformer model with computer awareness" },
      episodes: [
        { summary: "Fine-tuned Robbie-25M Stage 2 dialogue transformer with 115 passing tests.", tags: ["training", "stage2"], timestamp: Date.now() / 1000 - 3600 }
      ],
      episodes_count: 1
    });
  }

  if (cmd === 'run_research') {
    return JSON.stringify({
      topic: args.topic || "Computer Awareness & Telemetry",
      summary: "Computer awareness integrates operating system performance counters, GPU hardware telemetry, and diagnostic heuristics to detect thermal throttling and VRAM saturation.",
      markdown: "### Research Report: " + (args.topic || "Computer Awareness") + "\n\n*3 sources consulted in 4.2s | Citations: 100% verified*\n\n**Overview:**\nComputer awareness integrates operating system performance counters, GPU hardware telemetry, and diagnostic heuristics.\n\n**Key Findings:**\n• VRAM saturation over 85% causes micro-stuttering during 3D rendering. [1, 2]\n• AC power connection allows full TDP GPU boost clocks compared to battery operation. [1, 3]\n\n**Sources & Citations:**\n[1] Windows Hardware Telemetry & Architecture — https://learn.microsoft.com\n[2] GPU VRAM Allocation and Gaming Frame Times — https://nvidia.com\n[3] Local Transformer Optimization — https://huggingface.co",
      html: "<article class=\"robbie-research-report\"><header class=\"report-header\"><h2>Research Report: " + (args.topic || "Computer Awareness") + "</h2><p class=\"report-meta\">3 sources consulted in 4.2s &bull; Grounded Citations: 100% verified</p></header><section class=\"report-overview\"><h3>Overview</h3><p>Computer awareness integrates operating system performance counters, GPU hardware telemetry, and diagnostic heuristics to detect thermal throttling and VRAM saturation.</p></section><section class=\"report-findings\"><h3>Key Findings</h3><ul><li>VRAM saturation over 85% causes micro-stuttering during 3D rendering. <sup><a href=\"#source-1\">[1]</a>, <a href=\"#source-2\">[2]</a></sup></li><li>AC power connection allows full TDP GPU boost clocks compared to battery operation. <sup><a href=\"#source-1\">[1]</a>, <a href=\"#source-3\">[3]</a></sup></li></ul></section></article>",
      sources: [
        { index: 1, title: "Windows Hardware Telemetry & Architecture", url: "https://learn.microsoft.com", domain: "learn.microsoft.com", score: 1.8, date: "2026-10-11", is_outdated: false },
        { index: 2, title: "GPU VRAM Allocation and Gaming Frame Times", url: "https://nvidia.com", domain: "nvidia.com", score: 1.7, date: "2026-10-10", is_outdated: false },
        { index: 3, title: "Local Small Language Model Optimization", url: "https://huggingface.co", domain: "huggingface.co", score: 1.5, date: "2026-10-09", is_outdated: false }
      ],
      findings: [
        "VRAM saturation over 85% causes micro-stuttering during 3D rendering. [1, 2]",
        "AC power connection allows full TDP GPU boost clocks compared to battery operation. [1, 3]"
      ],
      citations: [
        { index: 1, title: "Windows Hardware Telemetry & Architecture", url: "https://learn.microsoft.com", source: "microsoft" },
        { index: 2, title: "GPU VRAM Allocation and Gaming Frame Times", url: "https://nvidia.com", source: "nvidia" },
        { index: 3, title: "Local Small Language Model Optimization", url: "https://huggingface.co", source: "huggingface" }
      ],
      latency_sec: 4.2,
      citation_validation_rate: 1.0,
      sources_count: 3
    });
  }

  if (cmd === 'run_diagnostic') {
    const rawQ = args.query || args.diagType || args.diag_type || 'all';
    const q = rawQ.toLowerCase().trim();
    if (q.includes('fps') || q.includes('game') || q.includes('gpu') || q.includes('vram')) {
      return JSON.stringify({
        headline: "FPS & GPU Diagnostic: Quadro RTX 3000",
        primary_bottleneck: "Normal Headroom",
        findings: [
          { category: "gpu", severity: "info", title: "VRAM Headroom Healthy", detail: "Quadro RTX 3000 has 5.9 GB free out of 6.0 GB total.", culprits: [], recommendation: "VRAM capacity is optimal." },
          { category: "cpu", severity: "warning", title: "Background Application Activity", detail: "IDE and python processes active.", culprits: ["Antigravity IDE.exe"], recommendation: "Close background windows for maximum gaming frame rate." }
        ],
        summary_markdown: `### Game Frame Rate & GPU Diagnostic\n\n**Graphics Hardware:** Quadro RTX 3000 (CUDA Ready, 5.9 GB Free / 6.0 GB Total)\n**Power State:** AC Connected (100% Battery - High Performance Plan)\n**System RAM:** 18.2 GB Free / 31.7 GB Total\n\n**Troubleshooting Checklist:**\n* [INFO] **VRAM Headroom Healthy**: GPU textures reside entirely in dedicated video memory.\n* [WARNING] **Background Activity**: Non-game applications active (Antigravity IDE.exe, python.exe).\n  *Action:* Close unneeded background apps before starting games.`
      });
    } else if (q.includes('disk') || q.includes('storage') || q.includes('drive')) {
      return JSON.stringify({
        headline: "Disk Storage Diagnostic",
        primary_bottleneck: "Storage Healthy",
        findings: [
          { category: "disk", severity: "info", title: "Drive C: Healthy", detail: "50.6 GB free of 175.1 GB (71.1% used).", culprits: [], recommendation: "No action required." }
        ],
        summary_markdown: `### Storage & Partition Diagnosis\n\n* **Drive C:**: \`[##############------]\` **71.1%** (50.6 GB free / 175.1 GB total)\n* **Drive Z:**: \`[########------------]\` **43.2%** (170.8 GB free / 300.8 GB total)\n\n**Status:** Sufficient storage available on all mounted volumes.`
      });
    } else {
      return JSON.stringify({
        headline: `Diagnostic Analysis: ${rawQ}`,
        primary_bottleneck: "Nominal Load",
        findings: [
          { category: "cpu", severity: "info", title: "Processor Load Nominal", detail: "Processor utilization at 24.5% across 12 cores.", culprits: [], recommendation: "CPU headroom is healthy." }
        ],
        summary_markdown: `### System Diagnostic for: "${escapeHTML(rawQ)}"\n\n**Live Telemetry Evaluation:**\n* **Processor Load:** 24.5% across 12 logical cores (Healthy)\n* **System Memory:** 18.2 GB free of 31.7 GB total (Healthy)\n* **GPU Headroom:** Quadro RTX 3000 (5.9 GB VRAM free, CUDA Active)\n* **Storage C:**: 50.6 GB free (71.1% capacity)\n\n**Assessment:** All core hardware metrics are operating within expected parameters.`
      });
    }
  }

  if (cmd === 'launch_app') {
    return JSON.stringify({ success: true, app_name: args.app, command: `start ${args.app}`, message: `Launched ${args.app} successfully.` });
  }

  if (cmd === 'browse_files') {
    const action = args.action || 'list';
    if (action === 'list') {
      return JSON.stringify({
        path: args.path || '.',
        entries: [
          { name: "src", is_dir: true, size_bytes: null },
          { name: "editor", is_dir: true, size_bytes: null },
          { name: "data", is_dir: true, size_bytes: null },
          { name: "checkpoints", is_dir: true, size_bytes: null },
          { name: "tests", is_dir: true, size_bytes: null },
          { name: "pyproject.toml", is_dir: false, size_bytes: 1420 },
          { name: "README.md", is_dir: false, size_bytes: 4890 },
          { name: "PLAN.md", is_dir: false, size_bytes: 12500 }
        ],
        total: 8
      });
    } else if (action === 'read') {
      return JSON.stringify({
        path: args.path,
        content: `# Read-Only File Inspector: ${args.path}\n\nRobbie operates under a strict read-only safety policy.\nFiles can be inspected and executed in sandboxes safely without risking modification or deletion.`,
        lines: 4,
        size_bytes: 180
      });
    } else if (action === 'open') {
      return JSON.stringify({ success: true, message: `Opened "${args.path}" in native system viewer.` });
    }
  }

  if (cmd === 'read_project_file') {
    return `# Read-Only Project File: ${args.file_path}\n"""Controlled safe view from project root."""\nimport sys\n\ndef check_system():\n    return True\n`;
  }

  if (cmd === 'execute_code') {
    return JSON.stringify({
      success: true,
      stdout: ">>> Executing Python snippet (Sandboxed Safety Policy Active)...\nRobbie Architecture: Verified\nProcessor Headline: CPU Analysis: Intel64 Family 6 Model 165 (12 Cores)\nPrimary Bottleneck: Nominal Load\nDiagnostic Findings Count: 1\n",
      stderr: "",
      error: ""
    });
  }

  return JSON.stringify({ status: "ok" });
}

// ============================================================================
// 4. Initialization & Lifecycle
// ============================================================================
window.addEventListener('DOMContentLoaded', async () => {
  setupThemeSystem();
  setupNavigation();
  setupChatController();
  setupResearchController();
  setupComputerController();
  setupWorkspaceController();
  setupMemoryController();
  setupSettingsController();

  // Initial Data Synchronization
  await syncTelemetryData();
  await syncMemoryData();
  await loadExplorerPath('.');
  startTelemetryPolling();

  // Initialize header for starting view
  updateTopHeader(state.activeView);
});

// ============================================================================
// 5. Theme & Navigation Architecture
// ============================================================================
function setupThemeSystem() {
  const root = document.documentElement;
  const toggleBtn = document.getElementById('btn-theme-toggle');
  const btnDark = document.getElementById('btn-set-theme-dark');
  const btnLight = document.getElementById('btn-set-theme-light');
  const themeIcon = document.getElementById('theme-icon');

  function applyTheme(theme) {
    state.theme = theme;
    root.setAttribute('data-theme', theme);
    localStorage.setItem('robbie-theme', theme);

    if (themeIcon) {
      if (theme === 'dark') {
        themeIcon.innerHTML = `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>`;
      } else {
        themeIcon.innerHTML = `<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>`;
      }
    }
  }

  applyTheme(state.theme);

  toggleBtn?.addEventListener('click', () => {
    applyTheme(state.theme === 'dark' ? 'light' : 'dark');
  });

  btnDark?.addEventListener('click', () => applyTheme('dark'));
  btnLight?.addEventListener('click', () => applyTheme('light'));
}

function setupNavigation() {
  const navItems = document.querySelectorAll('.sidebar-nav .nav-item, .sidebar-footer .nav-item');
  const brandBtn = document.getElementById('nav-brand-btn');
  const toggleBtn = document.getElementById('btn-sidebar-toggle');
  const sidebar = document.getElementById('app-sidebar');

  toggleBtn?.addEventListener('click', () => {
    state.sidebarCollapsed = !state.sidebarCollapsed;
    sidebar.classList.toggle('collapsed', state.sidebarCollapsed);
  });

  brandBtn?.addEventListener('click', () => {
    switchView('chat');
  });

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetView = item.getAttribute('data-view');
      if (targetView) switchView(targetView);
    });
  });
}

function switchView(viewName) {
  state.activeView = viewName;

  // Update Navigation Active State
  document.querySelectorAll('.sidebar-nav .nav-item, .sidebar-footer .nav-item').forEach(item => {
    if (item.getAttribute('data-view') === viewName) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });

  // Switch View Panes
  document.querySelectorAll('.view-pane').forEach(pane => {
    pane.classList.remove('active');
  });
  const targetPane = document.getElementById(`view-${viewName}`);
  if (targetPane) targetPane.classList.add('active');

  // Update Contextual Top Header Controls
  updateTopHeader(viewName);
}

function updateTopHeader(viewName) {
  const titleEl = document.getElementById('header-view-title');
  const badgeEl = document.getElementById('header-badge-text');
  const controlsEl = document.getElementById('header-custom-controls');
  if (!controlsEl) return;

  controlsEl.innerHTML = '';

  const VIEW_TITLES = {
    chat: 'Chat',
    research: 'Research',
    computer: 'Computer',
    workspace: 'Workspace',
    memory: 'Memory',
    settings: 'Settings',
  };

  if (titleEl) titleEl.textContent = VIEW_TITLES[viewName] || 'Studio';

  if (viewName === 'chat') {
    if (badgeEl) badgeEl.textContent = 'Robbie-25M Model';
    controlsEl.innerHTML = `
      <button class="btn-secondary-sm" id="btn-hdr-history" title="Toggle Chat History Drawer">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 8v4l3 3m6-3a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"/></svg>
        <span>Sessions</span>
      </button>
      <button class="btn-primary-sm" id="btn-hdr-new-chat" title="Start fresh conversation">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        <span>New Chat</span>
      </button>
    `;

    document.getElementById('btn-hdr-history')?.addEventListener('click', toggleChatHistoryDrawer);
    document.getElementById('btn-hdr-new-chat')?.addEventListener('click', resetChatSession);
  } else if (viewName === 'research') {
    if (badgeEl) badgeEl.textContent = state.activeReport ? state.activeReport.topic : 'Web Synthesis';
    if (state.activeReport) {
      controlsEl.innerHTML = `
        <button class="btn-secondary-sm" id="btn-hdr-res-copy">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          <span>Copy Markdown</span>
        </button>
        <button class="btn-primary-sm" id="btn-hdr-res-new">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <span>New Query</span>
        </button>
      `;
      document.getElementById('btn-hdr-res-copy')?.addEventListener('click', () => {
        copyToClipboard(state.activeReport.markdown, 'Report Markdown copied to clipboard.');
      });
      document.getElementById('btn-hdr-res-new')?.addEventListener('click', closeResearchReportView);
    }
  } else if (viewName === 'computer') {
    if (badgeEl) badgeEl.textContent = state.computerSubtab === 'overview' ? 'Live Telemetry' : 'File Explorer';
    controlsEl.innerHTML = `
      <button class="btn-secondary-sm" id="btn-hdr-comp-refresh">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 4v6h-6M1 20v-6h6"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
        <span>Refresh</span>
      </button>
    `;
    document.getElementById('btn-hdr-comp-refresh')?.addEventListener('click', () => {
      syncTelemetryData();
      showToast('Telemetry refreshed.');
    });
  } else if (viewName === 'workspace') {
    if (badgeEl) badgeEl.textContent = state.workspaceActiveFile;
    controlsEl.innerHTML = `
      <button class="btn-primary-sm" id="btn-hdr-run-code">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        <span>Run (Ctrl+Enter)</span>
      </button>
      <button class="btn-secondary-sm" id="btn-hdr-ask-robbie">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <span>Ask Robbie</span>
      </button>
    `;
    document.getElementById('btn-hdr-run-code')?.addEventListener('click', handleRunCode);
    document.getElementById('btn-hdr-ask-robbie')?.addEventListener('click', toggleWorkspaceAskDrawer);
  } else if (viewName === 'memory') {
    if (badgeEl) badgeEl.textContent = 'Persistent Storage';
    controlsEl.innerHTML = `
      <button class="btn-primary-sm" id="btn-hdr-mem-add">+ Add Item</button>
      <button class="btn-danger-sm" id="btn-hdr-mem-clear">Clear All</button>
    `;
    document.getElementById('btn-hdr-mem-add')?.addEventListener('click', promptAddMemoryModal);
    document.getElementById('btn-hdr-mem-clear')?.addEventListener('click', promptClearMemoryModal);
  } else if (viewName === 'settings') {
    if (badgeEl) badgeEl.textContent = 'Preferences';
  }
}

// ============================================================================
// 6. View 1: Chat Workspace Controller
// ============================================================================
function setupChatController() {
  const textarea = document.getElementById('chat-composer-textarea');
  const btnSend = document.getElementById('btn-chat-send');
  const btnStop = document.getElementById('btn-chat-stop');
  const btnCloseDrawer = document.getElementById('btn-close-history-drawer');
  const btnDrawerNewChat = document.getElementById('btn-drawer-new-chat');
  const btnClearHistory = document.getElementById('btn-clear-chat-history');

  // Multiline Textarea Expansion & Keyboard Triggers
  textarea?.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
  });

  textarea?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleChatSendMessage();
    }
  });

  btnSend?.addEventListener('click', handleChatSendMessage);
  btnStop?.addEventListener('click', handleChatStopGeneration);

  // History Drawer Controls
  btnCloseDrawer?.addEventListener('click', () => toggleChatHistoryDrawer(false));
  btnDrawerNewChat?.addEventListener('click', resetChatSession);
  btnClearHistory?.addEventListener('click', () => {
    showModal('Clear Chat', 'Are you sure you want to clear the active conversation?', () => {
      resetChatSession();
      showToast('Conversation cleared.');
    });
  });

  // Welcome Hint Pills
  document.querySelectorAll('.hint-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const hint = pill.getAttribute('data-hint');
      if (hint && textarea) {
        textarea.value = hint;
        textarea.focus();
        handleChatSendMessage();
      }
    });
  });

  // Composer Tool Shortcuts
  document.querySelectorAll('.composer-tool-tag').forEach(tag => {
    tag.addEventListener('click', () => {
      const tool = tag.getAttribute('data-tool');
      if (!textarea) return;
      if (tool === 'diag') {
        textarea.value = 'Why is my CPU or GPU usage high?';
      } else if (tool === 'research') {
        textarea.value = 'Research ';
      } else if (tool === 'code') {
        textarea.value = 'Can you run a Python script to check system diagnostics?';
      } else if (tool === 'launch') {
        textarea.value = 'Open ';
      }
      textarea.focus();
    });
  });
}

function toggleChatHistoryDrawer(forcedState) {
  const drawer = document.getElementById('chat-history-drawer');
  if (!drawer) return;
  state.chatHistoryDrawerOpen = typeof forcedState === 'boolean' ? forcedState : !state.chatHistoryDrawerOpen;
  drawer.classList.toggle('closed', !state.chatHistoryDrawerOpen);
}

function resetChatSession() {
  state.chatMessages = [];
  const stream = document.getElementById('chat-messages-stream');
  const welcome = document.getElementById('chat-welcome-state');
  if (stream) stream.innerHTML = '';
  if (welcome) welcome.style.display = 'flex';
  const turnsMeta = document.getElementById('session-turns-meta');
  if (turnsMeta) turnsMeta.textContent = '0 turns • Active now';
  showToast('Started fresh conversation.');
}

async function handleChatSendMessage() {
  if (state.isGenerating) return;
  const textarea = document.getElementById('chat-composer-textarea');
  if (!textarea) return;
  const prompt = textarea.value.trim();
  if (!prompt) return;

  // Clear input
  textarea.value = '';
  textarea.style.height = 'auto';

  // Hide Welcome Screen
  const welcome = document.getElementById('chat-welcome-state');
  if (welcome) welcome.style.display = 'none';

  // Append User Message to Stream
  appendChatMessage('user', prompt);

  // Set Generating UI State
  setChatGeneratingState(true);

  // Temporary Thinking Indicator
  const thinkingId = appendChatThinkingIndicator();

  try {
    const rawRes = await invoke('ask_robbie', { prompt });
    let res = {};
    try {
      res = JSON.parse(rawRes);
    } catch {
      res = { reply: rawRes };
    }

    removeChatThinkingIndicator(thinkingId);

    // Append Assistant Message with Rich Formatting & Inline Tools
    appendChatMessage('assistant', res.reply || 'No response generated.', {
      tool_action: res.tool_action,
      tool_data: res.tool_data,
      model: res.model || 'Robbie-25M',
    });

    // Update Session Turn Count Meta
    const turnsMeta = document.getElementById('session-turns-meta');
    if (turnsMeta) turnsMeta.textContent = `${state.chatMessages.length} turns • Active now`;

    // Refresh Session Slots in Memory if provided
    if (res.slots) {
      state.memory.session = res.slots;
      renderMemorySessionSlots();
    }
  } catch (err) {
    removeChatThinkingIndicator(thinkingId);
    appendChatMessage('assistant', `⚠️ **Inference Error:** Failed to generate response from Robbie-25M: ${err.message || err}`);
  } finally {
    setChatGeneratingState(false);
  }
}

function setChatGeneratingState(isGen) {
  state.isGenerating = isGen;
  const btnSend = document.getElementById('btn-chat-send');
  const btnStop = document.getElementById('btn-chat-stop');
  if (btnSend) btnSend.style.display = isGen ? 'none' : 'flex';
  if (btnStop) btnStop.style.display = isGen ? 'flex' : 'none';
}

function handleChatStopGeneration() {
  setChatGeneratingState(false);
  showToast('Generation halted.');
}

function appendChatMessage(role, content, extras = {}) {
  const stream = document.getElementById('chat-messages-stream');
  if (!stream) return;

  const msgObj = { role, content, extras, timestamp: Date.now() };
  state.chatMessages.push(msgObj);

  const msgDiv = document.createElement('div');
  msgDiv.className = `chat-message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = role === 'assistant' ? 'R' : 'U';

  const body = document.createElement('div');
  body.className = 'msg-body';
  body.innerHTML = renderMarkdown(content);

  // Render Inline Tool Cards if present
  if (extras.tool_action && extras.tool_data) {
    const toolCard = renderInlineToolCard(extras.tool_action, extras.tool_data);
    if (toolCard) body.appendChild(toolCard);
  }

  msgDiv.appendChild(avatar);
  msgDiv.appendChild(body);
  stream.appendChild(msgDiv);

  // Auto-scroll to bottom
  const viewport = document.getElementById('chat-scroll-viewport');
  if (viewport) viewport.scrollTop = viewport.scrollHeight;
}

function appendChatThinkingIndicator() {
  const stream = document.getElementById('chat-messages-stream');
  if (!stream) return null;
  const id = `thinking-${Date.now()}`;
  const div = document.createElement('div');
  div.className = 'chat-message assistant';
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar">R</div>
    <div class="msg-body" style="color: var(--text-dim); display: flex; align-items: center; gap: 8px;">
      <div class="progress-spinner" style="width: 10px; height: 10px;"></div>
      <span>Robbie-25M is evaluating context...</span>
    </div>
  `;
  stream.appendChild(div);
  const viewport = document.getElementById('chat-scroll-viewport');
  if (viewport) viewport.scrollTop = viewport.scrollHeight;
  return id;
}

function removeChatThinkingIndicator(id) {
  if (!id) return;
  const el = document.getElementById(id);
  if (el) el.remove();
}

function renderInlineToolCard(action, data) {
  const card = document.createElement('div');
  card.className = 'chat-tool-inline-card';

  if (action === 'diagnostic') {
    card.innerHTML = `
      <div class="tool-card-head">
        <span>⚡ Hardware Telemetry & Bottleneck Diagnostic</span>
        <span class="badge badge-accent">${escapeHTML(data.primary_bottleneck || 'Analyzed')}</span>
      </div>
      <div class="tool-card-body">
        <strong>${escapeHTML(data.headline || 'Diagnostic Report')}</strong>
        ${data.findings ? `<ul style="margin: 6px 0 0 16px; font-size: 11.5px;">${data.findings.map(f => `<li><strong>${escapeHTML(f.title)}:</strong> ${escapeHTML(f.detail)}</li>`).join('')}</ul>` : ''}
      </div>
    `;
  } else if (action === 'research') {
    card.innerHTML = `
      <div class="tool-card-head">
        <span>🔍 Research Evidence Synthesized</span>
        <span class="badge badge-success">${data.sources ? data.sources.length : 0} Sources</span>
      </div>
      <div class="tool-card-body">
        <p style="margin-bottom: 6px;">${escapeHTML(data.summary || '')}</p>
        ${data.sources ? `<div style="font-size: 11px; color: var(--text-dim);">${data.sources.map(s => `<div>• <a href="${escapeHTML(s.url)}" target="_blank" style="color: var(--accent-text);">${escapeHTML(s.title)}</a> (${escapeHTML(s.domain)})</div>`).join('')}</div>` : ''}
      </div>
    `;
  } else if (action === 'launch') {
    card.innerHTML = `
      <div class="tool-card-head">
        <span>🚀 Windows Process Launcher</span>
        <span class="badge badge-success">Executed</span>
      </div>
      <div class="tool-card-body">
        <span>Application target: <code>${escapeHTML(data.app || '')}</code></span>
      </div>
    `;
  } else {
    card.innerHTML = `
      <div class="tool-card-head">
        <span>Tool Action: ${escapeHTML(action)}</span>
      </div>
      <div class="tool-card-body">
        <pre style="font-size: 11px;">${escapeHTML(JSON.stringify(data, null, 2))}</pre>
      </div>
    `;
  }
  return card;
}

// ============================================================================
// 7. View 2: Research Workspace Controller
// ============================================================================
function setupResearchController() {
  const queryInput = document.getElementById('research-query-input');
  const countSelect = document.getElementById('research-sources-count');
  const btnStart = document.getElementById('btn-start-research');
  const btnReportBack = document.getElementById('btn-report-back');
  const btnCopyMarkdown = document.getElementById('btn-report-copy-markdown');
  const btnToggleSources = document.getElementById('btn-report-toggle-sources');

  btnStart?.addEventListener('click', handleStartResearch);
  queryInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleStartResearch();
  });

  btnReportBack?.addEventListener('click', closeResearchReportView);

  btnCopyMarkdown?.addEventListener('click', () => {
    if (state.activeReport?.markdown) {
      copyToClipboard(state.activeReport.markdown, 'Report Markdown copied to clipboard.');
    }
  });

  btnToggleSources?.addEventListener('click', () => {
    const sidebar = document.getElementById('report-sources-sidebar');
    if (sidebar) sidebar.classList.toggle('closed');
  });

  // Suggestion chips
  document.querySelectorAll('.research-suggestions-row .chip-btn').forEach(chip => {
    chip.addEventListener('click', () => {
      const topic = chip.getAttribute('data-topic');
      if (topic && queryInput) {
        queryInput.value = topic;
        handleStartResearch();
      }
    });
  });
}

async function handleStartResearch() {
  const queryInput = document.getElementById('research-query-input');
  if (!queryInput) return;
  const topic = queryInput.value.trim();
  if (!topic) return;

  const countSelect = document.getElementById('research-sources-count');
  const maxSources = countSelect ? parseInt(countSelect.value, 10) : 3;

  // Show Report View with Progress Banner
  const landing = document.getElementById('research-landing-view');
  const reportView = document.getElementById('research-report-view');
  const progressBanner = document.getElementById('research-progress-bar');
  const progressText = document.getElementById('research-progress-text');
  const topicLabel = document.getElementById('report-view-topic');

  if (landing) landing.style.display = 'none';
  if (reportView) reportView.style.display = 'flex';
  if (progressBanner) progressBanner.style.display = 'flex';
  if (topicLabel) topicLabel.textContent = topic;

  // Pipeline simulation stages while awaiting IPC response
  let stage = 1;
  const stageTimer = setInterval(() => {
    stage++;
    if (progressText) {
      if (stage === 2) progressText.textContent = '2. Fetching & extracting web source documents...';
      else if (stage === 3) progressText.textContent = '3. Synthesizing evidence & validating citations...';
    }
  }, 1200);

  try {
    const rawRes = await invoke('run_research', { topic });
    clearInterval(stageTimer);
    let report = {};
    try {
      report = JSON.parse(rawRes);
    } catch {
      report = { topic, summary: rawRes, markdown: rawRes, sources: [] };
    }

    if (progressBanner) progressBanner.style.display = 'none';

    // Store in State
    state.activeReport = report;
    state.researchReports.unshift(report);

    // Render Report Document
    renderResearchReport(report);
    renderResearchShelf();
    updateTopHeader('research');
  } catch (err) {
    clearInterval(stageTimer);
    if (progressBanner) progressBanner.style.display = 'none';
    const article = document.getElementById('report-article-content');
    if (article) {
      article.innerHTML = `<h3>Research Execution Failed</h3><p style="color: var(--color-error);">${escapeHTML(err.message || err)}</p>`;
    }
  }
}

function renderResearchReport(report) {
  const article = document.getElementById('report-article-content');
  const sourcesList = document.getElementById('report-sources-list');
  const metaBadge = document.getElementById('report-badge-meta');
  const citationsBadge = document.getElementById('report-badge-citations');
  const sourcesCountBadge = document.getElementById('report-sources-count-badge');

  if (metaBadge) {
    metaBadge.textContent = `${report.sources ? report.sources.length : 0} sources • ${report.latency_sec || '4.2'}s`;
  }
  if (citationsBadge) {
    const rate = report.citation_validation_rate !== undefined ? Math.round(report.citation_validation_rate * 100) : 100;
    citationsBadge.textContent = `${rate}% Citations Verified`;
  }
  if (sourcesCountBadge) {
    sourcesCountBadge.textContent = `${report.sources ? report.sources.length : 0} Sources`;
  }

  // Render Report Article (HTML if provided, else Markdown)
  if (article) {
    if (report.html) {
      article.innerHTML = report.html;
    } else {
      article.innerHTML = `
        <h2>Research Report: ${escapeHTML(report.topic || '')}</h2>
        <div style="margin: 12px 0;"><strong>Overview:</strong><p>${escapeHTML(report.summary || '')}</p></div>
        ${report.findings && report.findings.length ? `<h3>Key Findings</h3><ul>${report.findings.map(f => `<li>${escapeHTML(f)}</li>`).join('')}</ul>` : ''}
      `;
    }

    // Attach interactive citation clicks to highlight corresponding source card
    article.querySelectorAll('sup a').forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const href = link.getAttribute('href');
        if (href) {
          const targetId = href.replace('#', '');
          highlightSourceCard(targetId);
        }
      });
    });
  }

  // Render Right Sources Sidebar
  if (sourcesList) {
    sourcesList.innerHTML = '';
    const sources = report.sources || [];
    if (!sources.length) {
      sourcesList.innerHTML = '<div style="color: var(--text-dim); font-size: 11.5px;">No sources consulted.</div>';
    } else {
      sources.forEach(src => {
        const card = document.createElement('div');
        card.className = 'source-entry-card';
        card.id = `source-${src.index || 1}`;
        card.innerHTML = `
          <div class="source-entry-top">
            <span class="source-domain-badge">[${src.index || 1}] ${escapeHTML(src.domain || 'web')}</span>
            <span class="source-score-badge">Score: ${src.score || '1.0'}</span>
          </div>
          <div class="source-entry-title">${escapeHTML(src.title || 'Untitled Source')}</div>
          <a class="source-entry-link" href="${escapeHTML(src.url || '#')}" target="_blank" rel="noopener noreferrer">
            <span>Open Link</span>
            ${ICONS.external}
          </a>
        `;
        sourcesList.appendChild(card);
      });
    }
  }
}

function highlightSourceCard(id) {
  document.querySelectorAll('.source-entry-card').forEach(c => c.classList.remove('highlighted'));
  const card = document.getElementById(id);
  if (card) {
    card.classList.add('highlighted');
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function closeResearchReportView() {
  state.activeReport = null;
  const landing = document.getElementById('research-landing-view');
  const reportView = document.getElementById('research-report-view');
  if (landing) landing.style.display = 'flex';
  if (reportView) reportView.style.display = 'none';
  updateTopHeader('research');
}

function renderResearchShelf() {
  const grid = document.getElementById('research-history-grid');
  const countBadge = document.getElementById('shelf-reports-count');
  if (!grid) return;

  if (countBadge) countBadge.textContent = `${state.researchReports.length} Reports`;

  if (!state.researchReports.length) {
    grid.innerHTML = '<div class="empty-shelf-message">No research reports conducted in this session yet.</div>';
    return;
  }

  grid.innerHTML = '';
  state.researchReports.forEach(r => {
    const card = document.createElement('div');
    card.className = 'research-history-card';
    card.innerHTML = `
      <div>
        <div class="rh-topic">${escapeHTML(r.topic)}</div>
        <div class="rh-meta">${r.sources ? r.sources.length : 0} sources • ${r.latency_sec || '4.2'}s</div>
      </div>
      <button class="btn-secondary-xs">View Report</button>
    `;
    card.addEventListener('click', () => {
      state.activeReport = r;
      const landing = document.getElementById('research-landing-view');
      const reportView = document.getElementById('research-report-view');
      if (landing) landing.style.display = 'none';
      if (reportView) reportView.style.display = 'flex';
      renderResearchReport(r);
      updateTopHeader('research');
    });
    grid.appendChild(card);
  });
}

// ============================================================================
// 8. View 3: Computer Workspace Controller
// ============================================================================
function setupComputerController() {
  // Sub-workspace tab switching
  const btnSubtabOverview = document.getElementById('btn-subtab-overview');
  const btnSubtabFiles = document.getElementById('btn-subtab-files');
  const paneOverview = document.getElementById('subtab-pane-overview');
  const paneFiles = document.getElementById('subtab-pane-files');

  btnSubtabOverview?.addEventListener('click', () => {
    state.computerSubtab = 'overview';
    btnSubtabOverview.classList.add('active');
    btnSubtabFiles?.classList.remove('active');
    if (paneOverview) paneOverview.style.display = 'block';
    if (paneFiles) paneFiles.style.display = 'none';
    updateTopHeader('computer');
  });

  btnSubtabFiles?.addEventListener('click', () => {
    state.computerSubtab = 'files';
    btnSubtabFiles.classList.add('active');
    btnSubtabOverview?.classList.remove('active');
    if (paneFiles) paneFiles.style.display = 'block';
    if (paneOverview) paneOverview.style.display = 'none';
    updateTopHeader('computer');
  });

  // Diagnostics input
  const diagInput = document.getElementById('diag-freeform-input');
  const btnRunDiag = document.getElementById('btn-run-diag');
  const btnClearDiag = document.getElementById('btn-clear-diag');

  btnRunDiag?.addEventListener('click', () => runDiagnostics(diagInput?.value));
  diagInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runDiagnostics(diagInput.value);
  });

  btnClearDiag?.addEventListener('click', () => {
    const emptyState = document.getElementById('diag-empty-placeholder');
    const activeReport = document.getElementById('diag-active-report');
    if (emptyState) emptyState.style.display = 'block';
    if (activeReport) activeReport.style.display = 'none';
  });

  // Quick Diagnostic Pills
  document.querySelectorAll('.quick-diag-pills .pill-btn-xs').forEach(pill => {
    pill.addEventListener('click', () => {
      const diagType = pill.getAttribute('data-diag');
      runDiagnostics(diagType);
    });
  });

  // Refresh Telemetry Button
  document.getElementById('btn-comp-refresh-telemetry')?.addEventListener('click', async () => {
    await syncTelemetryData();
    showToast('Telemetry updated.');
  });

  // Quick Launcher & Speed Dial
  const launchInput = document.getElementById('quick-launch-input');
  const btnLaunch = document.getElementById('btn-quick-launch');
  const btnOpenAdd = document.getElementById('btn-open-add-shortcut');
  const addDrawer = document.getElementById('add-shortcut-drawer');
  const btnCancelAdd = document.getElementById('btn-cancel-add-shortcut');
  const btnSaveAdd = document.getElementById('btn-save-new-shortcut');

  btnLaunch?.addEventListener('click', () => {
    const app = launchInput?.value.trim();
    if (app) launchApplication(app);
  });
  launchInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const app = launchInput.value.trim();
      if (app) launchApplication(app);
    }
  });

  btnOpenAdd?.addEventListener('click', () => {
    if (addDrawer) addDrawer.style.display = addDrawer.style.display === 'none' ? 'block' : 'none';
  });
  btnCancelAdd?.addEventListener('click', () => {
    if (addDrawer) addDrawer.style.display = 'none';
  });
  btnSaveAdd?.addEventListener('click', () => {
    const name = document.getElementById('new-sc-name')?.value.trim();
    const target = document.getElementById('new-sc-target')?.value.trim();
    if (name && target) {
      state.speedDialApps.push({ id: `custom-${Date.now()}`, name, target, sub: 'Custom App', icon: 'generic' });
      saveSpeedDialToStorage(state.speedDialApps);
      renderSpeedDialGrid();
      if (addDrawer) addDrawer.style.display = 'none';
      showToast(`Added ${name} to shortcuts.`);
    }
  });

  renderSpeedDialGrid();

  // File Explorer Controls
  document.getElementById('btn-file-up')?.addEventListener('click', () => {
    const current = state.explorerPath;
    const parent = current === '.' ? '..' : `${current}/..`;
    loadExplorerPath(parent);
  });
  document.getElementById('btn-file-browse')?.addEventListener('click', () => {
    const inputPath = document.getElementById('explorer-path-input')?.value;
    if (inputPath) loadExplorerPath(inputPath);
  });
  document.getElementById('btn-file-refresh')?.addEventListener('click', () => {
    loadExplorerPath(state.explorerPath);
  });
  document.getElementById('btn-preview-open-system')?.addEventListener('click', async () => {
    if (state.activePreviewFile) {
      await invoke('browse_files', { action: 'open', path: state.activePreviewFile });
      showToast(`Opened ${state.activePreviewFile} in Windows.`);
    }
  });
}

async function runDiagnostics(query) {
  const target = (query || 'all').trim();
  const emptyState = document.getElementById('diag-empty-placeholder');
  const activeReport = document.getElementById('diag-active-report');
  const headlineEl = document.getElementById('diag-report-headline');
  const bottleneckBadge = document.getElementById('diag-bottleneck-badge');
  const findingsList = document.getElementById('diag-findings-list');
  const markdownBox = document.getElementById('diag-markdown-box');

  if (emptyState) emptyState.style.display = 'none';
  if (activeReport) activeReport.style.display = 'block';

  try {
    const rawRes = await invoke('run_diagnostic', { query: target });
    let diag = {};
    try {
      diag = JSON.parse(rawRes);
    } catch {
      diag = { headline: `Diagnostic: ${target}`, summary_markdown: rawRes };
    }

    state.diagnosticReport = diag;

    if (headlineEl) headlineEl.textContent = diag.headline || 'System Performance Diagnostic';
    if (bottleneckBadge) bottleneckBadge.textContent = diag.primary_bottleneck || 'Nominal Load';

    if (findingsList) {
      findingsList.innerHTML = '';
      const findings = diag.findings || [];
      if (!findings.length) {
        findingsList.innerHTML = '<div style="color: var(--text-dim); font-size: 11px;">No bottlenecks or critical hardware faults detected.</div>';
      } else {
        findings.forEach(f => {
          const row = document.createElement('div');
          row.className = 'finding-row';
          const sevClass = f.severity === 'critical' ? 'badge-danger' : (f.severity === 'warning' ? 'badge-warning' : 'badge-neutral');
          row.innerHTML = `
            <span class="badge ${sevClass}">${f.severity.toUpperCase()}</span>
            <div class="finding-content">
              <h5>${escapeHTML(f.title)}</h5>
              <p>${escapeHTML(f.detail)}</p>
              ${f.culprits && f.culprits.length ? `<div class="culprits">Culprits: ${escapeHTML(f.culprits.join(', '))}</div>` : ''}
              ${f.recommendation ? `<div style="font-size: 11px; color: var(--text-dim); margin-top: 2px;">Recommendation: ${escapeHTML(f.recommendation)}</div>` : ''}
            </div>
          `;
          findingsList.appendChild(row);
        });
      }
    }

    if (markdownBox) {
      markdownBox.innerHTML = renderMarkdown(diag.summary_markdown || '');
    }
  } catch (err) {
    if (findingsList) {
      findingsList.innerHTML = `<div style="color: var(--color-error);">Diagnostic Error: ${escapeHTML(err.message || err)}</div>`;
    }
  }
}

async function launchApplication(appName) {
  try {
    await invoke('launch_app', { app: appName });
    showToast(`Sent start command for "${appName}".`);
  } catch (err) {
    showToast(`Failed to launch "${appName}": ${err.message || err}`);
  }
}

function renderSpeedDialGrid() {
  const grid = document.getElementById('speed-dial-grid');
  if (!grid) return;
  grid.innerHTML = '';
  state.speedDialApps.forEach(app => {
    const tile = document.createElement('div');
    tile.className = 'speed-dial-tile';
    tile.title = `Launch ${app.name} (${app.target})`;
    tile.innerHTML = `
      <div class="tile-icon">${ICONS[app.icon] || ICONS.generic}</div>
      <div class="tile-name">${escapeHTML(app.name)}</div>
    `;
    tile.addEventListener('click', () => launchApplication(app.target));
    grid.appendChild(tile);
  });
}

async function loadExplorerPath(path) {
  state.explorerPath = path;
  const pathInput = document.getElementById('explorer-path-input');
  const tableBody = document.getElementById('explorer-table-body');
  if (pathInput) pathInput.value = path;
  if (!tableBody) return;

  tableBody.innerHTML = '<tr><td colspan="4" class="table-loading">Inspecting directory...</td></tr>';

  try {
    const rawRes = await invoke('browse_files', { action: 'list', path });
    let res = {};
    try {
      res = JSON.parse(rawRes);
    } catch {
      res = { entries: [] };
    }

    const entries = res.entries || [];
    tableBody.innerHTML = '';

    if (!entries.length) {
      tableBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-dim); padding: 12px;">Empty directory.</td></tr>';
      return;
    }

    entries.forEach(item => {
      const row = document.createElement('tr');
      const sizeStr = item.is_dir ? 'DIR' : (item.size_bytes !== null && item.size_bytes !== undefined ? formatBytes(item.size_bytes) : '--');
      row.innerHTML = `
        <td>${item.is_dir ? ICONS.folder : ICONS.file}</td>
        <td><strong>${escapeHTML(item.name)}</strong></td>
        <td style="font-family: var(--font-mono); font-size: 11px;">${sizeStr}</td>
        <td>
          ${item.is_dir ? `<button class="btn-secondary-xs btn-open-dir" data-name="${escapeHTML(item.name)}">Open</button>` : `<button class="btn-secondary-xs btn-inspect-file" data-name="${escapeHTML(item.name)}">Read</button>`}
        </td>
      `;

      row.querySelector('.btn-open-dir')?.addEventListener('click', () => {
        const nextPath = path === '.' ? item.name : `${path}/${item.name}`;
        loadExplorerPath(nextPath);
      });

      row.querySelector('.btn-inspect-file')?.addEventListener('click', () => {
        const filePath = path === '.' ? item.name : `${path}/${item.name}`;
        inspectFilePreview(filePath);
      });

      tableBody.appendChild(row);
    });
  } catch (err) {
    tableBody.innerHTML = `<tr><td colspan="4" style="color: var(--color-error); text-align: center;">Error: ${escapeHTML(err.message || err)}</td></tr>`;
  }
}

async function inspectFilePreview(filePath) {
  state.activePreviewFile = filePath;
  const nameEl = document.getElementById('preview-file-name');
  const codeEl = document.getElementById('preview-file-content');
  const openBtn = document.getElementById('btn-preview-open-system');

  if (nameEl) nameEl.textContent = filePath;
  if (openBtn) openBtn.style.display = 'inline-flex';
  if (codeEl) codeEl.textContent = 'Loading file contents safely...';

  try {
    const rawRes = await invoke('browse_files', { action: 'read', path: filePath });
    let res = {};
    try {
      res = JSON.parse(rawRes);
    } catch {
      res = { content: rawRes };
    }
    if (codeEl) codeEl.textContent = res.content || '(File is empty)';
  } catch (err) {
    if (codeEl) codeEl.textContent = `Could not read file: ${err.message || err}`;
  }
}

// ============================================================================
// 9. View 4: Developer Workspace Controller (3-Region IDE)
// ============================================================================
function setupWorkspaceController() {
  const textarea = document.getElementById('editor-textarea');
  const gutter = document.getElementById('editor-line-numbers');
  const cursorPos = document.getElementById('editor-cursor-pos');
  const btnRun = document.getElementById('btn-editor-run');
  const btnAsk = document.getElementById('btn-editor-ask');
  const btnCopy = document.getElementById('btn-editor-copy');
  const btnClear = document.getElementById('btn-editor-clear');
  const btnConsoleClear = document.getElementById('btn-console-clear');
  const btnCloseAsk = document.getElementById('btn-close-ask-drawer');

  // Initialize Starter Scratchpad Code
  if (textarea && !textarea.value) {
    textarea.value = STARTER_CODE;
    updateEditorGutter();
  }

  // Auto-Update Gutter & Cursor Tracker
  textarea?.addEventListener('input', updateEditorGutter);
  textarea?.addEventListener('keyup', updateCursorTracking);
  textarea?.addEventListener('click', updateCursorTracking);

  // Tab key indents with 4 spaces instead of losing focus
  textarea?.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      textarea.value = textarea.value.substring(0, start) + '    ' + textarea.value.substring(end);
      textarea.selectionStart = textarea.selectionEnd = start + 4;
      updateEditorGutter();
    } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleRunCode();
    }
  });

  btnRun?.addEventListener('click', handleRunCode);
  btnAsk?.addEventListener('click', toggleWorkspaceAskDrawer);
  btnCloseAsk?.addEventListener('click', () => toggleWorkspaceAskDrawer(false));

  btnCopy?.addEventListener('click', () => {
    if (textarea) copyToClipboard(textarea.value, 'Code copied to clipboard.');
  });
  btnClear?.addEventListener('click', () => {
    if (textarea) {
      textarea.value = '';
      updateEditorGutter();
    }
  });

  btnConsoleClear?.addEventListener('click', () => {
    const consoleOutput = document.getElementById('console-output');
    if (consoleOutput) consoleOutput.textContent = 'Console cleared.';
  });

  // Project Explorer File Selection
  document.querySelectorAll('#workspace-file-tree .file-tree-item').forEach(item => {
    item.addEventListener('click', async () => {
      document.querySelectorAll('#workspace-file-tree .file-tree-item').forEach(f => f.classList.remove('active'));
      item.classList.add('active');

      const fileName = item.getAttribute('data-file');
      const fileType = item.getAttribute('data-type');
      state.workspaceActiveFile = fileName;

      const tabName = document.getElementById('editor-active-tab-name');
      const saveIndicator = document.getElementById('editor-save-indicator');
      if (tabName) tabName.textContent = fileName.split('/').pop();
      if (saveIndicator) saveIndicator.textContent = fileType === 'editable' ? 'Local Scratchpad (Editable)' : 'Project File (Read-Only)';

      if (fileType === 'editable') {
        if (textarea) {
          textarea.readOnly = false;
          textarea.value = STARTER_CODE;
          updateEditorGutter();
        }
      } else {
        if (textarea) {
          textarea.readOnly = true;
          textarea.value = `# Loading ${fileName} in safe read-only viewer...\n`;
          try {
            const content = await invoke('read_project_file', { file_path: fileName });
            textarea.value = content;
          } catch {
            textarea.value = `# Read-only inspection for: ${fileName}\n"""Controlled safe view from project root."""\nimport sys\n\ndef check_system():\n    return True\n`;
          }
          updateEditorGutter();
        }
      }
      updateTopHeader('workspace');
    });
  });

  // Ask Robbie Assistant in Workspace
  setupWorkspaceAskAssistant();
}

function updateEditorGutter() {
  const textarea = document.getElementById('editor-textarea');
  const gutter = document.getElementById('editor-line-numbers');
  if (!textarea || !gutter) return;

  const lineCount = textarea.value.split('\n').length;
  let linesStr = '';
  for (let i = 1; i <= lineCount; i++) {
    linesStr += i + '\n';
  }
  gutter.textContent = linesStr;
}

function updateCursorTracking() {
  const textarea = document.getElementById('editor-textarea');
  const cursorPos = document.getElementById('editor-cursor-pos');
  if (!textarea || !cursorPos) return;

  const val = textarea.value.substr(0, textarea.selectionStart);
  const lines = val.split('\n');
  const row = lines.length;
  const col = lines[lines.length - 1].length + 1;
  cursorPos.textContent = `Ln ${row}, Col ${col}`;
}

async function handleRunCode() {
  const textarea = document.getElementById('editor-textarea');
  const consoleOutput = document.getElementById('console-output');
  const consoleBadge = document.getElementById('console-status-badge');
  if (!textarea || !consoleOutput) return;

  const code = textarea.value;
  consoleOutput.textContent = '>>> Executing Python snippet in sandboxed runtime...\n';
  if (consoleBadge) {
    consoleBadge.textContent = 'Running...';
    consoleBadge.className = 'badge badge-warning';
  }

  const startTime = performance.now();

  try {
    const rawRes = await invoke('execute_code', { code });
    let res = {};
    try {
      res = JSON.parse(rawRes);
    } catch {
      res = { stdout: rawRes, success: true };
    }

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

    if (res.success) {
      consoleOutput.textContent = res.stdout || 'Program exited cleanly with no stdout output.';
      if (consoleBadge) {
        consoleBadge.textContent = `Completed (0) in ${elapsed}s`;
        consoleBadge.className = 'badge badge-success';
      }
    } else {
      consoleOutput.innerHTML = `<span style="color: var(--color-error);">${escapeHTML(res.error || res.stderr || 'Execution failed.')}</span>`;
      if (consoleBadge) {
        consoleBadge.textContent = `Error (1) in ${elapsed}s`;
        consoleBadge.className = 'badge badge-danger';
      }
    }
  } catch (err) {
    consoleOutput.innerHTML = `<span style="color: var(--color-error);">${escapeHTML(err.message || err)}</span>`;
    if (consoleBadge) {
      consoleBadge.textContent = 'Runtime Exception';
      consoleBadge.className = 'badge badge-danger';
    }
  }
}

function toggleWorkspaceAskDrawer(forcedState) {
  const drawer = document.getElementById('dev-robbie-drawer');
  if (!drawer) return;
  state.workspaceAskDrawerOpen = typeof forcedState === 'boolean' ? forcedState : !state.workspaceAskDrawerOpen;
  drawer.style.display = state.workspaceAskDrawerOpen ? 'flex' : 'none';
}

function setupWorkspaceAskAssistant() {
  const input = document.getElementById('ask-drawer-input');
  const btnSend = document.getElementById('btn-ask-drawer-send');
  const box = document.getElementById('ask-response-box');

  async function queryRobbieAboutCode(customPrompt) {
    const textarea = document.getElementById('editor-textarea');
    const code = textarea ? textarea.value : '';
    const q = customPrompt || (input ? input.value.trim() : '');
    if (!q) return;

    if (input) input.value = '';
    if (box) box.innerHTML = '<div style="color: var(--text-dim); display: flex; align-items: center; gap: 6px;"><div class="progress-spinner" style="width: 10px; height: 10px;"></div> Robbie-25M analyzing code...</div>';

    const fullPrompt = `${q}\n\nHere is the code:\n\`\`\`python\n${code}\n\`\`\``;

    try {
      const rawRes = await invoke('ask_robbie', { prompt: fullPrompt });
      let res = {};
      try { res = JSON.parse(rawRes); } catch { res = { reply: rawRes }; }
      if (box) box.innerHTML = renderMarkdown(res.reply || 'Analysis complete.');
    } catch (err) {
      if (box) box.innerHTML = `<div style="color: var(--color-error);">Analysis error: ${escapeHTML(err.message || err)}</div>`;
    }
  }

  btnSend?.addEventListener('click', () => queryRobbieAboutCode());
  input?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') queryRobbieAboutCode();
  });

  document.querySelectorAll('.ask-quick-prompts .pill-btn-xs').forEach(pill => {
    pill.addEventListener('click', () => {
      const prompt = pill.getAttribute('data-prompt');
      queryRobbieAboutCode(prompt);
    });
  });
}

// ============================================================================
// 10. View 5: Memory Manager Controller
// ============================================================================
function setupMemoryController() {
  // Segmented tab switching
  const tabs = document.querySelectorAll('.segmented-tabs .segmented-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const memtab = tab.getAttribute('data-memtab');
      state.memoryActiveTab = memtab;

      document.querySelectorAll('.memory-tab-pane').forEach(p => p.classList.remove('active'));
      const activePane = document.getElementById(`pane-mem-${memtab}`);
      if (activePane) activePane.classList.add('active');
    });
  });

  // Search filter input
  const searchInput = document.getElementById('memory-search-filter');
  searchInput?.addEventListener('input', () => {
    state.memorySearchFilter = searchInput.value.toLowerCase().trim();
    renderActiveMemoryTab();
  });

  // Action buttons
  document.getElementById('btn-add-memory-item')?.addEventListener('click', promptAddMemoryModal);
  document.getElementById('btn-clear-all-memory')?.addEventListener('click', promptClearMemoryModal);
}

async function syncMemoryData() {
  try {
    const rawRes = await invoke('get_memory_profile', {});
    let res = {};
    try { res = JSON.parse(rawRes); } catch { res = {}; }

    state.memory.profile = res.profile || {};
    state.memory.notes = res.notes || {};
    state.memory.episodes = res.episodes || [];

    renderActiveMemoryTab();
  } catch (err) {
    console.warn('[Memory] Sync error:', err);
  }
}

function renderActiveMemoryTab() {
  renderMemoryProfileTable();
  renderMemoryNotesGrid();
  renderMemoryEpisodesTimeline();
  renderMemorySessionSlots();
}

function renderMemoryProfileTable() {
  const tbody = document.getElementById('mem-profile-table-body');
  if (!tbody) return;

  const profile = state.memory.profile || {};
  const filter = state.memorySearchFilter;
  const entries = Object.entries(profile).filter(([k, v]) => {
    if (!filter) return true;
    return k.toLowerCase().includes(filter) || String(v).toLowerCase().includes(filter);
  });

  tbody.innerHTML = '';
  if (!entries.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="table-loading">No profile attributes stored.</td></tr>';
    return;
  }

  entries.forEach(([key, val]) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td><code>${escapeHTML(key)}</code></td>
      <td><strong>${escapeHTML(String(val))}</strong></td>
      <td style="font-family: var(--font-mono); font-size: 11px;">${typeof val}</td>
      <td>
        <button class="btn-text-danger btn-delete-profile-key" data-key="${escapeHTML(key)}">Delete</button>
      </td>
    `;
    row.querySelector('.btn-delete-profile-key')?.addEventListener('click', async () => {
      await invoke('manage_memory', { action: 'delete', key });
      delete state.memory.profile[key];
      renderMemoryProfileTable();
      showToast(`Deleted attribute "${key}".`);
    });
    tbody.appendChild(row);
  });
}

function renderMemoryNotesGrid() {
  const grid = document.getElementById('mem-notes-grid');
  if (!grid) return;

  const notes = state.memory.notes || {};
  const filter = state.memorySearchFilter;
  const entries = Object.entries(notes).filter(([title, body]) => {
    if (!filter) return true;
    return title.toLowerCase().includes(filter) || String(body).toLowerCase().includes(filter);
  });

  grid.innerHTML = '';
  if (!entries.length) {
    grid.innerHTML = '<div class="empty-shelf-message">No notes found. Click "+ Add Item" to save technical notes.</div>';
    return;
  }

  entries.forEach(([title, content]) => {
    const card = document.createElement('div');
    card.className = 'note-card';
    card.innerHTML = `
      <div class="note-card-head">
        <span class="note-card-title">${escapeHTML(title)}</span>
        <button class="btn-text-danger btn-del-note">Delete</button>
      </div>
      <div class="note-card-body">${escapeHTML(String(content))}</div>
    `;
    card.querySelector('.btn-del-note')?.addEventListener('click', async () => {
      await invoke('manage_memory', { action: 'delete', key: title });
      delete state.memory.notes[title];
      renderMemoryNotesGrid();
      showToast(`Deleted note "${title}".`);
    });
    grid.appendChild(card);
  });
}

function renderMemoryEpisodesTimeline() {
  const container = document.getElementById('mem-episodes-timeline');
  if (!container) return;

  const episodes = state.memory.episodes || [];
  container.innerHTML = '';
  if (!episodes.length) {
    container.innerHTML = '<div class="empty-shelf-message">No conversational episodes recorded yet.</div>';
    return;
  }

  episodes.forEach(ep => {
    const item = document.createElement('div');
    item.className = 'episode-item';
    const dateStr = ep.timestamp ? new Date(ep.timestamp * 1000).toLocaleString() : 'Past Session';
    item.innerHTML = `
      <div class="episode-summary">${escapeHTML(ep.summary || '')}</div>
      <div class="episode-date">${dateStr}</div>
    `;
    container.appendChild(item);
  });
}

function renderMemorySessionSlots() {
  const container = document.getElementById('mem-session-slots');
  if (!container) return;

  const slots = state.memory.session || {};
  const entries = Object.entries(slots);
  container.innerHTML = '';
  if (!entries.length) {
    container.innerHTML = '<div class="empty-shelf-message">No active session context slots yet. Start chatting to register entity slots.</div>';
    return;
  }

  entries.forEach(([k, v]) => {
    const card = document.createElement('div');
    card.className = 'slot-card';
    card.innerHTML = `
      <span class="slot-name">${escapeHTML(k)}</span>
      <span class="slot-val">${escapeHTML(String(v))}</span>
    `;
    container.appendChild(card);
  });
}

function promptAddMemoryModal() {
  const html = `
    <div style="display: flex; flex-direction: column; gap: 10px;">
      <div>
        <label style="font-size: 11px; color: var(--text-dim); display: block; margin-bottom: 4px;">Item Type</label>
        <select id="modal-mem-type" class="input-text-sm">
          <option value="profile">Profile Attribute (Key-Value)</option>
          <option value="note">Project Knowledge Note</option>
        </select>
      </div>
      <div>
        <label style="font-size: 11px; color: var(--text-dim); display: block; margin-bottom: 4px;">Key / Title</label>
        <input type="text" id="modal-mem-key" class="input-text-sm" placeholder="e.g. preferred_model, project_milestone" />
      </div>
      <div>
        <label style="font-size: 11px; color: var(--text-dim); display: block; margin-bottom: 4px;">Value / Content</label>
        <textarea id="modal-mem-val" class="input-text-sm" rows="3" placeholder="Enter content..."></textarea>
      </div>
    </div>
  `;

  showModal('Add Memory Item', html, async () => {
    const type = document.getElementById('modal-mem-type')?.value;
    const key = document.getElementById('modal-mem-key')?.value.trim();
    const val = document.getElementById('modal-mem-val')?.value.trim();

    if (key && val) {
      if (type === 'profile') {
        await invoke('manage_memory', { action: 'set', key, value: val });
        state.memory.profile[key] = val;
      } else {
        await invoke('manage_memory', { action: 'add_note', title: key, content: val });
        state.memory.notes[key] = val;
      }
      renderActiveMemoryTab();
      showToast(`Saved memory item: ${key}`);
    }
  });
}

function promptClearMemoryModal() {
  showModal('Clear Persistent Memory', '⚠️ Are you sure you want to delete all long-term user profile attributes, notes, and conversation episodes? This action cannot be undone.', async () => {
    await invoke('manage_memory', { action: 'clear' });
    state.memory = { profile: {}, notes: {}, episodes: [], session: {} };
    renderActiveMemoryTab();
    showToast('Memory database cleared.');
  });
}

// ============================================================================
// 11. View 6: Settings Controller
// ============================================================================
function setupSettingsController() {
  const intervalSelect = document.getElementById('settings-telemetry-interval');
  if (intervalSelect) {
    intervalSelect.value = String(state.telemetryInterval);
    intervalSelect.addEventListener('change', () => {
      state.telemetryInterval = parseInt(intervalSelect.value, 10);
      localStorage.setItem('robbie-telemetry-interval', String(state.telemetryInterval));
      startTelemetryPolling();
      showToast(`Telemetry polling set to ${state.telemetryInterval / 1000}s.`);
    });
  }
}

// ============================================================================
// 12. Hardware Telemetry Polling System
// ============================================================================
async function syncTelemetryData() {
  try {
    const rawRes = await invoke('get_system_telemetry', {});
    let data = {};
    try {
      data = JSON.parse(rawRes);
    } catch {
      data = {};
    }

    state.telemetry = data;

    // Update Top Telemetry Chip & Inline Strip
    const cpuPct = data.cpu ? Math.round(data.cpu.cpu_usage_percent || data.cpu.usage_percent || 0) : 0;
    const ramFree = data.memory ? (data.memory.available_gb || 0).toFixed(1) : 0;
    const ramPct = data.memory ? Math.round(data.memory.percent_used || 0) : 0;

    const inlineCpu = document.getElementById('comp-inline-cpu');
    const inlineRam = document.getElementById('comp-inline-ram');
    if (inlineCpu) inlineCpu.textContent = `${cpuPct}%`;
    if (inlineRam) inlineRam.textContent = `${ramFree} GB Free`;

    // Update Computer Overview Rows
    const cpuProcEl = document.getElementById('comp-cpu-processor');
    const cpuBarEl = document.getElementById('comp-cpu-bar-fill');
    const cpuPctEl = document.getElementById('comp-cpu-pct');
    if (cpuProcEl && data.cpu) cpuProcEl.textContent = `${data.cpu.processor || 'Intel/AMD'} (${data.cpu.logical_cores || 12} Cores)`;
    if (cpuBarEl) cpuBarEl.style.width = `${cpuPct}%`;
    if (cpuPctEl) cpuPctEl.textContent = `${cpuPct}%`;

    const ramDetailEl = document.getElementById('comp-ram-detail');
    const ramBarEl = document.getElementById('comp-ram-bar-fill');
    const ramPctEl = document.getElementById('comp-ram-pct');
    if (ramDetailEl && data.memory) ramDetailEl.textContent = `${data.memory.available_gb} GB Free / ${data.memory.total_gb} GB Total`;
    if (ramBarEl) ramBarEl.style.width = `${ramPct}%`;
    if (ramPctEl) ramPctEl.textContent = `${ramPct}%`;

    const gpuDetailEl = document.getElementById('comp-gpu-detail');
    const gpuBarEl = document.getElementById('comp-gpu-bar-fill');
    const gpuPctEl = document.getElementById('comp-gpu-pct');
    if (data.gpu) {
      const vramTotal = data.gpu.vram_total_gb || 6.0;
      const vramFree = data.gpu.vram_free_gb || 5.9;
      const vramUsed = vramTotal - vramFree;
      const vramPct = Math.round((vramUsed / vramTotal) * 100);
      if (gpuDetailEl) gpuDetailEl.textContent = `${data.gpu.device_name || 'GPU'} (CUDA Active • ${vramFree.toFixed(1)} GB Free / ${vramTotal.toFixed(1)} GB)`;
      if (gpuBarEl) gpuBarEl.style.width = `${vramPct}%`;
      if (gpuPctEl) gpuPctEl.textContent = `${vramPct}%`;
    }

    const diskDetailEl = document.getElementById('comp-disk-detail');
    const diskBarEl = document.getElementById('comp-disk-bar-fill');
    const diskPctEl = document.getElementById('comp-disk-pct');
    if (data.disks && data.disks.length) {
      const primary = data.disks[0];
      const summaryParts = data.disks.map(d => `${d.drive} ${d.free_gb} GB free`).join(' • ');
      if (diskDetailEl) diskDetailEl.textContent = summaryParts;
      if (diskBarEl) diskBarEl.style.width = `${Math.round(primary.percent_used || 50)}%`;
      if (diskPctEl) diskPctEl.textContent = `${Math.round(primary.percent_used || 50)}%`;
    }

    // Update Top Processes Monitor Table
    const procTbody = document.getElementById('processes-table-body');
    if (procTbody && data.processes) {
      procTbody.innerHTML = '';
      data.processes.slice(0, 5).forEach(p => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${p.pid}</td>
          <td><strong>${escapeHTML(p.name)}</strong></td>
          <td>${(p.memory_mb || 0).toFixed(1)} MB</td>
        `;
        procTbody.appendChild(row);
      });
    }

    // Update Timestamp Pill
    const tsPill = document.getElementById('comp-timestamp-pill');
    if (tsPill) tsPill.textContent = new Date().toLocaleTimeString();
  } catch (err) {
    console.warn('[Telemetry] Sync error:', err);
  }
}

function startTelemetryPolling() {
  if (state.telemetryTimer) clearInterval(state.telemetryTimer);
  state.telemetryTimer = setInterval(syncTelemetryData, state.telemetryInterval);
}

// ============================================================================
// 13. UI Utilities: Markdown, Modal, Toast & Helpers
// ============================================================================
function renderMarkdown(md) {
  if (!md) return '';
  let out = escapeHTML(md);

  // Headers
  out = out.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  out = out.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  out = out.replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // Bold & Italic
  out = out.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
  out = out.replace(/\*(.*?)\*/gim, '<em>$1</em>');

  // Code Blocks
  out = out.replace(/```([\s\S]*?)```/gim, (match, code) => {
    return `<div class="code-block-wrapper"><div class="code-block-header"><span>Python / Output</span><button class="btn-copy-code" onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.innerText);">${ICONS.copy} Copy</button></div><pre class="code-block-content">${code.trim()}</pre></div>`;
  });

  // Inline Code
  out = out.replace(/`([^`]+)`/gim, '<code>$1</code>');

  // Lists
  out = out.replace(/^\* (.*$)/gim, '<li>$1</li>');
  out = out.replace(/^• (.*$)/gim, '<li>$1</li>');
  out = out.replace(/((?:<li>.*<\/li>\s*)+)/gim, '<ul>$1</ul>');

  // Paragraphs
  out = out.replace(/\n\n/gim, '</p><p>');
  out = `<p>${out}</p>`;
  out = out.replace(/<p><\/p>/gim, '');

  return out;
}

function escapeHTML(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function copyToClipboard(text, successMsg = 'Copied to clipboard.') {
  navigator.clipboard.writeText(text).then(() => {
    showToast(successMsg);
  });
}

function showModal(title, bodyHTML, onConfirm) {
  const backdrop = document.getElementById('app-modal-backdrop');
  const titleEl = document.getElementById('modal-title');
  const bodyEl = document.getElementById('modal-body');
  const btnConfirm = document.getElementById('modal-btn-confirm');
  const btnCancel = document.getElementById('modal-btn-cancel');

  if (!backdrop) return;
  if (titleEl) titleEl.textContent = title;
  if (bodyEl) bodyEl.innerHTML = bodyHTML;

  backdrop.classList.add('active');

  const cleanup = () => {
    backdrop.classList.remove('active');
    btnConfirm.replaceWith(btnConfirm.cloneNode(true));
    btnCancel.replaceWith(btnCancel.cloneNode(true));
  };

  btnCancel?.addEventListener('click', cleanup);
  btnConfirm?.addEventListener('click', () => {
    cleanup();
    if (onConfirm) onConfirm();
  });
}

function showToast(msg) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = msg;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(6px)';
    toast.style.transition = 'opacity 0.2s, transform 0.2s';
    setTimeout(() => toast.remove(), 200);
  }, 2600);
}
