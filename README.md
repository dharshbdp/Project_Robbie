# Project Robbie

> **Status:** Active / Robbie-300M Transformer & Desktop Studio Operational (212/212 Tests Passing ✅)  
> **Project Type:** Personal AI / Custom 300M LLM / Computer-Aware Agent / Desktop Application & Code Studio  
> **Primary Goal:** Build a personal AI character from the ground up, including a custom 300M Transformer LLM, persistent memory, web research, computer awareness, interactive desktop code execution, and safe tool use.

---

## 1. Project Overview

**Project Robbie** is a long-term personal side project to build an AI character named **Robbie**.

Robbie is intended to be more than a chatbot. The final system should combine:

- A custom language model trained from scratch
- A distinct and consistent AI personality
- Conversational memory
- Web research and current-information retrieval
- Awareness of the user's computer and its current state
- Controlled access to files, applications, and system information
- Tool use and agent-style decision making
- Optional voice interaction
- A desktop interface

The project is intentionally ambitious and will be developed incrementally.

The objective is **not** to recreate a frontier model such as GPT-5. The objective is to understand and build the major pieces of a modern AI system ourselves, then combine them into one coherent personal assistant.

---

# 2. Vision

The long-term vision for Robbie is:

> **A personal AI character that can converse naturally, understand its user, research information, remember relevant context, understand what is happening on the user's computer, and use controlled tools to help with everyday tasks.**

Robbie should feel like a persistent character rather than a generic AI interface.

For example:

**User:**  
"Robbie, why is my computer running so slowly?"

**Robbie:**  
"Give me a second."

Robbie checks system information and reports:

- CPU usage
- RAM usage
- GPU usage
- disk activity
- running processes
- active applications

Then it explains what appears relevant.

Another example:

**User:**  
"Robbie, what's new with ESP32 development?"

Robbie searches current sources, reads relevant information, compares sources, and summarizes the changes.

Another:

**User:**  
"Robbie, what was I working on yesterday?"

Robbie searches its permitted memory and project context and provides the relevant information.

---

# 3. Core Principles

### 3.1 Build, Don't Just Integrate

Where practical, core AI components should be implemented and understood by us rather than treated as black boxes.

This includes learning and implementing:

- Tokenization
- Embeddings
- Self-attention
- Transformer blocks
- Training
- Inference
- Fine-tuning
- Evaluation

External libraries may still be used for infrastructure and acceleration.

### 3.2 Incremental Development

Robbie will be built in small, working versions.

Every major version should produce something usable or demonstrably functional.

### 3.3 Modular Architecture

The LLM, memory, tools, computer-awareness system, web research system, voice system, and desktop interface should remain separate components.

This allows individual components to be improved or replaced without rebuilding the entire project.

### 3.4 Controlled Computer Access

Robbie should never have unrestricted access to the computer by default.

Actions should be exposed through explicit tools and permissions.

Examples:

| Tool | Default Permission |
|---|---|
| Read system statistics | Allowed |
| Read active window | Allowed |
| Search files | Allowed |
| Read files | Allowed |
| Open application | Confirmation |
| Modify files | Confirmation |
| Execute commands | Confirmation |
| Delete files | Explicit confirmation |
| Shutdown/restart | Explicit confirmation |

---

# 4. High-Level Architecture

```text
                         ┌───────────────────────┐
                         │        ROBBIE         │
                         │     AI Character      │
                         └───────────┬───────────┘
                                     │
                         ┌───────────▼───────────┐
                         │       AI Agent        │
                         │                       │
                         │ Planning / Tool Use   │
                         └───────────┬───────────┘
                                     │
             ┌───────────────────────┼───────────────────────┐
             │                       │                       │
             ▼                       ▼                       ▼
       ┌───────────┐           ┌───────────┐           ┌───────────┐
       │   LLM     │           │  Memory   │           │   Tools   │
       │           │           │           │           │           │
       │ Transformer│          │ Short-term │          │ Web       │
       │ Model     │           │ Long-term  │          │ Computer  │
       └─────┬─────┘           └───────────┘           │ Files     │
             │                                         │ Apps      │
             │                                         └─────┬─────┘
             │                                               │
             └───────────────────────┬───────────────────────┘
                                     │
                           ┌─────────▼─────────┐
                           │   User Interface  │
                           │ Desktop / Voice   │
                           └───────────────────┘
```

---

# 5. Development Phases

## Phase 0 — Project Foundation

### Goal

Establish the project, development environment, repository structure, documentation, and development philosophy.

### Work

- Create repository
- Establish project structure
- Set up Python environment
- Set up Git
- Establish experiment tracking
- Create initial documentation
- Decide hardware constraints
- Create testing strategy

### Deliverable

A clean project foundation ready for AI/LLM development.

---

# Phase 1 — Understanding and Building the Tokenizer

### Goal

Build the first component of the language model.

### Work

- Understand tokens and vocabulary
- Experiment with character-level tokenization
- Experiment with subword tokenization
- Build a tokenizer
- Encode text into token IDs
- Decode token IDs back into text
- Build vocabulary handling
- Create tokenizer tests

### Version

**Robbie 0.1 — Tokenizer**

### Deliverable

A functioning tokenizer that Robbie's future model can use.

---

# Phase 2 — First Language Model

### Goal

Build a very small language model capable of predicting the next token.

### Work

- Dataset preparation
- Embeddings
- Positional information
- Output projection
- Next-token prediction
- Cross-entropy loss
- Training loop
- Validation loop
- Sampling

Initially, the model will probably produce nonsense.

That is expected.

### Version

**Robbie 0.2 — Tiny Language Model**

### Deliverable

A small model capable of learning statistical patterns in text.

---

# Phase 3 — Transformer From Scratch

### Goal

Implement the architecture that forms the foundation of modern LLMs.

### Work

- Self-attention
- Causal masking
- Multi-head attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Transformer blocks
- Stacked transformer architecture
- Model configuration system
- Checkpoint saving/loading

### Version

**Robbie 0.3 — Transformer**

### Deliverable

A working small Transformer language model implemented and understood from the ground up.

---

# Phase 4 — Robbie Base Model

### Goal

Train a more capable model using a larger and cleaner dataset.

### Work

- Dataset collection
- Dataset cleaning
- Deduplication
- Data filtering
- Train/validation/test splits
- Larger model configuration
- GPU training
- Checkpoint management
- Evaluation
- Generation benchmarks

### Version

**Robbie 0.5 — Base Model**

### Deliverable

A usable small-scale pretrained language model.

The exact parameter count will depend on available hardware and training results.

---

# Phase 5 — Conversational Robbie

### Goal

Turn the base language model into a conversational model.

### Work

- Conversation formatting
- Instruction data
- Dialogue datasets
- Supervised fine-tuning
- Conversation evaluation
- Response formatting
- Context handling
- System/user/assistant roles

### Version

**Robbie 0.7 — Conversational Model**

### Deliverable

Robbie can hold basic conversations and follow instructions.

---

# Phase 6 — Robbie's Character

### Goal

Give Robbie a stable personality and identity.

### Work

Define:

- Name
- Personality
- Speaking style
- Humor
- Social behavior
- Boundaries
- Character traits
- Preferred response style
- Interaction rules

Training data will include examples demonstrating the desired behavior.

### Version

**Robbie 0.8 — Character**

### Deliverable

Robbie begins to behave consistently as a distinct AI character rather than simply acting as a generic assistant.

---

# Phase 7 — Memory System

### Goal

Give Robbie persistent, controlled memory.

### Memory Types

#### Short-Term Memory

Current conversation and immediate context.

#### Long-Term Memory

Information deliberately retained across conversations.

#### Semantic Memory

Searchable knowledge stored outside the model itself.

### Work

- Memory schema
- Memory storage
- Memory retrieval
- Relevance ranking
- Memory summarization
- Memory deletion
- User-controlled memory
- Context injection

### Version

**Robbie 0.9 — Memory**

### Deliverable

Robbie can maintain useful continuity across conversations.

---

# Phase 8 — Web Research

### Goal

Allow Robbie to obtain current information from the internet.

### Tools

- Web search
- Page retrieval
- Text extraction
- Source comparison
- Citation handling
- Research summarization

### Example

```text
User
  ↓
"What's new in ESP32 development?"
  ↓
Robbie decides research is required
  ↓
Search web
  ↓
Read sources
  ↓
Compare information
  ↓
Summarize
  ↓
Answer with sources
```

### Version

**Robbie 1.0 — Research Assistant**

### Deliverable

Robbie can perform current web research rather than relying entirely on its training data.

---

# Phase 9 — Computer Awareness

### Goal

Allow Robbie to understand what is happening on the user's computer.

### Possible Context

- CPU usage
- GPU usage
- RAM
- Storage
- Network
- Battery
- Running processes
- Active window
- Open applications
- Connected devices
- Recent files
- Development environment
- Terminal state
- Project/workspace information

### Version

**Robbie 1.1 — Computer Awareness**

### Deliverable

Robbie can answer questions about the current state of the computer.

Example:

> "What's using all my RAM?"

Robbie gathers system information and explains the relevant processes.

---

# Phase 10 — Computer Tools

### Goal

Allow Robbie to interact with the computer through controlled tools.

### Possible Tools

```text
read_system_stats()
get_active_window()
list_processes()
search_files()
read_file()
open_application()
open_file()
run_command()
modify_file()
```

Dangerous operations should require explicit user confirmation.

### Version

**Robbie 1.2 — Computer Agent**

### Deliverable

Robbie can perform useful computer-related actions through a permission-controlled tool system.

---

# Phase 11 — Agentic Robbie

### Goal

Allow Robbie to decide which tools it needs to accomplish a task.

Instead of:

```text
User → Tool → User
```

Robbie can perform:

```text
User
 ↓
Robbie
 ↓
Determine required information
 ↓
Select tools
 ↓
Execute tools
 ↓
Evaluate results
 ↓
Use another tool if required
 ↓
Generate response
```

### Version

**Robbie 1.5 — Agent**

### Deliverable

Robbie can independently perform multi-step tasks within its permitted environment.

---

# Phase 12 — Voice

### Goal

Make conversation more natural.

### Components

- Speech-to-text
- Voice activity detection
- Text generation
- Text-to-speech
- Voice interruption handling
- Wake word / activation system

### Version

**Robbie 1.6 — Voice**

### Deliverable

The user can speak naturally with Robbie.

---

# Phase 13 — Desktop Application

### Goal

Create Robbie's dedicated interface.

Possible technology:

- Electron
- React
- TypeScript

### Interface

Potential components:

- Conversation panel
- Current activity
- Memory viewer
- Tool activity
- System status
- Research sources
- Permissions
- Settings
- Model controls
- Voice controls

### Version

**Robbie 1.8 — Desktop**

### Deliverable

A dedicated Robbie desktop application.

---

# Phase 14 — Integrated Robbie

### Goal

Combine the major components into one stable system.

### Version

**Robbie 2.0**

```text
                    ┌─────────────────┐
                    │     ROBBIE      │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
      LLM                 Memory                Agent
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
           Web            Computer          Files
         Research         Awareness         & Apps
            │                │                │
            └────────────────┼────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Desktop + Voice │
                    └─────────────────┘
```

---

# 6. Version Roadmap

| Version | Milestone |
|---|---|
| 0.1 | Tokenizer |
| 0.2 | Tiny Language Model |
| 0.3 | Transformer |
| 0.5 | Base Language Model |
| 0.7 | Conversational Model |
| 0.8 | Character |
| 0.9 | Memory |
| 1.0 | Web Research |
| 1.1 | Computer Awareness |
| 1.2 | Computer Tools |
| 1.5 | Agent |
| 1.6 | Voice |
| 1.8 | Desktop Application |
| **2.0** | **Integrated Robbie** |

These versions are milestones rather than strict release dates.

---

# 7. What Robbie Should Be Able to Do at the End

A successful Robbie 2.0 should be able to:

### Conversation

- Hold natural conversations
- Maintain conversational context
- Follow instructions
- Maintain a consistent personality
- Explain technical subjects
- Discuss projects and ideas

### Memory

- Remember useful user-provided information
- Retrieve relevant past information
- Maintain project context
- Allow the user to inspect and control memory

### Research

- Search the internet
- Read web pages
- Compare multiple sources
- Answer questions about current information
- Identify sources used during research

### Computer Awareness

- Know what applications are running
- Identify the active application
- Read system statistics
- Understand basic computer state
- Inspect permitted project files
- Understand development workspaces

### Computer Assistance

- Search files
- Open applications
- Open files
- Read project files
- Run approved commands
- Assist with debugging
- Perform multi-step computer tasks

### Voice

- Listen to the user
- Speak responses
- Handle natural back-and-forth conversation
- Optionally use a wake word

### Personality

Robbie should feel like a persistent character.

Not merely:

> "How may I assist you?"

But something closer to:

> "You have three problems here. Fortunately, only one of them is actually interesting."

The personality should remain useful rather than becoming an obstacle to getting things done.

---

# 8. What Robbie Will NOT Initially Be

Robbie is not intended to initially be:

- A frontier-scale LLM
- A replacement for every existing AI model
- A fully autonomous computer controller
- An unrestricted system administrator
- A perfect source of factual information
- A sentient system
- A system that can safely perform arbitrary actions without confirmation

The project will prioritize **understanding, control, experimentation, and usefulness** over raw model scale.

---

# 9. Initial Technology Direction

The exact stack can evolve, but the initial direction is:

### AI / Model

- Python
- PyTorch
- Custom tokenizer
- Custom Transformer implementation
- CUDA/GPU acceleration where available

### Data

- Text datasets
- Conversation datasets
- Curated character/personality data
- Evaluation datasets

### Memory

Potentially:

- SQLite
- Vector database / embeddings
- Structured memory store

### Application

- Electron
- React
- TypeScript

### Computer Integration

Platform-specific APIs and controlled subprocess/system interfaces.

### Web Research

Search and webpage retrieval tools.

### Voice

Speech-to-text and text-to-speech components can be introduced later.

---

# 10. Development Philosophy

The project should follow a simple rule:

> **Understand first. Optimize second. Scale third.**

We should not jump immediately into a massive model.

Instead:

```text
Understand
    ↓
Implement
    ↓
Test
    ↓
Train
    ↓
Evaluate
    ↓
Improve
    ↓
Scale
```

Every major AI component should have a small experiment demonstrating that it works before it becomes part of Robbie.

---

# 11. Implementation Progress & Milestone Matrix

All core foundations, transformer models, conversational engine, research pipeline, computer awareness, safe tools, desktop application, and evaluation systems are built, verified, and operational:

| Phase | Subsystem | Key Components & Artifacts | Status |
|:---|:---|:---|:---:|
| **Phase 1** | **Tokenization** | Character & Byte-Pair Encoding (BPE) tokenizers with special token handling (`<|endoftext|>`, `<|user|>`, `<|assistant|>`, `<|pad|>`). Vocab: 2,048 tokens (`checkpoints/bpe_stage1_vocab.json`). | Completed ✅ |
| **Phase 2** | **Neural Baseline** | Bigram neural model, context embedding lookup, and multi-layer perceptron (MLP). | Completed ✅ |
| **Phase 3** | **Transformer Core** | Multi-Head Self-Attention, causal triangular masking, RMSNorm/LayerNorm, SwiGLU/GELU feedforward blocks, and residual streams. | Completed ✅ |
| **Phase 4** | **Robbie-300M Architecture** | Scaled transformer from 25M $\rightarrow$ 50M $\rightarrow$ **304.8M parameters** (`n_layer=24`, `n_head=16`, `n_embd=1024`, `block_size=512`), weight tying, cosine learning rate decay, and AMP FP16. | Completed ✅ |
| **Phase 5** | **Dialogue & Persona** | Multi-turn sliding window budget, role formatting (`User:` / `Robbie:`), and consistent conversational persona. | Completed ✅ |
| **Phase 6** | **Dataset & SFT Fine-Tuning** | Ingested `alespalla/chatbot_instruction_prompts` (258k pairs) combined with Robbie curated persona, computer systems Q&A, and Dolly/No-Robots datasets. Achieved **Val Loss: 1.5449** and **Perplexity: 4.69** on Robbie-300M under strict 4.8 GB VRAM guard. | Completed ✅ |
| **Phase 7** | **Memory & Context Separation** | Distinct separation of ephemeral session context from persistent cross-session storage (`robbie_memory.json`). | Completed ✅ |
| **Phase 8** | **Web Research & Grounded Citations** | Multi-source search (DuckDuckGo Lite, Wikipedia API), recency extraction, domain authority ranking, citation validation, and dual Markdown/HTML reports. | Completed ✅ |
| **Phase 9** | **Computer Awareness** | Live hardware telemetry (CPU, RAM, GPU, storage), top process monitor, battery & network state, active window context, and git workspace tracker. | Completed ✅ |
| **Phase 10** | **Safe Computer Tools** | Read-only default guard, path sandboxing (blocking credentials and system roots), dangerous command blacklisting, user confirmations, and audit trail. | Completed ✅ |
| **Phase 11** | **Repeatable Evaluation Suite** | Automated benchmark scorecard testing model diversity, factual grounding, citation integrity, and physical hardware invariants (**212/212 tests passing**). | Completed ✅ |
| **Phase 13** | **Desktop Application & Code Space** | **Robbie Studio** built with Tauri v2 + Rust backend + modern web UI. Features live Python code blocks with `▷ Run` executor, syntax highlighting, hardware telemetry, memory inspector, and research source viewer. | Completed ✅ |

---

# 12. End-to-End Model Training Guide & Reproducible Steps

Project Robbie supports full end-to-end training of custom transformer models from scratch—scaling from lightweight 25M/50M baselines up to **Robbie-300M (~304.8M parameters)**.

### 12.1 Hardware Safety & VRAM Safeguards
Training 300M parameters on consumer or mobile GPUs (such as 6 GB laptop GPUs like the NVIDIA Quadro RTX 3000) requires strict memory and thermal pacing safeguards:
* **Hard VRAM Allocator Ceiling (`SafeVRAMGuard`)**: Limits CUDA memory allocation to **$\le$ 4.8 GB** (leaving $>1.2\text{ GB}$ of dedicated video memory for Windows DWM and display tasks to avoid out-of-memory crashes).
* **8-Bit AdamW Optimizer (`bitsandbytes`)**: Stores optimizer first and second momentum states in 8-bit precision instead of 32-bit float, reducing optimizer memory overhead from $\approx 3.6\text{ GB}$ down to $\approx 600\text{ MB}$.
* **Automatic Mixed Precision (AMP FP16)**: Forward activations and tensor core matmuls run in half-precision with `torch.amp.GradScaler`, doubling compute throughput and halving activation memory.
* **Thermal Pacing Cooldown Cycles**: Every 100 training steps, a scheduled 12-second pause triggers CUDA cache flushing and allows GPU silicon temps to stabilize, preventing thermal throttling.

```text
+--------------------------------------------------------------------------------+
|                         Hardware & VRAM Architecture                           |
|                                                                                |
|  Total GPU VRAM: 6.0 GB                                                        |
|  +--------------------------------------------+-----------------------------+  |
|  | Robbie-300M Allocation (Max 4.8 GB Cap)     | OS / Display / DWM Overhead |  |
|  |  * Model Weights (FP16): ~610 MB           | (Reserved: ~1.2 GB)         |  |
|  |  * 8-Bit AdamW Optimizer: ~600 MB          |                             |  |
|  |  * Micro-Batch Activations: ~800 MB        |                             |  |
|  |  * CUDA Workspace & Buffers: ~400 MB       |                             |  |
|  +--------------------------------------------+-----------------------------+  |
+--------------------------------------------------------------------------------+
```

---

### 12.2 Step 1: Train the Byte-Pair Encoding (BPE) Tokenizer
The custom BPE tokenizer builds subword representations from raw corpora, generating vocabulary and merge tables:
```powershell
# Run from Project_Robbie directory
.\.venv\Scripts\python.exe scripts/train_bpe.py --vocab-size 2048 --output checkpoints/bpe_stage1_vocab.json
```
* **Output**: `checkpoints/bpe_stage1_vocab.json`
* **Special Tokens Registered**: `<|endoftext|>`, `<|user|>`, `<|assistant|>`, `<|pad|>`
* **Coverage**: Complete ASCII, Python syntax tokens, and common conversational subwords.

---

### 12.3 Step 2: Ingest & Prepare the SFT Training Curriculum
Prepare the multi-stage conversational dataset by combining instruction data, character identity, and domain knowledge:
```powershell
.\.venv\Scripts\python.exe scripts/prepare_300m_curriculum.py
```
This script automatically:
1. Streams and extracts instruction pairs from `alespalla/chatbot_instruction_prompts` (258,000 instruction-response pairs).
2. Blends Robbie's core personality, character rules, and conversational etiquette (`data/curated/robbie_50m_curated.json`, `data/curated/chat_basics.json`).
3. Integrates computer systems, hardware diagnostics, and Python programming Q&A (`data/curated/computer_answers.json`).
4. Ingests clean multi-turn dialogue from Dolly-15k and No-Robots.
5. Formats pairs into `<|user|> ... <|assistant|> ... <|endoftext|>` token streams.
6. Applies **assistant-only loss masking** (user prompts receive target index `-100` so loss is computed solely on the assistant's generated output).
7. Outputs prepared train and validation tensors to `data/processed/chat_300m_v1/answers/`.

---

### 12.4 Step 3: Launch Robbie-300M Training
Train the 304.8M parameter transformer model using the dedicated training harness:
```powershell
.\.venv\Scripts\python.exe scripts/train_300m_chat.py `
  --data-dir data/processed/chat_300m_v1/answers `
  --output-dir checkpoints/chat_300m_v1 `
  --epochs 6 `
  --batch-size 2 `
  --accumulation-steps 16 `
  --lr 1.8e-4 `
  --min-lr 1.8e-5 `
  --vram-limit-gb 4.8 `
  --eval-every 200
```

Alternatively, you can run the provided one-click Windows batch script:
```cmd
.\train_300m.bat
```

#### Training Parameters & Configuration:
* **Model Dimensions**: `n_layer=24`, `n_head=16` (head dim 64), `n_embd=1024`, `block_size=512`.
* **Parameter Count**: $304,834,560$ parameters.
* **Effective Batch Size**: $\text{Micro Batch } (2) \times \text{Accumulation Steps } (16) = 32$.
* **Learning Rate Schedule**: Cosine annealing decay from $1.8 \times 10^{-4}$ down to $1.8 \times 10^{-5}$ with a 4% linear warmup.
* **Validation & Checkpointing**: Evaluates validation loss every 200 steps. Automatically saves the checkpoint whenever validation loss improves:
  * Checkpoint path: `checkpoints/chat_300m_v1/best_model.pt`
  * Example milestone: Step 5,600 achieved **Val Loss: 1.5449** and **Perplexity: 4.69**.

---

### 12.5 Step 4: Register & Deploy Canonical Checkpoint
Copy or link the best trained checkpoint to Robbie's canonical model path so the terminal assistant and desktop editor pick it up automatically:
```powershell
Copy-Item "checkpoints/chat_300m_v1/best_model.pt" "checkpoints/stage2_300m_conversational_transformer.pt"
```

The model loader hierarchy in `robbie.main` and `editor/bridge.py` automatically detects and prioritizes models in the following order:
1. `checkpoints/stage2_300m_conversational_transformer.pt` (Robbie-300M Canonical)
2. `checkpoints/chat_300m_v1/best_model.pt` (Robbie-300M Best Val Checkpoint)
3. `checkpoints/stage2_50m_conversational_transformer.pt` (Robbie-50M Baseline)
4. `checkpoints/stage2_25m_conversational_transformer.pt` (Robbie-25M Baseline)

---

### 12.6 Step 5: Quantization & Deployment Modes
Robbie supports multiple precision formats depending on target hardware constraints:
* **16-bit Master Checkpoints (FP16/BF16)**: The canonical trained format ($\approx 1.2\text{ GB}$ on disk). Preserves all gradient and weight fidelities for resumed fine-tuning and reference evaluation.
* **8-bit Quantization (INT8)**: Reduces runtime memory footprint to $\approx 310\text{ MB}$ with $<0.5\%$ perplexity deviation.
* **4-bit Quantization (INT4 / NF4)**: Fits Robbie-300M into $\approx 160\text{ MB}$ of VRAM, enabling execution on low-power edge devices and ultra-thin laptops.

> [!NOTE]
> Always retain the 16-bit master `.pt` checkpoint file. Quantization to 4-bit or 8-bit is a lossy deployment step; further fine-tuning or scaling must always branch from the 16-bit master weights.

---

# 13. Quickstart & Usage

### 13.1 Robbie Studio Desktop Application (Recommended)
Robbie Studio is the modern desktop interface featuring an interactive chat workspace, live computer telemetry dashboard, persistent memory manager, and an executable Python Code Space:

```powershell
# From the Project_Robbie directory
.\launch_editor.bat
```
Key features in Robbie Studio:
* **Executable Python Code Blocks**: Code snippets render in dedicated dark graphite containers (`#282a2e`) with syntax highlighting, copy button, send-to-editor button, and an interactive **`▷ Run` button** that executes code in a sandboxed local Python process and displays live stdout/stderr.
* **Hardware Telemetry**: Displays real-time GPU VRAM headroom, CPU core loads, storage capacity, and thermal states.
* **Memory & Research**: Inspect and edit long-term memory facts or trigger grounded web research with Markdown & HTML synthesis.

---

### 13.2 Interactive Terminal Chat
Launch the conversational assistant directly in PowerShell or Bash:
```powershell
.\.venv\Scripts\python.exe -m robbie.main chat
```

Available slash commands and natural language triggers:
* `/help` — Display list of commands.
* `/system` or `/sys` — Live computer awareness report (hardware, power, workspace, active window).
* `/processes [N]` — Inspect top N running processes ranked by RAM usage.
* `/search <query>` — Multi-source web search with domain authority ranking.
* `/research <topic>` — In-depth web research with webpage scraping, grounded citations, and HTML report export.
* `/memory` — Inspect active session working context vs. persistent disk storage.
* `/memory set <key> <val>` — Store a fact in long-term memory.
* `/clear` — Reset active conversation turn history while preserving persistent profile facts.
* Natural language queries:
  * *"Why is my computer running slowly?"* $\rightarrow$ Analyzes CPU, RAM, and top processes.
  * *"Research transformer self-attention"* $\rightarrow$ Executes grounded research.
  * *"What is my name?"* or *"What do you remember about me?"* $\rightarrow$ Retrieves stored context.

---

### 13.3 Running the Test Suite
Execute the entire test suite across all subsystems:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```
Output:
```text
============================== test session starts ==============================
rootdir: Z:\Project_Robbie\Project_Robbie
configfile: pyproject.toml
plugins: anyio-4.14.2
collected 212 items

........................................................................ [ 33%]
........................................................................ [ 67%]
....................................................................     [100%]
============================== 212 passed in 29.21s ==============================
```

To run the frontend test suite for Robbie Studio:
```powershell
cd editor
npm test
```

---

### 13.4 Running the Unified Evaluation Benchmark
```powershell
.\.venv\Scripts\python.exe -c "from robbie.evaluation import RobbieEvaluationSuite; print(RobbieEvaluationSuite.run_all().to_markdown())"
```

---

# 14. Project Status

**Current Version:** 0.12 — Robbie-300M Model & Desktop Studio  
**Active Checkpoint:** `checkpoints/stage2_300m_conversational_transformer.pt` (304.8M parameters)  
**Test Pass Rate:** 212 / 212 Passed (100%)  
**Hardware Verification:** NVIDIA Quadro RTX 3000 (CUDA 12.8, 6.0 GB VRAM) / Windows 10 & 11  
**Desktop Environment:** Robbie Studio (Tauri v2 + Rust + Obsidian Glass Web UI)

