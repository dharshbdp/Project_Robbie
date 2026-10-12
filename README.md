# Project Robbie

> **Status:** Active / Core AI & Agent Modules Operational (115/115 Tests Passing ✅)  
> **Project Type:** Personal AI / Custom LLM / Computer-Aware Agent / Desktop Application  
> **Primary Goal:** Build a personal AI character from the ground up, including a custom LLM, persistent memory, web research, computer awareness, and safe tool execution.

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

All core foundations, transformer models, conversational engine, research pipeline, computer awareness, safe tools, and evaluation systems are built, verified, and operational:

| Phase | Subsystem | Key Components & Artifacts | Status |
|:---|:---|:---|:---:|
| **Phase 1** | **Tokenization** | Character & Byte-Pair Encoding (BPE) tokenizers with special token handling (`<|endoftext|>`, `<|user|>`, `<|assistant|>`). | Completed ✅ |
| **Phase 2** | **Neural Baseline** | Bigram neural model, context embedding lookup, and multi-layer perceptron (MLP). | Completed ✅ |
| **Phase 3** | **Transformer Core** | Multi-Head Self-Attention, causal triangular masking, RMSNorm/LayerNorm, SwiGLU/GELU feedforward blocks, and residual streams. | Completed ✅ |
| **Phase 4** | **Robbie-25M Architecture** | 25-Million parameter scaled causal transformer (`d_model=384`, `n_layers=12`, `n_heads=12`), weight tying, and learning rate scheduling. | Completed ✅ |
| **Phase 5** | **Dialogue & Persona** | Multi-turn sliding window budget, role formatting (`User:` / `Robbie:`), and native Gen-Z conversational personality. | Completed ✅ |
| **Phase 6** | **Dataset & Fine-Tuning** | Ingested 11.1M BPE tokens across pre-training and 9,816 multi-turn Stage 2 dialogues. Achieved **Val Loss: 1.2942** and **Perplexity: 3.65**. | Completed ✅ |
| **Phase 7** | **Memory & Context Separation** | Distinct separation of ephemeral session context from persistent cross-session storage (`robbie_memory.json`). | Completed ✅ |
| **Phase 8** | **Web Research & Grounded Citations** | Multi-source search (DuckDuckGo Lite, Wikipedia API), recency extraction, domain authority ranking, citation validation, and dual Markdown/HTML reports. | Completed ✅ |
| **Phase 9** | **Computer Awareness** | Live hardware telemetry (CPU, RAM, GPU, storage), top process monitor, battery & network state, active window context, and git workspace tracker. | Completed ✅ |
| **Phase 10** | **Safe Computer Tools** | Read-only default guard, path sandboxing (blocking credentials and system roots), dangerous command blacklisting, user confirmations, and audit trail. | Completed ✅ |
| **Phase 11** | **Repeatable Evaluation Suite** | Automated benchmark scorecard testing model diversity, factual grounding, citation integrity, and physical hardware invariants (115/115 tests passing). | Completed ✅ |

---

# 12. Quickstart & Usage

### 12.1 Interactive Terminal Chat
Launch the Robbie conversational assistant:
```powershell
.\.venv\Scripts\python.exe -m robbie.main chat
```

Inside the interactive chat interface, the following commands and natural language triggers are available:

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

### 12.2 Running the Test Suite
Execute the entire test suite across all subsystems:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```
Output:
```text
tests\test_computer_tools.py ..........                                  [  8%]
tests\test_config.py ..                                                  [ 10%]
tests\test_conversation.py ................                              [ 24%]
tests\test_data_pipeline.py ......                                       [ 29%]
tests\test_device.py ..                                                  [ 31%]
tests\test_evaluation_suite.py .........                                 [ 39%]
tests\test_inspector.py .                                                [ 40%]
tests\test_research.py ....................                              [ 57%]
tests\test_system.py .............                                       [ 68%]
tests\test_tiny_llm.py .......                                           [ 74%]
tests\test_tokenizer.py ................                                 [ 88%]
tests\test_transformer.py .............                                  [100%]
============================ 115 passed in 15.98s =============================
```

### 12.3 Running the Unified Evaluation Benchmark
```powershell
.\.venv\Scripts\python.exe -c "from robbie.evaluation import RobbieEvaluationSuite; print(RobbieEvaluationSuite.run_all().to_markdown())"
```

---

# 13. Project Status

**Current Version:** 0.10 — Computer-Aware Agent  
**Next Milestone:** 1.0 — Desktop Application & Optional Voice Interaction  
**Test Pass Rate:** 115 / 115 Passed (100%)  
**Hardware Verification:** NVIDIA Quadro RTX 3000 (CUDA 12.8) / Windows 10

