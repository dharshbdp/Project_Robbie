# Project Robbie — Architecture & Foundation

## 1. System Overview

Project Robbie is a personal AI system designed as an interconnected, modular hierarchy:

```text
                     ┌───────────────────────┐
                     │        ROBBIE         │
                     │     AI Character      │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │       AI Agent        │
                     │ Planning / Tool Use   │
                     └───────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
   ┌───────────┐           ┌───────────┐           ┌───────────┐
   │    LLM    │           │  Memory   │           │   Tools   │
   │Transformer│           │Short/Long │           │Web/OS/Apps│
   └─────┬─────┘           └───────────┘           └─────┬─────┘
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 │
                       ┌─────────▼─────────┐
                       │  Desktop / Voice  │
                       └───────────────────┘
```

## 2. Directory Layout

- `src/robbie/core/`: Language modeling subsystem
  - `tokenizer/`: Custom text tokenizers, vocabularies, BPE encoders/decoders (Phase 1)
  - `models/`: Neural architectures from Tiny LLM to Transformer blocks (Phases 2-4)
  - `training/`: Training loops, loss calculation, validation, checkpointing (Phases 2-4)
  - `conversation/`: Conversation formatting, instruction tuning, personality (Phases 5-6)
- `src/robbie/memory/`: Conversational context and persistent vector memory (Phase 7)
- `src/robbie/research/`: Search, web page retrieval, multi-source citations (Phase 8)
- `src/robbie/computer/`:
  - `awareness/`: System telemetry (CPU, RAM, GPU, processes, active window) (Phase 9)
  - `tools/`: Permission-controlled commands, application launcher, file ops (Phase 10)
- `src/robbie/agent/`: Autonomous multi-step planning, tool selection, error recovery (Phase 11)
- `src/robbie/voice/`: Speech-to-text, text-to-speech, wake word (Phase 12)
- `src/robbie/interface/`: Desktop bridge and API (Phase 13)
- `configs/`: YAML configurations for training, models, and runtime
- `data/`: Datasets (`raw/`, `processed/`)
- `checkpoints/`: Model weights and training checkpoints
- `tests/`: Automated unit and integration test suites

## 3. Hardware Profile

- **GPU:** NVIDIA Quadro RTX 3000 (6 GB VRAM, Turing architecture, CUDA compute capability 7.5)
- **CPU:** Intel Core i7-10850H (6 Cores, 12 Threads @ 2.70 GHz)
- **RAM:** 32 GB DDR4
- **Storage:** Dedicated high-capacity volume (Z: Drive, >240 GB free)
