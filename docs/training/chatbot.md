# Train Robbie on broad conversations

This run trains Robbie's own approximately 26M-parameter transformer. It does
not replace it with a downloaded model or enlarge it to 85M. The base checkpoint
has limited pretraining, so instruction tuning is a measurable next experiment,
not a promise of a broadly capable assistant. More pretraining may be necessary.

**Observed result for `chat_25m_v1`: unsuccessful chatbot experiment.** All
2,865 updates completed, and validation answer loss fell from 4.8766 to 1.7418.
Reserved test answer loss was 1.6014, yet generated answers repeated phrases,
invented Python functions, and confused basic computer concepts. This checkpoint
is not a usable chatbot. See [the diagnosis](chat-v1-diagnosis.md). Do not interpret
these losses as answer accuracy or repeat the same run expecting a proven fix.

## Prepared data

`scripts/prepare_chat.py` downloads pinned revisions of:

- [SmolTalk for small models](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), Apache-2.0, synthetic instruction conversations.
- [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1), Apache-2.0, reviewed English human conversations. Preparation selects rank-zero assistant branches.

Both datasets' original terms and provenance remain applicable. Download
revisions, file hashes, source counts, tokenizer hash, and split statistics are
recorded in `data/processed/chat_v1/manifest.json`.

A seeded reservoir samples 100,000 SmolTalk conversations across **all four**
training shards; eligible OpenAssistant paths are added. Full answers that fit
512 tokens of question, context, and answer become training examples. Earlier
exchanges can be dropped; the immediately preceding exchange, current question,
system instruction, and answer are preserved. Follow-ups are skipped when that
context cannot fit. This favors short answers and does not represent the full datasets.
Only assistant answer tokens and an explicit EOS contribute to the loss. Dynamic
padding avoids computing a full 512 tokens for every short example.

Opening user prompts are normalized and hashed into 97% train, 2% validation,
and 1% test groups. Branches of one opening prompt stay in one split. Exact
tokenized samples are deduplicated. This does not detect semantic paraphrases;
held-out loss is not a general reasoning or factual accuracy benchmark.

The old RAM templates and slang corpus are not mixed into this run. General
conversation SFT does not teach a complete live-computer tool policy. Real process
lists and clock values still require observations from the computer.

## PowerShell

Run from the repository directory (the inner `Project_Robbie` folder):

```powershell
Set-Location 'Z:\Project_Robbie\Project_Robbie'
.\.venv\Scripts\python.exe -u scripts\train_chat.py
```

Close the editor/other GPU model sessions first to free VRAM. Defaults use the
original **stage-1 base**, rather than the tiny overfit answer checkpoint:

- 512-token context; existing positional embeddings preserved, new positions learned.
- 3 epochs, microbatch 4, 8 accumulation steps (effective batch 32 answers).
- AdamW, peak learning rate 0.0001, 5% warmup, cosine decay, gradient clipping.
- BF16 on supported CUDA hardware, FP16 with scaling otherwise; float32 on CPU.
- Training progress every 10 updates; save every 100; validation and raw sample
  replies every 250 updates and at epoch ends. Step-zero validation establishes a baseline.

`best_chat.pt` contains the model with the lowest validation answer loss.
`last.pt` contains weights, optimizer/scaler state, learning-rate position,
shuffle position, and RNG states. Writes replace files atomically. Original
checkpoints are preserved. The training and data configuration is saved in
`run.json`; live metrics are appended to `events.jsonl`; sample replies are saved
as `samples_*.json`.

Subsequent runs also write `generation_*.json` and print an explicit failure
when sample outputs are empty or contain obvious phrase loops. This is a
topic-independent repetition check, not an answer rule or a correctness grader.
Passing it means the answers still need manual review. It does not certify readiness.

Press **Ctrl+C once** to finish the current optimizer update and save. After a
forced termination or power loss, resume from the last completed periodic save:

```powershell
.\.venv\Scripts\python.exe -u scripts\train_chat.py --resume checkpoints\chat_25m_v1\last.pt
```

Resume restores the saved training settings, including batch size and schedule;
it refuses changed data or architecture. CUDA floating-point kernels may vary
slightly across runs/devices. To run a different experiment, choose a new output
directory. If VRAM is insufficient, use a **new** run with microbatch 2 and
accumulation 16. This keeps the effective batch at 32.

## Judge the learned model

```powershell
.\.venv\Scripts\python.exe scripts\chat_model.py --checkpoint checkpoints\chat_25m_v1\best_chat.pt
.\.venv\Scripts\python.exe scripts\evaluate_chat.py --checkpoint checkpoints\chat_25m_v1\best_chat.pt
```

The chat script always generates from the weights and retains conversation
history. It does not use the application's greeting, RAM, memory, or tool
response branches. `/clear` resets the conversation. Long questions are rejected
with an explanation when they cannot fit the prompt budget.

Evaluation reads the reserved test split and prints multi-turn replies generated
using the model's own previous answers. Review relevance, factual correctness,
stopping, and follow-up understanding. A falling loss alone is insufficient.
Do not tune repeatedly against the final test set; use validation and separate
development conversations for iteration.

After training, restart the desktop editor to load the most recently saved best
checkpoint from `chat_*` or `answers_*` directories. An explicit
`ROBBIE_CHECKPOINT` environment variable takes precedence. The editor retains
its existing live-computer tools; use the direct chat command above to assess
the learned model without those response branches.

## Rebuild the setup elsewhere

The environment needs its existing CUDA PyTorch installation plus the `training`
extra (`pyarrow>=19,<24`). Prepare once, then check a full-length forward/backward
pass without updating or saving model weights:

```powershell
.\.venv\Scripts\python.exe -u scripts\prepare_chat.py
.\.venv\Scripts\python.exe -u scripts\train_chat.py --dry-run
```

Preparation refuses to overwrite a completed dataset, and training refuses to
overwrite an existing run. Downloaded files are cached by dataset revision. Pick
a new `--output-dir`/`--data-dir` for a different preparation; keep active runs'
data immutable. Dataset and checkpoint artifacts are Git-ignored.
