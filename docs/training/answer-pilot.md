# Computer-answer fine-tuning pilot

This run improves the existing 26.3M-parameter checkpoint before attempting a
larger model. It is a small supervised experiment, not broad pretraining and
not a guarantee of correct answers. The corpus contains 48 reviewed computer
and conversation topics, observed-memory examples, and summary follow-ups.
Validation holds out one question phrasing per topic and 24 memory states.
There are 368 assistant training examples and 72 validation examples.

The original checkpoints remain intact. The run extends the context from
128 to 256 tokens, preserves the existing learned positional embeddings, and
learns the additional positions during fine-tuning. Loss is applied only to
assistant answers and a real EOS token, never to user questions or padding.
The best validation epoch remains separate from the final epoch.

## Start in PowerShell

```powershell
Set-Location 'Z:\Project_Robbie\Project_Robbie'
$robbiePython = '.\.venv\Scripts\python.exe'
$robbieRun = '.\checkpoints\answers_25m_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Force '.\logs' | Out-Null

& $robbiePython -u .\scripts\train_answers.py --output-dir $robbieRun --epochs 20 --batch-size 8 --log-every 5 2>&1 | Tee-Object -FilePath '.\logs\answers-training.log'
```

Progress is printed every five batches. Each epoch prints training loss,
validation loss, and validation perplexity. A decreasing validation loss is
useful evidence, but actual generated answers still need review. Early stopping
can stop before 20 epochs if validation stops improving. Ctrl+C interrupts the
run; checkpoints are saved after completed epochs, not every batch. The log is
overwritten on each run, while the timestamped checkpoint folder is unique.

## Review the actual answers

In the same PowerShell session:

```powershell
$robbieCheckpoint = Join-Path $robbieRun 'best_answers.pt'
& $robbiePython -u .\scripts\evaluate_answers.py --checkpoint $robbieCheckpoint --output (Join-Path $robbieRun 'answer_review.json')
```

The evaluation calls the learned model directly with unseen question wording.
It bypasses keyword routing, personality replies, and computer report templates.
It includes two observations with different RAM values; answers should change
with the evidence. It prints outputs for manual review rather than assigning an
unverified accuracy score. The pre-training baseline is stored locally at
`data/processed/answer_baseline.json`.

## Chat with the selected checkpoint

```powershell
& $robbiePython -m robbie.main --chat --checkpoint $robbieCheckpoint --temperature 0 --repetition-penalty 1.0 --max-tokens 100
```

Chat retains the existing live computer tools. `/system`, `/processes`, and `/time` gather
real measurements and format them in code; their success does not demonstrate
learned model quality. Conceptual questions such as "What is RAM?" are routed
to the learned model. Independent concept questions use a focused prompt to
avoid interference from prior tool reports; the complete session history is
retained, and explicit follow-up explanations still receive it.
The observed-memory examples are a training/evaluation
pilot; the chat does not yet attach those observations to general model replies.

To watch GPU usage from another PowerShell window:

```powershell
nvidia-smi -l 2
```

## Verification

```powershell
& $robbiePython .\scripts\train_answers.py --dry-run
& $robbiePython -m pytest -q
```

The setup check loads the checkpoint, builds both datasets, and performs one
forward pass on the available device. It does not optimize or save weights.
