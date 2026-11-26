# Chat v1 diagnosis

The requested training completed, but the resulting model did not become a
usable chatbot. No replacement training run was started during this diagnosis.
The trained weights and original checkpoints are preserved.

## Evidence

- The best checkpoint and final checkpoint both identify update 2,865, epoch 3.
- Validation answer loss fell from 4.8766 to 1.7418.
- Evaluation on 305 reserved test examples produced answer loss 1.6014.
- Actual multi-turn generations repeatedly looped and invented facts and APIs.
  The raw outputs and repetition measurements are in
  `checkpoints/chat_25m_v1/chat_review.json`.
- Separate checks at temperatures 0, 0.8, and 1.0 did not restore correct answers.
  Higher temperature sometimes reduced repetition while retaining factual errors.
- The stage-1 base continued "Once upon a time," into a coherent simple story,
  but its continuation of "A computer uses RAM to" was not a sound explanation.
- The stage-1 corpus metadata lists approximately 17.9 million characters and
  3.45 million words from WikiText and a TinyStories sample. These are corpus
  sizes, not a count of total training-token exposures across epochs.
- Inspected training examples retain complete answers and EOS and use the
  intended next-token shift and assistant-only supervision. All training token
  IDs were within the tokenizer vocabulary. The data integrity checks and
  accumulation/resume regression tests passed.

## Interpretation and limits

The measured failure is weak question answering and unstable generation after
fine-tuning. Insufficient and narrow base pretraining is a plausible major
contributor. The existing checks do not isolate that factor from data difficulty,
optimization choices, or the expanded positional context. There is no evidence
that another identical three epochs or an 85M architecture would fix it.

Teacher-forced loss supplies the correct preceding answer tokens at every
prediction. Free generation uses the model's own preceding predictions; this
experiment demonstrates that low held-out token loss alone does not certify
coherent, relevant, or correct replies.

Small model size alone is not a diagnosis: the
[TinyStories research](https://arxiv.org/abs/2305.07759) demonstrates fluent
generation in a deliberately narrow domain with much smaller models. That
result does not establish broad assistant ability for Robbie.

## Next experiment requirements

Keep Robbie's custom architecture, but establish a base-model checkpoint that
can generate coherent text beyond simple stories before broad assistant tuning.
Use broader, relevant language pretraining and a short-answer conversation
curriculum; compare candidates using held-out prompts and actual generated
answers. Treat corpus size, training duration, and curriculum choices as
experimental settings, not guarantees. Verify learning on a small controlled
subset before committing to a long run, and preserve a separate held-out set.

Live RAM usage is a separate integration requirement: a direct model-only chat
has no current memory observation. A correct assistant must obtain that reading
from the computer through the existing tools, rather than inventing a value.

Generation repetition checks now run alongside training samples and evaluation.
They identify this class of visible failure without adding question-specific
responses. Semantic correctness still requires review.
