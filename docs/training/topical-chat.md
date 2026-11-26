# Topical-Chat comparison candidate

This experiment adds a conservative subset of human dialogue to the language
stage and preserves the existing clean assistant-reply training set. It is a
small data comparison, not an established fix for Robbie's off-topic replies.

## What was obtained

The original [Amazon Alexa Topical-Chat repository](https://github.com/alexa/Topical-Chat)
was downloaded at revision `7c939229cbcf6f55f6977b341a5a2f2fe982d53f`.
The 95.5 MB download contains all five official conversation splits, available
reading-set metadata, Wikipedia text, the source README, and original notices.
Downloads are cached in `data/raw/chat/Topical-Chat/<revision>`; hashes are recorded
in `download_manifest.json`. No downloaded source code is executed.

The dataset's default agreement is CDLA-Sharing-1.0. Wikipedia content has a
separate notice identifying CC BY-SA 3.0. The original agreement and notices are
preserved. The prepared dataset carries an attribution/modification notice;
its baseline portions retain their original source terms.

## Selection and limitations

The official reading-set metadata refers to Reddit posts and news articles whose
text is not included locally. A section marked FS can contain both available
Wikipedia text and unavailable Reddit facts, so an FS tag is not treated as
proof of complete grounding.

The importer therefore retains only four consecutive turns that:

- belong to a conversation rated Good or Excellent by both speakers;
- each have a Good or Excellent turn rating;
- each have only the `Personal Knowledge` source annotation;
- fit the existing short-turn word/token/unknown-token limits and context.

No messages are joined over rejected gaps. Speaker order, conversation IDs,
original turn indices, and source splits are retained. Only one speaker
orientation is used. Exact window duplicates are removed with held-out examples
taking precedence. Source IDs cannot cross official splits.

The source annotations are not independent truth checks. A selected window can
still refer to earlier conversation, contain informal grammar, or express a
human's personal experience or opinion. Reviewed training examples exhibited
these limitations. Consequently, **Topical-Chat is used only for language
adaptation; it adds no assistant-only instruction targets** in this experiment.

## Prepared counts

| Stage | Train | Validation | Test |
|---|---:|---:|---:|
| Language adaptation, baseline plus selected Topical-Chat | 4,900 | 251 | 289 |
| Assistant replies, unchanged baseline inputs | 4,527 | 217 | 226 |

Topical-Chat adds 373 train windows, 34 validation windows, and 63 test windows.
Its frequent/rare validation and test categories are retained in per-example
metadata. The reply split files are copied byte for byte from the baseline.

The language stage grows from 1,245,524 to 1,315,482 supervised tokens per epoch.
This modest increase does not replace substantial language pretraining. In
particular, the saved baseline samples at update 1,136 already show EOS stopping
but off-topic or nonsensical answers. There is no evidence yet that this variant
will repair those failures; a full retrain should be treated as an optional
comparison, not a promised improvement.

The processed files live in `data/processed/clean_chat_topical_v1`.
`topical_filter_report.json` records exclusions; `topical_review_examples.json`
contains training-only windows for inspection. The baseline data and checkpoints
are preserved.

## Reproduce the setup

```powershell
Set-Location 'Z:\Project_Robbie\Project_Robbie'
.\.venv\Scripts\python.exe -u scripts\download_topical_chat.py
.\.venv\Scripts\python.exe -u scripts\prepare_topical_chat.py
.\.venv\Scripts\python.exe -u scripts\train_clean_chat.py --data-dir data\processed\clean_chat_topical_v1 --output-dir checkpoints\clean_chat_topical_v1 --dry-run
```

Preparation refuses to overwrite an existing experiment. Both objectives passed
the full-length CUDA forward/backward setup check without optimizer updates or
saved model weights. This verifies execution, not chatbot quality.

## Optional comparison run

Training has not been started by the setup. To run this candidate yourself:

```powershell
.\.venv\Scripts\python.exe -u scripts\train_clean_chat.py --data-dir data\processed\clean_chat_topical_v1 --output-dir checkpoints\clean_chat_topical_v1
```

The command uses the same 8-epoch language stage and 8-epoch reply stage as the
baseline. Ctrl+C requests a save; running the same command resumes the experiment.
It uses the original stage-1 checkpoint, not the existing baseline's trained
weights, so the initial model is held constant for the comparison.

After training, assess generated conversation from the exact candidate:

```powershell
.\.venv\Scripts\python.exe scripts\chat_model.py --checkpoint checkpoints\clean_chat_topical_v1\replies\best_chat.pt
```

Only after development review, evaluate on the reserved baseline reply test set:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_chat.py --checkpoint checkpoints\clean_chat_topical_v1\replies\best_chat.pt --data-dir data\processed\clean_chat_topical_v1\answers --profile basic
```

The shared reply test inputs support a comparison of reply behavior. Changed
language-stage validation mixtures do not provide directly comparable language
losses; added training tokens also change compute exposure. Neither a low loss
nor a passed repetition/stopping check establishes relevance or correctness.
