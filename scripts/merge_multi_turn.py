"""Merge sequential follow-ups into genuine multi-turn conversations in robbie_50m_curated.json."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
curated_path = ROOT / "data/curated/robbie_50m_curated.json"

raw = json.loads(curated_path.read_text(encoding="utf-8"))
print(f"Total entries before: {len(raw)}")

# Explicit multi-turn dialogues to include
multi_turns = [
    {
        "conversation": [
            {"role": "user", "content": "What is the difference between RAM and storage?"},
            {"role": "assistant", "content": "RAM (Random Access Memory) is fast, temporary working memory that holds data currently in use by active programs, which is wiped when your computer restarts. Storage (such as an SSD or hard drive) is non-volatile, meaning it permanently preserves your files, documents, and programs even when powered off."},
            {"role": "user", "content": "Explain that more simply."},
            {"role": "assistant", "content": "Think of RAM as the top of your work desk where you keep tools you are actively using right now. Storage is the filing cabinet where you put things away to keep them permanently."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "What makes a good friend?"},
            {"role": "assistant", "content": "A good friend offers honesty, mutual trust, reliable support during difficult times, and active listening without judgment."},
            {"role": "user", "content": "Please say that in one short sentence."},
            {"role": "assistant", "content": "A good friend is someone you can trust, who supports you and values you for who you are."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "I have a folder of photos to organize. Suggest three steps."},
            {"role": "assistant", "content": "Here are three simple steps:\n1. Delete duplicates, blurry shots, and screenshots.\n2. Group the remaining photos into folders sorted by year, month, or event.\n3. Back up the organized folder to an external SSD or cloud storage."},
            {"role": "user", "content": "Can you summarize those steps in one sentence?"},
            {"role": "assistant", "content": "Remove unwanted duplicates, group the keepers into date-based folders, and create a secure backup."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "I have a free afternoon. Suggest something simple to do."},
            {"role": "assistant", "content": "You could take a refreshing walk in a nearby park, read a few chapters of an engaging book, sketch in a notebook, or try preparing a simple new recipe."},
            {"role": "user", "content": "I would rather stay indoors. Any other idea?"},
            {"role": "assistant", "content": "Staying indoors can be wonderful! You could listen to your favorite music, organize your desk, build a mini programming project, or watch an inspiring documentary."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "I want to learn Python. Where should I begin?"},
            {"role": "assistant", "content": "Begin by installing Python and writing basic scripts. Master variables, basic data types (strings, integers, lists, dictionaries), and control structures like `if` conditions and `for` loops. Build small interactive programs like a guessing game or calculator to gain confidence."},
            {"role": "user", "content": "How do I create a virtual environment in Python?"},
            {"role": "assistant", "content": "Run `python -m venv .venv` in your project folder. On Windows, activate it with `.venv\\Scripts\\activate`. On Linux or macOS, run `source .venv/bin/activate`."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "What is Git?"},
            {"role": "assistant", "content": "Git is a version control system that records changes to your source code over time, making it easy to create branches, review history, and collaborate without losing work."},
            {"role": "user", "content": "What does git status do?"},
            {"role": "assistant", "content": "`git status` displays the current branch and shows modified, staged, or untracked files in your working directory."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why do programmers always prefer dark mode? Because light attracts bugs!"},
            {"role": "user", "content": "Tell me another joke."},
            {"role": "assistant", "content": "There are 10 types of people in the world: those who understand binary, and those who don't!"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "What is RAM?"},
            {"role": "assistant", "content": "RAM (Random Access Memory) is high-speed temporary working memory that your computer uses to hold data and program instructions currently in use. When you shut down or restart, RAM is cleared."},
            {"role": "user", "content": "Why is it faster than storage?"},
            {"role": "assistant", "content": "RAM connects directly to the CPU's memory controller via a high-speed memory bus, providing nanosecond access times with gigabytes-per-second throughput, whereas storage drives must traverse slower bus interfaces."}
        ]
    }
]

# Keep single-turns, but remove single-turns that are meaningless out of context like "Explain that more simply." alone
clean_entries = []
for entry in raw:
    first_q = entry["conversation"][0]["content"].strip().lower()
    if first_q in ("explain that more simply.", "please say that in one short sentence.", "can you summarize those steps in one sentence?", "i would rather stay indoors. any other idea?", "tell me another joke."):
        continue
    clean_entries.append(entry)

final_entries = clean_entries + multi_turns
print(f"Total entries after merging: {len(final_entries)}")
curated_path.write_text(json.dumps(final_entries, indent=2), encoding="utf-8")
print(f"Successfully updated {curated_path}")
