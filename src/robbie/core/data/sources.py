"""Dataset acquisition and source loaders for Stage 1 (Pretrain) and Stage 2 (Dialogue)."""

import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from robbie.utils.logger import logger

# Public mirrors for WikiText-2
WIKITEXT_TRAIN_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
WIKITEXT_VALID_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"

# Stage 1 Offline Fallback Corpus (Clean, multi-domain encyclopedia articles)
STAGE1_FALLBACK_ARTICLES = [
    """ = Artificial Intelligence = 
Artificial intelligence is the intelligence of machines or software, as opposed to the intelligence of living beings, primarily of humans. It is a field of study in computer science that develops and studies intelligent machines. Such machines may be called AIs.
AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines, recommendation systems, understanding human speech, autonomous vehicles, and generative neural networks.
Alan Turing was the first to conduct substantial research in machine learning. In his 1950 paper, Turing considered whether machines can think and introduced the Turing test. Deep learning utilizes artificial neural networks with multiple layers to extract higher-level features progressively from raw input.
Modern language models process text by converting words into numeric token vectors. Through multi-head self-attention mechanisms, transformers model dependencies across tokens regardless of their positional distance in the sequence.""",

    """ = Computer Programming = 
Computer programming is the process of performing specific computations, usually by designing and building executable computer programs. Programming involves tasks such as analysis, generating algorithms, profiling algorithms' accuracy and resource consumption, and the implementation of algorithms.
The source code of a program is written in one or more programming languages. The purpose of programming is to find a sequence of instructions that will automate the performance of a task on a computer.
Different programming languages support different styles of programming. Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.""",

    """ = Solar System = 
The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud.
The vast majority of the system's mass is in the Sun, with the remainder contained mainly in the planet Jupiter. The four inner system planets—Mercury, Venus, Earth, and Mars—are terrestrial planets, being composed primarily of rock and metal. The four giant planets of the outer system are substantially larger and more massive than the terrestrials.
The two largest, Jupiter and Saturn, are gas giants, being composed mainly of hydrogen and helium; the next two, Uranus and Neptune, are ice giants, being composed mostly of volatile substances with relatively high melting points compared with hydrogen and helium, such as water, ammonia, and methane.""",

    """ = Deep Learning = 
Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.
Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks, convolutional neural networks and transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, and bioinformatics.
In transformer architectures, multi-head causal attention allows the model to simultaneously attend to information from different representation subspaces at different positions. Positional embeddings are added to token embeddings to convey sequence order.""",
]

# Stage 2 Curated Multi-Turn Dialogue & Assistant Instruction Pairs
STAGE2_ASSISTANT_DIALOGUES = [
    {
        "conversation": [
            {"role": "user", "content": "Hello! Who are you and what can you do?"},
            {"role": "assistant", "content": "Hello! I am Robbie, your personal artificial intelligence assistant. I am designed to assist you with reasoning, answering questions, writing and debugging code, organizing information, and executing tasks."},
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "Can you explain what a transformer attention mechanism is in simple terms?"},
            {"role": "assistant", "content": "Certainly! An attention mechanism allows a language model to decide which previous words are most relevant to predicting the next word. Instead of reading sequentially like a human, attention creates direct mathematical connections between all words in its context window, focusing on the most important relationships regardless of how far apart they are."},
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "How do I reverse a list in Python?"},
            {"role": "assistant", "content": "In Python, you can reverse a list in several ways:\n1. Using slicing: `reversed_list = my_list[::-1]` (creates a new reversed list).\n2. Using the `.reverse()` method: `my_list.reverse()` (reverses the list in-place).\n3. Using the `reversed()` iterator: `reversed_list = list(reversed(my_list))`."},
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "What is the difference between training loss and validation loss?"},
            {"role": "assistant", "content": "Training loss measures how well the model predicts data it is actively studying and optimizing on. Validation loss measures how well the model generalizes to completely new, unseen data. If training loss decreases while validation loss increases, the model is starting to overfit."},
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "What is an epoch in neural network training?"},
            {"role": "assistant", "content": "An epoch represents one complete pass through the entire training dataset. For example, if a dataset contains 10,000 examples and you train for 5 epochs, the network will have examined every example 5 times during training."},
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "Good morning Robbie. What is on our agenda today?"},
            {"role": "assistant", "content": "Good morning! Today we are working on Project Robbie's training pipeline, preparing Stage 1 base pretraining and Stage 2 conversational tuning. How would you like to start?"},
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "Can you check if this function has any errors?\n```python\ndef add(a, b):\nreturn a + b\n```"},
            {"role": "assistant", "content": "Yes, there is an `IndentationError` on line 2. The return statement must be indented inside the function body:\n\n```python\ndef add(a, b):\n    return a + b\n```"},
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "What is Byte-Pair Encoding (BPE)?"},
            {"role": "assistant", "content": "Byte-Pair Encoding (BPE) is a subword tokenization algorithm. It begins with individual characters and iteratively merges the most frequently occurring pairs of characters or subwords into new tokens. This allows common words to be represented by a single token while rare words are broken down into familiar syllables, keeping vocabulary size compact and eliminating out-of-vocabulary errors."},
        ]
    },
]


def download_file(url: str, dest_path: Path, timeout: int = 15) -> bool:
    """Download a file with timeout and logging."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Downloading from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Project-Robbie/0.4"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(dest_path, "wb") as f:
                f.write(response.read())
        logger.info(f"Successfully downloaded to {dest_path} ({dest_path.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as e:
        logger.warning(f"Failed to download from {url}: {e}")
        return False


def fetch_stage1_pretrain_data(raw_dir: Path, allow_download: bool = True) -> Tuple[Path, Path]:
    """Acquire Stage 1 Pretraining data (WikiText-2 or clean fallback)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_dest = raw_dir / "wikitext_train.txt"
    val_dest = raw_dir / "wikitext_val.txt"

    # Attempt download if allowed and not present
    if not train_dest.exists():
        success = download_file(WIKITEXT_TRAIN_URL, train_dest) if allow_download else False
        if not success:
            logger.info("Using local Stage 1 fallback encyclopedia corpus.")
            with open(train_dest, "w", encoding="utf-8") as f:
                f.write("\n\n".join(STAGE1_FALLBACK_ARTICLES * 200))

    if not val_dest.exists():
        success = download_file(WIKITEXT_VALID_URL, val_dest) if allow_download else False
        if not success:
            with open(val_dest, "w", encoding="utf-8") as f:
                f.write("\n\n".join(STAGE1_FALLBACK_ARTICLES * 20))

    return train_dest, val_dest


def fetch_stage2_dialogue_data(raw_dir: Path) -> Path:
    """Acquire or export Stage 2 Conversational & Assistant instruction data."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest_file = raw_dir / "assistant_dialogues.jsonl"

    if not dest_file.exists() or dest_file.stat().st_size == 0:
        logger.info(f"Writing curated Stage 2 assistant dialogues to {dest_file}")
        with open(dest_file, "w", encoding="utf-8") as f:
            for item in STAGE2_ASSISTANT_DIALOGUES:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return dest_file
