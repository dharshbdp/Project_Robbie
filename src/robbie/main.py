"""Main CLI entrypoint for Project Robbie."""

import sys
import argparse
from robbie import __version__
from robbie.config import RobbieConfig
from robbie.utils.logger import logger
from robbie.utils.device import get_hardware_info
from robbie.core.tokenizer import CharTokenizer, BPETokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Project Robbie CLI")
    parser.add_argument("--version", action="version", version=f"Project Robbie v{__version__}")
    parser.add_argument("--info", action="store_true", help="Print system and hardware report")
    parser.add_argument(
        "--demo-tokenizer",
        action="store_true",
        help="Demonstrate and compare CharTokenizer vs BPETokenizer",
    )
    return parser.parse_args()

def run_tokenizer_demo():
    print("\n" + "=" * 50)
    print("      PROJECT ROBBIE — TOKENIZER DEMONSTRATION")
    print("=" * 50)

    sample_training_corpus = [
        "Robbie is a personal AI character built from the ground up.",
        "Robbie has conversational memory, web research, and tool awareness.",
        "The language model learns to predict the next token in sequence.",
        "Deep learning transformers process attention across tokens.",
    ]
    test_phrase = "Robbie is learning language models and deep attention!"

    print(f"\n[Test Input String]:\n\"{test_phrase}\"\n")

    # 1. Character Tokenizer
    char_tok = CharTokenizer()
    char_tok.train(sample_training_corpus)
    char_encoded = char_tok.encode(test_phrase, add_special_tokens=True)
    char_decoded = char_tok.decode(char_encoded, skip_special_tokens=True)

    print("-" * 50)
    print("1. Character-Level Tokenizer (CharTokenizer)")
    print(f"   Vocabulary Size:    {char_tok.vocab_size} tokens")
    print(f"   Encoded Sequence:   {len(char_encoded)} token IDs")
    print(f"   Token IDs (first 15): {char_encoded[:15]}...")
    print(f"   Decoded Text:       \"{char_decoded}\"")

    # 2. BPE Subword Tokenizer
    bpe_tok = BPETokenizer(target_vocab_size=120)
    bpe_tok.train(sample_training_corpus)
    bpe_encoded = bpe_tok.encode(test_phrase, add_special_tokens=True)
    bpe_decoded = bpe_tok.decode(bpe_encoded, skip_special_tokens=True)

    print("\n2. Byte-Pair Encoding Subword Tokenizer (BPETokenizer)")
    print(f"   Learned Merges:     {len(bpe_tok.merges)} merges")
    print(f"   Vocabulary Size:    {bpe_tok.vocab_size} tokens")
    print(f"   Encoded Sequence:   {len(bpe_encoded)} token IDs")
    print(f"   Token IDs:          {bpe_encoded}")
    print(f"   Decoded Text:       \"{bpe_decoded}\"")
    
    compression = (1 - (len(bpe_encoded) / len(char_encoded))) * 100
    print(f"\n   Sequence Compression: {compression:.1f}% fewer tokens than character-level!")
    print("=" * 50 + "\n")

def main():
    args = parse_args()
    logger.info(f"Starting Project Robbie (v{__version__})")

    if args.info:
        report = get_hardware_info()
        print("\n=== Project Robbie Hardware Report ===")
        print(f"Device Type:     {report.device_type}")
        print(f"Device Name:     {report.device_name}")
        print(f"CUDA Available:  {report.cuda_available}")
        print(f"GPU Count:       {report.device_count}")
        if report.cuda_available:
            print(f"Total VRAM:      {report.vram_total_gb} GB")
            print(f"Available VRAM:  {report.vram_free_gb} GB")
        print("======================================\n")

    if args.demo_tokenizer:
        run_tokenizer_demo()

if __name__ == "__main__":
    main()
