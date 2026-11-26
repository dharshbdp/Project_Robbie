"""Talk directly to trained Robbie weights, with no canned replies or routing."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from robbie.core.conversation.model_chat import ModelChat
from robbie.core.models.transformer import RobbieTransformer
from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer


def load_chat(checkpoint, max_tokens=160, temperature=0.4, device=None):
    torch.set_num_threads(4)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = RobbieTransformer(payload["config"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    if not 0 < max_tokens < model.config.block_size:
        raise ValueError("max-tokens must be between 1 and the checkpoint context size minus 1")
    tokenizer = BPETokenizer.load(ROOT / "checkpoints/bpe_stage1_vocab.json")
    dialogue_fmt = payload.get("dialogue_format")
    assistant_eos = dialogue_fmt == "assistant-eos-v1" or dialogue_fmt is None
    return ModelChat(model, tokenizer, max_tokens, temperature,
                     assistant_eos=assistant_eos)


def main():
    default_ckpt = ROOT / "checkpoints/stage2_300m_conversational_transformer.pt"
    if not default_ckpt.is_file():
        default_ckpt = ROOT / "checkpoints/chat_300m_v1/best_model.pt"
    if not default_ckpt.is_file():
        default_ckpt = ROOT / "checkpoints/stage2_50m_conversational_transformer.pt"
    if not default_ckpt.is_file():
        default_ckpt = ROOT / "checkpoints/stage2_25m_conversational_transformer.pt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_ckpt)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.4)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found at {args.checkpoint}. Train the 50M model first.")
    chat = load_chat(args.checkpoint, args.max_tokens, args.temperature)
    print(f"Loaded {args.checkpoint}\nDirect model chat. /clear resets the conversation; exit quits.", flush=True)
    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question == "/clear":
            chat.messages.clear()
            continue
        if question:
            try:
                print("Robbie:", chat.reply(question), flush=True)
            except ValueError as error:
                print(error, flush=True)


if __name__ == "__main__":
    main()
