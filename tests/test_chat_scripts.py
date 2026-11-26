"""Check actual model chat and public dataset conversion without downloading."""

import importlib.util
from pathlib import Path

import pytest
import torch

from robbie.core.tokenizer.char_tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_chat_generates_every_reply_and_keeps_previous_answer():
    from types import SimpleNamespace
    module = script("chat_model")
    tok = CharTokenizer()
    tok.train(["User: Robbie: Hi Hello Why? What is RAM? \n"])

    class Model:
        config = SimpleNamespace(block_size=256)
        prompts = []

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def generate(self, ids, **kwargs):
            self.prompts.append(tok.decode(ids[0].tolist(), skip_special_tokens=True))
            suffix = torch.tensor([tok.encode(" Hello") + [tok.eos_token_id]])
            return torch.cat([ids, suffix], dim=1)

    model = Model()
    chat = module.ModelChat(model, tok, max_tokens=32, temperature=0)
    assert chat.reply("Hi") == "Hello"
    assert chat.reply("Why?") == "Hello"
    assert "User: Hi\nRobbie: Hello\nUser: Why?\nRobbie:" == model.prompts[-1]
    # A hardware keyword still reaches the model in this assessment command.
    chat.reply("What is RAM?")
    assert len(model.prompts) == 3
    before = list(chat.messages)
    with pytest.raises(ValueError, match="prompt budget"):
        chat.reply("a" * 300)
    assert chat.messages == before


def test_oasst_uses_reviewed_english_best_rank_answers(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    module = script("prepare_chat")
    rows = []
    for name, parent, role, text, rank, lang, reviewed in [
        ("root", None, "prompter", "Hi", None, "en", True),
        ("best", "root", "assistant", "Hello", 0, "en", True),
        ("other", "root", "assistant", "Other", 1, "en", True),
        ("unreviewed", "root", "assistant", "Unreviewed", 0, "en", False),
        ("french", "root", "assistant", "Bonjour", 0, "fr", True),
    ]:
        rows.append({"message_id": name, "parent_id": parent, "role": role,
                     "text": text, "rank": rank, "lang": lang, "review_result": reviewed,
                     "deleted": False, "synthetic": False})
    path = tmp_path / "oasst.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert module.oasst_records(path) == [{"messages": [
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
    ], "source": "oasst1/human"}]


def test_reservoir_reads_every_shard_and_is_reproducible(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    module = script("prepare_chat")
    paths = []
    for shard in range(4):
        path = tmp_path / f"{shard}.parquet"
        rows = [{"messages": [{"role": "user", "content": f"Question {shard}/{i}"},
                               {"role": "assistant", "content": "Answer"}], "source": str(shard)}
                for i in range(20)]
        pq.write_table(pa.Table.from_pylist(rows), path)
        paths.append(path)
    first = module.smol_records(paths, maximum=20, seed=42)
    assert first == module.smol_records(paths, maximum=20, seed=42)
    assert {row["source"] for row in first} == {f"smol/{i}" for i in range(4)}
