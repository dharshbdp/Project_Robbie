"""Protect source split boundaries, grounding filters, and the clean baseline."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from robbie.core.data.chat_data import digest_file, encode_conversation
from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer
from robbie.core.tokenizer.char_tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def importer(monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("prepare_topical_test", ROOT / "scripts/prepare_topical_chat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokenizer():
    tok = CharTokenizer()
    tok.train(["".join(chr(i) for i in range(32, 127)) + "\n"])
    return tok


def conversation(topic="drawing", first="agent_1", pairs=2):
    other = "agent_2" if first == "agent_1" else "agent_1"
    messages = [f"Do you enjoy {topic}?", f"Yes I enjoy {topic}.",
                "What do you like about it?", "It is relaxing."] * (pairs // 2)
    return {"conversation_rating": {"agent_1": "Good", "agent_2": "Excellent"},
            "content": [{"agent": first if i % 2 == 0 else other, "message": text,
                         "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]}
                        for i, text in enumerate(messages)]}


def test_grounding_filter_rejects_partial_or_missing_sources(importer):
    assert importer.personal_only({"knowledge_source": ["Personal Knowledge"]})
    for tags in [[], ["FS1"], ["AS2"], ["Personal Knowledge", "FS3"], ["Unknown"]]:
        assert not importer.personal_only({"knowledge_source": tags})
    assert not importer.personal_only({})


def test_windows_preserve_original_turns_and_do_not_stitch_over_grounding_gaps(importer):
    from collections import Counter
    value = conversation(pairs=4)
    value["content"][2]["knowledge_source"] = ["FS1"]
    counters = Counter()
    selected = list(importer.selected_windows(value, tokenizer(), counters))
    assert len(selected) == 1
    assert selected[0][1] == [4, 5, 6, 7]
    assert counters["window_external_or_unknown_grounding_rejected"] == 2


def test_original_first_speaker_defines_user_without_mirroring_or_reordering(importer):
    from collections import Counter
    selected = list(importer.selected_windows(conversation(first="agent_2"), tokenizer(), Counter()))
    assert selected[0][2] == "agent_1"
    assert [message["role"] for message in selected[0][0]] == ["user", "assistant", "user", "assistant"]


def test_quality_and_speaker_order_rejections(importer):
    from collections import Counter
    bad = conversation()
    bad["conversation_rating"]["agent_2"] = "Poor"
    assert not list(importer.selected_windows(bad, tokenizer(), Counter()))
    bad = conversation()
    bad["content"][2]["agent"] = "agent_2"
    assert not list(importer.selected_windows(bad, tokenizer(), Counter()))
    bad = conversation()
    bad["content"][1]["turn_rating"] = "Not Good"
    assert not list(importer.selected_windows(bad, tokenizer(), Counter()))


def test_turn_indices_distinguish_a_skipped_final_target():
    turns = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
             {"role": "user", "content": "Why?"}, {"role": "assistant", "content": "a" * 600}]
    samples = encode_conversation(turns, tokenizer(), 256, assistant_eos=True, with_turn_index=True)
    assert [sample["assistant_turn"] for sample in samples] == [1]


def test_full_import_keeps_official_splits_and_copies_reply_data_unchanged(importer, tmp_path, monkeypatch):
    from prepare_clean_chat import write_stage
    monkeypatch.setattr(importer, "ROOT", tmp_path)
    vocab = tmp_path / "checkpoints/bpe_stage1_vocab.json"
    tok = BPETokenizer(target_vocab_size=128)
    tok.train(["Do you enjoy drawing? Yes I enjoy drawing. What do you like about it? It is relaxing."])
    tok.save(vocab)
    baseline = tmp_path / "baseline"
    records = {name: [{"tokens": torch.tensor([4, 5 + index, 7, 3], dtype=torch.int32),
                      "answer_start": 1, "source": "baseline/fixture"}]
               for index, name in enumerate(("train", "validation", "test"))}
    groups = {name: {"baseline:" + name} for name in records}
    for stage, objective in [("dialogue", "dialogue_lm"), ("answers", "assistant_sft")]:
        write_stage(baseline / stage, records, groups, vocab, {}, 512, objective)
    raw = tmp_path / "raw"
    (raw / "conversations").mkdir(parents=True)
    (raw / "src/wiki").mkdir(parents=True)
    for filename in ("README.md", "DATALICENSE", "src/wiki/NOTICE.md"):
        (raw / filename).write_text("Original source notice", encoding="utf-8")
    files = []
    for source_split in importer.SPLITS:
        path = raw / "conversations" / (source_split + ".json")
        entries = {source_split + ":unique": conversation(topic=source_split.replace("_", " "))}
        if source_split == "train":
            # A train conversation with different ID but identical test window
            # is excluded, rather than leaking the held-out window into train.
            entries["train:duplicate"] = deepcopy(conversation(topic="test freq"))
        path.write_text(json.dumps(entries), encoding="utf-8")
        files.append({"path": str(path.relative_to(raw)), "sha256": digest_file(path)})
    (raw / "download_manifest.json").write_text(json.dumps({"revision": importer.REVISION, "files": files}), encoding="utf-8")
    output = tmp_path / "experiment"
    monkeypatch.setattr(importer.sys, "argv", ["prepare_topical_chat.py", "--raw-dir", str(raw),
                                              "--baseline-dir", str(baseline), "--output-dir", str(output)])
    importer.main()
    for name in records:
        assert digest_file(output / "answers" / (name + ".pt")) == digest_file(baseline / "answers" / (name + ".pt"))
    samples = {name: torch.load(output / "dialogue" / (name + ".pt"), weights_only=True) for name in records}
    assert [len(samples[name]) for name in ("train", "validation", "test")] == [2, 3, 3]
    assert samples["train"][-1]["original_split"] == "train"
    assert samples["train"][-1]["original_turn_indices"] == [0, 1, 2, 3]
    assert samples["train"][-1]["answer_start"] == 0
    stats = json.loads((output / "topical_filter_report.json").read_text())
    assert stats["train"]["exact_window_duplicate_rejected"] == 1
    assert (output / "TOPICAL_CHAT_DATALICENSE").read_text() == "Original source notice"
