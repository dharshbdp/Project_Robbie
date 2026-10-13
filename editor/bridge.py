"""Robbie Desktop Studio / Tauri JSON IPC Bridge.

Allows the Tauri Rust backend and Desktop UI to communicate seamlessly
with Robbie's ChatEngine, ComputerAwareness, PersistentMemory, and ResearchAssistant.
"""

import sys
import json
from dataclasses import asdict
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_cached_engine = None

def get_engine():
    global _cached_engine
    if _cached_engine is not None:
        return _cached_engine

    import torch
    from robbie.config import CHECKPOINT_DIR, DATA_DIR
    from robbie.core.tokenizer import BPETokenizer
    from robbie.core.models.transformer import RobbieTransformer, TransformerConfig
    from robbie.core.training.trainer import Trainer
    from robbie.core.conversation import ChatEngine, ConversationContext
    from robbie.memory.store import PersistentMemory
    from robbie.utils.device import get_best_device

    vocab_path = CHECKPOINT_DIR / "bpe_stage1_vocab.json"
    tokenizer = BPETokenizer.load(vocab_path)

    ckpt_candidates = [
        CHECKPOINT_DIR / "stage2_25m_conversational_transformer.pt",
        CHECKPOINT_DIR / "stage2_conversational_transformer.pt",
        CHECKPOINT_DIR / "stage1_25m_base_transformer.pt",
    ]
    selected_ckpt = None
    for c in ckpt_candidates:
        if c.exists():
            selected_ckpt = c
            break

    device = get_best_device()
    if selected_ckpt:
        raw_data = torch.load(selected_ckpt, map_location="cpu", weights_only=False)
        saved_config = raw_data.get("config", None)
        if isinstance(saved_config, TransformerConfig):
            config = saved_config
        elif isinstance(saved_config, dict):
            config = TransformerConfig(**saved_config)
        else:
            config = TransformerConfig(vocab_size=tokenizer.vocab_size, block_size=128, n_layer=8, n_head=8, n_embd=512)
        model = RobbieTransformer(config)
        Trainer.load_checkpoint(selected_ckpt, model)
    else:
        config = TransformerConfig(vocab_size=tokenizer.vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128)
        model = RobbieTransformer(config)

    model.to(device)
    model.eval()

    memory = PersistentMemory()
    context = ConversationContext(
        tokenizer=tokenizer,
        max_context_tokens=128,
        reserved_generation_tokens=40,
        persistent_memory=memory,
    )
    _cached_engine = ChatEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        context=context,
        max_tokens=40,
        temperature=0.7,
        top_k=25,
        repetition_penalty=1.15,
    )
    return _cached_engine

def handle_chat(payload: dict) -> dict:
    engine = get_engine()
    user_msg = payload.get("message", "")
    reply = engine.respond(user_msg)
    slots = dict(engine.context.slots)
    return {
        "reply": reply,
        "slots": slots,
        "turns": engine.context.turn_count,
        "model": "Robbie-25M (Stage 2 Conversational)",
    }

def handle_system(_payload: dict) -> dict:
    from robbie.system import ComputerAwareness
    ca = ComputerAwareness()
    snap = ca.snapshot()
    data = asdict(snap)
    data["markdown"] = snap.to_markdown()
    return data

def handle_memory(payload: dict) -> dict:
    from robbie.memory.store import PersistentMemory
    pm = PersistentMemory()
    action = payload.get("action", "get")
    if action == "clear":
        pm.clear()
        return {
            "status": "cleared",
            "profile": {},
            "notes": {},
            "episodes": [],
            "episodes_count": 0,
        }
    elif action == "set":
        key = payload.get("key")
        value = payload.get("value")
        if key and value is not None:
            pm.set(key, value)
    elif action == "delete":
        key = payload.get("key")
        if key:
            pm.delete(key)
    elif action == "add_note":
        title = payload.get("title")
        content = payload.get("content")
        if title and content:
            pm.add_note(title, content)

    return {
        "profile": pm.profile,
        "notes": pm.notes,
        "episodes": pm.episodes,
        "episodes_count": len(pm.episodes),
    }

def handle_research(payload: dict) -> dict:
    from robbie.research import ResearchAssistant
    topic = payload.get("topic", "")
    ra = ResearchAssistant()
    report = ra.research(topic, max_sources=3)
    return {
        "topic": report.query,
        "summary": report.summary,
        "markdown": report.to_markdown(),
        "html": report.to_html(),
        "sources": [
            {
                "index": c["index"],
                "title": c["title"],
                "url": c["url"],
                "domain": c.get("source", "web"),
                "score": round(c.get("authority_score", 1.0), 2),
                "date": c.get("published_date", ""),
                "is_outdated": c.get("is_outdated", False),
            }
            for c in report.citations
        ],
        "findings": report.key_findings,
        "citations": report.citations,
        "latency_sec": round(report.latency_sec, 2),
        "citation_validation_rate": report.citation_validation_rate,
        "sources_count": report.sources_count,
    }

def handle_execute(payload: dict) -> dict:
    code = payload.get("code", "")
    import io
    import contextlib
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    success = True
    error_msg = ""
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            safe_globals = {"__name__": "__main__", "__builtins__": __builtins__}
            exec(code, safe_globals)
    except Exception as e:
        success = False
        error_msg = str(e)
    
    return {
        "success": success,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "error": error_msg,
    }

def handle_launch(payload: dict) -> dict:
    from robbie.computer.launcher import AppLauncher
    app_query = payload.get("app", payload.get("query", ""))
    res = AppLauncher.launch(app_query)
    return res.to_dict()

def handle_diagnose(payload: dict) -> dict:
    from robbie.system.diagnostics import SystemDiagnostics
    raw_query = payload.get("query", payload.get("type", "all"))
    query = raw_query.lower().strip()

    if any(k in query for k in ("fps", "game", "stutter", "lag", "frame", "gpu", "vram", "render")):
        return SystemDiagnostics.diagnose_fps_drop().to_dict()
    elif any(k in query for k in ("disk", "drive", "storage", "space", "partition", "c:", "z:")):
        return SystemDiagnostics.diagnose_disk().to_dict()
    elif any(k in query for k in ("cpu", "processor", "load", "usage", "thread", "fan", "heat", "hot", "slow")):
        return SystemDiagnostics.diagnose_cpu().to_dict()
    elif query and query != "all":
        # General question or diagnostic query: route through ChatEngine with system context
        engine = get_engine()
        reply = engine.respond(raw_query)
        return {
            "headline": f"Diagnostic Analysis: {raw_query}",
            "primary_bottleneck": "Live Telemetry Evaluation",
            "summary_markdown": reply,
            "metrics": {},
        }
    else:
        return SystemDiagnostics.diagnose_cpu().to_dict()

def handle_files(payload: dict) -> dict:
    from robbie.computer.tools.file_ops import FileOps
    fops = FileOps()
    action = payload.get("action", "list")
    path = payload.get("path", ".")

    if action == "list":
        res = fops.list_dir(path)
        return res.to_dict()
    elif action == "read":
        res = fops.read_file(path)
        return res.to_dict()
    elif action == "open":
        res = fops.open_in_system(path)
        return res.to_dict()
    elif action == "search":
        pattern = payload.get("pattern", "*")
        res = fops.search_files(pattern=pattern, root_dir=path)
        return res.to_dict()
    else:
        return {"success": False, "error": f"Unknown file action: {action}"}

HANDLERS = {
    "chat": handle_chat,
    "system": handle_system,
    "memory": handle_memory,
    "research": handle_research,
    "execute": handle_execute,
    "launch": handle_launch,
    "diagnose": handle_diagnose,
    "files": handle_files,
}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command provided"}))
        return

    cmd = sys.argv[1]
    payload = {}
    if len(sys.argv) >= 3:
        raw_arg = sys.argv[2]
        if raw_arg == "-":
            raw_arg = sys.stdin.read()
        try:
            payload = json.loads(raw_arg)
        except Exception:
            payload = {"message": raw_arg, "topic": raw_arg, "code": raw_arg, "query": raw_arg, "app": raw_arg, "type": raw_arg}
    elif not sys.stdin.isatty():
        try:
            stdin_text = sys.stdin.read().strip()
            if stdin_text:
                payload = json.loads(stdin_text)
        except Exception:
            pass

    handler = HANDLERS.get(cmd)
    if not handler:
        print(json.dumps({"error": f"Unknown command: {cmd}"}))
        return

    try:
        result = handler(payload)
        print(json.dumps(result, default=str))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
