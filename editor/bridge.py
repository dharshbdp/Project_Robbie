"""Robbie Desktop Studio / Tauri JSON IPC Bridge.

Allows the Tauri Rust backend and Desktop UI to communicate seamlessly
with Robbie's ChatEngine, ComputerAwareness, PersistentMemory, and ResearchAssistant.
"""

import sys
import json
import os
import contextlib
from dataclasses import asdict
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_cached_engine = None
_model_info = None


def get_memory():
    from robbie.memory.store import PersistentMemory
    return PersistentMemory(store_path=os.environ.get("ROBBIE_MEMORY_PATH"))


def select_checkpoint(checkpoint_dir: Path) -> Path:
    explicit = os.environ.get("ROBBIE_CHECKPOINT")
    if explicit:
        selected = Path(explicit).expanduser()
        if not selected.is_absolute():
            selected = PROJECT_ROOT / selected
        if not selected.is_file():
            raise FileNotFoundError(f"Configured Robbie checkpoint does not exist: {selected}")
        return selected.resolve()
    trained = sorted(
        list(checkpoint_dir.glob("answers_*/best_answers.pt"))
        + list(checkpoint_dir.glob("chat_*/best_chat.pt")),
        key=lambda path: path.stat().st_mtime, reverse=True,
    )
    candidates = [
        checkpoint_dir / "stage2_300m_conversational_transformer.pt",
        checkpoint_dir / "chat_300m_v1" / "best_model.pt",
        checkpoint_dir / "stage2_50m_conversational_transformer.pt",
    ] + trained + [
        checkpoint_dir / "stage2_25m_conversational_transformer.pt",
        checkpoint_dir / "stage2_11m_conversational_transformer.pt",
        checkpoint_dir / "stage2_conversational_transformer.pt",
    ]
    for selected in candidates:
        if selected.is_file():
            return selected.resolve()
    raise FileNotFoundError("No trained Robbie checkpoint found. Train a model or set ROBBIE_CHECKPOINT.")

def get_engine():
    global _cached_engine, _model_info
    if _cached_engine is not None:
        return _cached_engine

    import torch
    from robbie.config import CHECKPOINT_DIR, DATA_DIR
    from robbie.core.tokenizer import BPETokenizer
    from robbie.core.models.transformer import RobbieTransformer, TransformerConfig
    from robbie.core.conversation import ChatEngine, ConversationContext
    from robbie.utils.device import get_best_device

    vocab_path = CHECKPOINT_DIR / "bpe_stage1_vocab.json"
    tokenizer = BPETokenizer.load(vocab_path)

    selected_ckpt = select_checkpoint(CHECKPOINT_DIR)
    device = get_best_device()
    raw_data = torch.load(selected_ckpt, map_location="cpu", weights_only=False)
    saved_config = raw_data.get("config")
    if isinstance(saved_config, TransformerConfig):
        config = saved_config
    elif isinstance(saved_config, dict):
        config = TransformerConfig(**saved_config)
    else:
        raise ValueError("The checkpoint has no supported model configuration.")
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("The checkpoint and BPE tokenizer vocabulary sizes do not match.")
    model = RobbieTransformer(config)
    model.load_state_dict(raw_data["model_state_dict"])
    model.to(device)
    model.eval()
    del raw_data

    memory = get_memory()
    chat_hist_path = Path(os.environ.get("ROBBIE_CHAT_HISTORY_PATH", PROJECT_ROOT / "chat_history.json"))
    context = ConversationContext(
        tokenizer=tokenizer,
        max_context_tokens=config.block_size,
        reserved_generation_tokens=min(140, config.block_size // 2),
        persistent_memory=memory,
        history_file=chat_hist_path,
        auto_load=True,
        auto_save=True,
    )
    _cached_engine = ChatEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        context=context,
        max_tokens=min(140, config.block_size // 2),
        temperature=0.5,
        top_k=40,
        repetition_penalty=1.2,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    _model_info = {
        "model": f"Robbie {parameters / 1_000_000:.1f}M",
        "checkpoint": str(selected_ckpt), "device": str(device),
        "context_tokens": config.block_size, "max_tokens": _cached_engine.max_tokens,
        "temperature": _cached_engine.temperature, "parameters": parameters,
        "loaded": True,
    }
    return _cached_engine

def handle_chat(payload: dict) -> dict:
    engine = get_engine()
    user_msg = payload.get("message", "")
    if not isinstance(user_msg, str) or not user_msg.strip():
        raise ValueError("Enter a message for Robbie.")
    reply = engine.respond(user_msg)
    slots = dict(engine.context.slots)
    return {
        "reply": reply,
        "slots": slots,
        "turns": engine.context.turn_count,
        **_model_info,
    }


def _format_history_message(m) -> dict:
    d = m.to_dict()
    from robbie.core.conversation.messages import Role
    role_str = str(d.get("role", "")).lower().strip()
    is_assistant = (hasattr(m, "role") and m.role == Role.ASSISTANT) or role_str in ("robbie", "assistant", "ai", "bot")
    d["ui_role"] = "assistant" if is_assistant else "user"
    d["role"] = "assistant" if is_assistant else "user"
    d["display_role"] = "Robbie" if is_assistant else "User"
    return d


def handle_status(_payload: dict) -> dict:
    engine = get_engine()
    return {
        **_model_info,
        "turns": engine.context.turn_count,
        "history_file": str(engine.context.history_file) if engine.context.history_file else "",
        "history": [_format_history_message(m) for m in engine.context.messages],
    }


def handle_reset(payload: dict) -> dict:
    if _cached_engine is not None:
        _cached_engine.context.delete_history()
    else:
        chat_hist_path = Path(os.environ.get("ROBBIE_CHAT_HISTORY_PATH", PROJECT_ROOT / "chat_history.json"))
        if chat_hist_path.is_file():
            chat_hist_path.unlink()
    return {"status": "cleared", "turns": 0, "history_deleted": True}


def handle_get_chat_history(_payload: dict) -> dict:
    engine = get_engine()
    return {
        "file": str(engine.context.history_file) if engine.context.history_file else "",
        "exists": engine.context.history_file.is_file() if engine.context.history_file else False,
        "turns": engine.context.turn_count,
        "messages": [_format_history_message(m) for m in engine.context.messages],
    }


def handle_delete_chat_history(payload: dict) -> dict:
    return handle_reset(payload)

def handle_system(_payload: dict) -> dict:
    from robbie.system import ComputerAwareness
    ca = ComputerAwareness()
    snap = ca.snapshot()
    data = asdict(snap)
    data["markdown"] = snap.to_markdown()
    return data

def handle_memory(payload: dict) -> dict:
    pm = get_memory()
    action = payload.get("action", "get")
    if action == "clear":
        pm.clear()
    elif action == "set":
        key = payload.get("key")
        value = payload.get("value")
        if key and value is not None:
            pm.set(key, value)
    elif action == "delete":
        key = payload.get("key")
        if key:
            if not pm.delete(key) and key in pm.notes:
                del pm.notes[key]
                pm.save()
    elif action == "add_note":
        title = payload.get("title")
        content = payload.get("content")
        if title and content:
            pm.add_note(title, content)

    if _cached_engine is not None:
        # The memory UI and chat must use the same current disk state.
        _cached_engine.context.persistent_memory = pm
        _cached_engine.context.persistent_slots.clear()
        _cached_engine.context.persistent_slots.update(pm.list_profile())
        _cached_engine.context.slots.sync()
    return {
        "profile": pm.profile,
        "notes": pm.notes,
        "episodes": pm.episodes,
        "episodes_count": len(pm.episodes),
        "session": dict(_cached_engine.context.slots) if _cached_engine is not None else {},
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
    import subprocess
    if not isinstance(code, str):
        raise ValueError("Python code must be text")
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8",
                   "PYTHONPATH": str(PROJECT_ROOT / "src")}
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", code], capture_output=True, text=True,
            encoding="utf-8", errors="replace", cwd=PROJECT_ROOT, env=environment,
            timeout=30, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return {"success": result.returncode == 0, "stdout": result.stdout,
                "stderr": result.stderr, "error": "" if result.returncode == 0 else f"Python exited with code {result.returncode}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "", "error": "Python execution exceeded 30 seconds."}

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
    "status": handle_status,
    "reset": handle_reset,
    "chat": handle_chat,
    "get_chat_history": handle_get_chat_history,
    "delete_chat_history": handle_delete_chat_history,
    "system": handle_system,
    "memory": handle_memory,
    "research": handle_research,
    "execute": handle_execute,
    "launch": handle_launch,
    "diagnose": handle_diagnose,
    "files": handle_files,
}

def dispatch(command, payload):
    if not isinstance(payload, dict):
        raise ValueError("Bridge payload must be a JSON object")
    handler = HANDLERS.get(command)
    if handler is None:
        raise ValueError(f"Unknown bridge command: {command}")
    # Keep stdout exclusively for the IPC protocol, even if a handler logs.
    with contextlib.redirect_stdout(sys.stderr):
        from robbie.utils.logger import logger
        streams = [(item, item.stream) for item in logger.handlers if hasattr(item, "setStream")]
        try:
            for handler_log, _stream in streams:
                handler_log.stream = sys.stderr
            return handler(payload)
        finally:
            for handler_log, stream in streams:
                handler_log.stream = stream


def serve():
    """One JSON request and response per line; reuse the model and session."""
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Bridge request must be an object")
            request_id = request.get("id")
            result = dispatch(request.get("command"), request.get("payload", {}))
        except Exception as error:
            result = {"error": str(error)}
        print(json.dumps({"id": request_id, "result": result}, default=str), flush=True)


def main():
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stdin is not None and hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    os.chdir(PROJECT_ROOT)
    if "--serve" in sys.argv:
        serve()
        return
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

    try:
        result = dispatch(cmd, payload)
        print(json.dumps(result, default=str))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
