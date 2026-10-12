"""ChatEngine orchestrating multi-turn conversation and model generation."""

import re
from typing import Any, Dict, Optional, Tuple
import torch

from robbie.core.conversation.context import ConversationContext
from robbie.core.conversation.messages import Role
from robbie.core.conversation.personality import RobbiePersonality


class ChatEngine:
    """Orchestrates multi-turn conversation with sliding window context and Robbie transformer."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        context: Optional[ConversationContext] = None,
        max_tokens: int = 60,
        temperature: float = 0.5,
        top_k: int = 25,
        repetition_penalty: float = 1.25,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        block_size = getattr(getattr(model, "config", None), "block_size", 128)
        self.context = context or ConversationContext(
            tokenizer=tokenizer,
            max_context_tokens=block_size,
            reserved_generation_tokens=max_tokens,
        )

    def handle_command(self, cmd: str) -> Optional[str]:
        """Process special in-chat slash commands."""
        parts = cmd.strip().split()
        if not parts:
            return None
        c = parts[0].lower()

        if c == "/clear":
            self.context.clear()
            return "Conversation history cleared! Let's start fresh."

        if c == "/memory":
            if len(parts) == 1 or parts[1].lower() == "list":
                if not self.context.slots:
                    return "Working memory is currently empty."
                items = "\n".join(f"  • {k} = '{v}'" for k, v in self.context.slots.items())
                return f"Remembered facts:\n{items}"
            if parts[1].lower() == "set" and len(parts) >= 4:
                k = parts[2]
                v = " ".join(parts[3:])
                self.context.slots[k] = v
                if self.context.persistent_memory is not None:
                    self.context.persistent_memory.set(k, v)
                return f"Saved to memory: {k} = '{v}'"
            if parts[1].lower() == "del" and len(parts) >= 3:
                k = parts[2]
                if k in self.context.slots:
                    del self.context.slots[k]
                    if self.context.persistent_memory is not None:
                        self.context.persistent_memory.delete(k)
                    return f"Deleted from memory: '{k}'"
                return f"Fact '{k}' was not found in memory."
            if parts[1].lower() == "clear":
                self.context.slots.clear()
                if self.context.persistent_memory is not None:
                    self.context.persistent_memory.clear()
                return "All memory attributes cleared!"
            return "Usage: /memory [list | set <key> <val> | del <key> | clear]"

        if c == "/history":
            if not self.context.messages:
                return "No messages in history yet."
            history_lines = [f"[{m.role.value}] {m.content}" for m in self.context.messages]
            return "\n".join(history_lines)

        if c == "/research":
            if len(parts) < 2:
                return "Usage: /research <topic or question>"
            query_str = " ".join(parts[1:])
            from robbie.research import ResearchAssistant
            ra = ResearchAssistant()
            report = ra.research(query_str, max_sources=4)
            return report.to_markdown()

        if c == "/search":
            if len(parts) < 2:
                return "Usage: /search <query>"
            query_str = " ".join(parts[1:])
            from robbie.research import ResearchAssistant
            ra = ResearchAssistant()
            results = ra.quick_search(query_str, max_results=5)
            if not results:
                return f"No search results found for '{query_str}'."
            res_lines = [f"Web search results for '{query_str}':"]
            for i, r in enumerate(results, 1):
                snippet_preview = (r.snippet[:120] + "...") if len(r.snippet) > 120 else r.snippet
                res_lines.append(f"  [{i}] {r.title} ({r.source})\n      URL: {r.url}\n      {snippet_preview}")
            return "\n".join(res_lines)

        if c in ("/system", "/sys", "/stats"):
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            return ca.snapshot().to_markdown()

        if c in ("/processes", "/top", "/procs"):
            limit = 10
            if len(parts) >= 2 and parts[1].isdigit():
                limit = int(parts[1])
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            procs = ca.process_monitor.list_processes(limit=limit)
            if not procs:
                return "No process information available."
            lines = [f"Top {len(procs)} Processes by Memory Usage:"]
            for i, p in enumerate(procs, 1):
                lines.append(f"  [{i:2d}] {p.name:<24} (PID {p.pid:<5}) - {p.memory_mb:>7.1f} MB RAM")
            return "\n".join(lines)

        if c == "/help":
            return (
                "Available commands:\n"
                "  /clear              — Clear current conversation history\n"
                "  /memory             — Inspect remembered facts (like your name)\n"
                "  /memory set <k> <v> — Store a custom fact in memory\n"
                "  /history            — View all conversation turns\n"
                "  /system             — Live computer awareness report (hardware, power, workspace)\n"
                "  /processes [N]      — List top N processes by memory consumption\n"
                "  /research <topic>   — Conduct multi-source web research with citations\n"
                "  /search <query>     — Quick web search across sources\n"
                "  exit/quit           — End chat session"
            )
        return None

    def respond(self, user_input: str) -> str:
        """Process a user message and generate Robbie's reply."""
        cleaned = user_input.strip().strip("\"' ")
        if not cleaned:
            return ""

        # Check for commands
        cmd_result = self.handle_command(cleaned)
        if cmd_result is not None:
            return cmd_result

        user_name = self.context.slots.get("user_name")

        # Check if user is introducing their name (e.g. "My name is Dharsh", "I am Dharsh")
        intro_slots = self.context.extract_slots(cleaned)
        intro_name = intro_slots.get("user_name")
        lower_cleaned = cleaned.lower()
        is_self_intro = intro_name and (
            lower_cleaned.startswith("my name is")
            or lower_cleaned.startswith("i am ")
            or lower_cleaned.startswith("call me ")
            or lower_cleaned.startswith("name's ")
            or "my name is " in lower_cleaned
        )
        if is_self_intro:
            self.context.add_user_message(cleaned)
            intro_reply = f"Nice to meet you, {intro_name}! I'll remember that. How can I help you today?"
            self.context.add_assistant_message(intro_reply)
            return intro_reply

        # Check for character affirmations & compliments (Phase 6)
        affirmation_reply = RobbiePersonality.get_affirmation_response(cleaned, user_name=user_name)
        if affirmation_reply is not None:
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(affirmation_reply)
            return affirmation_reply

        # Check for direct slot retrieval (e.g. 'what is my name?')
        slot_answer = self.context.check_direct_slot_answer(cleaned)
        if slot_answer is not None:
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(slot_answer)
            return slot_answer

        # If user gives a warm direct greeting and name is already known, personalize greeting
        if cleaned.lower() in ("hello", "hi", "hey", "hello robbie", "hey robbie") and user_name:
            greeting_reply = RobbiePersonality.get_personalized_greeting(user_name)
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(greeting_reply)
            return greeting_reply

        # Check for natural language research capability questions
        research_capability_triggers = (
            "can you research on the internet",
            "can you research the internet",
            "can you research online",
            "can you search the web",
            "can you search online",
            "can you search the internet",
            "can you browse the web",
            "do you have internet access",
            "can you look up things online",
            "can you search for things",
        )
        if any(trig in lower_cleaned for trig in research_capability_triggers):
            reply = (
                "Yes! I have a built-in research assistant. I can search the web and Wikipedia, "
                "read articles, and synthesize findings with citations.\n"
                "You can ask me to research any topic by typing `/research <topic>` or `/search <query>`, "
                "or simply say 'research <topic>'!"
            )
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # Normalize query and expand common contractions / spelling variations
        norm_query = lower_cleaned.rstrip("?.! ").strip()
        norm_expanded = (
            norm_query
            .replace("what's", "what is")
            .replace("whats", "what is")
            .replace("how's", "how is")
            .replace("hows", "how is")
            .replace("who's", "who is")
            .replace("whos", "who is")
            .replace("processess", "processes")
        )

        # Flexible natural language research triggers (Phase 8)
        research_patterns = [
            r"^(?:please\s+|can\s+you\s+)?(?:do\s+(?:a\s+)?)?research\s+(?:on\s+|about\s+)?(.+)$",
            r"^(?:search\s+(?:the\s+web\s+)?for|look\s+up)\s+(.+)$",
        ]
        for pat in research_patterns:
            m = re.match(pat, norm_expanded, re.IGNORECASE)
            if m:
                topic = m.group(1).strip()
                if topic:
                    self.context.add_user_message(cleaned)
                    from robbie.research import ResearchAssistant
                    ra = ResearchAssistant()
                    report = ra.research(topic, max_sources=4)
                    report_md = report.to_markdown()
                    self.context.add_assistant_message(report_md)
                    return report_md

        # Computer awareness capability questions (Phase 9)
        awareness_capability_triggers = (
            "can you monitor my system",
            "can you check my computer",
            "can you check my pc",
            "do you have computer awareness",
            "can you check system specs",
            "can you see my computer",
            "can you check my hardware",
            "what can you see about my computer",
        )
        if any(trig in norm_expanded for trig in awareness_capability_triggers):
            reply = (
                "Yes! I have built-in computer awareness. I can monitor CPU, RAM, GPU, storage, "
                "battery, running processes, active window, and development workspace status.\n"
                "You can type `/system` for a full status report, `/processes` for top processes, "
                "or simply ask me questions like 'How much RAM is free?' or 'What is my battery level?'!"
            )
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # Storage / Disk queries (e.g. "How much disk space do I have?")
        if any(w in norm_expanded for w in ("disk", "storage", "drive space", "hard drive", "free space")):
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            disks = ca.hardware.get_disks_info()
            disk_strs = [f"{d.drive} ({d.free_gb:.1f} GB free out of {d.total_gb:.1f} GB)" for d in disks]
            reply = f"Your storage status across drives: {', '.join(disk_strs)}."
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # Battery / Power queries (e.g. "what's my battery level?")
        if any(w in norm_expanded for w in ("battery", "charging", "power source")):
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            bat = ca.env_monitor.get_battery_info()
            if bat.present:
                charging_tag = "charging" if bat.is_charging else "discharging"
                reply = f"Your battery is at {bat.percent}% ({bat.power_source}, {charging_tag})."
            else:
                reply = f"System is currently running on {bat.power_source} (no battery detected)."
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # Process queries (e.g. "what processess are running?", "top processes")
        if any(w in norm_expanded for w in ("process", "processes", "running tasks", "tasklist", "top tasks")):
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            procs = ca.process_monitor.list_processes(limit=5)
            lines = ["Here are the top active processes by memory usage:"]
            for i, p in enumerate(procs, 1):
                lines.append(f"  {i}. {p.name} (PID {p.pid}): {p.memory_mb:.1f} MB RAM")
            reply = "\n".join(lines)
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # Window / Tabs queries (e.g. "What tabs are open in my system?", "What window is currently open?")
        if any(w in norm_expanded for w in ("window", "tabs", "active app", "foreground app", "foreground window")):
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            win = ca.env_monitor.get_active_window()
            reply = f"Your active foreground window is '{win.title}'."
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # RAM / Memory queries (e.g. "How much RAM is free?")
        if "ram" in norm_expanded or ("memory" in norm_expanded and "process" not in norm_expanded):
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            mem = ca.hardware.get_memory_info()
            reply = f"You have {mem.available_gb:.2f} GB free out of {mem.total_gb:.2f} GB total RAM ({mem.percent_used:.1f}% used)."
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # GPU queries (e.g. "What GPU do I have?")
        if "gpu" in norm_expanded or "graphics card" in norm_expanded or "vram" in norm_expanded:
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            gpu = ca.hardware.get_gpu_info()
            if gpu.cuda_available:
                reply = f"You have an NVIDIA {gpu.device_name} with {gpu.vram_free_gb:.1f} GB free out of {gpu.vram_total_gb:.1f} GB total VRAM."
            else:
                reply = "CUDA is not detected on your system; Robbie is currently running on CPU."
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # CPU queries (e.g. "What CPU do I have?")
        if "cpu" in norm_expanded or "processor" in norm_expanded:
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            cpu = ca.hardware.get_cpu_info()
            reply = f"Your system has an {cpu.processor} with {cpu.logical_cores} logical cores running on {cpu.system} ({cpu.architecture})."
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # Git / Workspace queries
        if any(w in norm_expanded for w in ("git branch", "git status", "git repo", "workspace status", "project status")):
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            ws = ca.workspace_monitor.get_workspace_context()
            clean_str = "clean" if ws.is_git_clean else f"{ws.modified_files_count} modified files"
            reply = f"You are working on '{ws.project_name}' on git branch '{ws.git_branch or 'main'}' ({ws.git_commit or 'unknown'}, {clean_str})."
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # Network queries
        if any(w in norm_expanded for w in ("network status", "internet status", "wifi status", "ip address", "ping latency")):
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            net = ca.env_monitor.get_network_info()
            net_tag = "online" if net.internet_connected else "offline"
            lat_str = f" with {net.latency_ms:.1f}ms latency" if net.latency_ms else ""
            reply = f"Network is {net_tag}{lat_str} on hostname '{net.hostname}' (Local IP: {net.local_ip})."
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # General System Specs / Computer Report
        if any(w in norm_expanded for w in ("specs", "hardware", "system report", "system status", "computer report", "computer status", "system info", "computer info", "check my computer", "check my system", "monitor my system")):
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            reply = ca.snapshot().to_markdown()
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # Add user turn to context
        self.context.add_user_message(cleaned)

        # Build token-budgeted prompt
        prompt, prompt_ids = self.context.get_context_prompt()
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Generate response from model
        generated_tensor = self.model.generate(
            input_tensor,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            stop_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=self.repetition_penalty,
        )

        gen_ids = generated_tensor[0].tolist()[len(prompt_ids):]
        response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Stop token string post-processing
        if response_text.startswith("Robbie:"):
            response_text = response_text[len("Robbie:"):].strip()
        if response_text.startswith("Assistant:"):
            response_text = response_text[len("Assistant:"):].strip()

        stop_markers = [
            "\nUser:",
            "\nRobbie:",
            "\nAssistant:",
            "\nSystem:",
            "User:",
            "Robbie:",
            "Assistant:",
            "<|endoftext|>",
            "<|endoftext",
            "<isendoftextis>",
            "<|im_end|>",
        ]
        for marker in stop_markers:
            if marker in response_text:
                response_text = response_text.split(marker)[0].strip()

        # Clean trailing incomplete boundary tokens or tags
        response_text = response_text.rstrip("<\n ").strip()

        # Fallback if empty response generated
        if not response_text:
            response_text = "I am listening! How can I assist you with your project?"

        # Save assistant turn to context
        self.context.add_assistant_message(response_text)
        return response_text
