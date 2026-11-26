"""ChatEngine orchestrating multi-turn conversation and model generation."""

import re
from datetime import datetime
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
        self.pending_search_query: Optional[str] = None

    def handle_command(self, cmd: str) -> Optional[str]:
        """Process special in-chat slash commands."""
        parts = cmd.strip().split()
        if not parts:
            return None
        c = parts[0].lower()

        if c in ("/clear", "/delete"):
            self.context.clear(delete_file=True)
            return "Conversation history cleared and deleted from disk! Let's start fresh."

        if c == "/memory":
            if len(parts) == 1 or parts[1].lower() == "list":
                bk = self.context.get_memory_breakdown()
                sess = bk["session_context"]
                pers = bk["persistent_storage"]
                lines = [
                    "🧠 **Active Session Context (Working Memory):**",
                    f"  • Turns in session: {sess['turn_count']}",
                    f"  • Current topic: {sess['current_topic'] or 'None'}",
                ]
                for k, v in sess["session_slots"].items():
                    if k not in ("session_start_time", "current_topic"):
                        lines.append(f"  • {k} = '{v}'")

                lines.append("\n💾 **Persistent Storage (Cross-Session Disk Memory):**")
                if pers["profile"]:
                    for k, v in pers["profile"].items():
                        lines.append(f"  • {k} = '{v}'")
                else:
                    lines.append("  • (No persistent profile attributes stored yet)")
                lines.append(f"  • Saved notes: {pers['notes_count']} | Episodes: {pers['episodes_count']}")
                if pers["store_path"] != "none":
                    lines.append(f"  • File: {pers['store_path']}")
                return "\n".join(lines)

            if parts[1].lower() == "set" and len(parts) >= 4:
                k = parts[2]
                v = " ".join(parts[3:])
                self.context.persistent_slots[k] = v
                if self.context.persistent_memory is not None:
                    self.context.persistent_memory.set(k, v)
                return f"Saved to persistent memory: {k} = '{v}'"

            if parts[1].lower() == "del" and len(parts) >= 3:
                k = parts[2]
                deleted = False
                if k in self.context.persistent_slots:
                    del self.context.persistent_slots[k]
                    if self.context.persistent_memory is not None:
                        self.context.persistent_memory.delete(k)
                    deleted = True
                if k in self.context.session_slots:
                    del self.context.session_slots[k]
                    deleted = True
                if deleted:
                    return f"Deleted from memory: '{k}'"
                return f"Fact '{k}' was not found in memory."

            if parts[1].lower() == "clear":
                self.context.clear(keep_slots=False, keep_persistent=False)
                return "All memory attributes (session and persistent) cleared!"

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

        if c in ("/time", "/date"):
            return self.current_time_reply(include_time=c == "/time")

        if c == "/help":
            return (
                "Available commands:\n"
                "  /clear              — Clear current conversation history\n"
                "  /memory             — Inspect remembered facts (like your name)\n"
                "  /memory set <k> <v> — Store a custom fact in memory\n"
                "  /history            — View all conversation turns\n"
                "  /system             — Live computer awareness report (hardware, power, workspace)\n"
                "  /processes [N]      — List top N processes by memory consumption\n"
                "  /time               — Current local date and time\n"
                "  /research <topic>   — Conduct multi-source web research with citations\n"
                "  /search <query>     — Quick web search across sources\n"
                "  exit/quit           — End chat session"
            )
        return None

    @staticmethod
    def current_time_reply(include_time: bool = True) -> str:
        """Read the computer's clock; a language model cannot know live time."""
        now = datetime.now().astimezone()
        if not include_time:
            return f"Today is {now:%d %B %Y}."
        offset = now.strftime("%z")
        zone = f"UTC{offset[:3]}:{offset[3:]}"
        return f"It is {now:%H:%M:%S} on {now:%d %B %Y} ({zone})."

    def remember_reply(self, user_input: str, reply: str) -> str:
        self.context.add_user_message(user_input)
        self.context.add_assistant_message(reply)
        return reply

    def respond(self, user_input: str) -> str:
        """Process a user message and generate Robbie's reply."""
        cleaned = user_input.strip().strip("\"' ")
        if not cleaned:
            return ""

        # Check for commands
        cmd_result = self.handle_command(cleaned)
        if cmd_result is not None:
            return cmd_result

        # Check if user is confirming or declining a pending internet search offer
        if self.pending_search_query:
            lower_clean = cleaned.lower().strip("?.! ")
            affirmative_pattern = r"^(?:yes|yeah|yep|sure|ok|okay|please|go ahead|search|look it up|do it|find out|yes please|y|do that)\b"
            negative_pattern = r"^(?:no|nope|nah|cancel|nevermind|never mind|don't|stop|n)\b"
            if re.search(affirmative_pattern, lower_clean):
                topic = self.pending_search_query
                self.pending_search_query = None
                self.context.add_user_message(cleaned)
                from robbie.research import ResearchAssistant
                ra = ResearchAssistant()
                web_ans = ra.answer_query(topic)
                if not web_ans or "couldn't retrieve live results" in web_ans:
                    reply = f"I attempted to look up **{topic}** online, but couldn't retrieve results right now. Please verify your internet connection."
                else:
                    reply = web_ans
                self.context.add_assistant_message(reply)
                return reply
            elif re.search(negative_pattern, lower_clean):
                self.pending_search_query = None
                self.context.add_user_message(cleaned)
                reply = "Understood! What else would you like to discuss or work on?"
                self.context.add_assistant_message(reply)
                return reply
            else:
                self.pending_search_query = None

        # Check for conversational follow-up directives referencing the previous turn
        followup_reply = self.handle_followup_directive(cleaned)
        if followup_reply is not None:
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(followup_reply)
            return followup_reply

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

        # If user gives a warm direct greeting, personalize if name is known or offer warm welcome
        if lower_cleaned in ("hello", "hi", "hey", "hello robbie", "hey robbie", "hi robbie", "good morning", "good afternoon", "good evening"):
            if user_name:
                greeting_reply = RobbiePersonality.get_personalized_greeting(user_name)
            else:
                greeting_reply = "Hi there! I am Robbie, your personal artificial intelligence assistant. What would you like to work on today?"
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

        # Word boundaries matter: "programs" contains the characters "ram",
        # and "VRAM" is GPU memory, not system RAM.
        def has_topic(*topics):
            return any(re.search(r"\b" + re.escape(topic) + r"\b", norm_expanded)
                       for topic in topics)

        if re.fullmatch(
            r"(?:please )?(?:what (?:is )?(?:the )?(?:current )?time(?: is it)?(?: (?:right )?now)?|"
            r"what time is it(?: (?:right )?now)?|tell me (?:the )?(?:current )?time|"
            r"current (?:local )?time|time (?:right )?now)", norm_expanded
        ):
            return self.remember_reply(cleaned, self.current_time_reply())
        if re.fullmatch(r"(?:what (?:is )?(?:the )?(?:current )?date(?: today)?|"
                        r"what date is it|today'?s date)", norm_expanded):
            return self.remember_reply(cleaned, self.current_time_reply(include_time=False))

        # Definitions and explanations are model questions, not requests for
        # live telemetry. "What is RAM?" must not become "Your RAM is ...".
        concept_question = re.search(
            r"\b(explain|define|definition|difference|different|what does .+ mean|"
            r"what (?:is|are) (?:a |an |the )?(?:ram|vram|cpu|gpu|git|"
            r"memory|storage|a process|processes|a branch|branches|a disk|disk|"
            r"python|code|programming|the web|web|the internet|internet|an api|api|http|cuda|an os|an operating system))\b",
            norm_expanded,
        )
        single_concept = norm_expanded in (
            "web", "python", "code", "internet", "git", "ram", "cpu", "gpu", "vram", "cuda", "storage", "disk"
        )
        # Generic hardware capability questions also belong to the model.
        # E.g. a question about computers without CUDA is not GPU inspection.
        general_hardware_question = (
            has_topic("ram", "vram", "memory", "gpu", "cuda", "cpu", "processor",
                      "storage", "disk", "process", "processes", "git", "battery")
            and re.search(r"^(?:can|does|do|is|are|why|how does|how do)\b", norm_expanded)
            and not re.search(r"\b(my|mine|you|currently|right now|this (?:computer|pc))\b",
                              norm_expanded)
        )
        explicit_explanation = re.search(
            r"\b(?:explain|define|definition|difference|different|what does .+ mean)\b",
            norm_expanded,
        )
        is_joke = has_topic("joke", "jokes")
        if (single_concept or explicit_explanation or is_joke or
                (concept_question and not has_topic("running", "currently", "right now")) or
                general_hardware_question):
            # Definitions remain definitions even when the user says "my".
            # A literal inventory question ("what is my CPU?") does not match
            # the definition expression above.
            refers_to_history = re.search(
                r"\b(that|it|this|those|them|above|earlier|previous|another|again)\b", norm_expanded
            )
            return self.generate_reply(cleaned, use_history=bool(refers_to_history))

        process_topic = has_topic("process", "processes", "program", "programs",
                                  "application", "applications", "app", "apps", "tasklist")
        running_list = process_topic and has_topic("running", "open", "active", "list")
        # Attribution questions can omit the noun after a process listing:
        # "and what is taking the most RAM?" still asks for a process.
        memory_attribution = (
            re.search(r"\b(?:what|which|who)\b", norm_expanded)
            and has_topic("using", "uses", "taking", "consuming", "consumes", "hogging", "eating")
        )
        most_memory = (
            (process_topic or memory_attribution)
            and has_topic("ram", "memory")
            and has_topic("most", "highest", "biggest", "largest", "top")
            and not has_topic("gpu", "vram", "graphics card")
        )
        if most_memory or running_list or has_topic("processes", "running tasks", "top tasks", "tasklist"):
            from robbie.system import ComputerAwareness
            procs = ComputerAwareness().process_monitor.list_processes(limit=10)
            if not procs:
                return self.remember_reply(cleaned, "No process information is available right now.")
            procs = sorted(procs, key=lambda p: p.memory_mb, reverse=True)
            if most_memory:
                proc = procs[0]
                reply = (
                    f"The running process using the most memory is {proc.name} "
                    f"(PID {proc.pid}), at {proc.memory_mb:.1f} MB RAM. "
                    "This compares individual processes; an application can have several."
                )
            else:
                lines = [f"Here are the top active processes by memory usage ({len(procs)} shown):"]
                for i, proc in enumerate(procs, 1):
                    lines.append(f"  {i}. {proc.name} (PID {proc.pid}): {proc.memory_mb:.1f} MB RAM")
                reply = "\n".join(lines)
            return self.remember_reply(cleaned, reply)

        # Flexible natural language research triggers (Phase 8)
        research_patterns = [
            r"^(?:please\s+|can\s+you\s+)?(?:do\s+(?:a\s+)?)?research\s+(?:on\s+|about\s+)?(.+)$",
            r"^(?:search\s+(?:the\s+web\s+)?for|look\s+up)\s+(.+)$",
        ]
        for pat in research_patterns:
            m = re.match(pat, norm_expanded, re.IGNORECASE)
            if m:
                topic = m.group(1).strip()
                # Clean filler verbs and prefixes (e.g. 'and give 10 nolan movies' -> '10 nolan movies')
                clean_topic = re.sub(
                    r"^(?:and\s+)?(?:give(?:\s+me)?|name|list|show(?:\s+me)?|find(?:\s+me)?|tell(?:\s+me)?)\s+",
                    "",
                    topic,
                    flags=re.IGNORECASE,
                ).strip()
                if not clean_topic:
                    clean_topic = topic
                if clean_topic:
                    from robbie.research import ResearchAssistant
                    ra = ResearchAssistant()
                    reply = ra.answer_query(clean_topic)
                    wants_sources = any(w in norm_expanded for w in ("source", "sources", "citation", "citations", "link", "links"))
                    if not wants_sources and "\n**Sources:**" in reply:
                        reply = reply.split("\n**Sources:**")[0].strip()
                    return self.remember_reply(cleaned, reply)

        # Application launching (e.g. "open Google Chrome", "open photos", "open file manager")
        if any(norm_expanded.startswith(p) for p in ("open ", "launch ", "start ")):
            from robbie.computer.launcher import AppLauncher
            launch_res = AppLauncher.launch(norm_expanded)
            if launch_res.success:
                self.context.add_user_message(cleaned)
                self.context.add_assistant_message(launch_res.message)
                return launch_res.message

        # Diagnostic: Why is game FPS dropping / stuttering?
        if any(w in norm_expanded for w in ("fps drop", "fps dropping", "game lagging", "game lag", "frame drops", "stuttering", "game stutter", "fps low")):
            from robbie.system.diagnostics import SystemDiagnostics
            diag = SystemDiagnostics.diagnose_fps_drop()
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(diag.summary_markdown)
            return diag.summary_markdown

        # Diagnostic: Why is CPU usage high?
        if ("why" in norm_expanded and "cpu" in norm_expanded) or "high cpu" in norm_expanded or "cpu pegged" in norm_expanded or "running slow" in norm_expanded:
            from robbie.system.diagnostics import SystemDiagnostics
            diag = SystemDiagnostics.diagnose_cpu()
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(diag.summary_markdown)
            return diag.summary_markdown

        # Diagnostic: Disk usage breakdown
        if any(w in norm_expanded for w in ("disk usage", "drive usage", "storage usage", "disk space", "disk capacity")):
            from robbie.system.diagnostics import SystemDiagnostics
            diag = SystemDiagnostics.diagnose_disk()
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(diag.summary_markdown)
            return diag.summary_markdown

        # Safe Read-Only File Operations
        if norm_expanded.startswith("read file ") or norm_expanded.startswith("check file ") or norm_expanded.startswith("view file "):
            file_target = cleaned.split("file", 1)[1].strip()
            from robbie.computer.tools.file_ops import FileOps
            fops = FileOps()
            fres = fops.read_file(file_target)
            if fres.success:
                reply = f"Contents of `{fres.path}`:\n```\n{fres.content[:3000]}\n```"
            else:
                reply = f"Could not read `{file_target}`: {fres.error}"
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        if any(w in norm_expanded for w in ("write file", "modify file", "delete file", "overwrite file")) or ("delete" in norm_expanded and "file" in norm_expanded) or ("write" in norm_expanded and "file" in norm_expanded) or ("modify" in norm_expanded and "file" in norm_expanded):
            reply = "My computer tool safety policy enforces strict read-only access. I can inspect, read, and open files for you, but I cannot modify, write to, or delete any files. This ensures your data remains protected."
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # Priorities query
        if "priority" in norm_expanded or "priorities" in norm_expanded or "primary focus" in norm_expanded:
            reply = (
                "My priorities are strictly ordered:\n"
                "1. **Computer Awareness & System Diagnostics** (my first and most important choice)\n"
                "2. **Autonomous Web Research** (my second choice)\n"
                "3. **Conversational Companion** (my third choice)\n"
                "4. **Programming & Code** (optional auxiliary tool)"
            )
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

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
                "Yes! I have built-in computer awareness, which is my first and highest priority. I can monitor CPU, RAM, GPU, storage, "
                "battery, running processes, active window, and workspace status.\n"
                "I can diagnose why your CPU is high, why your game's FPS is dropping, report current disk usage, "
                "launch applications like Google Chrome or File Explorer, and read workspace files safely."
            )
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # Storage / Disk queries (e.g. "How much disk space do I have?")
        if has_topic("disk", "storage", "drive space", "hard drive", "free space"):
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            disks = ca.hardware.get_disks_info()
            disk_strs = [f"{d.drive} ({d.free_gb:.1f} GB free out of {d.total_gb:.1f} GB)" for d in disks]
            reply = f"Your storage status across drives: {', '.join(disk_strs)}."
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # Battery / Power queries (e.g. "what's my battery level?")
        if has_topic("battery", "charging", "power source"):
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
        if has_topic("ram", "memory") and not has_topic("vram", "gpu", "graphics card"):
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            mem = ca.hardware.get_memory_info()
            reply = f"You have {mem.available_gb:.2f} GB free out of {mem.total_gb:.2f} GB total RAM ({mem.percent_used:.1f}% used)."
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # GPU queries (e.g. "What GPU do I have?")
        if has_topic("gpu", "graphics card", "vram"):
            from robbie.system import ComputerAwareness
            ca = ComputerAwareness()
            gpu = ca.hardware.get_gpu_info()
            if gpu.cuda_available:
                reply = f"You have {gpu.device_name} with {gpu.vram_free_gb:.1f} GB free out of {gpu.vram_total_gb:.1f} GB total VRAM."
            else:
                reply = "CUDA is not detected on your system; Robbie is currently running on CPU."
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # CPU queries (e.g. "What CPU do I have?")
        if has_topic("cpu", "processor"):
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

        # Python Code Snippets & Scripts Assistant
        from robbie.core.conversation.code_assistant import PythonCodeAssistant
        if PythonCodeAssistant.is_code_request(norm_expanded):
            reply = PythonCodeAssistant.generate_snippet(cleaned)
            return self.remember_reply(cleaned, reply)

        # Real-time Web Knowledge & Current Information Queries (Phase 8 RAG)
        # Instead of hallucinating on recent events, releases, news, or internet lookups,
        # Robbie retrieves live information from the web and summarizes relevant facts.
        if self.is_web_knowledge_query(norm_expanded):
            from robbie.research import ResearchAssistant
            ra = ResearchAssistant()
            reply = ra.answer_query(cleaned)
            wants_sources = any(w in norm_expanded for w in ("source", "sources", "citation", "citations", "link", "links"))
            if not wants_sources and "\n**Sources:**" in reply:
                reply = reply.split("\n**Sources:**")[0].strip()
            return self.remember_reply(cleaned, reply)

        return self.generate_reply(cleaned)

    @staticmethod
    def is_web_knowledge_query(query: str) -> bool:
        """Detect whether a query asks for external, real-time, or internet knowledge."""
        q = query.lower()

        # 1. Awards, winners, championships (e.g. 'which game won goty 2019?', 'oscar winner')
        if re.search(r"\b(won|winner|winners|award|awards|goty|game of the year|oscar|oscars|emmy|grammy|championship|super bowl|world cup|tournament|champion|mvp)\b", q):
            return True

        # 2. Rankings, top lists, ratings (e.g. 'list top 10 imdb of all time?', 'highest rated')
        if re.search(r"\b(top\s*\d+|best\s*\d+|highest\s+rated|top\s+rated|imdb|box\s*office|ranking|rankings|leaderboard|all\s*time|best\s*selling|highest\s*grossing)\b", q):
            return True

        # 3. Creator, director, studio, or entity media catalogs (filmographies, discographies, games, movies)
        creators_and_entities = (
            r"\b(nolan|christopher nolan|tarantino|quentin tarantino|scorsese|martin scorsese|"
            r"spielberg|steven spielberg|cameron|james cameron|fincher|david fincher|"
            r"kubrick|stanley kubrick|villeneuve|denis villeneuve|hitchcock|alfred hitchcock|"
            r"miyazaki|hayao miyazaki|ridley scott|wes anderson|coen brothers|tim burton|"
            r"fromsoftware|rockstar|rockstar games|bethesda|kojima|hideo kojima|"
            r"naughty dog|valve|capcom|nintendo|cd projekt|square enix|bioware|bungie|ubisoft)\b"
        )
        media_terms = r"\b(movies|films|filmography|discography|bibliography|games|video games|tracks|albums|songs|releases)\b"
        if re.search(creators_and_entities, q) and (re.search(media_terms, q) or re.search(r"\b\d+\b", q)):
            return True

        if re.search(r"\b(filmography|discography|bibliography)\b", q):
            return True

        # 4. Numbered item count requests (e.g. 'name 10 nolan movies', '10 best sci-fi movies', '5 movies by spielberg')
        if re.search(r"\b\d+\s+(?:movies|films|games|video games|songs|albums|books|shows|series|directors|actors|inventions)\b", q):
            return True

        # 5. New / recent / upcoming releases or schedules
        has_recency = bool(re.search(
            r"\b(today|todays|today's|new|latest|recent|recently|upcoming|newest|released|relesed|drops|dropping|announced|current|now|this week|this month)\b", q
        ))
        has_media_topic = bool(re.search(
            r"\b(game|games|gaming|movie|movies|film|films|show|shows|series|anime|album|albums|"
            r"song|songs|book|books|phone|phones|smartphone|console|consoles|gpu|gpus|cpu|cpus|"
            r"patch|patches|update|updates|driver|drivers|release|releases|calendar)\b", q
        ))
        if has_recency and has_media_topic:
            return True

        # 6. Media queries with a specific year (e.g. 'game 2019', 'movies of 2020')
        if has_media_topic and re.search(r"\b(19\d\d|20\d\d)\b", q):
            return True

        # 7. Latest news / current events
        has_news = bool(re.search(
            r"\b(news|headlines|event|events|update|updates|patch|patches|happening|happened|what's up with|whats up with)\b", q
        ))
        if (has_recency or "today" in q or "current" in q or "right now" in q) and has_news:
            return True

        # 8. Real-time external stats, prices, weather
        if re.search(
            r"\b(latest version of|newest version of|current version of|weather in|temperature in|"
            r"stock price of|current price of|who won|which .* won)\b", q
        ):
            return True

        # 9. Natural web lookup / browse / fact phrasing
        if re.search(
            r"\b(look up|find out about|search (?:the )?(?:web|internet) for|browse the web for|"
            r"check (?:the )?(?:web|internet) for|who is the current|who directed|who wrote|who invented|"
            r"who created|cast of|capital of|when was .* created|when was .* founded)\b", q
        ):
            return True

        # 10. Open-domain science, natural history, world facts, and informational inquiries
        # E.g. 'why cant penguins fly', 'what is quantum computing', 'who was Marie Curie', 'can dogs eat chocolate'
        is_local_concept = bool(re.search(
            r"\b(python|code|programming|git|ram|vram|cpu|gpu|cuda|memory|disk|storage|battery|process|processes|api|os|operating system|time complexity)\b", q
        ))
        is_local_telemetry = bool(re.search(
            r"\b(my|mine|your|you|computer|pc|laptop|time|date|running|currently|right now|open window|tasklist)\b", q
        ))
        if not (is_local_concept or is_local_telemetry):
            # List / enumeration requests (e.g. 'name 10 nolan movies', 'list some horror movies', 'give me 5 adventure games')
            if re.search(
                r"^(?:please\s+)?(?:can\s+you\s+)?(?:name|list|give(?:\s+me)?|show(?:\s+me)?|tell\s+me)\s+(?:the\s+)?(?:\d+\s+|some\s+|top\s+|best\s+)?([a-z0-9_\s'-]+)$", q
            ):
                return True

            if re.search(
                r"^(?:please\s+)?(?:can\s+you\s+)?(?:why|how do|how does|how can|how to|how did|what causes|what makes|"
                r"where do|where does|where is|where are|where was|when did|when was|when is|"
                r"who is|who was|who were|what is|what are|what was|what were|"
                r"can [a-z]+|do [a-z]+|does [a-z]+|is [a-z]+|are [a-z]+|tell me about|explain\b)\b", q
            ):
                return True

        return False

    def generate_reply(self, user_input: str, use_history: bool = True) -> str:
        """Run the learned model directly, without keyword or personality replies.

        This is also the evaluation entry point: model quality must not include
        successful deterministic tool reports or memorized routing responses.
        """
        cleaned = user_input.strip()
        # Add user turn to context
        self.context.add_user_message(cleaned)

        # A standalone definition should not inherit a previous hardware report.
        # Preserve the actual session history; only narrow the generation prompt.
        if use_history:
            prompt, prompt_ids = self.context.get_context_prompt(assistant_eos=True)
        else:
            _, prompt, prompt_ids = self.context.formatter.fit_to_context_budget(
                [self.context.messages[-1]], self.tokenizer,
                max_prompt_tokens=self.context.max_prompt_budget,
                assistant_eos=True,
            )
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

        # Hallucination & Gibberish Guard (Phase 8.5)
        # If the local model degenerates into repetitive loops, nonsensical babble,
        # or fails quality checks on an unknown query, intercept before display!
        if self.is_hallucination_or_gibberish(response_text):
            self.pending_search_query = cleaned
            response_text = (
                "I don't know the answer to that from memory. "
                "Should I look it up on the internet, Dharsh?"
            )
        elif not response_text:
            response_text = "I am listening! How can I assist you with your project?"

        # Save assistant turn to context
        self.context.add_assistant_message(response_text)
        return response_text

    @staticmethod
    def is_hallucination_or_gibberish(text: str) -> bool:
        """Detect generation failures, repetitive loops, and word-salad hallucination."""
        cleaned = text.strip()
        if not cleaned or len(cleaned) < 5:
            return True

        # 0. Check for unspaced babble or corrupt non-word tokens (e.g. 28+ letters without space/hyphen)
        if any(len(w) > 28 for w in re.findall(r"\b[A-Za-z0-9_]+\b", cleaned) if not (w.startswith("http") or "/" in w)):
            return True

        if "\ufffd" in cleaned or "\x00" in cleaned:
            return True

        # Check known degeneration patterns, word-salad, and grammatical collapse early
        degen_patterns = [
            r"\ba good day\b.*\ba good day\b",
            r"\bset of parking\b",
            r"\bto shop the park\b",
            r"\bflower proof\b",
            r"\btime directly resulting in the same list\b",
            r"\b(voliders?|arix|kiday|curm|cosaurus|curmproper|trainate|yespecial)\b",
            r"\b(are|is|was|were)\s+the\s+[a-z]+\s+(is|are|was|were)\b",
            r"\b([a-z]{3,})\s+(?:of|in|to|for|with|by)\s+(?:the\s+|a\s+|an\s+)?\1\b",
            r"\ba normal ness\b",
            # Broken English & ungrammatical generation artifacts
            r"\b(don't|doesn't|didn't|not)\s+(?:have\s+been|been\s+not)\b",
            r"\b(have|has|had)\s+been\s+not\b",
            r"\bnot\s+exist\s+(?:you|me|him|her|them|us)\b",
            r"\bcan\s+(?:also\s+)?be\s+able\b",
            r"\b(is|are|was|were)\s+(?:the\s+)?[a-z]+\s+(?:provide|give|take|make)\b",
            r"\bwant\s+to\s+(?:useful|good|bad)\b",
            r"\b(a|an)\s+[a-z]+\s+(a|an)\s+[a-z]+\s+(a|an)\b",
            r"\bwith\s+an\s+[b-df-hj-np-tv-z][a-z]+\b",
            r"\b(cands|exist you|our got)\b",
        ]
        for pat in degen_patterns:
            if re.search(pat, cleaned, re.IGNORECASE):
                return True

        # Check for dangling broken preposition/article endings (e.g. 'and creative on a')
        if re.search(r"\b(on|at|in|to|for|with|of|a|an|the|and|or)\s*$", cleaned.rstrip(" .?!"), re.IGNORECASE):
            return True

        words = re.findall(r"\b[A-Za-z0-9]+\b", cleaned.lower())
        if len(words) < 3:
            if len(cleaned) > 25 and not cleaned.startswith("http"):
                return True
            return False

        # 1. Repeated n-grams (3-grams appearing 2+ times in text)
        if len(words) >= 8:
            from collections import Counter
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            tri_counts = Counter(trigrams)
            if any(cnt >= 2 for cnt in tri_counts.values()):
                return True

        # 2. Content word repetition / dominance
        stopwords = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
            "is", "are", "it", "its", "i", "you", "that", "this", "be", "with"
        }
        content_words = [w for w in words if w not in stopwords]
        if len(content_words) >= 4:
            from collections import Counter
            content_counts = Counter(content_words)
            _, max_count = content_counts.most_common(1)[0]
            if max_count >= 3 and (max_count / len(content_words)) > 0.35:
                return True

        # 3. Overall lexical diversity for longer text
        if len(words) >= 14:
            diversity = len(set(words)) / len(words)
            if diversity < 0.50:
                return True

        # 5. Short non-sentence fabricated fragment (e.g. 'Arix, Kiday (L)')
        if len(words) <= 4 and re.search(r"\b[A-Z][a-z]+,\s+[A-Z][a-z]+", cleaned):
            common_verbs = {"is", "are", "was", "were", "has", "have", "had", "can", "could", "will", "would", "do", "does", "did", "see", "show", "here"}
            if not any(w in common_verbs for w in words):
                return True

        return False

    def handle_followup_directive(self, user_input: str) -> Optional[str]:
        """Resolve conversational follow-up directives referencing the active dialogue exchange.

        Handles natural multi-turn requests like 'just the name', 'who was it?', 'what year?',
        'where was that?', 'in short', and pronoun queries like 'where was he born?'.
        """
        if len(self.context.messages) < 2:
            return None

        # Collect recent assistant messages for multi-turn context (up to last 6 turns)
        recent_assistant_texts = [
            m.content for m in self.context.messages[-6:]
            if m.role == Role.ASSISTANT and m.content
        ]
        if not recent_assistant_texts:
            return None

        last_content = recent_assistant_texts[-1]
        combined_context = "\n".join(reversed(recent_assistant_texts))
        q = user_input.lower().strip("?.! ")

        # 0. Conversational repair / clarification triggers ('what?', 'huh?', 'pardon?', 'what do you mean?')
        repair_triggers = (
            "what", "what?", "huh", "huh?", "pardon", "pardon?",
            "what do you mean", "what do you mean?", "i don't understand",
            "come again", "come again?", "what is that"
        )
        if q in repair_triggers:
            # Find the most recent substantive user message prior to this repair turn
            substantive_user_turns = [
                m.content for m in self.context.messages
                if m.role == Role.USER and m.content and m.content.lower().strip("?.! ") not in repair_triggers
            ]
            if substantive_user_turns:
                prior_q = substantive_user_turns[-1]
                if self.is_web_knowledge_query(prior_q) or len(prior_q.split()) >= 3:
                    from robbie.research import ResearchAssistant
                    ra = ResearchAssistant()
                    clarified = ra.answer_query(prior_q)
                    wants_sources = any(w in prior_q.lower() for w in ("source", "sources", "citation", "citations", "link", "links"))
                    if not wants_sources and "\n**Sources:**" in clarified:
                        clarified = clarified.split("\n**Sources:**")[0].strip()
                    return f"Let me clarify that with accurate information:\n\n{clarified}"
                else:
                    return f"I was responding to your previous question: \"{prior_q}\". Could you specify what details you'd like me to look into?"

        # 1. Name query follow-ups: 'just the name', 'only the name', 'give me the name', 'who?', 'who was it'
        name_triggers = (
            "just the name", "only the name", "the name", "what is the name", "whats the name",
            "give me the name", "tell me the name", "who", "who was it", "who is it", "who invented it",
            "who made it", "whose name", "his name", "her name", "name please"
        )
        is_asking_for_name = (
            q in name_triggers
            or bool(re.search(r"^(?:just|only|give me|tell me)\s+(?:the\s+)?name\s*$", q))
            or bool(re.search(r"^(?:what is|whats)\s+(?:the|his|her|its|their)\s+name\s*$", q))
            or bool(re.search(r"^(?:the\s+)?name(?:\s+please|\s+only)?\s*$", q))
        )
        if is_asking_for_name and not re.search(r"\b(movies|films|games|books|songs|albums|tracks|directors|actors|inventions|\d+)\b", q):
            for target_text in (last_content, combined_context):
                m_by = re.search(r"\b(?:invented|created|directed|written|designed|founded|discovered|won|awarded to)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", target_text)
                if m_by:
                    return f"{m_by.group(1).strip()}."
                m_was = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:who\s+)?(?:invented|created|directed|wrote|founded|developed|was a\b)", target_text)
                if m_was:
                    candidate = m_was.group(1).strip()
                    if not any(w in candidate.lower() for w in ("who", "the", "what", "here", "sources", "list")):
                        return f"{candidate}."
                m_was2 = re.search(r"\bwas\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", target_text)
                if m_was2:
                    candidate = m_was2.group(1).strip()
                    if not any(w in candidate.lower() for w in ("who", "the", "what", "here", "sources", "list")):
                        return f"{candidate}."
                matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", target_text)
                filtered = [
                    m for m in matches
                    if not any(w in m.lower() for w in ("invented", "history", "weirdo", "who", "here is", "sources", "based on", "overview"))
                ]
                if filtered:
                    return f"{filtered[0]}."

        # 2. Date / Year follow-ups: 'what year?', 'which year?', 'when was it?', 'when?'
        year_triggers = ("when", "when was it", "when was that", "what year", "which year", "what date", "which date")
        is_asking_for_year = (
            q in year_triggers
            or bool(re.search(r"^(?:just|only|give me|tell me)\s+(?:the\s+)?(?:year|date)\s*$", q))
            or bool(re.search(r"^(?:what|which)\s+(?:year|date)\s*$", q))
            or bool(re.search(r"^(?:when)\s+(?:was\s+it|was\s+that)\s*$", q))
        )
        if is_asking_for_year and not re.search(r"\b(movies|films|games|books|songs|albums|\d+)\b", q):
            for target_text in (last_content, combined_context):
                m_full_date = re.search(r"\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b", target_text)
                if m_full_date:
                    return f"{m_full_date.group(1)}."
                m_year = re.findall(r"\b(1[4-9]\d\d|20\d\d)\b", target_text)
                if m_year:
                    for clause in re.split(r"(?: - |\n|(?<=[.!?])\s+)", target_text):
                        if m_year[0] in clause and not clause.lower().startswith("sources"):
                            cleaned_c = re.sub(r"^[-\s*•]+", "", clause).strip()
                            if len(cleaned_c) >= 15:
                                return cleaned_c
                    return f"In {m_year[0]}."

        # 3. Location / Place follow-ups: 'where?', 'which city?', 'which country?', 'just the country', 'just the city'
        place_triggers = (
            "where", "where was that", "where was it", "which city", "what city",
            "which country", "what country", "which state", "what state", "just the city",
            "just the country", "the location", "the place", "where at", "which place"
        )
        is_asking_for_place = (
            q in place_triggers
            or bool(re.search(r"^(?:just|only|give me|tell me)\s+(?:the\s+)?(?:city|country|location|place)\s*$", q))
            or bool(re.search(r"^(?:what|which)\s+(?:city|country|place|location)\s*$", q))
            or bool(re.search(r"^(?:where)\s+(?:was\s+it|was\s+that|at)\s*$", q))
        )
        if is_asking_for_place and not re.search(r"\b(movies|films|games|books|songs|albums|\d+)\b", q):
            for target_text in (last_content, combined_context):
                m_place = re.search(r"\b(?:born on [^.]+ in|born in|located in|based in|held in|in)\s+([A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+)?)\b", target_text)
                if m_place:
                    cand = m_place.group(1).strip()
                    if not any(w in cand.lower() for w in ("the", "here", "sources", "early", "late", "what", "which")):
                        return f"{cand}."

        # 4. Summary / Short version: 'in short', 'summarize that', 'briefly', 'tldr'
        summary_triggers = ("in short", "short version", "briefly", "summarize", "summary", "in one sentence", "in a nutshell", "tldr")
        if any(q.startswith(trig) for trig in summary_triggers):
            for target_text in (last_content, combined_context):
                lines = [l.strip().lstrip("-•* ") for l in target_text.split("\n") if len(l.strip()) > 30 and not l.lower().startswith("sources") and not l.lower().startswith("here is what")]
                if lines:
                    return lines[0]

        # Helper to extract the primary named entity subject from context
        def get_active_subject() -> Optional[str]:
            for target_text in (last_content, combined_context):
                m_was = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:who\s+)?(?:invented|created|directed|wrote|founded|developed|was a\b)", target_text)
                if m_was:
                    cand = m_was.group(1).strip()
                    if not any(w in cand.lower() for w in ("who", "the", "what", "here", "sources", "list")):
                        return cand
                matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", target_text)
                filtered = [
                    m for m in matches
                    if not any(w in m.lower() for w in ("invented", "history", "weirdo", "who", "here is", "sources", "based on", "overview"))
                ]
                if filtered:
                    return filtered[0]
            return None

        # 5. Elaboration / Continuation: 'tell me more', 'more details', 'expand on that', 'what else'
        elaboration_triggers = (
            "tell me more", "more details", "expand on that", "can you elaborate", "elaborate",
            "what else", "anything else", "more info", "tell me more about that", "give me more info",
            "give me more details", "keep going", "continue"
        )
        if any(q == trig or q.startswith(trig) for trig in elaboration_triggers):
            subject = get_active_subject()
            recent_user_queries = [
                m.content for m in self.context.messages[-6:]
                if m.role == Role.USER and m.content and m.content != user_input
            ]
            if subject:
                from robbie.research import ResearchAssistant
                ra = ResearchAssistant()
                return ra.answer_query(f"Tell me more details about {subject}")
            elif recent_user_queries:
                from robbie.research import ResearchAssistant
                ra = ResearchAssistant()
                return ra.answer_query(f"More details on {recent_user_queries[-1]}")

        # 6. Anaphora resolution: If query uses third-person pronouns ('he', 'she', 'it', 'they', 'him', 'her')
        # resolve against the subject from recent turns and execute web research
        has_pronoun = bool(re.search(r"\b(he|she|it|they|him|her|his|their)\b", q))
        is_question = bool(re.search(r"^(?:where|when|what|why|how|who|did|does|is|was)\b", q) or "?" in user_input)
        if has_pronoun and is_question:
            subject = get_active_subject()
            if subject:
                resolved_q = re.sub(r"\b(he|she|it|they)\b", subject, user_input, flags=re.IGNORECASE)
                resolved_q = re.sub(r"\b(his|her|their)\b", f"{subject}'s", resolved_q, flags=re.IGNORECASE)
                from robbie.research import ResearchAssistant
                ra = ResearchAssistant()
                return ra.answer_query(resolved_q)

        # 7. Recommendation / Best Pick follow-ups:
        # e.g. 'Which one would you recommend for a person who like thrillers?',
        #      'Which one should I watch/play first?', 'What do you recommend?', 'Which is the best?'
        is_rec_query = bool(re.search(
            r"\b(recommend|recommendation|should i (?:watch|play|read|start with|check out)|which (?:one )?is (?:the )?best|which (?:one )?do you (?:like|suggest|recommend)|top pick|best pick)\b",
            q,
        ))
        if is_rec_query:
            # Check if recent assistant context contains a list of items
            for target_text in (last_content, combined_context):
                numbered_items = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s*\*\*?([^*]+?)\*\*?\s*(?:\((\d{4})\))?\s*(?:-\s*([^\n]+))?", target_text)
                if not numbered_items:
                    numbered_items = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s*([^(\n]+?)\s*\((19\d\d|20\d\d)\)\s*(?:-\s*([^\n]+))?", target_text)
                if numbered_items:
                    subject = get_active_subject()
                    subject_str = f" by **{subject}**" if subject else ""
                    genres = [
                        ("thriller", ["Seven", "Zodiac", "Gone Girl", "Panic Room", "The Game", "Memento", "Insomnia", "Death Stranding"]),
                        ("mystery", ["Seven", "Zodiac", "Memento", "The Prestige", "Policenauts", "Snatcher"]),
                        ("mind-bending", ["Inception", "Fight Club", "The Game", "Memento", "The Prestige"]),
                        ("twist", ["Fight Club", "The Game", "Seven", "The Prestige"]),
                        ("action", ["The Dark Knight", "Inception", "Metal Gear Solid", "Aliens", "The Killer"]),
                        ("sci-fi", ["Interstellar", "Inception", "Death Stranding", "Policenauts", "Snatcher"]),
                        ("stealth", ["Metal Gear Solid", "Metal Gear", "Metal Gear 2"]),
                        ("psychological", ["Fight Club", "Seven", "Gone Girl", "Memento", "Silent Hills"]),
                        ("horror", ["Alien 3", "Seven", "P.T."]),
                    ]
                    matched_genre = None
                    recommended_candidates = []
                    for genre_name, genre_titles in genres:
                        if genre_name in q:
                            matched_genre = genre_name
                            for item_tuple in numbered_items:
                                title = item_tuple[0].strip()
                                yr = item_tuple[1] if len(item_tuple) > 1 else ""
                                desc = item_tuple[2] if len(item_tuple) > 2 else ""
                                if any(gt.lower() in title.lower() for gt in genre_titles) or (desc and genre_name in desc.lower()):
                                    yr_str = f" ({yr})" if yr else ""
                                    recommended_candidates.append(f"**{title}**{yr_str}")
                            break

                    if recommended_candidates:
                        picks = ", ".join(recommended_candidates[:2])
                        extra = f" (along with {recommended_candidates[2]})" if len(recommended_candidates) >= 3 else ""
                        return (
                            f"For fans of {matched_genre}s{subject_str}, I would especially recommend {picks}{extra}. "
                            "They stand out for their gripping tension, masterclass pacing, and atmospheric depth."
                        )
                    else:
                        top_picks = []
                        for item_tuple in numbered_items[:2]:
                            title = item_tuple[0].strip()
                            yr = item_tuple[1] if len(item_tuple) > 1 else ""
                            yr_str = f" ({yr})" if yr else ""
                            top_picks.append(f"**{title}**{yr_str}")
                        return (
                            f"From the list above{subject_str}, my top recommendations are {' and '.join(top_picks)}. "
                            "They represent the definitive entry points and most acclaimed highlights."
                        )

        return None
