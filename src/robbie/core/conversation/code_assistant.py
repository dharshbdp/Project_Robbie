"""Python Code Snippet Assistant for Project Robbie.

Provides clean, structured, executable Python code snippets formatted for Robbie's
Chat and Code Space Workspace. Prevents degeneration on programming queries.
"""

import re
from typing import Optional, Tuple


class PythonCodeAssistant:
    """Detects and fulfills Python code generation and snippet requests."""

    CODE_PATTERNS = [
        r"\b(?:can you\s+)?(?:write|give|give me|show|show me|create|generate|provide)\s+(?:a\s+|an\s+|some\s+)?(?:small\s+|quick\s+|simple\s+)?(?:python\s+)?(?:code\s+snippets?|coding\s+snippets?|code\s+samples?|snippets?|scripts?|codes?|functions?|programs?)\b",
        r"\b(?:python\s+)?(?:code\s+snippets?|coding\s+snippets?|code\s+samples?|snippets?)\b",
        r"\bpython\s+(?:scripts?|codes?|functions?|programs?)\b",
        r"\b(?:write\s+)?hello\s+world\b",
        r"\b(?:blender|bpy)\b",
        r"\b(?:timsort|bubble\s+sort|quick\s+sort|merge\s+sort)\b",
        r"\b(?:sort|sorting)\s+(?:an?\s+)?(?:array|list|numbers|in\s+python|algorithm)\b",
        r"\bhow\s+to\s+(?:read|write|parse|fetch|sort|reverse|loop|filter|calculate)\s+.*(?:in\s+python)?\b",
        r"\bpython\s+.*to\s+(?:read|write|parse|fetch|sort|reverse|loop|filter|calculate)\b",
    ]

    @classmethod
    def is_code_request(cls, query: str) -> bool:
        """Return True if the query is asking for a code snippet, script, or programming example."""
        q = query.lower().strip()
        # Avoid matching system commands or telemetry
        if any(q.startswith(cmd) for cmd in ("/sys", "/mem", "/search", "/research", "/proc", "/time", "/date")):
            return False

        # Exclude questions about definitions (handled by definition / concept handler)
        if re.search(r"^(?:what is|define|explain)\s+(?:python|code|programming)$", q):
            return False

        return any(re.search(pat, q) for pat in cls.CODE_PATTERNS)

    @classmethod
    def generate_snippet(cls, query: str) -> str:
        """Generate a structured markdown response with executable Python code."""
        q = query.lower().strip()

        # 1. Hello World / Basic Greeting
        if "hello world" in q or "hello-world" in q:
            return (
                "Here is a classic **Hello World** Python snippet:\n\n"
                "```python\n"
                "# Hello World Demonstration\n"
                "def main() -> None:\n"
                "    message: str = \"Hello, World! Welcome to Robbie Code Studio.\"\n"
                "    print(message)\n\n"
                "if __name__ == \"__main__\":\n"
                "    main()\n"
                "```\n\n"
                "You can copy this snippet or click **Send to Editor** to run or modify it in your Workspace."
            )

        # 2. File Reading
        if "read" in q and ("file" in q or "text" in q):
            return (
                "Here is a clean Python snippet to safely read a file using `pathlib`:\n\n"
                "```python\n"
                "from pathlib import Path\n\n"
                "def read_file_contents(file_path: str) -> str:\n"
                "    \"\"\"Safely read and return text from a file.\"\"\"\n"
                "    path = Path(file_path)\n"
                "    if not path.is_file():\n"
                "        raise FileNotFoundError(f\"File not found: {file_path}\")\n"
                "    return path.read_text(encoding=\"utf-8\")\n\n"
                "# Example usage:\n"
                "if __name__ == \"__main__\":\n"
                "    try:\n"
                "        data = read_file_contents(\"example.txt\")\n"
                "        print(f\"File content ({len(data)} characters):\\n{data}\")\n"
                "    except FileNotFoundError as err:\n"
                "        print(f\"Notice: {err}\")\n"
                "```\n\n"
                "Use `read_text(encoding='utf-8')` for automatic file handle closing and UTF-8 handling."
            )

        # 3. File Writing
        if "write" in q and "file" in q:
            return (
                "Here is a Python snippet to safely write text to a file:\n\n"
                "```python\n"
                "from pathlib import Path\n\n"
                "def write_to_file(file_path: str, content: str) -> int:\n"
                "    \"\"\"Write text content to file, creating parent directories if needed.\"\"\"\n"
                "    path = Path(file_path)\n"
                "    path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    return path.write_text(content, encoding=\"utf-8\")\n\n"
                "if __name__ == \"__main__\":\n"
                "    bytes_written = write_to_file(\"output.txt\", \"Hello from Project Robbie!\\n\")\n"
                "    print(f\"Wrote {bytes_written} characters successfully.\")\n"
                "```\n\n"
                "Click **Send to Editor** to test this directly in your workspace."
            )

        # 4. JSON parsing / handling
        if "json" in q:
            return (
                "Here is a Python snippet for JSON serialization and parsing:\n\n"
                "```python\n"
                "import json\n\n"
                "# Sample dictionary\n"
                "user_data = {\n"
                "    \"name\": \"Robbie Developer\",\n"
                "    \"model\": \"Robbie-300M\",\n"
                "    \"active\": True,\n"
                "    \"capabilities\": [\"Computer Awareness\", \"Web Research\", \"Code Studio\"]\n"
                "}\n\n"
                "# 1. Convert Python dict to JSON string\n"
                "json_str = json.dumps(user_data, indent=2)\n"
                "print(\"Serialized JSON:\")\n"
                "print(json_str)\n\n"
                "# 2. Parse JSON string back into Python dict\n"
                "parsed = json.loads(json_str)\n"
                "print(f\"\\nParsed back model name: {parsed['model']}\")\n"
                "```\n\n"
                "This uses Python's standard `json` module with pretty-print formatting."
            )

        # 5. Reverse String / Palindrome
        if "reverse" in q and "string" in q:
            return (
                "Here is a Python snippet demonstrating string reversal and palindrome checking:\n\n"
                "```python\n"
                "def reverse_string(text: str) -> str:\n"
                "    \"\"\"Reverse a string using Python slicing.\"\"\"\n"
                "    return text[::-1]\n\n"
                "def is_palindrome(text: str) -> bool:\n"
                "    \"\"\"Check if text reads the same forwards and backwards.\"\"\"\n"
                "    clean = \"\".join(c.lower() for c in text if c.isalnum())\n"
                "    return clean == clean[::-1]\n\n"
                "if __name__ == \"__main__\":\n"
                "    sample = \"Racecar\"\n"
                "    print(f\"Original: {sample}\")\n"
                "    print(f\"Reversed: {reverse_string(sample)}\")\n"
                "    print(f\"Is palindrome? {is_palindrome(sample)}\")\n"
                "```\n\n"
                "Slicing with `[::-1]` runs in $O(n)$ time in optimized CPython."
            )

        # 6. Fibonacci
        if "fibonacci" in q:
            return (
                "Here is an efficient Python Fibonacci generator snippet:\n\n"
                "```python\n"
                "from typing import Generator\n\n"
                "def fibonacci_sequence(count: int) -> Generator[int, None, None]:\n"
                "    \"\"\"Generate the first N Fibonacci numbers with O(1) memory.\"\"\"\n"
                "    a, b = 0, 1\n"
                "    for _ in range(count):\n"
                "        yield a\n"
                "        a, b = b, a + b\n\n"
                "if __name__ == \"__main__\":\n"
                "    first_10 = list(fibonacci_sequence(10))\n"
                "    print(f\"First 10 Fibonacci numbers: {first_10}\")\n"
                "```\n\n"
                "Using a generator ensures $O(1)$ auxiliary space complexity."
            )

        # 7. Web Requests / HTTP
        if any(w in q for w in ("http", "api", "request", "fetch", "url", "web")):
            return (
                "Here is a Python snippet to fetch data from an HTTP API using standard library:\n\n"
                "```python\n"
                "import urllib.request\n"
                "import json\n\n"
                "def fetch_json_api(url: str, timeout: float = 8.0) -> dict:\n"
                "    \"\"\"Fetch and decode JSON from an HTTP URL.\"\"\"\n"
                "    req = urllib.request.Request(\n"
                "        url,\n"
                "        headers={\"User-Agent\": \"RobbieStudio/1.0\"}\n"
                "    )\n"
                "    with urllib.request.urlopen(req, timeout=timeout) as resp:\n"
                "        raw_bytes = resp.read()\n"
                "        return json.loads(raw_bytes.decode(\"utf-8\"))\n\n"
                "# Example usage:\n"
                "if __name__ == \"__main__\":\n"
                "    try:\n"
                "        data = fetch_json_api(\"https://api.github.com/zen\")\n"
                "        print(\"API Result:\", data)\n"
                "    except Exception as err:\n"
                "        print(f\"Request failed: {err}\")\n"
                "```\n\n"
                "This relies entirely on Python standard library without external dependencies."
            )

        # 8. Sorting / Algorithms (matching user's preferred layout from Image 1)
        if "sort" in q:
            return (
                "If you want the best general-purpose sorting algorithm in Python, use Python's built-in `sorted()` or `.sort()` . "
                "Python uses **Timsort**, a highly optimized, stable hybrid sorting algorithm with $O(n \\log n)$ worst-case time complexity.\n\n"
                "1. Best for real-world use — Timsort\n\n"
                "```python\n"
                "def best_sort(arr):\n"
                "    return sorted(arr)\n\n"
                "numbers = [64, 34, 25, 12, 22, 11, 90]\n"
                "print(best_sort(numbers))\n"
                "# [11, 12, 22, 25, 34, 64, 90]\n"
                "```"
            )

        # 9. Blender Python / bpy
        if "blender" in q or "bpy" in q:
            return (
                "Here is an optimized Blender Python (`bpy`) script to procedurally create a smooth 3D object with a metallic material:\n\n"
                "1. Procedural 3D Mesh in Blender\n\n"
                "```python\n"
                "import bpy\n\n"
                "def create_procedural_object(name=\"RobbieCube\", location=(0, 0, 1)):\n"
                "    \"\"\"Spawn a smoothed primitive mesh with a Principled BSDF material in Blender.\"\"\"\n"
                "    # 1. Add cube mesh\n"
                "    bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)\n"
                "    obj = bpy.context.active_object\n"
                "    obj.name = name\n\n"
                "    # 2. Add Subdivision Surface modifier\n"
                "    subsurf = obj.modifiers.new(name=\"Subsurf\", type='SUBSURF')\n"
                "    subsurf.levels = 2\n\n"
                "    # 3. Create and assign material\n"
                "    mat = bpy.data.materials.new(name=f\"{name}_Material\")\n"
                "    mat.use_nodes = True\n"
                "    bsdf = mat.node_tree.nodes.get(\"Principled BSDF\")\n"
                "    if bsdf:\n"
                "        bsdf.inputs['Base Color'].default_value = (0.2, 0.6, 1.0, 1.0)\n"
                "        bsdf.inputs['Metallic'].default_value = 0.8\n"
                "        bsdf.inputs['Roughness'].default_value = 0.2\n\n"
                "    obj.data.materials.append(mat)\n"
                "    print(f\"Successfully generated {obj.name} in Blender scene!\")\n"
                "    return obj\n\n"
                "if __name__ == \"__main__\":\n"
                "    create_procedural_object()\n"
                "```\n\n"
                "You can run this directly in Blender's Scripting workspace or test it via the Robbie Workspace editor."
            )

        # 10. Random / Password Generator
        if "random" in q or "password" in q:
            return (
                "Here is a cryptographically secure random password generator in Python:\n\n"
                "```python\n"
                "import secrets\n"
                "import string\n\n"
                "def generate_secure_password(length: int = 16) -> str:\n"
                "    \"\"\"Generate a strong, cryptographically secure random password.\"\"\"\n"
                "    alphabet = string.ascii_letters + string.digits + \"!@#$%^&*()-_=+\"\n"
                "    return \"\".join(secrets.choice(alphabet) for _ in range(length))\n\n"
                "if __name__ == \"__main__\":\n"
                "    print(f\"Secure Password: {generate_secure_password(16)}\")\n"
                "```\n\n"
                "Uses `secrets` instead of `random` for cryptographic safety."
            )

        # 11. General / Default Python Snippet
        return (
            "Here is a Python code snippet with type annotations, docstrings, and a CLI test runner:\n\n"
            "1. Task Management Class in Python\n\n"
            "```python\n"
            "from typing import List, Dict, Any\n"
            "import time\n\n"
            "class TaskTracker:\n"
            "    \"\"\"Simple task manager tracking completion times.\"\"\"\n"
            "    def __init__(self, owner: str = \"Developer\") -> None:\n"
            "        self.owner = owner\n"
            "        self.tasks: List[Dict[str, Any]] = []\n\n"
            "    def add_task(self, title: str) -> None:\n"
            "        self.tasks.append({\n"
            "            \"title\": title,\n"
            "            \"timestamp\": time.time(),\n"
            "            \"completed\": False\n"
            "        })\n"
            "        print(f\"[TaskTracker] Added task: '{title}' for {self.owner}\")\n\n"
            "    def list_tasks(self) -> None:\n"
            "        for idx, task in enumerate(self.tasks, 1):\n"
            "            status = \"[x]\" if task[\"completed\"] else \"[ ]\"\n"
            "            print(f\"  {idx}. {status} {task['title']}\")\n\n"
            "if __name__ == \"__main__\":\n"
            "    tracker = TaskTracker(owner=\"Robbie\")\n"
            "    tracker.add_task(\"Evaluate 300M transformer context\")\n"
            "    tracker.add_task(\"Execute real-time web research\")\n"
            "    tracker.list_tasks()\n"
            "```\n\n"
            "You can click **Run** to execute this snippet or **Send to Editor** to open this in your Robbie Workspace code editor!"
        )
