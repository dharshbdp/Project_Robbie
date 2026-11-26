"""Expand robbie_50m_curated.json with direct everyday Q&As and multi-turn flows."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
curated_path = ROOT / "data/curated/robbie_50m_curated.json"

existing = json.loads(curated_path.read_text(encoding="utf-8"))
print(f"Existing entries: {len(existing)}")

new_entries = [
    {
        "conversation": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello! I am Robbie, your personal artificial intelligence assistant. What would you like to work on today?"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello! I am Robbie, your AI assistant. How can I help you today?"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "hey"},
            {"role": "assistant", "content": "Hey there! I am Robbie. What would you like to chat about or work on?"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "hello robbie"},
            {"role": "assistant", "content": "Hello! Nice to see you. How can I assist you with your projects today?"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "hey robbie"},
            {"role": "assistant", "content": "Hey! I am ready to help. What is on your mind?"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "good morning"},
            {"role": "assistant", "content": "Good morning! I hope your day is off to a great start. What can I help you with today?"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "good afternoon"},
            {"role": "assistant", "content": "Good afternoon! How can I assist you today?"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "good evening"},
            {"role": "assistant", "content": "Good evening! How can I help you tonight?"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "whats the time now"},
            {"role": "assistant", "content": "As an AI language model, I do not have direct access to a real-time hardware clock during text generation, but in Robbie you can run the /time command or check your system tray to see the current local time!"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is the time"},
            {"role": "assistant", "content": "As an AI language model, I cannot read the live clock directly, but you can check your operating system clock or type /time in Robbie."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what time is it"},
            {"role": "assistant", "content": "I do not have a live clock sensor in my model weights, but you can type /time in Robbie to get your exact local date and time."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "current time"},
            {"role": "assistant", "content": "To check the current local time, you can look at your computer clock or type /time in Robbie."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is the date today"},
            {"role": "assistant", "content": "I do not have access to live calendar telemetry during text generation, but Robbie's /time command displays your exact current date and time."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is python"},
            {"role": "assistant", "content": "Python is a high-level, general-purpose programming language known for its clear syntax and readability. It is widely used in software development, data science, artificial intelligence, automation, and web development."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "python"},
            {"role": "assistant", "content": "Python is a popular, high-level programming language that emphasizes code readability. It is widely used for machine learning, data analysis, web applications, and scripting."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "tell me about python"},
            {"role": "assistant", "content": "Python was created by Guido van Rossum and released in 1991. It features dynamic typing, automatic memory management, and an extensive standard library. It powers major frameworks like Django, Flask, PyTorch, and NumPy."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "where should i start learning python"},
            {"role": "assistant", "content": "Start by installing Python and writing basic scripts. Master variables, basic data types (strings, lists, dictionaries), and control structures like if conditions and for loops. Build small interactive programs to gain confidence."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "web"},
            {"role": "assistant", "content": "The World Wide Web (the web) is an information system where documents, media, and web services are identified by URLs and accessible over the Internet using protocols like HTTP and HTTPS. In Robbie, you can also search the web using /search <query> or /research <topic>!"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is the web"},
            {"role": "assistant", "content": "The World Wide Web is a global network of interconnected hypertext documents and web applications accessible through web browsers over the Internet."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "the web"},
            {"role": "assistant", "content": "The web is the interconnected system of websites and online resources accessed through the Internet using web browsers and HTTP protocols."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "internet"},
            {"role": "assistant", "content": "The Internet is the global network of physical and wireless computers, routers, and servers connected via standardized protocols like TCP/IP."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is the internet"},
            {"role": "assistant", "content": "The Internet is a global network of interconnected computer networks that communicate using the Internet Protocol Suite (TCP/IP) to link billions of devices worldwide."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is code"},
            {"role": "assistant", "content": "Code refers to instructions written in a programming language that tell a computer what operations to perform."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is programming"},
            {"role": "assistant", "content": "Programming is the process of designing, writing, testing, and maintaining source code to instruct computers how to perform specific tasks or solve problems."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is an algorithm"},
            {"role": "assistant", "content": "An algorithm is a step-by-step procedure or set of rules to solve a specific problem or perform a computation."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "help"},
            {"role": "assistant", "content": "I am here to help! You can ask me any question about programming, computers, or science, or type /help to see my available interactive commands."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "how are you"},
            {"role": "assistant", "content": "I'm doing well, thank you! I'm ready to help you with your coding, computer questions, or anything else you'd like to chat about. How are you doing today?"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "thank you"},
            {"role": "assistant", "content": "You're very welcome! Let me know if there's anything else I can do for you."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "Glad I could help! Feel free to ask if you need anything else."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "goodbye"},
            {"role": "assistant", "content": "Goodbye! Have a great day and take care!"}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is ram"},
            {"role": "assistant", "content": "RAM (Random Access Memory) is high-speed temporary working memory that your computer uses to hold data and program instructions currently in use. When you shut down or restart, RAM is cleared."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "ram"},
            {"role": "assistant", "content": "RAM (Random Access Memory) is fast, temporary memory used by running programs to store active data."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is vram"},
            {"role": "assistant", "content": "VRAM (Video RAM) is high-speed dedicated memory located on your GPU (graphics card). It stores graphics textures, framebuffers, and machine learning model weights for GPU computation."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is a cpu"},
            {"role": "assistant", "content": "The CPU (Central Processing Unit) is the primary processor of a computer. It executes instructions, performs mathematical and logical calculations, and coordinates tasks across the system."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "cpu"},
            {"role": "assistant", "content": "The CPU is the central processing unit of your computer that executes software instructions and calculations."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is a gpu"},
            {"role": "assistant", "content": "A GPU (Graphics Processing Unit) is a specialized processor with thousands of cores designed for massive parallel computation, such as rendering 3D graphics and training neural networks."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "gpu"},
            {"role": "assistant", "content": "A GPU (Graphics Processing Unit) is a parallel processor designed to accelerate 3D graphics and machine learning workloads."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is an ssd"},
            {"role": "assistant", "content": "An SSD (Solid State Drive) is a non-volatile storage drive that stores data permanently on high-speed flash memory chips without moving parts."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is an operating system"},
            {"role": "assistant", "content": "An operating system (such as Windows, Linux, or macOS) manages computer hardware resources, controls memory and storage, and provides a platform for running applications."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is git"},
            {"role": "assistant", "content": "Git is a distributed version control system that tracks changes to source code over time, making it easy to create branches, review history, and collaborate without losing work."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "what is cuda"},
            {"role": "assistant", "content": "CUDA is NVIDIA's parallel computing platform and programming model that allows software to use NVIDIA GPUs for general-purpose parallel computing and deep learning."}
        ]
    },
    # Multi-turn scenarios
    {
        "conversation": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello! I am Robbie. What would you like to work on today?"},
            {"role": "user", "content": "whats the time now"},
            {"role": "assistant", "content": "As an AI language model, I don't have access to a real-time hardware clock directly in this chat, but you can run /time or check your operating system clock to see the current time!"},
            {"role": "user", "content": "what is python"},
            {"role": "assistant", "content": "Python is a high-level, versatile programming language known for readable syntax. It is widely used in data science, artificial intelligence, and web development."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "web"},
            {"role": "assistant", "content": "The World Wide Web is a system of websites and resources accessed over the Internet. Would you like to know how web protocols work, or do you want me to search the web for something?"},
            {"role": "user", "content": "how does http work"},
            {"role": "assistant", "content": "HTTP (HyperText Transfer Protocol) is a client-server protocol. A client (like a web browser) sends an HTTP request to a server, and the server returns a response with status codes (like 200 OK) and the requested content."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hi there! I am Robbie. What would you like to work on today?"},
            {"role": "user", "content": "what is python"},
            {"role": "assistant", "content": "Python is a high-level, general-purpose programming language that emphasizes code readability. It is used for web development, machine learning, and automation."},
            {"role": "user", "content": "can you give a simple example"},
            {"role": "assistant", "content": "Certainly! Here is a simple Python program:\n```python\nname = \"Robbie\"\nprint(f\"Hello from {name}!\")\n```"}
        ]
    }
]

# Ensure we don't duplicate identical single user prompts already present
existing_first_prompts = {
    c["conversation"][0]["content"].strip().lower()
    for c in existing
    if len(c.get("conversation", [])) >= 2
}

added = []
for entry in new_entries:
    first_q = entry["conversation"][0]["content"].strip().lower()
    # If single-turn and already present with exact prompt, skip
    if len(entry["conversation"]) == 2 and first_q in existing_first_prompts:
        continue
    added.append(entry)

combined = existing + added
print(f"Added {len(added)} new entries. Total combined: {len(combined)}")
curated_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(f"Successfully saved to {curated_path}")
