# bsy-clippy

`bsy-clippy.py` is a lightweight Python client for interacting with an [Ollama](https://ollama.ai) server.  

It supports both **batch (stdin) mode** for one-shot prompts and **interactive mode** for chatting directly in the terminal.  
You can also load **system prompts** from a file to guide the LLM’s behavior.

---

## Features

- Connects to Ollama API over HTTP (`/api/generate`).
- Defaults to:
  - IP: `172.20.0.100`
  - Port: `11434`
  - Model: `qwen3:1.7b`
  - Mode: `batch` (wait for full output)
  - System prompt file: `bsy-clippy.txt`
- Configurable parameters:
  - `-i` / `--ip` → Ollama server IP
  - `-p` / `--port` → Ollama server port
  - `-M` / `--model` → model name
  - `-m` / `--mode` → output mode (`stream` or `batch`)
  - `-t` / `--temperature` → sampling temperature (default: `0.7`)
  - `-s` / `--system-file` → path to a text file with system instructions
  - `-u` / `--user-prompt` → extra user instructions prepended before the data payload
- Two modes of operation:
  - **Batch mode** (default) → waits until the answer is complete, then prints only the final result.
  - **Stream mode** → shows response in real-time, tokens appear as they are generated.
- Colored terminal output:
  - **Yellow** = streaming tokens (the model’s “thinking” in progress).
  - **Default terminal color** = final assembled answer.

---

## Installation

1. Clone or copy this repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### System prompt file

By default, `bsy-clippy.py` will load instructions from `bsy-clippy.txt` if it exists.  
You can change this with `--system-file`.

Example **bsy-clippy.txt**:

```
You are a helpful assistant specialized in cybersecurity.
Always explain your reasoning clearly, and avoid unnecessary markdown formatting.
```

These lines will be sent to the LLM before every user prompt.

### User prompt parameter

Use `--user-prompt "Classify the following log:"` when piping data so the model receives:

```
system prompt (if any)

user prompt text

data from stdin or interactive input
```

---

### Interactive mode (default = batch)

Run without piping input:

```bash
python3 bsy-clippy.py
```

Example session in **batch mode**:

```
You: Hello!
Hello! How can I assist you today? 😊
```

To force **streaming mode**:

```bash
python3 bsy-clippy.py --mode stream
```

Streaming session looks like:

```
You: Hello!
LLM (thinking): <think>
Reasoning step by step...
</think>
Hello! How can I assist you today? 😊
```

---

### Batch mode (stdin)

Pipe input directly:

```bash
echo "Tell me a joke" | python3 bsy-clippy.py
```

Output:

```
Why don’t scientists trust atoms? Because they make up everything!
```

---

### Forcing modes

```bash
python3 bsy-clippy.py --mode batch
python3 bsy-clippy.py --mode stream
```

---

### Adjusting temperature

```bash
python3 bsy-clippy.py --temperature 0.2
python3 bsy-clippy.py --temperature 1.2
```

---

### Custom server and model

```bash
python3 bsy-clippy.py --ip 127.0.0.1 --port 11434 --model llama2
```

---

## Requirements

See [`requirements.txt`](requirements.txt).
