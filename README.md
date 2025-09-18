# bsy-clippy

`bsy-clippy.py` is a lightweight Python client for interacting with an [Ollama](https://ollama.ai) server.  

It supports both **batch (stdin) mode** for one-shot prompts and **interactive mode** for chatting directly in the terminal.  
Responses are shown with **colored output** so you can distinguish when the model is *thinking* and when it has finished its answer.

---

## Features

- Connects to Ollama API over HTTP (`/api/generate`).
- Defaults to:
  - IP: `172.20.0.100`
  - Port: `11434`
  - Model: `qwen3:1.7b`
- Configurable parameters:
  - `--ip` → Ollama server IP
  - `--port` → Ollama server port
  - `--model` → model name
  - `--mode` → output mode (`stream` or `batch`)
- Two modes of operation:
  - **Batch mode** (stdin) → waits until the answer is complete, then prints only the final result.
  - **Interactive mode** → chat in the terminal with real-time token streaming.
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

### Interactive mode (default)

Run without piping input:

```bash
python3 bsy-clippy.py
```

Example session:

```
You: Hello!
LLM (thinking):  How are you doing today?
How are you doing today?
```

- The first line in **yellow** is the token-by-token streaming output.  
- The last line in **default color** is the final complete answer.  
- Exit with `exit` or `Ctrl+C`.

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

Here the response is only shown **once the model has finished** generating.

---

### Forcing modes

Use the `--mode` flag to override defaults:

- Force **batch** (wait until the end, only final answer):

```bash
python3 bsy-clippy.py --mode batch
```

- Force **streaming** (see tokens appear in yellow, then final output), even with stdin:

```bash
echo "Summarize the news" | python3 bsy-clippy.py --mode stream
```

---

### Custom server and model

Override the defaults if your Ollama server runs elsewhere:

```bash
python3 bsy-clippy.py --ip 127.0.0.1 --port 11434 --model llama2
```

---

## Requirements

See [`requirements.txt`](requirements.txt).

---

## Notes

- Works with Python 3.8+.  
- Relies on [Ollama](https://ollama.ai) running and serving the REST API at the specified IP/port.  
- ANSI colors (yellow for streaming, default for final output) may not display correctly on very old terminals.
