# bsy-clippy

`bsy-clippy` is a lightweight Python client for the [OpenAI](https://platform.openai.com/) Chat Completions API (and compatible deployments).  

It supports both **batch (stdin) mode** for one-shot prompts and **interactive mode** for chatting directly in the terminal.  
You can also load **system prompts** from a file to guide the LLM’s behavior.

---

## Features

- Speaks to the OpenAI Chat Completions API (or any compatible base URL).
- Loads credentials from `.env` (`OPENAI_API_KEY`) using `python-dotenv`.
- Reads defaults (profile, base URL, IP/port overrides, model) from `bsy-clippy.yaml`.
- Toggle endpoints by editing `api.profile` in `bsy-clippy.yaml` or passing `--profile` on the CLI.
- Defaults to:
  - Base URL: `http://172.20.0.100:11434/v1` (profile `ollama`)
  - Model: `qwen3:1.7b`
  - Mode: `stream` (see `--mode` to switch)
  - Bundled system prompt file that can be overridden with `--system-file`
- Configurable parameters:
  - `-b` / `--base-url` → explicit API endpoint
  - `-i` / `--ip` and `-p` / `--port` → override host/port when targeting compatible servers
  - `-M` / `--model` → model name
  - `-m` / `--mode` → output mode (`stream` or `batch`)
  - `-t` / `--temperature` → sampling temperature (default: `0.7`)
  - `-s` / `--system-file` → path to a text file with system instructions
  - `-u` / `--user-prompt` → extra user instructions prepended before the data payload
  - `-r` / `--memory-lines` → number of conversation lines to remember in interactive mode
  - `-c` / `--chat-after-stdin` → process stdin once, then drop into interactive chat
- Two modes of operation:
  - **Batch mode** → waits until the answer is complete, then prints only the final result.
  - **Stream mode** (default) → shows response in real-time, tokens appear as they are generated.
- Colored terminal output:
  - **Yellow** = streaming tokens (the model’s “thinking” in progress).
  - **Default terminal color** = final assembled answer.

---

## Installation

### pipx (recommended)

```bash
pipx install .
```

After updating the source, reinstall with `pipx reinstall bsy-clippy`.

### pip / virtual environments

```bash
pip install .
```

---

## Configuration

### API credentials (.env)

Create a `.env` file next to where you run `bsy-clippy` and add your key:

```
OPENAI_API_KEY=sk-...
```

The CLI loads this automatically via `python-dotenv`; environment variables from your shell work too.

### YAML defaults (`bsy-clippy.yaml`)

`bsy-clippy.yaml` selects which profile to use and what settings belong to it. The packaged example ships with an Ollama profile enabled and an OpenAI profile commented out for reference:

```
api:
  profile: ollama
  profiles:
    ollama:
      base_url: http://172.20.0.100:11434/v1
      model: qwen3:1.7b
    # openai:
    #   base_url: https://api.openai.com/v1
    #   model: gpt-4o-mini
```

Change `profile` (or pass `--profile openai`) to switch endpoints, or add more entries under `profiles` for additional deployments.

## Usage

### System prompt file

By default, `bsy-clippy` loads a bundled prompt (`Be very brief. Be very short.`).  
You can change this with `--system-file` or disable it via `--no-default-system`.

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

### Interactive memory

Set `--memory-lines 6` (or `-r 6`) to keep the last six conversation lines (user + assistant) while chatting.  
Only the final assistant reply (not the thinking traces) is stored and sent back on the next turn.

### Chat after stdin

Use `-c` / `--chat-after-stdin` to process piped data first and then remain in interactive mode with the response (and any configured memory) available:

```bash
cat sample.txt | bsy-clippy -u "Summarize this report" -r 6 -c
```

After the initial answer prints, you can continue the conversation while the tool remembers the piped data and the model’s reply.

---

### Interactive mode (default = stream)

Run without piping input:

```bash
bsy-clippy
```

Streaming session looks like:

```
You: Hello!
LLM (thinking): <think>
Reasoning step by step...
</think>
Hello! How can I assist you today? 😊
```

Prefer a single print at the end? Switch to batch mode:

```bash
bsy-clippy --mode batch
```

Batch output:

```
You: Hello!
Hello! How can I assist you today? 😊
```

---

### Batch mode (stdin)

Pipe input directly:

```bash
echo "Tell me a joke" | bsy-clippy
```

Output:

```
Why don’t scientists trust atoms? Because they make up everything!
```

---

### Forcing modes

```bash
bsy-clippy --mode batch
bsy-clippy --mode stream
```

---

### Adjusting temperature

```bash
bsy-clippy --temperature 0.2
bsy-clippy --temperature 1.2
```

---

### Custom server and model

```bash
bsy-clippy --base-url http://127.0.0.1:11434/v1 --model llama2
```

---

### Vector Database (RAG Mode)

When processing large files that exceed the LLM's context window, use `--vector` to enable RAG (Retrieval-Augmented Generation):

```bash
cat large_document.txt | bsy-clippy --vector
```

This mode:
1. Splits the input into semantic chunks (default: 500 chars with overlap)
2. Creates vector embeddings using a fast CPU-based model
3. Builds an HNSW index for efficient similarity search
4. Enters interactive mode where each question retrieves only relevant chunks

**Example workflow:**

```bash
# Process a large log file
cat server_logs.txt | bsy-clippy --vector --retrieve-chunks 6

# Interactive session starts
You: What errors occurred between 10am and 11am?
# LLM receives only the 6 most relevant log chunks

You: Show me database connection issues
# LLM receives different relevant chunks automatically
```

**Customize chunking:**

```bash
# Smaller chunks for dense technical content
cat api_docs.txt | bsy-clippy --vector --chunk-size 300

# Larger chunks for narrative content  
cat book.txt | bsy-clippy --vector --chunk-size 800 --retrieve-chunks 3
```

**Benefits:**
- Handle documents larger than LLM context window
- Faster responses (fewer tokens processed)
- More accurate answers (focused context)
- Runs on CPU (no GPU needed)

**Testing:**

```bash
# Run automated tests
python test_vector_full.py

# Test with sample data
cat test_data.txt | bsy-clippy --vector --profile localollama
```

**Note:** On first use, the embedding model (~66MB) will be downloaded from HuggingFace. Local endpoints (localhost, 127.0.0.1, 192.168.x.x, 172.x.x.x, 10.x.x.x) don't require an API key.

---

## Requirements

See [`requirements.txt`](requirements.txt).
