# bsy-clippy

`bsy-clippy` is a lightweight Python client for the [OpenAI](https://platform.openai.com/) Chat Completions API (and compatible deployments).  

It supports both **batch (stdin) mode** for one-shot prompts and **interactive mode** for chatting directly in the terminal.  
You can also load **system prompts** from a file to guide the LLM’s behavior.

---

## Features

- Speaks to the OpenAI Chat Completions API or any OpenAI-compatible deployment.
- Loads credentials from `.env` (`OPENAI_API_KEY`) via `python-dotenv` when the selected profile requires them.
- Reads defaults (profile, provider, base URL, model) from `bsy-clippy.yaml` and falls back to a packaged sample if none is found.
- Switch profiles by editing `api.profile` or passing `--profile` (e.g. `--profile openai`) on the CLI.
- Defaults to:
  - Provider: `ollama`
  - Base URL: `http://172.20.0.100:11434`
  - Model: `qwen3:1.7b`
  - Mode: `stream` (see `--mode` to switch)
  - Bundled system prompt file that can be overridden with `--system-file`
- Configurable parameters:
  - `--config` / `--profile` → select a profile file and profile name
  - `-b` / `--base-url`, `-i` / `--ip`, `-p` / `--port` → override connection details per run
  - `-M` / `--model`, `-t` / `--temperature`, `-m` / `--mode` → tuning controls
  - `-s` / `--system-file`, `-u` / `--user-prompt` → additional prompting knobs
  - `-r` / `--memory-lines`, `-c` / `--chat-after-stdin` → conversation persistence options
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

The CLI loads this automatically via `python-dotenv`; environment variables from your shell work too. Only the `openai` profile requires this token — `ollama` profiles can leave it unset.

### YAML defaults (`bsy-clippy.yaml`)

`bsy-clippy.yaml` selects which profile to use and what settings belong to it. The CLI looks for this file in the current working directory, then in `~/.config/bsy-clippy/`, and finally falls back to the bundled sample. Copy the sample to your project or config directory, edit it, and switch profiles as needed:

```
api:
  profile: ollama
  profiles:
    ollama:
      provider: ollama
      base_url: http://172.20.0.100:11434
      model: qwen3:1.7b
    # openai:
    #   provider: openai
    #   base_url: https://api.openai.com/v1
    #   model: gpt-4o-mini
    #   api_key_env: OPENAI_API_KEY
```

#### Switching profiles

- Keep `profile: ollama` (the default) to talk to an Ollama server; no API token is required. Adjust `base_url` if your host or port differs.
- To use OpenAI, either set `profile: openai` in the file or pass `--profile openai` on the command line. Make sure the `OPENAI_API_KEY` environment variable (or the name set in `api_key_env`) is populated before running the command.
- You can define additional profiles (for example, staging clusters) under `api.profiles` and select them with `--profile <name>`.

### Quick check

```bash
pip install -r requirements.txt          # or pipx install .
cp src/bsy_clippy/data/bsy-clippy.yaml ./bsy-clippy.yaml
python bsy-clippy.py --help              # confirms dependencies are in place
```

After copying the sample config you can edit it in-place and re-run `bsy-clippy` to target a different profile.
If you installed via `pipx` or `pip`, copy the bundled sample with:

```bash
python -c "import importlib.resources as r, pathlib; pathlib.Path('bsy-clippy.yaml').write_text(r.files('bsy_clippy').joinpath('data/bsy-clippy.yaml').read_text())"
```


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
bsy-clippy --profile ollama --base-url http://127.0.0.1:11434 --model llama2
```

---

## Requirements

See [`requirements.txt`](requirements.txt).
