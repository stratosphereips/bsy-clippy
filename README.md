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
  - `-r` / `--memory-lines`, `-c` / `--chat-after-stdin` → conversation persistence options (interactive mode remembers 12 entries by default)
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

To install the latest code directly from GitHub after it is pushed:

```bash
pip install --upgrade git+https://github.com/stratosphereips/bsy-clippy.git@main
```

---

## Configuration

### API credentials (.env)

Create a `.env` file next to where you run `bsy-clippy` and add your key:

```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

The CLI loads this automatically via `python-dotenv`; environment variables from your shell work too. The `openai` profile uses `OPENAI_API_KEY`, the `groq` profile uses `GROQ_API_KEY`, and `ollama` profiles need no API key. Keep `.env` out of version control.

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
    groq:
      provider: openai
      base_url: https://api.groq.com/openai/v1
      model: openai/gpt-oss-120b
      api_key_env: GROQ_API_KEY
    cesnet:
      provider: openai
      base_url: https://chat.ai.e-infra.cz/api
      #model: glm-5.3
      model: mini
      api_key_env: CESNET_API_KEY
```

#### Switching profiles

- Use `profile: ollama` to talk to an Ollama server; no API token is required. Adjust `base_url` if your host or port differs.
- To use OpenAI, either set `profile: openai` in the file or pass `--profile openai` on the command line. Make sure the `OPENAI_API_KEY` environment variable (or the name set in `api_key_env`) is populated before running the command.
- To use Groq, add `GROQ_API_KEY` to `.env` or your shell and select `--profile groq`. The repository config selects this profile by default.
- For CESNET/e-INFRA CZ, set `CESNET_API_KEY` and select `--profile cesnet`. The CLI maps the configured `https://chat.ai.e-infra.cz/api` URL to the current API endpoint, `https://llm.ai.e-infra.cz/v1`.
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
You can change this with `--system-file` or disable it via `--no-default-system`. The CLI always adds its tool instructions so models know when and how to request a local command.

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

Interactive mode remembers the last 12 conversation entries (user, assistant, and tool results) by default. Use `--memory-lines 6` (or `-r 6`) to keep six, or `-r 0` to disable memory. Only the final assistant reply and tool results are stored; thinking traces are not.

### Local command tool

When a request needs information from this machine, the model can request a command. The CLI displays it and asks `Execute command? [y/N]:` before running it. The command output is sent back to the model so it can answer in the same turn. A declined or blocked command is reported back as well. Up to three commands can run in one turn. Piped input without an interactive terminal cannot approve commands.

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
