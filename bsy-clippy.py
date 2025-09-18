#!/usr/bin/env python3
import argparse
import sys
import requests
import json
from pathlib import Path

# ANSI colors
YELLOW = "\033[93m"
RESET = "\033[0m"


def load_system_prompt(path):
    """Return the content of a system prompt file if it exists."""
    if not path:
        return ""

    file_path = Path(path)
    if not file_path.exists():
        return ""

    try:
        return file_path.read_text(encoding="utf-8").strip("\n")
    except OSError as exc:
        print(f"[Warning] Could not read system prompt file '{path}': {exc}", file=sys.stderr)
        return ""


def compose_prompt(system_prompt, user_prompt, data):
    """Combine system prompt, user prompt, and data into a single message."""
    parts = []
    for part in (system_prompt, user_prompt, data):
        if part and part.strip():
            parts.append(part.strip("\n"))
    return "\n\n".join(parts)

def call_ollama_batch(api_url, model, prompt, temperature):
    """Send a prompt to Ollama API and return response text (batch mode)."""
    try:
        response = requests.post(
            f"{api_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
            },
            stream=True,
            timeout=600
        )
        response.raise_for_status()

        output = []
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    output.append(data.get("response", ""))
                except Exception:
                    pass
        return "".join(output)

    except requests.RequestException as e:
        return f"[Error contacting Ollama API: {e}]"


def call_ollama_stream(api_url, model, prompt, temperature):
    """Send a prompt to Ollama API and stream response with color separation."""
    try:
        response = requests.post(
            f"{api_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
            },
            stream=True,
            timeout=600
        )
        response.raise_for_status()

        in_think = False
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    text = data.get("response", "")
                    if text:
                        # Detect <think> start
                        if "<think>" in text:
                            in_think = True
                        # Detect </think> end
                        if "</think>" in text:
                            in_think = False
                            print(f"{YELLOW}{text}{RESET}", end="", flush=True)
                            continue

                        if in_think:
                            print(f"{YELLOW}{text}{RESET}", end="", flush=True)
                        else:
                            print(f"{text}", end="", flush=True)

                    if data.get("done", False):
                        break
                except Exception:
                    continue
        print()  # newline after generation

    except requests.RequestException as e:
        print(f"[Error contacting Ollama API: {e}]")


def interactive_mode(api_url, model, mode, temperature, system_prompt, user_prompt):
    """Interactive chat mode with selectable output mode."""
    print(f"Interactive mode with model '{model}' at {api_url}")
    print(f"Mode: {mode}, Temperature: {temperature}")
    print("Type 'exit' or Ctrl+C to quit.")
    while True:
        try:
            prompt = input("You: ")
            if prompt.strip().lower() in {"exit", "quit"}:
                break
            final_prompt = compose_prompt(system_prompt, user_prompt, prompt)
            if not final_prompt:
                continue
            if mode == "stream":
                print("LLM (thinking): ", end="", flush=True)
                call_ollama_stream(api_url, model, final_prompt, temperature)
            else:  # batch
                response = call_ollama_batch(api_url, model, final_prompt, temperature)
                print(response)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="bsy-clippy: Ollama API Client")
    parser.add_argument("--ip", default="172.20.0.100", help="Ollama server IP (default: 172.20.0.100)")
    parser.add_argument("--port", default="11434", help="Ollama server port (default: 11434)")
    parser.add_argument("--model", default="qwen3:1.7b", help="Model name (default: qwen3:1.7b)")
    parser.add_argument("--mode", choices=["stream", "batch"], default=None,
                        help="Output mode: 'stream' = real-time, 'batch' = wait for final output")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7, higher = more random)")
    parser.add_argument("--system-file", default="bsy-clippy.txt",
                        help="Path to a system prompt file (default: bsy-clippy.txt)")
    parser.add_argument("--user-prompt", default="",
                        help="Additional user instructions to prepend before the data")

    args = parser.parse_args()
    api_url = f"http://{args.ip}:{args.port}"
    system_prompt = load_system_prompt(args.system_file)
    user_prompt = args.user_prompt

    # Detect mode if not specified
    mode = args.mode
    if mode is None:
        if not sys.stdin.isatty():
            mode = "batch"   # stdin defaults to batch
        else:
            mode = "stream"  # interactive defaults to stream

    if not sys.stdin.isatty():
        data = sys.stdin.read()
        full_prompt = compose_prompt(system_prompt, user_prompt, data)

        if not full_prompt:
            interactive_mode(api_url, args.model, mode, args.temperature, system_prompt, user_prompt)
            return

        if mode == "stream":
            call_ollama_stream(api_url, args.model, full_prompt, args.temperature)
        else:
            response = call_ollama_batch(api_url, args.model, full_prompt, args.temperature)
            print(response)
        return

    interactive_mode(api_url, args.model, mode, args.temperature, system_prompt, user_prompt)


if __name__ == "__main__":
    main()
