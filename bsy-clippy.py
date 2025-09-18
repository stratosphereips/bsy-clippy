#!/usr/bin/env python3
import argparse
import sys
import requests
import json
from pathlib import Path

# ANSI colors
YELLOW = "\033[93m"
ANSWER_COLOR = "\033[96m"
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


def colorize_response(text):
    """Return the response string with ANSI colors applied to think segments."""
    if not text:
        return ""

    idx = 0
    in_think = False
    output = []

    while idx < len(text):
        if in_think:
            close_idx = text.find("</think>", idx)
            if close_idx == -1:
                output.append(f"{YELLOW}{text[idx:]}{RESET}")
                break

            if close_idx > idx:
                output.append(f"{YELLOW}{text[idx:close_idx]}{RESET}")
            output.append(f"{YELLOW}</think>{RESET}")
            idx = close_idx + len("</think>")
            in_think = False
        else:
            open_idx = text.find("<think>", idx)
            if open_idx == -1:
                output.append(f"{ANSWER_COLOR}{text[idx:]}{RESET}")
                break

            if open_idx > idx:
                output.append(f"{ANSWER_COLOR}{text[idx:open_idx]}{RESET}")
            output.append(f"{YELLOW}<think>{RESET}")
            idx = open_idx + len("<think>")
            in_think = True

    return "".join(output)


def print_stream_chunk(text, in_think):
    """Stream a chunk of text with think/final color separation."""
    idx = 0
    while idx < len(text):
        if in_think:
            close_idx = text.find("</think>", idx)
            if close_idx == -1:
                segment = text[idx:]
                if segment:
                    print(f"{YELLOW}{segment}{RESET}", end="", flush=True)
                idx = len(text)
            else:
                segment = text[idx:close_idx]
                if segment:
                    print(f"{YELLOW}{segment}{RESET}", end="", flush=True)
                print(f"{YELLOW}</think>{RESET}", end="", flush=True)
                idx = close_idx + len("</think>")
                in_think = False
        else:
            open_idx = text.find("<think>", idx)
            if open_idx == -1:
                segment = text[idx:]
                if segment:
                    print(f"{ANSWER_COLOR}{segment}{RESET}", end="", flush=True)
                idx = len(text)
            else:
                segment = text[idx:open_idx]
                if segment:
                    print(f"{ANSWER_COLOR}{segment}{RESET}", end="", flush=True)
                print(f"{YELLOW}<think>{RESET}", end="", flush=True)
                idx = open_idx + len("<think>")
                in_think = True
    return in_think

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
        return colorize_response("".join(output))

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
                        in_think = print_stream_chunk(text, in_think)

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
    parser.add_argument("-i", "--ip", default="172.20.0.100",
                        help="Ollama server IP (default: 172.20.0.100)")
    parser.add_argument("-p", "--port", default="11434",
                        help="Ollama server port (default: 11434)")
    parser.add_argument("-M", "--model", default="qwen3:1.7b",
                        help="Model name (default: qwen3:1.7b)")
    parser.add_argument("-m", "--mode", choices=["stream", "batch"], default="stream",
                        help="Output mode: 'stream' = real-time, 'batch' = wait for final output")
    parser.add_argument("-t", "--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7, higher = more random)")
    parser.add_argument("-s", "--system-file", default="bsy-clippy.txt",
                        help="Path to a system prompt file (default: bsy-clippy.txt)")
    parser.add_argument("-u", "--user-prompt", default="",
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
