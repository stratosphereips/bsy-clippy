#!/usr/bin/env python3
import argparse
import os
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


def strip_think_segments(text):
    """Return text with <think> sections removed."""
    if not text:
        return ""

    result = []
    idx = 0
    in_think = False

    while idx < len(text):
        if in_think:
            close_idx = text.find("</think>", idx)
            if close_idx == -1:
                break
            idx = close_idx + len("</think>")
            in_think = False
        else:
            open_idx = text.find("<think>", idx)
            if open_idx == -1:
                result.append(text[idx:])
                break

            if open_idx > idx:
                result.append(text[idx:open_idx])
            idx = open_idx + len("<think>")
            in_think = True

    return "".join(result).strip()


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
    final_parts = []
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
                    final_parts.append(segment)
                idx = len(text)
            else:
                segment = text[idx:open_idx]
                if segment:
                    print(f"{ANSWER_COLOR}{segment}{RESET}", end="", flush=True)
                    final_parts.append(segment)
                print(f"{YELLOW}<think>{RESET}", end="", flush=True)
                idx = open_idx + len("<think>")
                in_think = True
    return in_think, "".join(final_parts)

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
        raw_text = "".join(output)
        return colorize_response(raw_text), strip_think_segments(raw_text)

    except requests.RequestException as e:
        error_text = f"[Error contacting Ollama API: {e}]"
        return error_text, ""


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
        final_parts = []
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    text = data.get("response", "")
                    if text:
                        in_think, segment = print_stream_chunk(text, in_think)
                        if segment:
                            final_parts.append(segment)

                    if data.get("done", False):
                        break
                except Exception:
                    continue
        print()  # newline after generation
        return strip_think_segments("".join(final_parts))

    except requests.RequestException as e:
        print(f"[Error contacting Ollama API: {e}]")
        return ""


def read_user_input(prompt_text, input_stream):
    """Read a line of input, supporting non-tty streams."""
    if input_stream is None:
        return input(prompt_text)

    print(prompt_text, end="", flush=True)
    line = input_stream.readline()
    if not line:
        raise EOFError
    return line.rstrip("\r\n")


def interactive_mode(api_url, model, mode, temperature, system_prompt, user_prompt,
                     memory_lines, memory_seed=None, input_stream=None):
    """Interactive chat mode with selectable output mode."""
    print(f"Interactive mode with model '{model}' at {api_url}")
    print(f"Mode: {mode}, Temperature: {temperature}")
    print("Type 'exit' or Ctrl+C to quit.")
    memory = list(memory_seed) if memory_seed else []
    if memory_lines > 0 and memory:
        memory[:] = memory[-memory_lines:]
    local_stream = input_stream
    close_stream = False
    if local_stream is None:
        if sys.stdin.isatty():
            local_stream = None
        else:
            tty_paths = ["CONIN$"] if os.name == "nt" else ["/dev/tty"]
            for path in tty_paths:
                try:
                    local_stream = open(path, "r", encoding="utf-8", errors="ignore")
                    close_stream = True
                    break
                except OSError:
                    local_stream = None
            if local_stream is None and sys.stdin.isatty():
                local_stream = None
            elif local_stream is None:
                local_stream = sys.stdin

    try:
        while True:
            try:
                prompt = read_user_input("You: ", local_stream)
            except EOFError:
                if local_stream is sys.stdin and not sys.stdin.isatty():
                    print("\n[Warning] No interactive input available; exiting.")
                else:
                    print("\nExiting.")
                break
            except KeyboardInterrupt:
                print("\nExiting.")
                break

            user_text = prompt.strip()
            if user_text.lower() in {"exit", "quit"}:
                break
            history_block = ""
            if memory:
                history_block = "History of Past Interaction:\n" + "\n".join(memory)

            current_block = ""
            if user_text:
                current_block = f"Current User Message:\n{user_text}"

            conversation_parts = [part for part in (history_block, current_block) if part]
            conversation_input = "\n\n".join(conversation_parts)
            final_prompt = compose_prompt(system_prompt, user_prompt, conversation_input)
            if not final_prompt:
                continue
            final_text = ""
            if mode == "stream":
                print("LLM (thinking): ", end="", flush=True)
                final_text = call_ollama_stream(api_url, model, final_prompt, temperature)
            else:  # batch
                response, final_text = call_ollama_batch(api_url, model, final_prompt, temperature)
                print(response)

            if memory_lines > 0:
                if user_text:
                    memory.append(f"User: {user_text}")
                if final_text:
                    memory.append(f"Assistant: {final_text.strip()}")
                if len(memory) > memory_lines:
                    memory[:] = memory[-memory_lines:]
    finally:
        if close_stream and local_stream not in {None, sys.stdin}:
            try:
                local_stream.close()
            except OSError:
                pass


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
    parser.add_argument("-r", "--memory-lines", type=int, default=0,
                        help="Remember this many lines of conversation in interactive mode")
    parser.add_argument("-c", "--chat-after-stdin", action="store_true",
                        help="After processing stdin, continue in interactive chat mode")

    args = parser.parse_args()
    api_url = f"http://{args.ip}:{args.port}"
    system_prompt = load_system_prompt(args.system_file)
    user_prompt = args.user_prompt
    memory_lines = max(0, args.memory_lines)
    chat_after_stdin = args.chat_after_stdin

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
            interactive_mode(api_url, args.model, mode, args.temperature, system_prompt, user_prompt, memory_lines)
            return

        memory_seed = []
        data_text = data.strip()
        if data_text:
            memory_seed.append(f"User: {data_text}")

        final_text = ""
        if mode == "stream":
            final_text = call_ollama_stream(api_url, args.model, full_prompt, args.temperature)
        else:
            response, final_text = call_ollama_batch(api_url, args.model, full_prompt, args.temperature)
            print(response)
        if chat_after_stdin:
            if final_text:
                memory_seed.append(f"Assistant: {final_text.strip()}")
            if memory_lines > 0 and memory_seed:
                memory_seed = memory_seed[-memory_lines:]
            interactive_mode(
                api_url,
                args.model,
                mode,
                args.temperature,
                system_prompt,
                user_prompt,
                memory_lines,
                memory_seed if memory_seed else None,
            )
        return

    interactive_mode(api_url, args.model, mode, args.temperature, system_prompt, user_prompt, memory_lines)


if __name__ == "__main__":
    main()
