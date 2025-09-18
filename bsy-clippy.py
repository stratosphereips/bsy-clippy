#!/usr/bin/env python3
import argparse
import sys
import requests
import json

# ANSI colors
YELLOW = "\033[93m"
RESET = "\033[0m"

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


def interactive_mode(api_url, model, mode, temperature):
    """Interactive chat mode with selectable output mode."""
    print(f"Interactive mode with model '{model}' at {api_url}")
    print(f"Mode: {mode}, Temperature: {temperature}")
    print("Type 'exit' or Ctrl+C to quit.")
    while True:
        try:
            prompt = input("You: ")
            if prompt.strip().lower() in {"exit", "quit"}:
                break
            if mode == "stream":
                print("LLM (thinking): ", end="", flush=True)
                call_ollama_stream(api_url, model, prompt, temperature)
            else:  # batch
                response = call_ollama_batch(api_url, model, prompt, temperature)
                print(response)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="bsy-clippy: Ollama API Client")
    parser.add_argument("--ip", default="172.20.0.100", help="Ollama server IP (default: 172.20.0.100)")
    parser.add_argument("--port", default="11434", help="Ollama server port (default: 11434)")
    parser.add_argument("--model", default="qwen3:1.7b", help="Model name (default: qwen3:1.7b)")
    parser.add_argument("--mode", choices=["stream", "batch"], default=stream,
                        help="Output mode: 'stream' = real-time, 'batch' = wait for final output")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7, higher = more random)")

    args = parser.parse_args()
    api_url = f"http://{args.ip}:{args.port}"

    # Detect mode if not specified
    mode = args.mode
    if mode is None:
        if not sys.stdin.isatty():
            mode = "batch"   # stdin defaults to batch
        else:
            mode = "stream"  # interactive defaults to stream

    if not sys.stdin.isatty() and args.mode is None:  # stdin input, no override
        prompt = sys.stdin.read().strip()
        if prompt:
            response = call_ollama_batch(api_url, args.model, prompt, args.temperature)
            print(response)
        else:
            interactive_mode(api_url, args.model, mode, args.temperature)
    elif not sys.stdin.isatty() and args.mode == "stream":
        # stdin but user forces streaming
        prompt = sys.stdin.read().strip()
        if prompt:
            call_ollama_stream(api_url, args.model, prompt, args.temperature)
    else:
        interactive_mode(api_url, args.model, mode, args.temperature)


if __name__ == "__main__":
    main()
