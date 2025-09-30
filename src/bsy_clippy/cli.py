"""Command-line interface for the bsy-clippy OpenAI client."""
from __future__ import annotations

import argparse
import os
import sys
from importlib import resources
from pathlib import Path
from typing import IO, Dict, List, Optional, Sequence, Tuple, Union

import yaml
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

YELLOW = "\033[93m"
ANSWER_COLOR = "\033[96m"
RESET = "\033[0m"

_DEFAULT_SYSTEM_PROMPT = "data/bsy-clippy.txt"
_DEFAULT_CONFIG = "bsy-clippy.yaml"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"

load_dotenv()


def _read_default_system_prompt() -> str:
    """Return the packaged default system prompt if it exists."""
    try:
        prompt_path = resources.files("bsy_clippy").joinpath(_DEFAULT_SYSTEM_PROMPT)
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        return ""

    try:
        return prompt_path.read_text(encoding="utf-8").strip("\n")
    except OSError:
        return ""


def load_system_prompt(path: Optional[str], allow_default: bool = True) -> str:
    """Return the content of a system prompt file, or the packaged default."""
    if path:
        file_path = Path(path)
        if not file_path.exists():
            return _read_default_system_prompt() if allow_default else ""
        try:
            return file_path.read_text(encoding="utf-8").strip("\n")
        except OSError as exc:
            print(f"[Warning] Could not read system prompt file '{path}': {exc}", file=sys.stderr)
            return ""
    if allow_default:
        return _read_default_system_prompt()
    return ""


def compose_prompt(*parts: Optional[str]) -> str:
    """Combine the provided string parts into a single message."""
    collected: List[str] = []
    for part in parts:
        if part and part.strip():
            collected.append(part.strip("\n"))
    return "\n\n".join(collected)


def strip_think_segments(text: str) -> str:
    """Return text with <think> sections removed."""
    if not text:
        return ""

    result: List[str] = []
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


def colorize_response(text: str) -> str:
    """Return the response string with ANSI colors applied to think segments."""
    if not text:
        return ""

    idx = 0
    in_think = False
    output: List[str] = []

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


def print_stream_chunk(text: str, in_think: bool) -> Tuple[bool, str]:
    """Stream a chunk of text with think/final color separation."""
    idx = 0
    final_parts: List[str] = []
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


def load_config(path: Optional[str]) -> Dict[str, object]:
    """Load YAML configuration from the provided path."""
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"[Warning] Failed to read configuration file '{path}': {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"[Warning] Configuration file '{path}' must contain a YAML mapping.", file=sys.stderr)
        return {}
    return data


def select_api_profile(api_config: Dict[str, object], override: Optional[str] = None) -> Tuple[Optional[str], Dict[str, object]]:
    """Return the active API profile settings and its name."""
    if not isinstance(api_config, dict):
        return None, {}

    base_settings: Dict[str, object] = {
        key: value for key, value in api_config.items() if key not in {"profiles", "profile"}
    }

    profiles = api_config.get("profiles")
    profile_name = api_config.get("profile")
    default_name = profile_name if isinstance(profile_name, str) else None
    selected_name: Optional[str] = override or default_name

    if isinstance(profiles, dict) and profiles:
        profile_map: Dict[str, Dict[str, object]] = {
            str(name): value for name, value in profiles.items() if isinstance(value, dict)
        }
        if selected_name and selected_name not in profile_map:
            print(
                f"[Warning] Profile '{selected_name}' not found in config; falling back to first available profile.",
                file=sys.stderr,
            )
            selected_name = None
        if not selected_name:
            selected_name = next(iter(profile_map), None)
        chosen_settings = profile_map.get(selected_name or "", {})
        merged: Dict[str, object] = {**base_settings, **chosen_settings}
        return selected_name, merged

    if selected_name:
        print(
            f"[Warning] Profile '{selected_name}' requested but config has no profiles section; using top-level settings.",
            file=sys.stderr,
        )
    return selected_name, base_settings


def resolve_base_url(
    settings: Dict[str, object],
    ip_override: Optional[str],
    port_override: Optional[str],
) -> Tuple[str, Optional[str], Optional[str]]:
    """Return base URL plus normalized host/port used for display or overrides."""
    base_url = str(settings.get("base_url", "")).strip()
    raw_ip = ip_override if ip_override is not None else settings.get("ip")
    raw_port = port_override if port_override is not None else settings.get("port")
    ip = str(raw_ip).strip() if raw_ip is not None else ""
    port = str(raw_port).strip() if raw_port is not None else ""

    if base_url:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(base_url)
            if parsed.scheme and parsed.netloc:
                host = parsed.hostname or (ip or None)
                port_from_url = parsed.port
                if host and port_from_url:
                    return base_url, host, str(port_from_url)
        except Exception:
            pass
        if ip and port:
            return base_url, ip, port
        return base_url, ip or None, port or None

    if ip and port:
        normalized = f"http://{ip}:{port}/v1"
        return normalized, ip, port

    return _DEFAULT_BASE_URL, None, None


def create_openai_client(base_url: str) -> OpenAI:
    """Create an OpenAI client using environment credentials."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "[Error] OPENAI_API_KEY is not set. Create a .env file with OPENAI_API_KEY=<token> or export it.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        return OpenAI(api_key=api_key, base_url=base_url)
    except OpenAIError as exc:
        print(f"[Error] Could not initialize OpenAI client: {exc}", file=sys.stderr)
        sys.exit(1)


def build_messages(system_prompt: str, user_content: str) -> List[Dict[str, str]]:
    """Construct chat-completion style messages for the OpenAI API."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_content:
        messages.append({"role": "user", "content": user_content})
    return messages


def _extract_content(raw: Union[str, List[object], None]) -> str:
    """Normalize OpenAI content payloads into a plain string."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: List[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif hasattr(item, "text"):
                text = getattr(item, "text")
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(raw)


def call_openai_batch(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> Tuple[str, str]:
    """Send a prompt to the OpenAI API and return the formatted and raw text."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    except OpenAIError as exc:
        error_text = f"[Error contacting OpenAI API: {exc}]"
        return error_text, ""

    if not response.choices:
        return "[No response received]", ""

    choice = response.choices[0]
    message = getattr(choice, "message", None)
    content = ""
    if message is not None:
        content = _extract_content(getattr(message, "content", None))
    return colorize_response(content), strip_think_segments(content)


def call_openai_stream(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> str:
    """Send a prompt to the OpenAI API and stream response with color separation."""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
    except OpenAIError as exc:
        print(f"[Error contacting OpenAI API: {exc}]")
        return ""

    in_think = False
    final_parts: List[str] = []
    try:
        for chunk in stream:
            for choice in getattr(chunk, "choices", []) or []:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                piece = _extract_content(getattr(delta, "content", None))
                if piece:
                    in_think, segment = print_stream_chunk(piece, in_think)
                    if segment:
                        final_parts.append(segment)
        print()
    finally:
        # Ensure the generator is closed to free resources
        try:
            stream.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    return strip_think_segments("".join(final_parts))


def read_user_input(prompt_text: str, input_stream: Optional[IO[str]]) -> str:
    """Read a line of input, supporting non-tty streams."""
    if input_stream is None:
        return input(prompt_text)

    print(prompt_text, end="", flush=True)
    line = input_stream.readline()
    if not line:
        raise EOFError
    return line.rstrip("\r\n")


def interactive_mode(
    client: OpenAI,
    base_url: str,
    model: str,
    mode: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    memory_lines: int,
    profile_name: Optional[str] = None,
    memory_seed: Optional[Sequence[str]] = None,
    input_stream: Optional[IO[str]] = None,
) -> None:
    """Interactive chat mode with selectable output mode."""
    profile_info = f" (profile '{profile_name}')" if profile_name else ""
    print(f"Interactive mode with model '{model}' via {base_url}{profile_info}")
    print(f"Mode: {mode}, Temperature: {temperature}")
    print("Type 'exit' or Ctrl+C to quit.")
    memory: List[str] = list(memory_seed) if memory_seed else []
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
            user_content = compose_prompt(user_prompt, conversation_input)
            messages = build_messages(system_prompt, user_content)
            if not messages:
                continue
            final_text = ""
            if mode == "stream":
                print("LLM (thinking): ", end="", flush=True)
                final_text = call_openai_stream(client, model, messages, temperature)
            else:
                response_text, final_text = call_openai_batch(client, model, messages, temperature)
                print(response_text)

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


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="bsy-clippy: OpenAI-compatible CLI client")
    parser.add_argument(
        "-cfg",
        "--config",
        default=_DEFAULT_CONFIG,
        help=f"Path to a YAML config file (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Profile name defined in the YAML config to use (overrides api.profile)",
    )
    parser.add_argument(
        "-i",
        "--ip",
        default=None,
        help="Override IP address for OpenAI-compatible endpoints (default: use config or base URL)",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=None,
        help="Override port for OpenAI-compatible endpoints (default: use config or base URL)",
    )
    parser.add_argument(
        "-b",
        "--base-url",
        default=None,
        help="Explicit OpenAI API base URL (default: derived from config or https://api.openai.com/v1)",
    )
    parser.add_argument(
        "-M",
        "--model",
        default=None,
        help=f"Model name (default: value from config or {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["stream", "batch"],
        default="stream",
        help="Output mode: 'stream' = real-time, 'batch' = wait for final output",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7, higher = more random)",
    )
    parser.add_argument(
        "-s",
        "--system-file",
        default=None,
        help="Path to a system prompt file (default: bundled prompt)",
    )
    parser.add_argument(
        "-u",
        "--user-prompt",
        default="",
        help="Additional user instructions to prepend before the data",
    )
    parser.add_argument(
        "-r",
        "--memory-lines",
        type=int,
        default=0,
        help="Remember this many lines of conversation in interactive mode",
    )
    parser.add_argument(
        "-c",
        "--chat-after-stdin",
        action="store_true",
        help="After processing stdin, continue in interactive chat mode",
    )
    parser.add_argument(
        "--no-default-system",
        action="store_true",
        help="Disable the packaged default system prompt",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_data = load_config(args.config)
    api_config = config_data.get("api") if isinstance(config_data.get("api"), dict) else {}
    active_profile, profile_settings = select_api_profile(api_config, args.profile)

    base_url, _, _ = resolve_base_url(
        profile_settings,
        args.ip if args.ip is not None else None,
        args.port if args.port is not None else None,
    )
    if args.base_url:
        base_url = args.base_url
    model = args.model or profile_settings.get("model") or _DEFAULT_MODEL

    allow_default = not args.no_default_system
    system_prompt = load_system_prompt(args.system_file, allow_default=allow_default)
    user_prompt = args.user_prompt
    memory_lines = max(0, args.memory_lines)
    chat_after_stdin = args.chat_after_stdin
    temperature = args.temperature

    client = create_openai_client(base_url)

    mode = args.mode
    if mode is None:
        if not sys.stdin.isatty():
            mode = "batch"
        else:
            mode = "stream"

    if not sys.stdin.isatty():
        data = sys.stdin.read()
        user_content = compose_prompt(user_prompt, data)
        messages = build_messages(system_prompt, user_content)

        if not messages:
            interactive_mode(
                client,
                base_url,
                model,
                mode,
                temperature,
                system_prompt,
                user_prompt,
                memory_lines,
                active_profile,
            )
            return

        memory_seed: List[str] = []
        data_text = data.strip()
        if data_text:
            memory_seed.append(f"User: {data_text}")

        final_text = ""
        if mode == "stream":
            final_text = call_openai_stream(client, model, messages, temperature)
        else:
            response_text, final_text = call_openai_batch(client, model, messages, temperature)
            print(response_text)
        if chat_after_stdin:
            if final_text:
                memory_seed.append(f"Assistant: {final_text.strip()}")
            if memory_lines > 0 and memory_seed:
                memory_seed = memory_seed[-memory_lines:]
            interactive_mode(
                client,
                base_url,
                model,
                mode,
                temperature,
                system_prompt,
                user_prompt,
                memory_lines,
                active_profile,
                memory_seed if memory_seed else None,
            )
        return

    interactive_mode(
        client,
        base_url,
        model,
        mode,
        temperature,
        system_prompt,
        user_prompt,
        memory_lines,
        active_profile,
    )


if __name__ == "__main__":
    main()
