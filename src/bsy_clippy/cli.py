"""Command-line interface for the bsy-clippy client."""
from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import resources
from pathlib import Path
from typing import IO, Dict, List, Optional, Sequence, Tuple

import requests
import yaml
from dotenv import dotenv_values, find_dotenv
from openai import OpenAI, OpenAIError

YELLOW = "\033[93m"
ANSWER_COLOR = "\033[96m"
RESET = "\033[0m"

_DEFAULT_SYSTEM_PROMPT_RESOURCE = "data/bsy-clippy.txt"
_DEFAULT_CONFIG_FILENAME = "bsy-clippy.yaml"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _read_text_resource(relative_path: str) -> str:
    try:
        return resources.files("bsy_clippy").joinpath(relative_path).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, AttributeError, OSError):
        return ""


def _load_default_system_prompt() -> str:
    return _read_text_resource(_DEFAULT_SYSTEM_PROMPT_RESOURCE).strip("\n")


def load_system_prompt(path: Optional[str], allow_default: bool = True) -> str:
    if path:
        file_path = Path(path).expanduser()
        if not file_path.exists():
            print(f"[Warning] System prompt file '{file_path}' not found; continuing without it.", file=sys.stderr)
        else:
            try:
                return file_path.read_text(encoding="utf-8").strip("\n")
            except OSError as exc:
                print(f"[Warning] Could not read system prompt file '{file_path}': {exc}", file=sys.stderr)
    if allow_default:
        default_path = Path.cwd() / Path(_DEFAULT_SYSTEM_PROMPT_RESOURCE).name
        if default_path.exists():
            try:
                return default_path.read_text(encoding="utf-8").strip("\n")
            except OSError:
                pass
        return _load_default_system_prompt()
    return ""


def compose_prompt(*parts: Optional[str]) -> str:
    collected: List[str] = []
    for part in parts:
        if part and part.strip():
            collected.append(part.strip("\n"))
    return "\n\n".join(collected)


def strip_think_segments(text: str) -> str:
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


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

SEARCH_CONFIG_LOCATIONS = (
    lambda: (Path.cwd() / _DEFAULT_CONFIG_FILENAME),
    lambda: (Path.home() / ".config" / "bsy-clippy" / _DEFAULT_CONFIG_FILENAME),
)


def _load_yaml_file(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"[Warning] Failed to read configuration file '{path}': {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"[Warning] Configuration file '{path}' must contain a YAML mapping.", file=sys.stderr)
        return None
    return data


def _load_packaged_config() -> Dict[str, object]:
    text = _read_text_resource("data/bsy-clippy.yaml")
    if not text:
        return {}
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        print(f"[Warning] Failed to parse packaged configuration: {exc}", file=sys.stderr)
        return {}
    return data if isinstance(data, dict) else {}


def load_config(path: Optional[str]) -> Tuple[Dict[str, object], Optional[str]]:
    if path:
        candidate = Path(path).expanduser()
        data = _load_yaml_file(candidate)
        if data is not None:
            return data, str(candidate)
        return {}, str(candidate)

    for resolver in SEARCH_CONFIG_LOCATIONS:
        candidate = resolver()
        data = _load_yaml_file(candidate)
        if data is not None:
            return data, str(candidate)

    return _load_packaged_config(), "packaged sample"


def select_api_profile(api_config: Dict[str, object], override: Optional[str] = None) -> Tuple[Optional[str], Dict[str, object]]:
    if not isinstance(api_config, dict):
        return None, {}

    base_settings: Dict[str, object] = {
        key: value for key, value in api_config.items() if key not in {"profiles", "profile"}
    }

    profiles = api_config.get("profiles")
    profile_name = str(api_config.get("profile")) if isinstance(api_config.get("profile"), str) else None
    selected_name = override or profile_name

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


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def call_ollama_batch(base_url: str, model: str, prompt: str, temperature: float) -> Tuple[str, str]:
    endpoint = base_url.rstrip("/") + "/api/generate"
    try:
        response = requests.post(
            endpoint,
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
            },
            stream=True,
            timeout=600,
        )
        response.raise_for_status()
        output: List[str] = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            output.append(data.get("response", ""))
            if data.get("done"):
                break
        raw_text = "".join(output)
        return colorize_response(raw_text), strip_think_segments(raw_text)
    except requests.RequestException as exc:
        error_text = f"[Error contacting Ollama API: {exc}]"
        return error_text, ""


def call_ollama_stream(base_url: str, model: str, prompt: str, temperature: float) -> str:
    endpoint = base_url.rstrip("/") + "/api/generate"
    try:
        response = requests.post(
            endpoint,
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
            },
            stream=True,
            timeout=600,
        )
        response.raise_for_status()
        in_think = False
        final_parts: List[str] = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            text = data.get("response", "")
            if text:
                in_think, segment = print_stream_chunk(text, in_think)
                if segment:
                    final_parts.append(segment)
            if data.get("done"):
                break
        print()
        return strip_think_segments("".join(final_parts))
    except requests.RequestException as exc:
        print(f"[Error contacting Ollama API: {exc}]")
        return ""


def _extract_content(raw) -> str:
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
    message = getattr(response.choices[0], "message", None)
    content = _extract_content(getattr(message, "content", None)) if message is not None else ""
    return colorize_response(content), strip_think_segments(content)


def call_openai_stream(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> str:
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
        try:
            stream.close()  # type: ignore[attr-defined]
        except Exception:
            pass
    return strip_think_segments("".join(final_parts))


class Provider:
    def __init__(self, name: str, base_url: str, model: str) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model

    def stream(self, system_prompt: str, user_prompt: str, conversation_input: str, temperature: float) -> str:
        raise NotImplementedError

    def batch(self, system_prompt: str, user_prompt: str, conversation_input: str, temperature: float) -> Tuple[str, str]:
        raise NotImplementedError


class OllamaProvider(Provider):
    def __init__(self, base_url: str, model: str) -> None:
        super().__init__("ollama", base_url, model)

    def stream(self, system_prompt: str, user_prompt: str, conversation_input: str, temperature: float) -> str:
        prompt = compose_prompt(system_prompt, user_prompt, conversation_input)
        return call_ollama_stream(self.base_url, self.model, prompt, temperature)

    def batch(self, system_prompt: str, user_prompt: str, conversation_input: str, temperature: float) -> Tuple[str, str]:
        prompt = compose_prompt(system_prompt, user_prompt, conversation_input)
        return call_ollama_batch(self.base_url, self.model, prompt, temperature)


class OpenAIProvider(Provider):
    def __init__(self, client: OpenAI, base_url: str, model: str) -> None:
        super().__init__("openai", base_url, model)
        self.client = client

    def stream(self, system_prompt: str, user_prompt: str, conversation_input: str, temperature: float) -> str:
        user_content = compose_prompt(user_prompt, conversation_input)
        messages = build_messages(system_prompt, user_content)
        return call_openai_stream(self.client, self.model, messages, temperature)

    def batch(self, system_prompt: str, user_prompt: str, conversation_input: str, temperature: float) -> Tuple[str, str]:
        user_content = compose_prompt(user_prompt, conversation_input)
        messages = build_messages(system_prompt, user_content)
        return call_openai_batch(self.client, self.model, messages, temperature)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def build_messages(system_prompt: str, user_content: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_content:
        messages.append({"role": "user", "content": user_content})
    return messages


def read_user_input(prompt_text: str, input_stream: Optional[IO[str]]) -> str:
    if input_stream is None:
        return input(prompt_text)
    print(prompt_text, end="", flush=True)
    line = input_stream.readline()
    if not line:
        raise EOFError
    return line.rstrip("\r\n")


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def interactive_mode(
    provider: Provider,
    mode: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    memory_lines: int,
    profile_name: Optional[str] = None,
    memory_seed: Optional[Sequence[str]] = None,
    input_stream: Optional[IO[str]] = None,
) -> None:
    profile_info = f" (profile '{profile_name}')" if profile_name else ""
    print(f"Interactive mode with model '{provider.model}' via {provider.base_url}{profile_info}")
    print(f"Mode: {mode}, Temperature: {temperature} (provider: {provider.name})")
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

            history_block = "History of Past Interaction:\n" + "\n".join(memory) if memory else ""
            current_block = f"Current User Message:\n{user_text}" if user_text else ""
            conversation_parts = [part for part in (history_block, current_block) if part]
            conversation_input = "\n\n".join(conversation_parts)

            final_text = ""
            if mode == "stream":
                print("LLM (thinking): ", end="", flush=True)
                final_text = provider.stream(system_prompt, user_prompt, conversation_input, temperature)
            else:
                response_text, final_text = provider.batch(system_prompt, user_prompt, conversation_input, temperature)
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


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


def _iter_dotenv_candidates(config_source: Optional[str]) -> List[Path]:
    discovered = find_dotenv(usecwd=True)
    candidates: List[Path] = []
    if discovered:
        candidates.append(Path(discovered))
    candidates.append(Path.cwd() / ".env")
    if config_source:
        source_path = Path(config_source).expanduser()
        base_dir = source_path.parent if source_path.is_file() else source_path
        candidates.append(base_dir / ".env")

    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            unique.append(resolved)
    return unique


def _resolve_api_key(env_var: str, config_source: Optional[str]) -> Optional[str]:
    direct = os.getenv(env_var)
    if direct:
        return direct

    for dotenv_path in _iter_dotenv_candidates(config_source):
        values = dotenv_values(str(dotenv_path))
        if not values:
            continue
        candidate = values.get(env_var)
        if candidate:
            return candidate
        if env_var != "OPENAI_API_KEY":
            fallback = values.get("OPENAI_API_KEY")
            if fallback:
                return fallback
    return None


def create_openai_client(
    base_url: str,
    require_api_key: bool,
    env_var: str = "OPENAI_API_KEY",
    config_source: Optional[str] = None,
) -> OpenAI:
    api_key = _resolve_api_key(env_var, config_source)
    if not api_key:
        if require_api_key:
            print(
                f"[Error] {env_var} is not set. Export it or add it to a .env file.",
                file=sys.stderr,
            )
            sys.exit(1)
        api_key = "not-set"
    try:
        return OpenAI(api_key=api_key, base_url=base_url)
    except OpenAIError as exc:
        print(f"[Error] Could not initialize OpenAI client: {exc}", file=sys.stderr)
        sys.exit(1)


def determine_provider(profile_settings: Dict[str, object], profile_name: Optional[str]) -> str:
    provider = profile_settings.get("provider")
    if isinstance(provider, str) and provider.strip():
        return provider.strip().lower()
    if profile_name and profile_name.lower() == "ollama":
        return "ollama"
    return "openai"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="bsy-clippy: configurable chat CLI")

    cwd_config = (Path.cwd() / _DEFAULT_CONFIG_FILENAME).resolve()
    sample_path = resources.files("bsy_clippy").joinpath("data/bsy-clippy.yaml")
    config_help = (
        "Path to a YAML config file controlling profiles. "
        f"Default search order: {cwd_config}, ~/.config/bsy-clippy/{_DEFAULT_CONFIG_FILENAME}. "
        f"Sample bundled at {sample_path}."
    )
    parser.add_argument(
        "-cfg",
        "--config",
        default=None,
        help=config_help,
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
        help="Override IP address for Ollama-compatible endpoints",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=None,
        help="Override port for Ollama-compatible endpoints",
    )
    parser.add_argument(
        "-b",
        "--base-url",
        default=None,
        help="Explicit API base URL (overrides config)",
    )
    parser.add_argument(
        "-M",
        "--model",
        default=None,
        help="Model name (default: value from profile or provider default)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["stream", "batch"],
        default="stream",
        help="Output mode: 'stream' = real-time, 'batch' = wait for full output",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_data, config_source = load_config(args.config)
    api_config = config_data.get("api") if isinstance(config_data.get("api"), dict) else {}
    active_profile, profile_settings = select_api_profile(api_config, args.profile)

    provider_name = determine_provider(profile_settings, active_profile)

    base_url = args.base_url or str(profile_settings.get("base_url", "")).strip()
    if provider_name == "ollama":
        ip = args.ip or profile_settings.get("ip")
        port = args.port or profile_settings.get("port")
        if not base_url:
            if ip and port:
                base_url = f"http://{ip}:{port}"
            else:
                base_url = "http://127.0.0.1:11434"
    else:
        if not base_url:
            base_url = _DEFAULT_OPENAI_BASE_URL

    model = args.model or str(profile_settings.get("model", "")).strip() or (
        "qwen3:1.7b" if provider_name == "ollama" else "gpt-4o-mini"
    )

    allow_default = not args.no_default_system
    system_prompt = load_system_prompt(args.system_file, allow_default)

    user_prompt = args.user_prompt
    memory_lines = max(0, args.memory_lines)
    chat_after_stdin = args.chat_after_stdin
    temperature = args.temperature

    if provider_name == "ollama":
        provider: Provider = OllamaProvider(base_url, model)
    else:
        require_key = provider_name == "openai"
        env_var = str(profile_settings.get("api_key_env", "OPENAI_API_KEY"))
        client = create_openai_client(base_url, require_key, env_var, config_source)
        provider = OpenAIProvider(client, base_url, model)

    mode = args.mode
    if mode is None:
        mode = "batch" if not sys.stdin.isatty() else "stream"

    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if not data.strip():
            interactive_mode(
                provider,
                mode,
                temperature,
                system_prompt,
                user_prompt,
                memory_lines,
                active_profile,
            )
            return

        final_text = ""
        if mode == "stream":
            print("LLM (thinking): ", end="", flush=True)
            final_text = provider.stream(system_prompt, user_prompt, data, temperature)
        else:
            response_text, final_text = provider.batch(system_prompt, user_prompt, data, temperature)
            print(response_text)

        if chat_after_stdin:
            memory_seed: List[str] = []
            data_text = data.strip()
            if data_text:
                memory_seed.append(f"User: {data_text}")
            if final_text:
                memory_seed.append(f"Assistant: {final_text.strip()}")
            if memory_lines > 0 and memory_seed:
                memory_seed = memory_seed[-memory_lines:]
            interactive_mode(
                provider,
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
        provider,
        mode,
        temperature,
        system_prompt,
        user_prompt,
        memory_lines,
        active_profile,
    )


if __name__ == "__main__":
    main()
