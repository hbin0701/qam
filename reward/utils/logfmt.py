"""Small ANSI formatter for consistent reward pipeline logs."""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Tuple


_RESET = "\033[0m"
_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "gray": "\033[90m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
}


def _color_mode() -> str:
    mode = str(os.getenv("QAM_LOG_COLOR", "auto")).strip().lower()
    if mode in {"always", "never", "auto"}:
        return mode
    return "auto"


def use_color() -> bool:
    mode = _color_mode()
    if mode == "always":
        return True
    if mode == "never":
        return False
    if os.getenv("NO_COLOR") is not None:
        return False
    if os.getenv("TERM", "").lower() in {"", "dumb"}:
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def color(text: str, *styles: str) -> str:
    if not use_color() or not styles:
        return text
    prefix = "".join(_CODES.get(s, "") for s in styles)
    if not prefix:
        return text
    return f"{prefix}{text}{_RESET}"


def level_prefix(name: str, level: str = "info") -> str:
    lvl = str(level).lower()
    if lvl == "warn":
        return color(f"[{name}]", "bold", "yellow")
    if lvl == "error":
        return color(f"[{name}]", "bold", "red")
    if lvl == "success":
        return color(f"[{name}]", "bold", "green")
    return color(f"[{name}]", "bold", "cyan")


def kv(key: str, value: Any, tone: str = "muted") -> str:
    key_style = ("bold", "gray") if tone == "muted" else ("bold", "blue")
    key_txt = color(str(key), *key_style)
    val_txt = color(str(value), "magenta") if tone == "money" else str(value)
    return f"{key_txt}={val_txt}"


def join_kv(items: Iterable[Tuple[str, Any]], tone: str = "muted") -> str:
    return " ".join(kv(k, v, tone=tone) for k, v in items)

