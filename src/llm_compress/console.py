from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def section(title: str) -> None:
    print(f"\n== {title} ==", flush=True)


def subsection(title: str) -> None:
    print(f"\n-- {title} --", flush=True)


def step(message: str) -> None:
    print(f"→ {message}", flush=True)


def info(message: str, *, indent: int = 0) -> None:
    print(f"{' ' * indent}{message}", flush=True)


def ok(message: str) -> None:
    print(f"✓ {message}", flush=True)


def warn(message: str) -> None:
    print(f"! {message}", flush=True)


def fail(message: str) -> None:
    print(f"✗ {message}", flush=True)


def detail(label: str, value: Any, *, indent: int = 2, label_width: int = 18) -> None:
    label_text = f"{label}:"
    lines = str(value).splitlines() or [""]
    prefix = " " * indent
    print(f"{prefix}{label_text:<{label_width}} {lines[0]}", flush=True)
    continuation_prefix = " " * (indent + label_width + 1)
    for line in lines[1:]:
        print(f"{continuation_prefix}{line}", flush=True)


def details(rows: Iterable[tuple[str, Any]], *, indent: int = 2, label_width: int = 18) -> None:
    for label, value in rows:
        detail(label, value, indent=indent, label_width=label_width)


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def format_bytes(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(size) < 1024.0 or unit == "GiB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def format_ratio(value: float) -> str:
    return f"{value:.3f}x"


def format_duration(seconds: float) -> str:
    return f"{seconds:.1f}s"


def preview_list(items: Iterable[Any], *, limit: int = 6) -> str:
    values = [str(item) for item in items]
    if not values:
        return "none"
    shown = values[:limit]
    suffix = f" (+{len(values) - limit} more)" if len(values) > limit else ""
    return ", ".join(shown) + suffix


def print_error_map(errors: Mapping[str, Any], *, limit: int = 5, indent: int = 4) -> None:
    if not errors:
        return
    for index, (path, error) in enumerate(errors.items()):
        if index >= limit:
            info(f"... {len(errors) - limit} more error(s)", indent=indent)
            break
        info(f"- {path}: {one_line(error, limit=500)}", indent=indent)


def output_tail(text: str, *, max_lines: int = 12, max_chars: int = 2_000) -> str:
    if not text:
        return ""
    tail = text[-max_chars:]
    tail_lines = tail.splitlines()
    lines = tail_lines[-max_lines:]
    truncated = len(text) > max_chars or len(tail_lines) > max_lines
    prefix = "...\n" if truncated else ""
    return prefix + "\n".join(lines)


def print_output_tail(text: str, *, indent: int = 4, max_lines: int = 12, max_chars: int = 2_000) -> None:
    tail = output_tail(text, max_lines=max_lines, max_chars=max_chars)
    if not tail:
        return
    info("output tail:", indent=indent)
    for line in tail.splitlines():
        info(line, indent=indent + 2)


def one_line(value: Any, *, limit: int = 160) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"
