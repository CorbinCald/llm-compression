from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .files import is_excluded_dir_name

_CONTEXT_FILE_ORDER = (
    "pyproject.toml",
    "program.md",
    "PROMPTS.md",
    "src/llm_compress/compressor.py",
    "src/llm_compress/decompressor.py",
    "src/llm_compress/prompts.py",
    "src/llm_compress/files.py",
    "src/llm_compress/artifact.py",
    "src/llm_compress/repair.py",
    "src/llm_compress/openrouter.py",
    "src/llm_compress/verify.py",
    "src/llm_compress/pipeline.py",
    "src/llm_compress/research.py",
    "src/llm_compress/global_research.py",
    "src/llm_compress/benchmarks.py",
    "src/llm_compress/app.py",
    "src/llm_compress/cli.py",
)

_PROTECTED_PATCH_PATHS = {
    "src/llm_compress/iteration_runner.py",
    "src/llm_compress/tool_patch.py",
}

_EXTRA_EXCLUDED_DIRS = {
    ".git",
    ".llm-compress",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
}


def tool_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_tool_repo_context(*, max_chars: int = 180_000, max_tree_entries: int = 400) -> dict[str, Any]:
    root = tool_repo_root()
    tree = _repo_tree(root, max_entries=max_tree_entries)
    files: list[dict[str, Any]] = []
    remaining = max(0, max_chars)
    for rel in _context_file_order(root):
        if remaining <= 0:
            break
        path = root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        truncated = len(text) > remaining
        content = text[:remaining]
        files.append(
            {
                "path": rel,
                "chars": len(text),
                "truncated": truncated,
                "content": content,
            }
        )
        remaining -= len(content)
    return {
        "root": str(root),
        "tree": tree,
        "selected_files": files,
        "patch_capability": {
            "enabled": True,
            "field": "code_patch",
            "format": "unified diff rooted at the llm-compression repo, with a/ and b/ paths",
            "temporary": "The harness applies the patch to an isolated copy for one experiment, then discards it.",
            "effective_scope": (
                "The patched copy runs compression and decompression for that experiment. "
                "The main harness still performs scoring and verification so patches cannot fake success."
            ),
            "protected_paths": sorted(_PROTECTED_PATCH_PATHS),
        },
    }


def copy_tool_repo(destination: Path) -> Path:
    root = tool_repo_root()
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(root, destination, symlinks=True, ignore=_copy_ignore)
    return destination


def apply_code_patch(repo_root: Path, patch_text: str) -> None:
    patch = patch_text.strip()
    if not patch:
        return
    validate_code_patch(patch)
    patch_path = repo_root / ".llm-compress-experiment.patch"
    patch_path.write_text(patch + "\n", encoding="utf-8")
    completed = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch_path)],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"code_patch did not apply:\n{completed.stdout[-4000:]}")
    compile_completed = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "src"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    if compile_completed.returncode != 0:
        raise RuntimeError(f"code_patch produced invalid Python:\n{compile_completed.stdout[-4000:]}")


def validate_code_patch(patch_text: str, *, max_chars: int = 80_000) -> None:
    if len(patch_text) > max_chars:
        raise ValueError(f"code_patch is too large ({len(patch_text)} chars > {max_chars})")
    paths = _patch_paths(patch_text)
    if not paths:
        raise ValueError("code_patch must be a unified diff with file paths")
    for path in paths:
        _validate_patch_path(path)


def _context_file_order(root: Path) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for rel in _CONTEXT_FILE_ORDER:
        if (root / rel).exists() and rel not in seen:
            result.append(rel)
            seen.add(rel)
    src_root = root / "src" / "llm_compress"
    if src_root.exists():
        for path in sorted(src_root.glob("*.py")):
            rel = path.relative_to(root).as_posix()
            if rel not in seen and rel not in _PROTECTED_PATCH_PATHS:
                result.append(rel)
                seen.add(rel)
    tests_root = root / "tests"
    if tests_root.exists():
        for path in sorted(tests_root.glob("test_*.py")):
            rel = path.relative_to(root).as_posix()
            if rel not in seen:
                result.append(rel)
                seen.add(rel)
    return result


def _repo_tree(root: Path, *, max_entries: int) -> list[str]:
    entries: list[str] = []
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs[:] = [name for name in sorted(dirs) if not _is_excluded(name)]
        current_path = Path(current)
        rel_dir = current_path.relative_to(root).as_posix() if current_path != root else "."
        if rel_dir != ".":
            entries.append(rel_dir + "/")
        for name in sorted(files):
            if _is_excluded(name):
                continue
            entries.append((current_path / name).relative_to(root).as_posix())
            if len(entries) >= max_entries:
                entries.append(f"... truncated after {max_entries} entries")
                return entries
    return entries


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if _is_excluded(name)}


def _is_excluded(name: str) -> bool:
    return name in _EXTRA_EXCLUDED_DIRS or is_excluded_dir_name(name) or name.endswith(".egg-info")


def _patch_paths(patch_text: str) -> set[str]:
    paths: set[str] = set()
    for raw_line in patch_text.splitlines():
        line = raw_line.strip()
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                paths.add(_normalize_patch_path(parts[2]))
                paths.add(_normalize_patch_path(parts[3]))
        elif line.startswith("--- ") or line.startswith("+++ "):
            parts = line.split(maxsplit=2)
            if len(parts) >= 2:
                paths.add(_normalize_patch_path(parts[1]))
    paths.discard("/dev/null")
    return paths


def _normalize_patch_path(raw: str) -> str:
    path = raw.strip().strip('"')
    if path in {"/dev/null", "dev/null"}:
        return "/dev/null"
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path


def _validate_patch_path(path: str) -> None:
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe code_patch path: {path}")
    parts = set(rel.parts)
    if ".git" in parts or ".llm-compress" in parts:
        raise ValueError(f"code_patch may not modify internal directories: {path}")
    normalized = rel.as_posix()
    if normalized in _PROTECTED_PATCH_PATHS:
        raise ValueError(f"code_patch may not modify protected harness file: {path}")
