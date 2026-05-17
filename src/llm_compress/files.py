from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".eggs",
    ".cache",
    ".gradle",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "coverage",
    ".coverage",
    "htmlcov",
    ".next",
    ".nuxt",
    ".svelte-kit",
    "target",
    "vendor",
    "Pods",
    "DerivedData",
    ".llm-compress-venv",
    ".llm-compress",
}

LOSSLESS_FILENAMES = {
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lockb",
    "bun.lock",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "requirements-dev.txt",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "tox.ini",
    "pytest.ini",
    "ruff.toml",
    ".ruff.toml",
    "mypy.ini",
    "Cargo.toml",
    "Cargo.lock",
    "go.mod",
    "go.sum",
    "Gemfile",
    "Gemfile.lock",
    "composer.json",
    "composer.lock",
    "tsconfig.json",
    "tsconfig.build.json",
    "jsconfig.json",
    "Makefile",
    "Dockerfile",
    ".gitignore",
    ".npmrc",
    ".python-version",
    ".node-version",
}

SOURCE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".kts",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",
    ".svelte",
}

TEXT_EXTENSIONS = SOURCE_EXTENSIONS | {
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".md",
    ".txt",
    ".rst",
    ".csv",
    ".xml",
    ".svg",
}

TEST_DIR_NAMES = {"test", "tests", "spec", "specs", "__tests__", "testdata", "fixtures"}


def posix_rel(path: Path, root: Path) -> str:
    return "./" + path.relative_to(root).as_posix()


def is_excluded_dir_name(name: str) -> bool:
    return name in EXCLUDED_DIRS or name.endswith(".egg-info")


def iter_repo_paths(root: Path) -> Iterable[Path]:
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs[:] = [d for d in dirs if not is_excluded_dir_name(d)]
        current_path = Path(current)
        for name in sorted(files):
            yield current_path / name


def iter_empty_dirs(root: Path) -> Iterable[Path]:
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs[:] = [d for d in dirs if not is_excluded_dir_name(d)]
        if not dirs and not files:
            current_path = Path(current)
            if current_path != root:
                yield current_path


def is_probably_binary(data: bytes) -> bool:
    if b"\0" in data[:4096]:
        return True
    if not data:
        return False
    sample = data[:4096]
    control = sum(1 for byte in sample if byte < 9 or (13 < byte < 32))
    return control / len(sample) > 0.30


def can_decode_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def is_test_path(rel_path: str) -> bool:
    rel = rel_path[2:] if rel_path.startswith("./") else rel_path
    parts = [part.lower() for part in Path(rel).parts]
    if any(part in TEST_DIR_NAMES for part in parts[:-1]):
        return True
    name = parts[-1] if parts else ""
    test_markers = (
        "test_",
        "_test.py",
        ".test.js",
        ".test.ts",
        ".test.jsx",
        ".test.tsx",
        ".spec.js",
        ".spec.ts",
        ".spec.jsx",
        ".spec.tsx",
        "_test.go",
        "_test.rs",
        "spec.rb",
    )
    return name.startswith("test_") or any(name.endswith(marker) for marker in test_markers)


def should_use_llm(rel_path: str, data: bytes, *, max_llm_bytes: int) -> tuple[bool, str]:
    rel = rel_path[2:] if rel_path.startswith("./") else rel_path
    path = Path(rel)
    name = path.name
    suffix = path.suffix.lower()

    if len(data) == 0:
        return False, "empty"
    if len(data) > max_llm_bytes:
        return False, "too_large"
    if is_probably_binary(data):
        return False, "binary"
    if not can_decode_utf8(data):
        return False, "non_utf8"
    if name in LOSSLESS_FILENAMES:
        return False, "critical_config"
    if is_test_path(rel_path):
        return False, "tests_exact"
    if suffix not in SOURCE_EXTENSIONS:
        return False, "not_source"
    return True, "source_llm"


def language_hint(rel_path: str) -> str:
    suffix = Path(rel_path).suffix.lower()
    return {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".c": "c",
        ".h": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".scala": "scala",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "zsh",
        ".fish": "fish",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".vue": "vue",
        ".svelte": "svelte",
    }.get(suffix, "text")
