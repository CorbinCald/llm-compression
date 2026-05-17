from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .files import is_excluded_dir_name

INDEX_PATH = Path(".llm-compress/index.json")


@dataclass
class PreparedTarget:
    target: str
    label: str
    source_root: Path
    run_dir: Path
    cloned: bool


def is_git_url(target: str) -> bool:
    return (
        target.startswith("https://")
        or target.startswith("http://")
        or target.startswith("git@")
        or target.endswith(".git")
    )


def slugify_target(target: str) -> str:
    if is_git_url(target):
        parsed = urlparse(target)
        base = (parsed.netloc + parsed.path).strip("/") if parsed.netloc else target
    else:
        base = str(Path(target).expanduser().resolve()) if Path(target).exists() else target
    base = base.removesuffix(".git")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-._")
    return slug[-96:] or "target"


def prepare_target(target: str, runs_root: Path) -> PreparedTarget:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = slugify_target(target)
    run_dir = runs_root / f"{label}-{timestamp}"
    source_root = run_dir / "source"
    run_dir.mkdir(parents=True, exist_ok=False)

    if is_git_url(target):
        _clone(target, source_root)
        cloned = True
    else:
        src = Path(target).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Target does not exist and is not a git URL: {target}")
        if src.is_file():
            source_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, source_root / src.name)
        else:
            shutil.copytree(src, source_root, ignore=_copy_ignore, symlinks=True)
        cloned = False

    return PreparedTarget(
        target=target,
        label=label,
        source_root=source_root,
        run_dir=run_dir,
        cloned=cloned,
    )


def _clone(url: str, destination: Path) -> None:
    command = ["git", "clone", "--depth", "1", url, str(destination)]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git clone failed for {url}:\n{completed.stdout[-4000:]}")


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if is_excluded_dir_name(name):
            ignored.add(name)
    return ignored


def normalize_target_key(target: str) -> str:
    if is_git_url(target):
        return target.rstrip("/").removesuffix(".git")
    path = Path(target).expanduser()
    if path.exists():
        return str(path.resolve())
    return target


def update_index(target: str, run_dir: Path, artifact_path: Path) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index = read_index()
    index[normalize_target_key(target)] = {
        "run_dir": str(run_dir),
        "artifact_path": str(artifact_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def read_index() -> dict[str, dict[str, str]]:
    if not INDEX_PATH.exists():
        return {}
    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def resolve_artifact_reference(reference: str) -> Path:
    path = Path(reference).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        direct = path / "compressed.jsonl"
        if direct.exists():
            return direct.resolve()
        final = path / "final" / "compressed.jsonl"
        if final.exists():
            return final.resolve()
        candidates = sorted(path.glob("iter-*/compressed.jsonl"))
        if candidates:
            return candidates[-1].resolve()

    key = normalize_target_key(reference)
    index = read_index()
    if key in index:
        artifact = Path(index[key]["artifact_path"]).expanduser()
        if artifact.exists():
            return artifact.resolve()
    raise FileNotFoundError(
        f"Could not resolve artifact from {reference!r}. Pass a compressed.jsonl, run dir, "
        "or a target that was previously compressed."
    )
