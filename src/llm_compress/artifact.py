from __future__ import annotations

import base64
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

FORMAT_NAME = "llm-compress-jsonl"
FORMAT_VERSION = 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_lossless(data: bytes) -> str:
    return base64.b64encode(gzip.compress(data, compresslevel=9, mtime=0)).decode("ascii")


def decode_lossless(data: str) -> bytes:
    return gzip.decompress(base64.b64decode(data.encode("ascii")))


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def write_artifact(path: Path, meta: dict[str, Any], records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    meta_record = {"type": "meta", "format": FORMAT_NAME, "version": FORMAT_VERSION, **meta}
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(compact_json(meta_record) + "\n")
        for record in records:
            handle.write(compact_json(record) + "\n")
    tmp.replace(path)


def read_artifact(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid artifact JSON on line {line_no}: {path}") from exc
            if line_no == 1 and record.get("type") == "meta":
                meta = record
            else:
                records.append(record)
    if meta.get("format") != FORMAT_NAME:
        raise ValueError(f"Unsupported artifact format in {path}")
    return meta, records


def artifact_size(path: Path) -> int:
    return path.stat().st_size


def copy_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def safe_output_path(root: Path, artifact_path: str) -> Path:
    rel = artifact_path[2:] if artifact_path.startswith("./") else artifact_path
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError(f"Unsafe artifact path: {artifact_path}")
    return root / rel_path
