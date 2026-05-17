from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compressor import CompressionOptions, CompressionSummary, Compressor
from .console import details, section, step, yes_no
from .openrouter import OpenRouterClient
from .reporting import log_compression_summary
from .targets import PreparedTarget, prepare_target, update_index


@dataclass
class DirectCompressionOptions:
    runs_root: Path = Path(".llm-compress/runs")
    workers: int = 4
    max_llm_bytes: int = 20_000
    use_llm: bool = True
    chunking_strategy: str = "file"
    chunk_size_lines: int = 120
    format_variant: str = "component_v1"
    compression_prompt_extra: str = ""


@dataclass
class DirectCompressionResult:
    target: PreparedTarget
    compression: CompressionSummary
    artifact_path: Path

    @property
    def success(self) -> bool:
        return not self.compression.errors

    def to_json(self) -> dict[str, Any]:
        return {
            "target": {
                "target": self.target.target,
                "label": self.target.label,
                "source_root": str(self.target.source_root),
                "run_dir": str(self.target.run_dir),
                "cloned": self.target.cloned,
            },
            "success": self.success,
            "artifact_path": str(self.artifact_path),
            "compression": self.compression.to_json(),
        }


def compress_target_once(
    target: str,
    *,
    client: OpenRouterClient | None,
    options: DirectCompressionOptions,
) -> DirectCompressionResult:
    prepared = prepare_target(target, options.runs_root)
    artifact_path = prepared.run_dir / "compressed.jsonl"
    section("Compression run")
    details(
        (
            ("target", prepared.target),
            ("source", prepared.source_root),
            ("run dir", prepared.run_dir),
            ("autoresearch", "off"),
            ("LLM enabled", yes_no(bool(client and client.available and options.use_llm))),
            ("workers", options.workers),
            ("max LLM bytes", options.max_llm_bytes),
            ("format", options.format_variant),
            ("chunking", options.chunking_strategy),
        )
    )
    step("Compressing source files")

    compression = Compressor(client).compress_repo(
        prepared.source_root,
        artifact_path,
        options=CompressionOptions(
            max_llm_bytes=options.max_llm_bytes,
            workers=options.workers,
            use_llm=bool(client and client.available and options.use_llm),
            chunking_strategy=options.chunking_strategy,
            chunk_size_lines=options.chunk_size_lines,
            format_variant=options.format_variant,
            compression_prompt_extra=options.compression_prompt_extra,
        ),
        target_label=prepared.target,
    )
    update_index(prepared.target, prepared.run_dir, artifact_path)
    result = DirectCompressionResult(
        target=prepared,
        compression=compression,
        artifact_path=artifact_path,
    )
    _write_json(prepared.run_dir / "summary.json", result.to_json())
    log_compression_summary(compression)
    return result


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
