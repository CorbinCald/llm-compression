from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifact import encode_lossless, sha256_bytes, write_artifact
from .files import iter_empty_dirs, iter_repo_paths, language_hint, posix_rel, should_use_llm
from .openrouter import LLMError, OpenRouterClient
from .prompts import COMPONENT_PROMPT_VERSION, COMPRESSION_PROMPT

_COMPONENT_RE = re.compile(r"<-Name:.+?\|Input:.+?\|Return:.+?\|Path:.+?\|Order:.+?->", re.DOTALL)


@dataclass
class CompressionOptions:
    max_llm_bytes: int = 20_000
    workers: int = 4
    use_llm: bool = True
    # Deprecated compatibility field. Path-specific source-file lossless overrides
    # are intentionally ignored; file eligibility decides unavoidable lossless records.
    lossless_overrides: set[str] = field(default_factory=set)
    lossless_fallback: bool = True
    chunking_strategy: str = "file"
    chunk_size_lines: int = 120
    format_variant: str = "component_v1"
    compression_prompt_extra: str = ""


@dataclass
class CompressionSummary:
    artifact_path: Path
    original_bytes: int
    artifact_bytes: int
    file_count: int
    llm_file_count: int
    lossless_file_count: int
    llm_paths: list[str]
    fallback_paths: list[str]
    errors: dict[str, str]
    chunking_strategy: str
    format_variant: str

    @property
    def ratio(self) -> float:
        return self.artifact_bytes / self.original_bytes if self.original_bytes else 1.0

    def to_json(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "original_bytes": self.original_bytes,
            "artifact_bytes": self.artifact_bytes,
            "ratio": self.ratio,
            "file_count": self.file_count,
            "llm_file_count": self.llm_file_count,
            "lossless_file_count": self.lossless_file_count,
            "llm_paths": self.llm_paths,
            "fallback_paths": self.fallback_paths,
            "errors": self.errors,
            "chunking_strategy": self.chunking_strategy,
            "format_variant": self.format_variant,
        }


class Compressor:
    def __init__(self, client: OpenRouterClient | None):
        self.client = client
        self._component_cache: dict[tuple[Any, ...], list[str]] = {}

    def compress_repo(
        self,
        source_root: Path,
        artifact_path: Path,
        *,
        options: CompressionOptions,
        target_label: str,
    ) -> CompressionSummary:
        source_root = source_root.resolve()
        records_by_path: dict[str, dict[str, Any]] = {}
        tasks: dict[Any, tuple[str, bytes]] = {}
        original_bytes = 0
        errors: dict[str, str] = {}
        fallback_paths: list[str] = []

        llm_available = bool(self.client and self.client.available and options.use_llm)

        with ThreadPoolExecutor(max_workers=max(1, options.workers)) as executor:
            for path in iter_repo_paths(source_root):
                rel = posix_rel(path, source_root)
                if path.is_symlink():
                    records_by_path[rel] = {
                        "type": "symlink",
                        "path": rel,
                        "target": path.readlink().as_posix(),
                    }
                    continue

                data = path.read_bytes()
                original_bytes += len(data)
                digest = sha256_bytes(data)
                wants_llm, reason = should_use_llm(rel, data, max_llm_bytes=options.max_llm_bytes)
                if wants_llm and llm_available:
                    future = executor.submit(self._compress_file, rel, data, digest, options)
                    tasks[future] = (rel, data)
                else:
                    if wants_llm and not llm_available:
                        reason = "llm_unavailable"
                    records_by_path[rel] = self._lossless_record(rel, data, reason)

            for empty_dir in iter_empty_dirs(source_root):
                rel = posix_rel(empty_dir, source_root)
                records_by_path[rel] = {"type": "empty_dir", "path": rel}

            for future in as_completed(tasks):
                rel, data = tasks[future]
                try:
                    records_by_path[rel] = future.result()
                except Exception as exc:  # noqa: BLE001 - fallback is part of the product behavior.
                    errors[rel] = str(exc)
                    if not options.lossless_fallback:
                        raise
                    fallback_paths.append(rel)
                    records_by_path[rel] = self._lossless_record(rel, data, "llm_fallback")

        records = [records_by_path[key] for key in sorted(records_by_path)]
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target": target_label,
            "model": self.client.model if self.client else None,
            "prompt_version": COMPONENT_PROMPT_VERSION,
            "strategy": "hybrid-llm-lossless-autoresearch",
            "chunking_strategy": options.chunking_strategy,
            "chunk_size_lines": options.chunk_size_lines,
            "format_variant": options.format_variant,
        }
        write_artifact(artifact_path, meta, records)
        artifact_bytes = artifact_path.stat().st_size
        llm_paths = [record["path"] for record in records if record.get("mode") == "llm"]
        lossless_count = sum(1 for record in records if record.get("mode") == "lossless")
        file_count = sum(1 for record in records if record.get("type") == "file")
        return CompressionSummary(
            artifact_path=artifact_path,
            original_bytes=original_bytes,
            artifact_bytes=artifact_bytes,
            file_count=file_count,
            llm_file_count=len(llm_paths),
            lossless_file_count=lossless_count,
            llm_paths=llm_paths,
            fallback_paths=fallback_paths,
            errors=errors,
            chunking_strategy=options.chunking_strategy,
            format_variant=options.format_variant,
        )

    @staticmethod
    def _lossless_record(rel: str, data: bytes, reason: str) -> dict[str, Any]:
        return {
            "type": "file",
            "path": rel,
            "mode": "lossless",
            "reason": reason,
            "size": len(data),
            "sha256": sha256_bytes(data),
            "data": encode_lossless(data),
        }

    def _compress_file(
        self,
        rel: str,
        data: bytes,
        digest: str,
        options: CompressionOptions,
    ) -> dict[str, Any]:
        if not self.client:
            raise LLMError("LLM client unavailable")
        cache_key = (
            rel,
            digest,
            self.client.model,
            options.chunking_strategy,
            str(options.chunk_size_lines),
            options.format_variant,
            options.compression_prompt_extra,
        )
        if cache_key in self._component_cache:
            components = self._component_cache[cache_key]
        else:
            text = data.decode("utf-8")
            components = self._compress_components(rel, text, options)
            self._component_cache[cache_key] = components

        return {
            "type": "file",
            "path": rel,
            "mode": "llm",
            "size": len(data),
            "sha256": digest,
            "components": components,
            "chunking_strategy": options.chunking_strategy,
            "format_variant": options.format_variant,
        }

    def _compress_components(
        self,
        rel: str,
        text: str,
        options: CompressionOptions,
    ) -> list[str]:
        chunks = _chunks_for_strategy(text, options)
        components: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            component = self._compress_component(rel, chunk, index, len(chunks), options)
            components.append(component)
        return components

    def _compress_component(
        self,
        rel: str,
        chunk: str,
        index: int,
        count: int,
        options: CompressionOptions,
    ) -> str:
        if not self.client:
            raise LLMError("LLM client unavailable")
        component_name = _component_name(rel)
        if count > 1:
            component_name = f"{component_name}_chunk_{index}"
        order = f"{index}a"
        user = (
            _variant_user_instruction(options.format_variant, count > 1)
            + "\n\n"
            f"Path: {rel}\n"
            f"Order: {order}\n"
            f"Chunk: {index}/{count}\n"
            f"Suggested Name: {component_name}\n\n"
            f"```{language_hint(rel)}\n{chunk}\n```"
        )
        max_tokens = min(16_384, max(1_024, len(chunk) // 2 + 1_024))
        response = self.client.chat(
            system=_compression_system_prompt(options),
            user=user,
            max_completion_tokens=max_tokens,
        )
        component = _extract_component(response)
        acceptable_paths = {rel, rel[2:] if rel.startswith("./") else rel}
        if not any(f"|Path:{path}|" in component or f"|Path: {path}|" in component for path in acceptable_paths):
            # The path is recoverable from the JSON record, but a wrong path inside the
            # component confuses decompression enough that falling back is safer.
            raise LLMError(f"Compressed component path mismatch for {rel}: {component[:200]}")
        return component


def _chunks_for_strategy(text: str, options: CompressionOptions) -> list[str]:
    if options.chunking_strategy != "line_chunks":
        return [text]
    lines = text.splitlines(keepends=True)
    if not lines:
        return [text]
    size = max(1, options.chunk_size_lines)
    return ["".join(lines[i : i + size]) for i in range(0, len(lines), size)]


def _compression_system_prompt(options: CompressionOptions) -> str:
    parts = [COMPRESSION_PROMPT, _variant_system_instruction(options.format_variant)]
    if options.compression_prompt_extra.strip():
        parts.append("Autoresearch prompt adjustment:\n" + options.compression_prompt_extra.strip())
    return "\n\n".join(part for part in parts if part)


def _variant_system_instruction(format_variant: str) -> str:
    if format_variant == "component_contracts":
        return (
            "Format variant: component_contracts. In Return, emphasize exact imports, exports, "
            "public signatures, data shapes, side effects, thrown errors, and call ordering."
        )
    if format_variant == "component_testsafe":
        return (
            "Format variant: component_testsafe. Optimize for passing an unknown existing test suite. "
            "Include exact edge cases, exceptions, async behavior, filesystem/network effects, and literals."
        )
    if format_variant == "component_literal_heavy":
        return (
            "Format variant: component_literal_heavy. Preserve every exact literal, selector, regex, URL, "
            "environment variable, protocol token, and numeric constant. Use compact code-shaped specs."
        )
    return "Format variant: component_v1. Use the base component syntax exactly."


def _variant_user_instruction(format_variant: str, chunked: bool) -> str:
    scope = "source chunk" if chunked else "complete source file"
    text = f"Compress this {scope} as one recoverable component. "
    if chunked:
        text += "The component must rebuild only this chunk; chunk order will be handled by the harness. "
    else:
        text += "The component must rebuild the entire file content. "
    if format_variant == "component_contracts":
        text += "Prioritize contracts, public behavior, imports/exports, and side effects."
    elif format_variant == "component_testsafe":
        text += "Prioritize details tests commonly assert: exact errors, branches, edge cases, and literals."
    elif format_variant == "component_literal_heavy":
        text += "Include all exact literals/constants and code-shaped behavior, even if less compressed."
    else:
        text += "Use the base component format."
    return text


def _extract_component(response: str) -> str:
    stripped = _strip_fence(response.strip())
    if stripped == "<FAILURE>":
        raise LLMError("Compression worker returned <FAILURE>")
    match = _COMPONENT_RE.search(stripped)
    if not match:
        raise LLMError(f"Compression worker did not return a component: {response[:500]}")
    component = " ".join(match.group(0).splitlines()).strip()
    if component == "<FAILURE>":
        raise LLMError("Compression worker returned <FAILURE>")
    return component


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _component_name(rel: str) -> str:
    stem = Path(rel).stem or "file"
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
    if not cleaned:
        cleaned = "file"
    if cleaned[0].isdigit():
        cleaned = "file_" + cleaned
    return cleaned
