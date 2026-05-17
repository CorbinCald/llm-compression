from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact import decode_lossless, read_artifact, safe_output_path, sha256_bytes
from .files import language_hint
from .openrouter import LLMError, OpenRouterClient
from .prompts import DECOMPRESSION_PROMPT


@dataclass
class DecompressionSummary:
    artifact_path: Path
    output_dir: Path
    file_count: int
    llm_file_count: int
    lossless_file_count: int
    hash_matches: int
    hash_mismatches: list[str]
    errors: dict[str, str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_json(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "output_dir": str(self.output_dir),
            "file_count": self.file_count,
            "llm_file_count": self.llm_file_count,
            "lossless_file_count": self.lossless_file_count,
            "hash_matches": self.hash_matches,
            "hash_mismatches": self.hash_mismatches,
            "errors": self.errors,
            "ok": self.ok,
        }


class Decompressor:
    def __init__(
        self,
        client: OpenRouterClient | None,
        *,
        prompt_extra: str = "",
        candidate_index: int = 0,
        candidate_count: int = 1,
    ):
        self.client = client
        self.prompt_extra = prompt_extra
        self.candidate_index = candidate_index
        self.candidate_count = candidate_count

    def decompress_artifact(
        self,
        artifact_path: Path,
        output_dir: Path,
        *,
        workers: int = 4,
        clean: bool = True,
    ) -> DecompressionSummary:
        artifact_path = artifact_path.resolve()
        _, records = read_artifact(artifact_path)
        if clean and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        errors: dict[str, str] = {}
        hash_mismatches: list[str] = []
        hash_matches = 0
        file_count = 0
        llm_file_count = 0
        lossless_file_count = 0
        tasks: dict[Any, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for record in records:
                record_type = record.get("type")
                rel = record.get("path")
                if not isinstance(rel, str):
                    errors[f"record:{len(errors)}"] = "record missing path"
                    continue
                try:
                    out_path = safe_output_path(output_dir, rel)
                except ValueError as exc:
                    errors[rel] = str(exc)
                    continue

                if record_type == "empty_dir":
                    out_path.mkdir(parents=True, exist_ok=True)
                elif record_type == "symlink":
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    target = record.get("target")
                    if not isinstance(target, str):
                        errors[rel] = "symlink missing target"
                        continue
                    try:
                        out_path.symlink_to(target)
                    except FileExistsError:
                        pass
                elif record_type == "file":
                    file_count += 1
                    mode = record.get("mode")
                    if mode == "lossless":
                        lossless_file_count += 1
                        try:
                            data = decode_lossless(record["data"])
                            _write_bytes(out_path, data)
                            if record.get("sha256") == sha256_bytes(data):
                                hash_matches += 1
                            else:
                                hash_mismatches.append(rel)
                        except Exception as exc:  # noqa: BLE001 - keep restoring other files.
                            errors[rel] = str(exc)
                    elif mode == "llm":
                        llm_file_count += 1
                        future = executor.submit(self._decompress_file, record)
                        tasks[future] = record
                    else:
                        errors[rel] = f"unknown file mode: {mode}"
                else:
                    errors[rel] = f"unknown record type: {record_type}"

            for future in as_completed(tasks):
                record = tasks[future]
                rel = record["path"]
                try:
                    text = future.result()
                    data = text.encode("utf-8")
                    _write_bytes(safe_output_path(output_dir, rel), data)
                    if record.get("sha256") == sha256_bytes(data):
                        hash_matches += 1
                    else:
                        hash_mismatches.append(rel)
                except Exception as exc:  # noqa: BLE001 - report all failed files.
                    errors[rel] = str(exc)

        return DecompressionSummary(
            artifact_path=artifact_path,
            output_dir=output_dir,
            file_count=file_count,
            llm_file_count=llm_file_count,
            lossless_file_count=lossless_file_count,
            hash_matches=hash_matches,
            hash_mismatches=hash_mismatches,
            errors=errors,
        )

    def _decompress_file(self, record: dict[str, Any]) -> str:
        if not self.client or not self.client.available:
            raise LLMError("OPENROUTER_API_KEY is required to decompress LLM components")
        rel = record["path"]
        components = record.get("components")
        if not isinstance(components, list) or not all(isinstance(c, str) for c in components):
            raise ValueError("LLM file record missing components")
        user = (
            f"Rebuild the complete file at {rel} from these compressed components.\n"
            "Return only the file content. Do not wrap it in Markdown fences.\n"
            f"Candidate: {self.candidate_index + 1}/{self.candidate_count}. If multiple valid reconstructions exist, "
            "choose the simplest one that satisfies the component specs.\n"
            f"Language hint: {language_hint(rel)}\n\n"
            + "\n".join(components)
        )
        max_tokens = min(32_768, max(2_048, int(record.get("size", 0)) * 2 + 1_024))
        response = self.client.chat(
            system=self._system_prompt(),
            user=user,
            max_completion_tokens=max_tokens,
        )
        text = _strip_fence(response.strip("\n"))
        if text.strip() == "<DECOMP_FAILURE>":
            raise LLMError("Decompression worker returned <DECOMP_FAILURE>")
        return text

    def _system_prompt(self) -> str:
        if not self.prompt_extra.strip():
            return DECOMPRESSION_PROMPT
        return DECOMPRESSION_PROMPT + "\n\nAutoresearch prompt adjustment:\n" + self.prompt_extra.strip()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return text
