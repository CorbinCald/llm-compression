from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifact import read_artifact, safe_output_path
from .openrouter import LLMError, OpenRouterClient
from .verify import VerificationReport

REPAIR_PROMPT = """You repair source files reconstructed from compressed component descriptions.

Input includes failed verification output, compressed components, and current reconstructed file contents.
Return only JSON:
{
  "repairs": [
    {"path": "./relative/path", "content": "full corrected file content"}
  ],
  "notes": "short explanation"
}

Rules:
- Repair only listed allowed paths.
- Return full file content for each repaired path, not a diff.
- Preserve exact behavior described by the compressed components.
- Use failed tests/lints/builds only as guidance to restore intended behavior.
- Do not invent unrelated features.
- If you cannot safely repair, return {"repairs": [], "notes": "..."}.
"""


@dataclass
class RepairSummary:
    attempted: bool
    repaired_paths: list[str] = field(default_factory=list)
    notes: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_json(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "repaired_paths": self.repaired_paths,
            "notes": self.notes,
            "error": self.error,
            "ok": self.ok,
        }


def repair_restored_candidate(
    *,
    client: OpenRouterClient | None,
    artifact_path: Path,
    restore_dir: Path,
    verification: VerificationReport | None,
    allowed_paths: list[str],
    prompt_extra: str = "",
    max_files: int = 4,
) -> RepairSummary:
    if not client or not client.available:
        return RepairSummary(attempted=False, notes="LLM unavailable")
    if verification is None:
        return RepairSummary(attempted=False, notes="verification unavailable")
    if verification.ok:
        return RepairSummary(attempted=False, notes="verification already passes")

    records_by_path = _llm_records_by_path(artifact_path)
    allowed = [path for path in allowed_paths if path in records_by_path]
    suspects = _suspect_paths(verification.all_output(), allowed)
    selected = (suspects or allowed)[:max_files]
    if not selected:
        return RepairSummary(attempted=False, notes="no LLM-restored paths available to repair")

    payload = {
        "failed_verification": verification.to_json(),
        "allowed_paths": selected,
        "files": [
            {
                "path": path,
                "components": records_by_path[path].get("components", []),
                "current_content": _read_current_content(restore_dir, path),
            }
            for path in selected
        ],
    }
    system = REPAIR_PROMPT
    if prompt_extra.strip():
        system += "\n\nAutoresearch repair guidance:\n" + prompt_extra.strip()
    try:
        response = client.chat(
            system=system,
            user=json.dumps(payload, indent=2, sort_keys=True),
            max_completion_tokens=16_384,
        )
        data = _extract_json_object(response)
        repairs = data.get("repairs", [])
        if not isinstance(repairs, list):
            raise LLMError("repair JSON field 'repairs' was not a list")
        repaired_paths: list[str] = []
        allowed_set = set(selected)
        for repair in repairs:
            if not isinstance(repair, dict):
                continue
            path = repair.get("path")
            content = repair.get("content")
            if not isinstance(path, str) or path not in allowed_set or not isinstance(content, str):
                continue
            output_path = safe_output_path(restore_dir, path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
            repaired_paths.append(path)
        return RepairSummary(
            attempted=True,
            repaired_paths=repaired_paths,
            notes=str(data.get("notes", ""))[:1000],
        )
    except Exception as exc:  # noqa: BLE001 - repair failure is data for the next experiment.
        return RepairSummary(attempted=True, error=str(exc))


def _llm_records_by_path(artifact_path: Path) -> dict[str, dict[str, Any]]:
    _, records = read_artifact(artifact_path)
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("type") == "file" and record.get("mode") == "llm":
            path = record.get("path")
            if isinstance(path, str):
                result[path] = record
    return result


def _read_current_content(root: Path, rel: str) -> str:
    try:
        path = safe_output_path(root, rel)
        return path.read_text(encoding="utf-8")[:80_000]
    except Exception as exc:  # noqa: BLE001 - include why repair lacks content.
        return f"<UNREADABLE:{exc}>"


def _suspect_paths(output: str, candidates: list[str]) -> list[str]:
    normalized = output.replace("\\", "/")
    suspects: list[str] = []
    for rel in candidates:
        bare = rel[2:] if rel.startswith("./") else rel
        name = Path(bare).name
        if bare in normalized or rel in normalized or re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(name)}(?![A-Za-z0-9_.-])", normalized):
            suspects.append(rel)
    return suspects


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise LLMError(f"Repair response did not contain JSON: {text[:500]}")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise LLMError("Repair response JSON was not an object")
    return data
