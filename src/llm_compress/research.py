from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .openrouter import LLMError, OpenRouterClient

DEFAULT_RESEARCH_PROGRAM = """# llm-compress autoresearch program

You are the autonomous researcher for llm-compress. Your job is to propose the next
compression experiment. You may also propose temporary changes to the llm-compression
tool codebase when doing so would improve compression/decompression reliability. The
fixed loss metric is the restored project's own verification result: tests, lints,
and builds that pass on the baseline should pass after decompression. Secondary
objective: minimize artifact bytes/original bytes.

Honor the autoresearch loop: propose one experiment, let the harness execute it,
inspect the metric and logs, then keep/discard/adjust in the next proposal.
Change every relevant variable when evidence suggests it could help: prompt wording,
model choice, chunking, compression format variant, decompression candidate count,
repair from failed tests, and temporary tool-code patches. Repair is diagnostic only:
a repaired candidate is not accepted as success, but its changed paths, notes, and
reverification result are fed into the next experiment. Do not use source-file
lossless overrides; if an LLM-eligible source file fails verification, improve the
compression/decompression strategy or tool implementation rather than preserving that
file exactly.
"""

RESEARCH_SYSTEM_PROMPT = """You run an autonomous software compression research loop.

You output exactly one JSON object that describes the next experiment for the harness.

You may temporarily change the llm-compression tool itself by setting code_patch to a unified diff. The context includes a directory tree and selected source files from this repository. The harness applies that diff to an isolated copy of this repository for one experiment, uses the patched copy for compression/decompression, then discards it. The main harness still performs scoring and verification, so patches must genuinely improve restored code behavior rather than bypass verification.

Core objective:
- Primary metric: minimize raw-decompression failure_units before any repair. failure_units use granular failed test counts when verification output exposes them, and fallback command-level units for failed setup/lint/build/test commands otherwise.
- Secondary metric: lower artifact_bytes/original_bytes is better.
- Prefer the most compressed verified strategy; do not mark LLM-eligible source files lossless.
- Diagnostic repair may run after failure, but repaired candidates are not accepted. When history contains repair_diagnostics, treat changed paths, notes, and repaired verification as clues about what information the next raw compression/decompression must preserve. Never describe repaired verification as success.

You may adjust all experiment variables:
- model: choose one allowed OpenRouter model.
- compression_prompt_extra: extra instructions appended to the compression worker prompt.
- decompression_prompt_extra: extra instructions appended to the decompression worker prompt.
- format_variant: one of component_v1, component_contracts, component_testsafe, component_literal_heavy.
- chunking_strategy: one of file, line_chunks.
- chunk_size_lines: positive integer used when chunking_strategy is line_chunks.
- candidate_count: number of independent decompressions that compete.
- repair_enabled: whether failed candidates should be diagnostically patched by an LLM using verification output. Repaired candidates are not accepted as success.
- max_llm_bytes: max source file bytes eligible for LLM compression; may be raised above the user default, but not lowered to avoid hard files.
- temperature: OpenRouter sampling temperature for compression/decompression/repair.
- code_patch: optional unified diff against this llm-compression repo. Use it to improve the compressor, decompressor, prompts, file classification, artifact handling, or other tool code when parameter tuning is not enough. Leave it empty when no tool-code change is needed.

The lossless_overrides field is kept only for backward-compatible JSON output. It must be [];
the harness ignores non-empty values from research plans.

Return only JSON with this schema:
{
  "hypothesis": "short reason for this experiment",
  "model": "one allowed model",
  "compression_prompt_extra": "string, may be empty",
  "decompression_prompt_extra": "string, may be empty",
  "format_variant": "component_v1|component_contracts|component_testsafe|component_literal_heavy",
  "chunking_strategy": "file|line_chunks",
  "chunk_size_lines": 80,
  "candidate_count": 1,
  "repair_enabled": true,
  "lossless_overrides": [],
  "max_llm_bytes": 20000,
  "temperature": 0.1,
  "code_patch": ""
}
"""

FORMAT_VARIANTS = {
    "component_v1",
    "component_contracts",
    "component_testsafe",
    "component_literal_heavy",
}
CHUNKING_STRATEGIES = {"file", "line_chunks"}


@dataclass
class ResearchPlan:
    hypothesis: str = "baseline hybrid compression"
    model: str = "openrouter/auto"
    compression_prompt_extra: str = ""
    decompression_prompt_extra: str = ""
    format_variant: str = "component_v1"
    chunking_strategy: str = "file"
    chunk_size_lines: int = 120
    candidate_count: int = 1
    repair_enabled: bool = True
    lossless_overrides: list[str] = field(default_factory=list)
    max_llm_bytes: int = 20_000
    temperature: float = 0.1
    code_patch: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "model": self.model,
            "compression_prompt_extra": self.compression_prompt_extra,
            "decompression_prompt_extra": self.decompression_prompt_extra,
            "format_variant": self.format_variant,
            "chunking_strategy": self.chunking_strategy,
            "chunk_size_lines": self.chunk_size_lines,
            "candidate_count": self.candidate_count,
            "repair_enabled": self.repair_enabled,
            "lossless_overrides": self.lossless_overrides,
            "max_llm_bytes": self.max_llm_bytes,
            "temperature": self.temperature,
            "code_patch": self.code_patch,
        }

    @classmethod
    def from_json(
        cls,
        data: dict[str, Any],
        *,
        default_model: str,
        allowed_models: list[str],
        max_candidates: int,
        default_max_llm_bytes: int,
    ) -> "ResearchPlan":
        allowed = allowed_models or [default_model]
        model = str(data.get("model") or default_model)
        if model not in allowed:
            model = default_model

        format_variant = str(data.get("format_variant") or "component_v1")
        if format_variant not in FORMAT_VARIANTS:
            format_variant = "component_v1"

        chunking_strategy = str(data.get("chunking_strategy") or "file")
        if chunking_strategy not in CHUNKING_STRATEGIES:
            chunking_strategy = "file"

        max_llm_bytes = _clamp_int(data.get("max_llm_bytes"), 1_000, 200_000, default_max_llm_bytes)
        max_llm_bytes = max(default_max_llm_bytes, max_llm_bytes)

        return cls(
            hypothesis=_short_string(data.get("hypothesis"), "LLM-proposed experiment", 400),
            model=model,
            compression_prompt_extra=_short_string(data.get("compression_prompt_extra"), "", 3000),
            decompression_prompt_extra=_short_string(data.get("decompression_prompt_extra"), "", 3000),
            format_variant=format_variant,
            chunking_strategy=chunking_strategy,
            chunk_size_lines=_clamp_int(data.get("chunk_size_lines"), 20, 400, 120),
            candidate_count=_clamp_int(data.get("candidate_count"), 1, max(1, max_candidates), 1),
            repair_enabled=bool(data.get("repair_enabled", True)),
            lossless_overrides=[],
            max_llm_bytes=max_llm_bytes,
            temperature=_clamp_float(data.get("temperature"), 0.0, 1.2, 0.1),
            code_patch=_short_string(data.get("code_patch"), "", 80_000),
        )


class ResearchAgent:
    def __init__(
        self,
        client: OpenRouterClient | None,
        *,
        program: str = DEFAULT_RESEARCH_PROGRAM,
        allowed_models: list[str] | None = None,
        max_candidates: int = 3,
    ):
        self.client = client
        self.program = program
        self.allowed_models = allowed_models or ([client.model] if client else ["openrouter/auto"])
        self.max_candidates = max(1, max_candidates)

    def initial_plan(self, *, context: dict[str, Any]) -> ResearchPlan:
        return self._propose(
            context=context,
            history=[],
            fallback=self._fallback_initial(context),
            phase="initial",
        )

    def next_plan(
        self,
        *,
        context: dict[str, Any],
        history: list[dict[str, Any]],
        fallback_lossless: list[str] | None = None,
    ) -> ResearchPlan:
        fallback = self._fallback_next(context, history)
        return self._propose(
            context=context,
            history=history,
            fallback=fallback,
            phase="next",
        )

    def _propose(
        self,
        *,
        context: dict[str, Any],
        history: list[dict[str, Any]],
        fallback: ResearchPlan,
        phase: str,
    ) -> ResearchPlan:
        if not self.client or not self.client.available:
            return fallback
        user = json.dumps(
            {
                "program_md": self.program,
                "phase": phase,
                "allowed_models": self.allowed_models,
                "context": context,
                "history": history[-6:],
                "fallback_if_uncertain": fallback.to_json(),
            },
            indent=2,
            sort_keys=True,
        )
        try:
            response = self.client.chat(
                system=RESEARCH_SYSTEM_PROMPT,
                user=user,
                max_completion_tokens=8_000,
            )
            data = _extract_json_object(response)
            return ResearchPlan.from_json(
                data,
                default_model=fallback.model,
                allowed_models=self.allowed_models,
                max_candidates=self.max_candidates,
                default_max_llm_bytes=int(context.get("default_max_llm_bytes", fallback.max_llm_bytes)),
            )
        except Exception as exc:  # noqa: BLE001 - failed research should not stop the harness.
            fallback.hypothesis = f"research LLM failed ({exc}); deterministic fallback"
            return fallback

    def _fallback_initial(self, context: dict[str, Any]) -> ResearchPlan:
        return ResearchPlan(
            hypothesis="initial file-level component_v1 run with diagnostic repair and candidate competition enabled",
            model=self.allowed_models[0],
            format_variant="component_v1",
            chunking_strategy="file",
            candidate_count=min(2, self.max_candidates),
            repair_enabled=True,
            max_llm_bytes=int(context.get("default_max_llm_bytes", 20_000)),
        )

    def _fallback_next(
        self,
        context: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> ResearchPlan:
        previous = _plan_from_history(history) or self._fallback_initial(context)
        failed_count = len(history)
        if failed_count == 1:
            return ResearchPlan(
                hypothesis="failure after file-level compression; try smaller chunks, a more literal prompt, and diagnostic repair",
                model=_next_model(previous.model, self.allowed_models),
                compression_prompt_extra=(
                    "Preserve exact public API, import/export names, error messages, branching conditions, "
                    "and every literal. If unsure, include more code-shaped detail instead of prose."
                ),
                decompression_prompt_extra=(
                    "Reconstruct runnable code conservatively. Preserve module boundaries and public behavior. "
                    "Prefer explicit simple code over clever rewrites."
                ),
                format_variant="component_literal_heavy",
                chunking_strategy="line_chunks",
                chunk_size_lines=80,
                candidate_count=min(3, self.max_candidates),
                repair_enabled=True,
                max_llm_bytes=max(
                    int(context.get("default_max_llm_bytes", previous.max_llm_bytes)),
                    previous.max_llm_bytes,
                ),
                temperature=0.2,
            )
        return ResearchPlan(
            hypothesis="verification still fails; use testsafe format, smaller chunks, diagnostic repair, and stronger reconstruction constraints",
            model=_next_model(previous.model, self.allowed_models),
            compression_prompt_extra=(
                "Tests failed before. Compress only when fully recoverable. Include exact edge cases, exceptions, "
                "types, async/sync behavior, environment variables, filesystem effects, and protocol shapes."
            ),
            decompression_prompt_extra=(
                "Use failed verification as a guide if repair runs. Generate minimal equivalent code and do not invent features."
            ),
            format_variant="component_testsafe",
            chunking_strategy="line_chunks",
            chunk_size_lines=60,
            candidate_count=min(3, self.max_candidates),
            repair_enabled=True,
            max_llm_bytes=max(
                int(context.get("default_max_llm_bytes", previous.max_llm_bytes)),
                previous.max_llm_bytes,
            ),
            temperature=0.1,
        )


def load_research_program(path: Path | None) -> str:
    if path and path.exists():
        return path.read_text(encoding="utf-8")
    default_path = Path("program.md")
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")
    return DEFAULT_RESEARCH_PROGRAM


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
            raise LLMError(f"Research response did not contain JSON: {text[:500]}")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise LLMError("Research response JSON was not an object")
    return data


def _plan_from_history(history: list[dict[str, Any]]) -> ResearchPlan | None:
    if not history:
        return None
    plan = history[-1].get("plan")
    if not isinstance(plan, dict):
        return None
    return ResearchPlan.from_json(
        plan,
        default_model=str(plan.get("model") or "openrouter/auto"),
        allowed_models=[str(plan.get("model") or "openrouter/auto")],
        max_candidates=10,
        default_max_llm_bytes=int(plan.get("max_llm_bytes") or 20_000),
    )


def _next_model(current: str, allowed_models: list[str]) -> str:
    if not allowed_models:
        return current
    if current not in allowed_models:
        return allowed_models[0]
    return allowed_models[(allowed_models.index(current) + 1) % len(allowed_models)]


def _short_string(value: Any, default: str, limit: int) -> str:
    if not isinstance(value, str):
        return default
    return value[:limit]


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))
