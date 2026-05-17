from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .openrouter import LLMError, OpenRouterClient
from .research import ResearchPlan

GLOBAL_RESEARCH_SYSTEM_PROMPT = """You run global meta-research for llm-compress across a seven-repo benchmark suite.

You output exactly one JSON object that updates the global strategy for future repos.

Seed plans may include a temporary code_patch unified diff against the llm-compression tool repository. The context includes a directory tree and selected source files from this repository. The harness applies the patch to an isolated copy for an experiment, uses it for compression/decompression, then discards it. Use code patches when global evidence suggests the tool implementation itself should change, not to bypass verification.

Goal:
- Learn lessons from every completed repo and apply them to later repos.
- Primary metric: minimize raw-decompression failure_units before repair. failure_units use granular failed test counts when verification output exposes them, and fallback command-level units for failed setup/lint/build/test commands otherwise.
- Secondary metric: lower artifact_bytes/original_bytes is better.
- Prefer lessons that generalize across languages/repos; source-file lossless overrides are banned.
- Lessons must be language-level or tool-level, not repo-specific. Do not store exact repo names, package names, file paths, or file names in lessons.
- Translate specific evidence into general patterns. Good: "Go files must preserve package declarations." Bad: "mux.go must start with package mux."

You may update the global seed experiment plan for the next repo. This seed plan can tune:
model, compression/decompression prompt addenda, format variant, chunking strategy, chunk size,
candidate count, diagnostic repair enablement, max LLM bytes, temperature, and optional temporary code_patch.
Repair is diagnostic only: repaired candidates are not accepted as benchmark success. If repair diagnostics
show changed paths or a repaired verification pass, treat that only as evidence of what the next raw
compression/decompression must encode better. max LLM bytes may be raised above the user default, but
not lowered to avoid hard files.

Return only JSON with this schema:
{
  "reason": "short explanation of what changed globally and why",
  "lessons": ["global lesson 1", "global lesson 2"],
  "seed_plan": {
    "hypothesis": "short reason for this global seed plan",
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
}
"""


@dataclass
class GlobalResearchState:
    reason: str = "initial global strategy"
    lessons: list[str] = field(default_factory=list)
    seed_plan: ResearchPlan = field(default_factory=ResearchPlan)

    def to_json(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "lessons": self.lessons,
            "seed_plan": self.seed_plan.to_json(),
        }


class GlobalResearchAgent:
    def __init__(
        self,
        client: OpenRouterClient | None,
        *,
        allowed_models: list[str],
        max_candidates: int,
    ):
        self.client = client
        self.allowed_models = allowed_models or ([client.model] if client else ["openrouter/auto"])
        self.max_candidates = max(1, max_candidates)

    def initial_state(self, *, suite_context: dict[str, Any], default_max_llm_bytes: int) -> GlobalResearchState:
        fallback = GlobalResearchState(
            reason="initial global benchmark strategy",
            lessons=[
                "Start with diagnostic repair and candidate competition enabled so failed decompressions produce useful next-plan feedback without counting repaired trees as success.",
                "Do not use path-specific source-file lossless overrides; tune prompts, chunking, models, candidates, repair, and thresholds instead.",
            ],
            seed_plan=ResearchPlan(
                hypothesis="global initial seed: file-level component_v1 with diagnostic repair and candidate competition",
                model=self.allowed_models[0],
                format_variant="component_v1",
                chunking_strategy="file",
                candidate_count=min(2, self.max_candidates),
                repair_enabled=True,
                max_llm_bytes=default_max_llm_bytes,
            ),
        )
        return self._propose(
            phase="initial_global_strategy",
            suite_context=suite_context,
            history=[],
            previous_state=None,
            fallback=fallback,
            default_max_llm_bytes=default_max_llm_bytes,
        )

    def resume_state(
        self,
        *,
        suite_context: dict[str, Any],
        history: list[dict[str, Any]],
        previous_state: GlobalResearchState,
        default_max_llm_bytes: int,
    ) -> GlobalResearchState:
        return self._propose(
            phase="resume_global_strategy",
            suite_context=suite_context,
            history=history,
            previous_state=previous_state,
            fallback=previous_state,
            default_max_llm_bytes=default_max_llm_bytes,
        )

    def update_state(
        self,
        *,
        suite_context: dict[str, Any],
        history: list[dict[str, Any]],
        previous_state: GlobalResearchState,
        default_max_llm_bytes: int,
    ) -> GlobalResearchState:
        fallback = self._fallback_update(previous_state, history, default_max_llm_bytes)
        return self._propose(
            phase="update_after_repo",
            suite_context=suite_context,
            history=history,
            previous_state=previous_state,
            fallback=fallback,
            default_max_llm_bytes=default_max_llm_bytes,
        )

    def _propose(
        self,
        *,
        phase: str,
        suite_context: dict[str, Any],
        history: list[dict[str, Any]],
        previous_state: GlobalResearchState | None,
        fallback: GlobalResearchState,
        default_max_llm_bytes: int,
    ) -> GlobalResearchState:
        if not self.client or not self.client.available:
            return fallback
        payload = {
            "phase": phase,
            "allowed_models": self.allowed_models,
            "suite_context": suite_context,
            "previous_global_state": previous_state.to_json() if previous_state else None,
            "completed_repo_history": _history_for_prompt(history[-10:]),
            "fallback_if_uncertain": fallback.to_json(),
        }
        try:
            response = self.client.chat(
                system=GLOBAL_RESEARCH_SYSTEM_PROMPT,
                user=json.dumps(payload, indent=2, sort_keys=True),
                max_completion_tokens=8_000,
            )
            data = _extract_json_object(response)
            return _state_from_json(
                data,
                default_model=fallback.seed_plan.model,
                allowed_models=self.allowed_models,
                max_candidates=self.max_candidates,
                default_max_llm_bytes=default_max_llm_bytes,
            )
        except Exception as exc:  # noqa: BLE001 - global research failure should not stop benchmarks.
            fallback.reason = f"global research LLM failed ({exc}); deterministic fallback"
            return fallback

    def _fallback_update(
        self,
        previous_state: GlobalResearchState,
        history: list[dict[str, Any]],
        default_max_llm_bytes: int,
    ) -> GlobalResearchState:
        if not history:
            return previous_state
        latest = history[-1]
        previous_plan = _latest_plan(history) or previous_state.seed_plan
        lessons = list(previous_state.lessons)
        repo = latest.get("repo", {})
        language = str(repo.get("language") or "benchmark") if isinstance(repo, dict) else "benchmark"
        if latest.get("success"):
            ratio = latest.get("ratio")
            lessons.append(
                f"A recent {language} run passed raw verification with {previous_plan.format_variant}/{previous_plan.chunking_strategy}; ratio={ratio}. Prefer this only while later runs improve."
            )
            seed = ResearchPlan.from_json(
                previous_plan.to_json(),
                default_model=previous_plan.model,
                allowed_models=self.allowed_models,
                max_candidates=self.max_candidates,
                default_max_llm_bytes=default_max_llm_bytes,
            )
            seed.lossless_overrides = []
            seed.hypothesis = "carry forward recent successful global strategy"
        else:
            lessons.append(
                f"A recent {language} run failed raw verification; strengthen generally recoverable component detail before later runs."
            )
            seed = ResearchPlan(
                hypothesis="after recent failure: use testsafe/literal-heavy chunked compression with diagnostic repair",
                model=_next_model(previous_plan.model, self.allowed_models),
                compression_prompt_extra=(
                    previous_plan.compression_prompt_extra + "\nPreserve exact public APIs, literals, exceptions, import/export names, "
                    "edge cases, side effects, and protocol shapes. If uncertain, include more code-shaped detail."
                ).strip()[-3000:],
                decompression_prompt_extra=(
                    previous_plan.decompression_prompt_extra + "\nUse verification failures as constraints. Reconstruct minimal conservative code; "
                    "do not invent behavior beyond the compressed components."
                ).strip()[-3000:],
                format_variant="component_testsafe",
                chunking_strategy="line_chunks",
                chunk_size_lines=80,
                candidate_count=min(max(previous_plan.candidate_count, 2), self.max_candidates),
                repair_enabled=True,
                lossless_overrides=[],
                max_llm_bytes=max(default_max_llm_bytes, previous_plan.max_llm_bytes),
                temperature=0.1,
            )
        return GlobalResearchState(
            reason="deterministic global fallback update",
            lessons=_dedupe_tail(lessons, 12),
            seed_plan=seed,
        )


def _history_for_prompt(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in history:
        cloned = dict(entry)
        plan = cloned.get("final_plan")
        if isinstance(plan, dict) and len(str(plan.get("code_patch", ""))) > 12_000:
            plan = dict(plan)
            plan["code_patch"] = str(plan["code_patch"])[:12_000] + "\n... <truncated in global prompt>"
            cloned["final_plan"] = plan
        result.append(cloned)
    return result


def _state_from_json(
    data: dict[str, Any],
    *,
    default_model: str,
    allowed_models: list[str],
    max_candidates: int,
    default_max_llm_bytes: int,
) -> GlobalResearchState:
    raw_lessons = data.get("lessons", [])
    lessons: list[str] = []
    if isinstance(raw_lessons, list):
        for item in raw_lessons:
            lesson = _generic_global_lesson(item)
            if lesson:
                lessons.append(lesson)
    seed_data = data.get("seed_plan", {})
    if not isinstance(seed_data, dict):
        seed_data = {}
    seed = ResearchPlan.from_json(
        seed_data,
        default_model=default_model,
        allowed_models=allowed_models,
        max_candidates=max_candidates,
        default_max_llm_bytes=default_max_llm_bytes,
    )
    # Source-file lossless overrides are disabled globally and per repo.
    seed.lossless_overrides = []
    seed.max_llm_bytes = max(default_max_llm_bytes, seed.max_llm_bytes)
    return GlobalResearchState(
        reason=str(data.get("reason") or "global LLM update")[:1000],
        lessons=_dedupe_tail(lessons, 20),
        seed_plan=seed,
    )


def _generic_global_lesson(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lesson = " ".join(value.strip().split())[:500]
    if not lesson:
        return None
    lowered = lesson.lower()
    repo_terms = {"nanoid", "itsdangerous", "mux", "itoa", "click", "express", "lodash"}
    if any(re.search(rf"(?<![a-z0-9_-]){re.escape(term)}(?![a-z0-9_-])", lowered) for term in repo_terms):
        return None
    if re.search(r"(?:^|[\s'\"])(?:\.?/?[\w.-]+/)+[\w.-]+", lesson):
        return None
    if re.search(r"\b[\w.-]+\.(?:py|js|jsx|ts|tsx|go|rs|c|h|css|html|json|toml|yaml|yml|md|sh)\b", lowered):
        return None
    return lesson


def load_global_research_file(
    path: Path,
    *,
    allowed_models: list[str],
    max_candidates: int,
    default_max_llm_bytes: int,
) -> tuple[GlobalResearchState, list[dict[str, Any]]] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    state_data = data.get("state")
    if not isinstance(state_data, dict):
        return None
    state = _state_from_json(
        state_data,
        default_model=(allowed_models[0] if allowed_models else "openrouter/auto"),
        allowed_models=allowed_models,
        max_candidates=max_candidates,
        default_max_llm_bytes=default_max_llm_bytes,
    )

    history_data = data.get("history", [])
    history = [item for item in history_data if isinstance(item, dict)] if isinstance(history_data, list) else []
    return state, history[-50:]


def _latest_plan(history: list[dict[str, Any]]) -> ResearchPlan | None:
    for entry in reversed(history):
        plan_data = entry.get("final_plan")
        if isinstance(plan_data, dict):
            return ResearchPlan.from_json(
                plan_data,
                default_model=str(plan_data.get("model") or "openrouter/auto"),
                allowed_models=[str(plan_data.get("model") or "openrouter/auto")],
                max_candidates=10,
                default_max_llm_bytes=int(plan_data.get("max_llm_bytes") or 20_000),
            )
    return None


def _next_model(current: str, allowed_models: list[str]) -> str:
    if not allowed_models:
        return current
    if current not in allowed_models:
        return allowed_models[0]
    return allowed_models[(allowed_models.index(current) + 1) % len(allowed_models)]


def _dedupe_tail(items: list[str], limit: int) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result[-limit:]


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
            raise LLMError(f"Global research response did not contain JSON: {text[:500]}")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise LLMError("Global research response JSON was not an object")
    return data
