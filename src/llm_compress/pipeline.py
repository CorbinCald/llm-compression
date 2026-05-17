from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifact import copy_artifact
from .compressor import CompressionOptions, CompressionSummary, Compressor
from .decompressor import DecompressionSummary, Decompressor
from .files import iter_repo_paths, posix_rel
from .openrouter import OpenRouterClient
from .repair import RepairSummary, repair_restored_candidate
from .research import ResearchAgent, ResearchPlan, load_research_program
from .targets import PreparedTarget, prepare_target, update_index
from .verify import VerificationPlan, VerificationReport, discover_verification_plan, run_verification


@dataclass
class RunOptions:
    runs_root: Path = Path(".llm-compress/runs")
    max_iterations: int = 3
    workers: int = 4
    max_llm_bytes: int = 20_000
    verify: bool = True
    install: bool = True
    timeout: int = 180
    setup_timeout: int = 600
    continue_on_baseline_fail: bool = False
    max_candidates: int = 3
    model_candidates: list[str] | None = None
    program_path: Path | None = None
    global_lessons: list[str] = field(default_factory=list)
    initial_research_plan: ResearchPlan | None = None


@dataclass
class CandidateResult:
    index: int
    restore_dir: Path
    decompression: DecompressionSummary
    verification: VerificationReport | None
    repair: RepairSummary | None
    repaired_verification: VerificationReport | None
    success: bool
    score: float

    @property
    def effective_verification(self) -> VerificationReport | None:
        return self.repaired_verification or self.verification

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "restore_dir": str(self.restore_dir),
            "success": self.success,
            "score": self.score,
            "decompression": self.decompression.to_json(),
            "verification": self.verification.to_json() if self.verification else None,
            "repair": self.repair.to_json() if self.repair else None,
            "repaired_verification": self.repaired_verification.to_json()
            if self.repaired_verification
            else None,
        }


@dataclass
class IterationResult:
    index: int
    artifact_path: Path
    restore_dir: Path
    compression: CompressionSummary
    decompression: DecompressionSummary
    verification: VerificationReport | None
    success: bool
    lossless_overrides: list[str]
    plan: ResearchPlan
    candidates: list[CandidateResult]

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "artifact_path": str(self.artifact_path),
            "restore_dir": str(self.restore_dir),
            "success": self.success,
            "lossless_overrides": self.lossless_overrides,
            "plan": self.plan.to_json(),
            "compression": self.compression.to_json(),
            "decompression": self.decompression.to_json(),
            "verification": self.verification.to_json() if self.verification else None,
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }


@dataclass
class PipelineResult:
    target: PreparedTarget
    baseline: VerificationReport | None
    iterations: list[IterationResult]
    success: bool
    final_artifact: Path | None
    final_restored: Path | None
    stopped_reason: str | None = None

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
            "final_artifact": str(self.final_artifact) if self.final_artifact else None,
            "final_restored": str(self.final_restored) if self.final_restored else None,
            "stopped_reason": self.stopped_reason,
            "baseline": self.baseline.to_json() if self.baseline else None,
            "iterations": [iteration.to_json() for iteration in self.iterations],
        }


def run_target(
    target: str,
    *,
    client: OpenRouterClient | None,
    options: RunOptions,
    research_client: OpenRouterClient | None = None,
) -> PipelineResult:
    prepared = prepare_target(target, options.runs_root)
    print(f"Run dir: {prepared.run_dir}", flush=True)
    print(f"Source:  {prepared.source_root}", flush=True)

    plan = discover_verification_plan(prepared.source_root, install=options.install)
    baseline: VerificationReport | None = None
    if options.verify:
        print("Baseline verification", flush=True)
        baseline = run_verification(
            prepared.source_root,
            plan,
            timeout=options.timeout,
            setup_timeout=options.setup_timeout,
            print_progress=True,
        )
        _write_json(prepared.run_dir / "baseline.json", baseline.to_json())
        if not baseline.ok and not options.continue_on_baseline_fail:
            reason = "baseline verification failed; rerun with --continue-on-baseline-fail to force compression"
            result = PipelineResult(
                target=prepared,
                baseline=baseline,
                iterations=[],
                success=False,
                final_artifact=None,
                final_restored=None,
                stopped_reason=reason,
            )
            _write_json(prepared.run_dir / "summary.json", result.to_json())
            print(reason, flush=True)
            return result
    else:
        print("Verification disabled", flush=True)

    model_candidates = _model_candidates(client, options.model_candidates)
    context = _research_context(prepared.source_root, options, model_candidates, plan)
    research_agent = ResearchAgent(
        research_client or client,
        program=load_research_program(options.program_path),
        allowed_models=model_candidates,
        max_candidates=options.max_candidates,
    )
    research_plan = _enforce_research_plan_policy(
        options.initial_research_plan or research_agent.initial_plan(context=context),
        options,
    )
    _write_json(prepared.run_dir / "research-initial-plan.json", research_plan.to_json())

    iterations: list[IterationResult] = []
    history: list[dict[str, Any]] = []

    for index in range(max(1, options.max_iterations)):
        iteration_dir = prepared.run_dir / f"iter-{index}"
        artifact_path = iteration_dir / "compressed.jsonl"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        _write_json(iteration_dir / "research-plan.json", research_plan.to_json())

        iteration_client = _client_for_plan(client, research_plan)
        print(
            f"Iteration {index}: experiment model={research_plan.model} "
            f"format={research_plan.format_variant} chunks={research_plan.chunking_strategy} "
            f"candidates={research_plan.candidate_count} repair={research_plan.repair_enabled}",
            flush=True,
        )
        print(f"Iteration {index}: compress", flush=True)
        compression = Compressor(iteration_client).compress_repo(
            prepared.source_root,
            artifact_path,
            options=CompressionOptions(
                max_llm_bytes=research_plan.max_llm_bytes,
                workers=options.workers,
                use_llm=bool(iteration_client and iteration_client.available),
                chunking_strategy=research_plan.chunking_strategy,
                chunk_size_lines=research_plan.chunk_size_lines,
                format_variant=research_plan.format_variant,
                compression_prompt_extra=research_plan.compression_prompt_extra,
            ),
            target_label=prepared.target,
        )
        print(
            f"  files={compression.file_count} llm={compression.llm_file_count} "
            f"lossless={compression.lossless_file_count} ratio={compression.ratio:.3f}",
            flush=True,
        )

        candidates: list[CandidateResult] = []
        for candidate_index in range(max(1, research_plan.candidate_count)):
            restore_dir = iteration_dir / f"candidate-{candidate_index}"
            print(f"Iteration {index}: decompress candidate {candidate_index + 1}/{research_plan.candidate_count}", flush=True)
            decompression = Decompressor(
                iteration_client,
                prompt_extra=research_plan.decompression_prompt_extra,
                candidate_index=candidate_index,
                candidate_count=max(1, research_plan.candidate_count),
            ).decompress_artifact(
                artifact_path,
                restore_dir,
                workers=options.workers,
                clean=True,
            )
            if decompression.errors:
                print(f"  decompression errors: {len(decompression.errors)}", flush=True)

            verification: VerificationReport | None = None
            if options.verify and not decompression.errors:
                print(f"Iteration {index}: verify candidate {candidate_index + 1}", flush=True)
                verification = run_verification(
                    restore_dir,
                    plan,
                    timeout=options.timeout,
                    setup_timeout=options.setup_timeout,
                    print_progress=True,
                )
            elif not options.verify:
                print(f"Iteration {index}: verification skipped", flush=True)

            repair: RepairSummary | None = None
            repaired_verification: VerificationReport | None = None
            if (
                options.verify
                and research_plan.repair_enabled
                and verification is not None
                and not verification.ok
                and not decompression.errors
            ):
                print(f"Iteration {index}: repair candidate {candidate_index + 1}", flush=True)
                repair = repair_restored_candidate(
                    client=iteration_client,
                    artifact_path=artifact_path,
                    restore_dir=restore_dir,
                    verification=verification,
                    allowed_paths=compression.llm_paths,
                    prompt_extra=research_plan.decompression_prompt_extra,
                )
                if repair.repaired_paths:
                    print(f"  repaired {len(repair.repaired_paths)} file(s); reverifying", flush=True)
                    repaired_verification = run_verification(
                        restore_dir,
                        plan,
                        timeout=options.timeout,
                        setup_timeout=options.setup_timeout,
                        print_progress=True,
                    )

            effective_verification = repaired_verification or verification
            success = _candidate_success(
                baseline=baseline,
                verification=effective_verification,
                decompression=decompression,
                verify=options.verify,
            )
            score = _candidate_score(
                verification=effective_verification,
                decompression=decompression,
                verify=options.verify,
            )
            candidates.append(
                CandidateResult(
                    index=candidate_index,
                    restore_dir=restore_dir,
                    decompression=decompression,
                    verification=verification,
                    repair=repair,
                    repaired_verification=repaired_verification,
                    success=success,
                    score=score,
                )
            )

        best = max(candidates, key=lambda candidate: candidate.score)
        iteration = IterationResult(
            index=index,
            artifact_path=artifact_path,
            restore_dir=best.restore_dir,
            compression=compression,
            decompression=best.decompression,
            verification=best.effective_verification,
            success=best.success,
            lossless_overrides=sorted(research_plan.lossless_overrides),
            plan=research_plan,
            candidates=candidates,
        )
        iterations.append(iteration)
        _write_json(iteration_dir / "summary.json", iteration.to_json())

        if best.success:
            final_dir = prepared.run_dir / "final"
            final_dir.mkdir(exist_ok=True)
            final_artifact = final_dir / "compressed.jsonl"
            final_restored = final_dir / "restored"
            copy_artifact(artifact_path, final_artifact)
            if final_restored.exists():
                shutil.rmtree(final_restored)
            shutil.copytree(best.restore_dir, final_restored, symlinks=True)
            update_index(prepared.target, prepared.run_dir, final_artifact)
            result = PipelineResult(
                target=prepared,
                baseline=baseline,
                iterations=iterations,
                success=True,
                final_artifact=final_artifact,
                final_restored=final_restored,
            )
            _write_json(prepared.run_dir / "summary.json", result.to_json())
            print(f"Success: {final_artifact}", flush=True)
            return result

        history.append(_history_entry(iteration))
        if index + 1 >= max(1, options.max_iterations):
            break

        print("Iteration {index}: autoresearch LLM proposing next experiment".format(index=index), flush=True)
        research_plan = _enforce_research_plan_policy(
            research_agent.next_plan(
                context=context,
                history=history,
            ),
            options,
        )
        _write_json(prepared.run_dir / f"research-plan-after-iter-{index}.json", research_plan.to_json())
        print(f"  autoresearch: {research_plan.hypothesis}", flush=True)

    final_artifact = iterations[-1].artifact_path if iterations else None
    final_restored = iterations[-1].restore_dir if iterations else None
    if final_artifact:
        update_index(prepared.target, prepared.run_dir, final_artifact)
    result = PipelineResult(
        target=prepared,
        baseline=baseline,
        iterations=iterations,
        success=False,
        final_artifact=final_artifact,
        final_restored=final_restored,
        stopped_reason="max iterations reached without a verified candidate",
    )
    _write_json(prepared.run_dir / "summary.json", result.to_json())
    return result


def _candidate_success(
    *,
    baseline: VerificationReport | None,
    verification: VerificationReport | None,
    decompression: DecompressionSummary,
    verify: bool,
) -> bool:
    if decompression.errors:
        return False
    if not verify:
        return True
    if verification is None:
        return False
    if baseline is not None and not baseline.ok:
        return verification.check_pass_count >= baseline.check_pass_count and verification.ok
    return verification.ok


def _candidate_score(
    *,
    verification: VerificationReport | None,
    decompression: DecompressionSummary,
    verify: bool,
) -> float:
    score = 0.0
    if not decompression.errors:
        score += 1.0
    score += min(1.0, decompression.hash_matches / max(1, decompression.file_count)) * 0.1
    score -= len(decompression.errors) * 0.25
    if not verify:
        return score
    if verification is None:
        return score - 1.0
    if verification.check_count:
        score += verification.check_pass_count / verification.check_count
    if verification.ok:
        score += 2.0
    return score


def _history_entry(iteration: IterationResult) -> dict[str, Any]:
    verification = iteration.verification
    failed_output = verification.all_output()[-12_000:] if verification else ""
    return {
        "iteration": iteration.index,
        "plan": iteration.plan.to_json(),
        "success": iteration.success,
        "compression": iteration.compression.to_json(),
        "best_candidate": {
            "score": max((candidate.score for candidate in iteration.candidates), default=0.0),
            "decompression_errors": iteration.decompression.errors,
            "verification_ok": verification.ok if verification else None,
            "check_pass_count": verification.check_pass_count if verification else 0,
            "check_count": verification.check_count if verification else 0,
            "failed_output_tail": failed_output,
        },
        "candidate_count": len(iteration.candidates),
    }


def _research_context(
    source_root: Path,
    options: RunOptions,
    model_candidates: list[str],
    plan: VerificationPlan,
) -> dict[str, Any]:
    files = list(iter_repo_paths(source_root))
    total_bytes = 0
    extensions: dict[str, int] = {}
    for path in files:
        if path.is_symlink():
            continue
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
        suffix = path.suffix.lower() or "<none>"
        extensions[suffix] = extensions.get(suffix, 0) + 1
    return {
        "source_root": str(source_root),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "top_extensions": dict(sorted(extensions.items(), key=lambda item: item[1], reverse=True)[:12]),
        "sample_paths": [posix_rel(path, source_root) for path in files[:40]],
        "verification_plan": plan.to_json(),
        "default_max_llm_bytes": options.max_llm_bytes,
        "max_iterations": options.max_iterations,
        "max_candidates": options.max_candidates,
        "allowed_models": model_candidates,
        "global_lessons": options.global_lessons,
        "global_seed_plan": options.initial_research_plan.to_json() if options.initial_research_plan else None,
    }


def _model_candidates(client: OpenRouterClient | None, configured: list[str] | None) -> list[str]:
    candidates = [item.strip() for item in (configured or []) if item.strip()]
    if client and client.model not in candidates:
        candidates.insert(0, client.model)
    if not candidates:
        candidates = ["openrouter/auto"]
    return candidates


def _client_for_plan(client: OpenRouterClient | None, plan: ResearchPlan) -> OpenRouterClient | None:
    if client is None:
        return None
    return client.with_overrides(model=plan.model, temperature=plan.temperature)


def _enforce_research_plan_policy(plan: ResearchPlan, options: RunOptions) -> ResearchPlan:
    # Autoresearch may not opt individual source files out of LLM compression.
    # Deterministic lossless handling for binaries/config/tests/oversized files still
    # happens inside file eligibility, but path-specific source overrides are banned.
    plan.lossless_overrides = []
    plan.max_llm_bytes = max(options.max_llm_bytes, plan.max_llm_bytes)
    return plan


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
