from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifact import copy_artifact, write_artifact
from .compressor import CompressionOptions, CompressionSummary, Compressor
from .console import detail, details, fail, ok, section, step, subsection, warn, yes_no
from .decompressor import DecompressionSummary, Decompressor
from .files import iter_repo_paths, posix_rel
from .openrouter import OpenRouterClient
from .repair import RepairSummary, repair_restored_candidate
from .reporting import (
    log_candidate_outcome,
    log_compression_summary,
    log_decompression_summary,
    log_research_plan,
    log_verification_plan,
    log_verification_report,
)
from .research import ResearchAgent, ResearchPlan, load_research_program
from .targets import PreparedTarget, prepare_target, update_index
from .tool_patch import apply_code_patch, build_tool_repo_context, copy_tool_repo
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
    iteration_count = max(1, options.max_iterations)
    section("Autoresearch run")
    details(
        (
            ("target", prepared.target),
            ("source", prepared.source_root),
            ("run dir", prepared.run_dir),
            ("iterations", iteration_count),
            ("workers", options.workers),
            ("verification", yes_no(options.verify)),
            ("install deps", yes_no(options.install)),
            ("LLM available", yes_no(bool(client and client.available))),
        )
    )

    plan = discover_verification_plan(prepared.source_root, install=options.install)
    section("Verification plan")
    log_verification_plan(plan)

    baseline: VerificationReport | None = None
    if options.verify:
        section("Baseline verification")
        baseline = run_verification(
            prepared.source_root,
            plan,
            timeout=options.timeout,
            setup_timeout=options.setup_timeout,
            print_progress=True,
        )
        log_verification_report(baseline, label="Baseline verification")
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
            fail(reason)
            return result
    else:
        warn("Verification disabled; candidates will be accepted if decompression succeeds.")

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
    section("Initial experiment")
    log_research_plan(research_plan, title="hypothesis")

    iterations: list[IterationResult] = []
    history: list[dict[str, Any]] = []

    for index in range(iteration_count):
        iteration_dir = prepared.run_dir / f"iter-{index}"
        artifact_path = iteration_dir / "compressed.jsonl"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        _write_json(iteration_dir / "research-plan.json", research_plan.to_json())

        iteration_client = _client_for_plan(client, research_plan)
        section(f"Iteration {index + 1}/{iteration_count}")
        log_research_plan(research_plan, title="trying")
        if research_plan.code_patch.strip():
            step("Applying temporary llm-compression code patch in an isolated copy")
            compression, decompressed_candidates = _run_iteration_with_code_patch(
                prepared=prepared,
                iteration_dir=iteration_dir,
                artifact_path=artifact_path,
                research_plan=research_plan,
                options=options,
            )
            log_compression_summary(compression)
            candidates: list[CandidateResult] = []
            candidate_count = len(decompressed_candidates)
            for candidate_index, restore_dir, decompression in decompressed_candidates:
                subsection(f"Candidate {candidate_index + 1}/{candidate_count}")
                candidates.append(
                    _evaluate_candidate(
                        candidate_index=candidate_index,
                        restore_dir=restore_dir,
                        decompression=decompression,
                        compression=compression,
                        artifact_path=artifact_path,
                        verification_plan=plan,
                        run_options=options,
                        research_plan=research_plan,
                        iteration_client=iteration_client,
                        baseline=baseline,
                    )
                )
        else:
            step("Compressing source files with this plan")
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
            log_compression_summary(compression)

            candidates = []
            candidate_count = max(1, research_plan.candidate_count)
            for candidate_index in range(candidate_count):
                restore_dir = iteration_dir / f"candidate-{candidate_index}"
                subsection(f"Candidate {candidate_index + 1}/{candidate_count}")
                step("Decompressing artifact")
                decompression = Decompressor(
                    iteration_client,
                    prompt_extra=research_plan.decompression_prompt_extra,
                    candidate_index=candidate_index,
                    candidate_count=candidate_count,
                ).decompress_artifact(
                    artifact_path,
                    restore_dir,
                    workers=options.workers,
                    clean=True,
                )
                candidates.append(
                    _evaluate_candidate(
                        candidate_index=candidate_index,
                        restore_dir=restore_dir,
                        decompression=decompression,
                        compression=compression,
                        artifact_path=artifact_path,
                        verification_plan=plan,
                        run_options=options,
                        research_plan=research_plan,
                        iteration_client=iteration_client,
                        baseline=baseline,
                    )
                )

        best = max(candidates, key=lambda candidate: candidate.score)
        if best.success:
            ok(f"Iteration {index + 1} succeeded with candidate {best.index + 1}.")
        else:
            fail(f"Iteration {index + 1} did not produce a verified candidate; best was candidate {best.index + 1}.")
        detail("best score", f"{best.score:.3f}")
        iteration = IterationResult(
            index=index,
            artifact_path=artifact_path,
            restore_dir=best.restore_dir,
            compression=compression,
            decompression=best.decompression,
            verification=best.verification,
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
            section("Run result")
            ok("Autoresearch succeeded.")
            details(
                (
                    ("final artifact", final_artifact),
                    ("final restored", final_restored),
                    ("winning iter", index + 1),
                    ("winning candidate", best.index + 1),
                    ("run dir", prepared.run_dir),
                )
            )
            return result

        history.append(_history_entry(iteration))
        if index + 1 >= iteration_count:
            break

        section("Autoresearch update")
        step("Asking the research controller for the next experiment")
        research_plan = _enforce_research_plan_policy(
            research_agent.next_plan(
                context=context,
                history=history,
            ),
            options,
        )
        _write_json(prepared.run_dir / f"research-plan-after-iter-{index}.json", research_plan.to_json())
        log_research_plan(research_plan, title="next hypothesis")

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
    section("Run result")
    fail("Autoresearch stopped without a verified candidate.")
    details(
        (
            ("reason", result.stopped_reason),
            ("final artifact", final_artifact),
            ("final restored", final_restored),
            ("run dir", prepared.run_dir),
        )
    )
    return result


def _run_iteration_with_code_patch(
    *,
    prepared: PreparedTarget,
    iteration_dir: Path,
    artifact_path: Path,
    research_plan: ResearchPlan,
    options: RunOptions,
) -> tuple[CompressionSummary, list[tuple[int, Path, DecompressionSummary]]]:
    candidate_count = max(1, research_plan.candidate_count)
    runner_log = iteration_dir / "code-patch-runner.log"
    try:
        patched_tool_dir = copy_tool_repo(iteration_dir / "patched-llm-compression")
        apply_code_patch(patched_tool_dir, research_plan.code_patch)
        detail("patched tool", patched_tool_dir)

        config_path = (iteration_dir / "code-patch-runner-input.json").resolve()
        output_json = (iteration_dir / "code-patch-runner-output.json").resolve()
        _write_json(
            config_path,
            {
                "source_root": str(prepared.source_root.resolve()),
                "artifact_path": str(artifact_path.resolve()),
                "iteration_dir": str(iteration_dir.resolve()),
                "target_label": prepared.target,
                "workers": options.workers,
                "research_plan": research_plan.to_json(),
                "output_json": str(output_json),
            },
        )
        env = os.environ.copy()
        patched_src = str(patched_tool_dir / "src")
        env["PYTHONPATH"] = patched_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        timeout = max(3600, options.setup_timeout + options.timeout * candidate_count)
        completed = subprocess.run(
            [sys.executable, "-m", "llm_compress.iteration_runner", str(config_path)],
            cwd=patched_tool_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        runner_log.write_text(completed.stdout or "", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"patched iteration runner failed with exit {completed.returncode}; see {runner_log}")
        data = json.loads(output_json.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError("patched iteration runner output was not a JSON object")
        raw_compression = data.get("compression", {})
        compression = _compression_from_json(raw_compression if isinstance(raw_compression, dict) else {})
        raw_decompressions = data.get("decompressions", [])
        if not isinstance(raw_decompressions, list):
            raw_decompressions = []
        decompressed_candidates: list[tuple[int, Path, DecompressionSummary]] = []
        for item in raw_decompressions:
            if not isinstance(item, dict):
                continue
            candidate_index = int(item.get("index", len(decompressed_candidates)))
            restore_dir = Path(str(item.get("restore_dir") or iteration_dir / f"candidate-{candidate_index}"))
            raw_decompression = item.get("decompression", {})
            decompressed_candidates.append(
                (
                    candidate_index,
                    restore_dir,
                    _decompression_from_json(raw_decompression if isinstance(raw_decompression, dict) else {}),
                )
            )
        if not decompressed_candidates:
            raise RuntimeError("patched iteration runner returned no decompression candidates")
        return compression, decompressed_candidates
    except Exception as exc:  # noqa: BLE001 - failed code patches are research feedback.
        fail(f"Temporary code patch failed: {exc}")
        if runner_log.exists():
            detail("runner log", runner_log)
        return _failed_code_patch_iteration(prepared, iteration_dir, artifact_path, research_plan, str(exc))


def _failed_code_patch_iteration(
    prepared: PreparedTarget,
    iteration_dir: Path,
    artifact_path: Path,
    research_plan: ResearchPlan,
    error: str,
) -> tuple[CompressionSummary, list[tuple[int, Path, DecompressionSummary]]]:
    write_artifact(
        artifact_path,
        {
            "target": prepared.target,
            "strategy": "failed-temporary-code-patch",
            "format_variant": research_plan.format_variant,
            "chunking_strategy": research_plan.chunking_strategy,
        },
        [],
    )
    restore_dir = iteration_dir / "candidate-0"
    restore_dir.mkdir(parents=True, exist_ok=True)
    compression = CompressionSummary(
        artifact_path=artifact_path,
        original_bytes=_source_byte_count(prepared.source_root),
        artifact_bytes=artifact_path.stat().st_size,
        file_count=0,
        llm_file_count=0,
        lossless_file_count=0,
        llm_paths=[],
        fallback_paths=[],
        errors={"__code_patch__": error},
        chunking_strategy=research_plan.chunking_strategy,
        format_variant=research_plan.format_variant,
    )
    decompression = DecompressionSummary(
        artifact_path=artifact_path,
        output_dir=restore_dir,
        file_count=0,
        llm_file_count=0,
        lossless_file_count=0,
        hash_matches=0,
        hash_mismatches=[],
        errors={"__code_patch__": error},
    )
    return compression, [(0, restore_dir, decompression)]


def _compression_from_json(data: dict[str, Any]) -> CompressionSummary:
    return CompressionSummary(
        artifact_path=Path(str(data.get("artifact_path") or "compressed.jsonl")),
        original_bytes=int(data.get("original_bytes") or 0),
        artifact_bytes=int(data.get("artifact_bytes") or 0),
        file_count=int(data.get("file_count") or 0),
        llm_file_count=int(data.get("llm_file_count") or 0),
        lossless_file_count=int(data.get("lossless_file_count") or 0),
        llm_paths=[str(item) for item in data.get("llm_paths", []) if isinstance(item, str)],
        fallback_paths=[str(item) for item in data.get("fallback_paths", []) if isinstance(item, str)],
        errors=_string_dict(data.get("errors", {})),
        chunking_strategy=str(data.get("chunking_strategy") or "file"),
        format_variant=str(data.get("format_variant") or "component_v1"),
    )


def _decompression_from_json(data: dict[str, Any]) -> DecompressionSummary:
    return DecompressionSummary(
        artifact_path=Path(str(data.get("artifact_path") or "compressed.jsonl")),
        output_dir=Path(str(data.get("output_dir") or "decompressed")),
        file_count=int(data.get("file_count") or 0),
        llm_file_count=int(data.get("llm_file_count") or 0),
        lossless_file_count=int(data.get("lossless_file_count") or 0),
        hash_matches=int(data.get("hash_matches") or 0),
        hash_mismatches=[str(item) for item in data.get("hash_mismatches", []) if isinstance(item, str)],
        errors=_string_dict(data.get("errors", {})),
    )


def _source_byte_count(root: Path) -> int:
    total = 0
    for path in iter_repo_paths(root):
        if path.is_symlink():
            continue
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _evaluate_candidate(
    *,
    candidate_index: int,
    restore_dir: Path,
    decompression: DecompressionSummary,
    compression: CompressionSummary,
    artifact_path: Path,
    verification_plan: VerificationPlan,
    run_options: RunOptions,
    research_plan: ResearchPlan,
    iteration_client: OpenRouterClient | None,
    baseline: VerificationReport | None,
) -> CandidateResult:
    log_decompression_summary(decompression)

    verification: VerificationReport | None = None
    if run_options.verify and not decompression.errors:
        step("Running candidate verification")
        verification = run_verification(
            restore_dir,
            verification_plan,
            timeout=run_options.timeout,
            setup_timeout=run_options.setup_timeout,
            print_progress=True,
        )
        log_verification_report(verification, label=f"Candidate {candidate_index + 1} verification")
    elif not run_options.verify:
        warn("Verification skipped for this candidate.")
    elif decompression.errors:
        warn("Verification skipped because decompression failed.")

    repair: RepairSummary | None = None
    repaired_verification: VerificationReport | None = None
    if (
        run_options.verify
        and research_plan.repair_enabled
        and verification is not None
        and not verification.ok
        and not decompression.errors
    ):
        step("Verification failed; attempting diagnostic LLM repair")
        repair_dir = restore_dir
        if iteration_client and iteration_client.available:
            repair_dir = restore_dir.parent / f"{restore_dir.name}-repair"
            if repair_dir.exists():
                shutil.rmtree(repair_dir)
            shutil.copytree(restore_dir, repair_dir, symlinks=True)
            detail("diagnostic repair dir", repair_dir)
        repair = repair_restored_candidate(
            client=iteration_client,
            artifact_path=artifact_path,
            restore_dir=repair_dir,
            verification=verification,
            allowed_paths=compression.llm_paths,
            prompt_extra=research_plan.decompression_prompt_extra,
        )
        if repair.repaired_paths:
            ok(f"Repair changed {len(repair.repaired_paths)} file(s): {', '.join(repair.repaired_paths[:4])}")
            step("Re-running verification after repair")
            repaired_verification = run_verification(
                repair_dir,
                verification_plan,
                timeout=run_options.timeout,
                setup_timeout=run_options.setup_timeout,
                print_progress=True,
            )
            log_verification_report(
                repaired_verification,
                label=f"Candidate {candidate_index + 1} repaired verification",
            )
            if repaired_verification.ok:
                warn(
                    "Diagnostic repair passed, but this candidate still counts as failed; "
                    "autoresearch must find a fresh unrepaired decompression that passes."
                )
        elif repair.error:
            fail(f"Repair failed: {repair.error}")
        elif repair.attempted:
            warn(f"Repair made no changes. {repair.notes}".strip())
        else:
            warn(f"Repair skipped. {repair.notes}".strip())

    success = _candidate_success(
        baseline=baseline,
        verification=verification,
        decompression=decompression,
        verify=run_options.verify,
    )
    score = _candidate_score(
        verification=verification,
        decompression=decompression,
        verify=run_options.verify,
    )
    candidate_result = CandidateResult(
        index=candidate_index,
        restore_dir=restore_dir,
        decompression=decompression,
        verification=verification,
        repair=repair,
        repaired_verification=repaired_verification,
        success=success,
        score=score,
    )
    log_candidate_outcome(
        index=candidate_index,
        success=success,
        score=score,
        verification=verification,
        decompression=decompression,
    )
    return candidate_result


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
    score -= verification.failure_units
    if verification.ok:
        score += 2.0
    return score


def _history_entry(iteration: IterationResult) -> dict[str, Any]:
    verification = iteration.verification
    failed_output = verification.all_output()[-12_000:] if verification else ""
    plan_json = iteration.plan.to_json()
    if len(str(plan_json.get("code_patch", ""))) > 12_000:
        plan_json["code_patch"] = str(plan_json["code_patch"])[:12_000] + "\n... <truncated in history>"
    repair_diagnostics = _repair_diagnostics_for_history(iteration.candidates)
    return {
        "iteration": iteration.index,
        "plan": plan_json,
        "success": iteration.success,
        "compression": iteration.compression.to_json(),
        "best_candidate": {
            "score": max((candidate.score for candidate in iteration.candidates), default=0.0),
            "decompression_errors": iteration.decompression.errors,
            "verification_ok": verification.ok if verification else None,
            "check_pass_count": verification.check_pass_count if verification else 0,
            "check_count": verification.check_count if verification else 0,
            "failed_test_count": verification.failed_test_count if verification else 0,
            "total_test_count": verification.total_test_count if verification else None,
            "failure_units": verification.failure_units if verification else 0,
            "failed_output_tail": failed_output,
        },
        "candidate_count": len(iteration.candidates),
        "repair_diagnostics": repair_diagnostics,
    }


def _repair_diagnostics_for_history(candidates: list[CandidateResult]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.repair is None and candidate.repaired_verification is None:
            continue
        repaired = candidate.repaired_verification
        diagnostics.append(
            {
                "candidate": candidate.index,
                "repair": candidate.repair.to_json() if candidate.repair else None,
                "repaired_verification_ok": repaired.ok if repaired else None,
                "repaired_check_pass_count": repaired.check_pass_count if repaired else 0,
                "repaired_check_count": repaired.check_count if repaired else 0,
                "repaired_failed_test_count": repaired.failed_test_count if repaired else 0,
                "repaired_total_test_count": repaired.total_test_count if repaired else None,
                "repaired_failure_units": repaired.failure_units if repaired else 0,
                "repaired_failed_output_tail": repaired.all_output()[-4_000:] if repaired and not repaired.ok else "",
                "accepted": candidate.success,
                "note": "repair is diagnostic only; accepted requires raw decompression verification",
            }
        )
    return diagnostics


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
        "llm_compression_tool_repo": build_tool_repo_context(),
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
