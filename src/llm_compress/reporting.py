from __future__ import annotations

from typing import Any

from .console import (
    detail,
    details,
    fail,
    format_bytes,
    format_duration,
    format_ratio,
    info,
    ok,
    one_line,
    preview_list,
    print_error_map,
    print_output_tail,
    warn,
    yes_no,
)


def log_compression_summary(summary: Any) -> None:
    if summary.errors:
        warn(f"Compression finished with {len(summary.errors)} error(s); affected files may have used lossless fallback.")
    else:
        ok("Compression completed.")
    details(
        (
            ("artifact", summary.artifact_path),
            ("original", format_bytes(summary.original_bytes)),
            ("artifact size", format_bytes(summary.artifact_bytes)),
            ("ratio", format_ratio(summary.ratio)),
            ("files", summary.file_count),
            ("LLM files", summary.llm_file_count),
            ("lossless files", summary.lossless_file_count),
            ("format", summary.format_variant),
            ("chunking", summary.chunking_strategy),
        )
    )
    if summary.fallback_paths:
        warn(f"Lossless fallback paths: {preview_list(summary.fallback_paths)}")
    if summary.errors:
        fail("Compression errors:")
        print_error_map(summary.errors)


def log_decompression_summary(summary: Any) -> None:
    if summary.ok:
        ok("Decompression completed.")
    else:
        fail(f"Decompression finished with {len(summary.errors)} error(s).")
    details(
        (
            ("output", summary.output_dir),
            ("files", summary.file_count),
            ("LLM files", summary.llm_file_count),
            ("lossless files", summary.lossless_file_count),
            ("hash matches", f"{summary.hash_matches}/{summary.file_count}"),
            ("mismatches", len(summary.hash_mismatches)),
            ("errors", len(summary.errors)),
        )
    )
    if summary.hash_mismatches:
        warn(f"Hash mismatches: {preview_list(summary.hash_mismatches)}")
    if summary.errors:
        fail("Decompression errors:")
        print_error_map(summary.errors)


def log_research_plan(plan: Any, *, title: str = "Experiment plan") -> None:
    detail(title, one_line(plan.hypothesis, limit=220), label_width=18)
    chunking = plan.chunking_strategy
    if plan.chunking_strategy == "line_chunks":
        chunking = f"line_chunks ({plan.chunk_size_lines} lines)"
    details(
        (
            ("model", plan.model),
            ("format", plan.format_variant),
            ("chunking", chunking),
            ("candidates", plan.candidate_count),
            ("repair", yes_no(plan.repair_enabled)),
            ("max LLM bytes", format_bytes(plan.max_llm_bytes)),
            ("temperature", plan.temperature),
            ("code patch", "yes" if getattr(plan, "code_patch", "").strip() else "no"),
        )
    )
    if plan.compression_prompt_extra.strip():
        detail("compress note", one_line(plan.compression_prompt_extra, limit=220))
    if plan.decompression_prompt_extra.strip():
        detail("decompress note", one_line(plan.decompression_prompt_extra, limit=220))


def log_verification_plan(plan: Any) -> None:
    details(
        (
            ("setup commands", len(plan.setup)),
            ("check commands", len(plan.checks)),
        )
    )
    if not plan.setup and not plan.checks:
        warn("No verification commands were detected for this target.")
        return
    for command in plan.setup:
        info(f"[setup] {command.name}: {command.command}", indent=4)
    for command in plan.checks:
        info(f"[check] {command.name}: {command.command}", indent=4)


def log_verification_report(report: Any, *, label: str = "Verification", include_output: bool = False) -> None:
    if report.ok:
        ok(f"{label} passed ({report.check_pass_count}/{report.check_count} checks).")
    else:
        fail(f"{label} failed ({report.check_pass_count}/{report.check_count} checks passed).")
    failed_results = [result for result in report.setup_results + report.check_results if not result.ok]
    if not failed_results:
        return
    for result in failed_results[:3]:
        timed_out = " timed out" if result.timed_out else ""
        fail(
            f"{result.kind} {result.name}{timed_out} failed "
            f"(exit {result.returncode}, {format_duration(result.seconds)}): {result.command}"
        )
        if include_output:
            print_output_tail(result.output)
    if len(failed_results) > 3:
        warn(f"{len(failed_results) - 3} more verification failure(s) omitted from console output.")


def log_candidate_outcome(*, index: int, success: bool, score: float, verification: Any | None, decompression: Any) -> None:
    if success:
        ok(f"Candidate {index + 1} is viable (score {score:.3f}).")
    else:
        fail(f"Candidate {index + 1} is not viable (score {score:.3f}).")
    if decompression.errors:
        detail("decomp errors", len(decompression.errors))
    if verification is not None:
        detail("checks", f"{verification.check_pass_count}/{verification.check_count}")
        detail("verification", "pass" if verification.ok else "fail")
