from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

from .console import detail, details, fail, ok, section, warn
from .global_research import GlobalResearchAgent, GlobalResearchState, load_global_research_file
from .openrouter import OpenRouterClient
from .pipeline import PipelineResult, RunOptions, run_target
from .reporting import log_research_plan
from .tool_patch import build_tool_repo_context


@dataclass(frozen=True)
class BenchmarkRepo:
    name: str
    url: str
    language: str
    size: str
    reason: str


@dataclass
class BenchmarkFailure:
    repo: BenchmarkRepo
    error: str

    def to_json(self) -> dict[str, Any]:
        return {"repo": self.repo.__dict__, "error": self.error}


@dataclass
class BenchmarkSuiteResult:
    results: list[PipelineResult]
    failures: list[BenchmarkFailure]
    global_history: list[dict[str, Any]]
    final_global_state: GlobalResearchState

    def to_json(self) -> dict[str, Any]:
        return {
            "results": [result.to_json() for result in self.results],
            "failures": [failure.to_json() for failure in self.failures],
            "global_history": self.global_history,
            "final_global_state": self.final_global_state.to_json(),
        }


BUILTIN_BENCHMARKS: tuple[BenchmarkRepo, ...] = (
    BenchmarkRepo(
        name="nanoid",
        url="https://github.com/ai/nanoid.git",
        language="JavaScript/TypeScript",
        size="small",
        reason="tiny package with npm scripts and tests",
    ),
    BenchmarkRepo(
        name="itsdangerous",
        url="https://github.com/pallets/itsdangerous.git",
        language="Python",
        size="small",
        reason="compact library with pytest/build metadata",
    ),
    BenchmarkRepo(
        name="mux",
        url="https://github.com/gorilla/mux.git",
        language="Go",
        size="small",
        reason="small Go router with go test/build/vet",
    ),
    BenchmarkRepo(
        name="itoa",
        url="https://github.com/dtolnay/itoa.git",
        language="Rust",
        size="small",
        reason="small Rust crate with cargo test/build/clippy when available",
    ),
    BenchmarkRepo(
        name="click",
        url="https://github.com/pallets/click.git",
        language="Python",
        size="medium",
        reason="CLI library with a broader test suite",
    ),
    BenchmarkRepo(
        name="express",
        url="https://github.com/expressjs/express.git",
        language="JavaScript",
        size="medium",
        reason="Node web framework with lint/test scripts",
    ),
    BenchmarkRepo(
        name="lodash",
        url="https://github.com/lodash/lodash.git",
        language="JavaScript",
        size="large",
        reason="larger utility library with many source files and tests",
    ),
)


def benchmark_table() -> list[dict[str, str]]:
    return [repo.__dict__ for repo in BUILTIN_BENCHMARKS]


def run_benchmarks(
    *,
    client: OpenRouterClient | None,
    options: RunOptions,
    repo_workers: int = 2,
    research_client: OpenRouterClient | None = None,
) -> BenchmarkSuiteResult:
    # Global benchmark learning is inherently sequential: repo N's outcome seeds repo N+1.
    if repo_workers != 1:
        warn("Global benchmark learning runs repos sequentially; --repo-workers is ignored.")
    options.runs_root = Path(options.runs_root)
    section("Global benchmark suite")
    details(
        (
            ("repos", len(BUILTIN_BENCHMARKS)),
            ("iterations/repo", options.max_iterations),
            ("max candidates", options.max_candidates),
            ("runs root", options.runs_root),
            ("verification", "yes" if options.verify else "no"),
        )
    )
    suite_context = _suite_context(options)
    model_candidates = _model_candidates(client, options.model_candidates)
    global_agent = GlobalResearchAgent(
        research_client or client,
        allowed_models=model_candidates,
        max_candidates=options.max_candidates,
    )
    persisted = load_global_research_file(
        _global_state_path(options.runs_root),
        allowed_models=model_candidates,
        max_candidates=options.max_candidates,
        default_max_llm_bytes=options.max_llm_bytes,
    )
    if persisted:
        persisted_state, global_history = persisted
        global_state = global_agent.resume_state(
            suite_context={**suite_context, "resumed_from": str(_global_state_path(options.runs_root))},
            history=global_history,
            previous_state=persisted_state,
            default_max_llm_bytes=options.max_llm_bytes,
        )
        ok("Loaded persisted global research state.")
        details((("lessons", len(global_state.lessons)), ("history", len(global_history))))
    else:
        global_state = global_agent.initial_state(
            suite_context=suite_context,
            default_max_llm_bytes=options.max_llm_bytes,
        )
        global_history: list[dict[str, Any]] = []
    results: list[PipelineResult] = []
    failures: list[BenchmarkFailure] = []
    _write_global_state(options.runs_root, "initial", global_state, global_history)
    section("Global seed strategy")
    detail("reason", global_state.reason)
    detail("lessons", len(global_state.lessons))
    log_research_plan(global_state.seed_plan, title="seed hypothesis")

    for index, repo in enumerate(BUILTIN_BENCHMARKS):
        section(f"Benchmark repo {index + 1}/{len(BUILTIN_BENCHMARKS)}: {repo.name}")
        details(
            (
                ("url", repo.url),
                ("language", repo.language),
                ("size", repo.size),
                ("why", repo.reason),
            )
        )
        log_research_plan(global_state.seed_plan, title="seed hypothesis")
        repo_options = dataclass_replace(
            options,
            initial_research_plan=global_state.seed_plan,
            global_lessons=list(global_state.lessons),
        )
        try:
            result = run_target(repo.url, client=client, research_client=research_client or client, options=repo_options)
            results.append(result)
            repo_summary = _repo_result_summary(repo, result)
        except Exception as exc:  # noqa: BLE001 - benchmark should report every repo.
            failure = BenchmarkFailure(repo=repo, error=str(exc))
            failures.append(failure)
            repo_summary = {
                "repo": repo.__dict__,
                "success": False,
                "error": str(exc),
                "global_seed_plan": global_state.seed_plan.to_json(),
            }
            fail(f"Benchmark {repo.name} failed before summary: {exc}")

        global_history.append(repo_summary)
        global_state = global_agent.update_state(
            suite_context={**suite_context, "completed_count": len(global_history)},
            history=global_history,
            previous_state=global_state,
            default_max_llm_bytes=options.max_llm_bytes,
        )
        _write_global_state(options.runs_root, f"after-{repo.name}", global_state, global_history)
        if repo_summary.get("success"):
            ok(f"Benchmark {repo.name} passed.")
        else:
            fail(f"Benchmark {repo.name} did not pass.")
        details(
            (
                ("ratio", repo_summary.get("ratio")),
                ("checks", f"{repo_summary.get('check_pass_count', 0)}/{repo_summary.get('check_count', 0)}"),
                ("lessons", len(global_state.lessons)),
            )
        )
        log_research_plan(global_state.seed_plan, title="next seed")

    return BenchmarkSuiteResult(
        results=results,
        failures=failures,
        global_history=global_history,
        final_global_state=global_state,
    )


def summarize_benchmarks(results: BenchmarkSuiteResult | list[PipelineResult]) -> dict[str, Any]:
    if isinstance(results, BenchmarkSuiteResult):
        suite = results
        pipeline_results = suite.results
        failures = suite.failures
        global_history = suite.global_history
        final_global_state = suite.final_global_state.to_json()
    else:
        suite = None
        pipeline_results = results
        failures = []
        global_history = []
        final_global_state = None

    rows: list[dict[str, Any]] = []
    for result in pipeline_results:
        final_iteration = result.iterations[-1] if result.iterations else None
        rows.append(
            {
                "target": result.target.target,
                "success": result.success,
                "run_dir": str(result.target.run_dir),
                "ratio": final_iteration.compression.ratio if final_iteration else None,
                "llm_files": final_iteration.compression.llm_file_count if final_iteration else 0,
                "lossless_files": final_iteration.compression.lossless_file_count if final_iteration else 0,
                "final_plan": final_iteration.plan.to_json() if final_iteration else None,
                "failed_test_count": final_iteration.verification.failed_test_count if final_iteration and final_iteration.verification else 0,
                "failure_units": final_iteration.verification.failure_units if final_iteration and final_iteration.verification else 0,
                "stopped_reason": result.stopped_reason,
            }
        )
    for failure in failures:
        rows.append(
            {
                "target": failure.repo.url,
                "success": False,
                "run_dir": None,
                "ratio": None,
                "llm_files": 0,
                "lossless_files": 0,
                "final_plan": None,
                "stopped_reason": failure.error,
            }
        )
    repo_count = len(pipeline_results) + len(failures)
    return {
        "repo_count": repo_count,
        "success_count": sum(1 for result in pipeline_results if result.success),
        "rows": rows,
        "global_history": global_history,
        "final_global_state": final_global_state,
        "global_learning": suite is not None,
    }


def _repo_result_summary(repo: BenchmarkRepo, result: PipelineResult) -> dict[str, Any]:
    final_iteration = result.iterations[-1] if result.iterations else None
    verification = final_iteration.verification if final_iteration else None
    failed_output = verification.all_output()[-8_000:] if verification and not verification.ok else ""
    repair_diagnostics = []
    if final_iteration:
        for candidate in final_iteration.candidates:
            if candidate.repair is None and candidate.repaired_verification is None:
                continue
            repaired = candidate.repaired_verification
            repair_diagnostics.append(
                {
                    "candidate": candidate.index,
                    "repair": candidate.repair.to_json() if candidate.repair else None,
                    "repaired_verification_ok": repaired.ok if repaired else None,
                    "repaired_check_pass_count": repaired.check_pass_count if repaired else 0,
                    "repaired_check_count": repaired.check_count if repaired else 0,
                    "repaired_failed_test_count": repaired.failed_test_count if repaired else 0,
                    "repaired_total_test_count": repaired.total_test_count if repaired else None,
                    "repaired_failure_units": repaired.failure_units if repaired else 0,
                    "accepted": candidate.success,
                    "note": "repair is diagnostic only; accepted requires raw decompression verification",
                }
            )
    return {
        "repo": repo.__dict__,
        "target": result.target.target,
        "success": result.success,
        "run_dir": str(result.target.run_dir),
        "ratio": final_iteration.compression.ratio if final_iteration else None,
        "llm_files": final_iteration.compression.llm_file_count if final_iteration else 0,
        "lossless_files": final_iteration.compression.lossless_file_count if final_iteration else 0,
        "final_plan": final_iteration.plan.to_json() if final_iteration else None,
        "iterations": len(result.iterations),
        "verification_ok": verification.ok if verification else None,
        "check_pass_count": verification.check_pass_count if verification else 0,
        "check_count": verification.check_count if verification else 0,
        "failed_test_count": verification.failed_test_count if verification else 0,
        "total_test_count": verification.total_test_count if verification else None,
        "failure_units": verification.failure_units if verification else 0,
        "failed_output_tail": failed_output,
        "repair_diagnostics": repair_diagnostics,
        "stopped_reason": result.stopped_reason,
    }


def _suite_context(options: RunOptions) -> dict[str, Any]:
    return {
        "benchmark_count": len(BUILTIN_BENCHMARKS),
        "repos": [repo.__dict__ for repo in BUILTIN_BENCHMARKS],
        "max_iterations_per_repo": options.max_iterations,
        "max_candidates": options.max_candidates,
        "default_max_llm_bytes": options.max_llm_bytes,
        "verify": options.verify,
        "llm_compression_tool_repo": build_tool_repo_context(),
    }


def _model_candidates(client: OpenRouterClient | None, configured: list[str] | None) -> list[str]:
    candidates = [item.strip() for item in (configured or []) if item.strip()]
    if client and client.model not in candidates:
        candidates.insert(0, client.model)
    return candidates or ["openrouter/auto"]


def _write_global_state(
    runs_root: Path,
    label: str,
    state: GlobalResearchState,
    history: list[dict[str, Any]],
) -> None:
    path = _global_state_path(runs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "state": state.to_json(),
        "history": history,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _global_state_path(runs_root: Path) -> Path:
    return runs_root.parent / "benchmark-global-research.json"
