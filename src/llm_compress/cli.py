from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .app import DirectCompressionOptions, compress_target_once
from .benchmarks import benchmark_table, run_benchmarks, summarize_benchmarks
from .decompressor import Decompressor
from .openrouter import OpenRouterClient, OpenRouterConfig
from .pipeline import RunOptions, run_target
from .targets import resolve_artifact_reference

FORMAT_VARIANTS = (
    "component_v1",
    "component_contracts",
    "component_testsafe",
    "component_literal_heavy",
)
CHUNKING_STRATEGIES = ("file", "line_chunks")
DEFAULT_RESEARCH_MODEL = "anthropic/claude-sonnet-4.6"


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] in {"-h", "--help"}:
        print(_root_help())
        return 0 if args_list else 2

    command = args_list[0]
    if command in {"autoresearch", "research"}:
        return _main_autoresearch(args_list[1:])
    if command in {"benchmark", "benchmarks"}:
        return _main_autoresearch(["--benchmarks", *args_list[1:]])
    if command == "compress":
        return _main_direct(args_list[1:])
    if command in {"decompress", "restore"}:
        return _main_direct(["--decompress", *args_list[1:]])

    # Default application path: compress/decompress only. Autoresearch is opt-in
    # through the separate `llm-compress autoresearch ...` command.
    return _main_direct(args_list)


def _main_direct(argv: list[str]) -> int:
    parser = _build_direct_parser()
    args = parser.parse_args(argv)

    if args.list_benchmarks:
        print(json.dumps(benchmark_table(), indent=2))
        return 0

    client = OpenRouterClient(OpenRouterConfig.from_env(model=args.model))
    if not client.available:
        print("OPENROUTER_API_KEY is not set; using lossless-only compression where possible.")

    if args.decompress:
        if not args.target:
            parser.error("target is required with -d/--decompress")
        artifact = resolve_artifact_reference(args.target)
        output = Path(args.output) if args.output else artifact.parent / "decompressed"
        summary = Decompressor(client).decompress_artifact(
            artifact,
            output,
            workers=args.workers,
            clean=True,
        )
        print(json.dumps(summary.to_json(), indent=2, sort_keys=True))
        return 0 if summary.ok else 1

    if not args.target:
        parser.error("target is required")

    result = compress_target_once(
        args.target,
        client=client,
        options=DirectCompressionOptions(
            runs_root=Path(args.work_dir),
            workers=args.workers,
            max_llm_bytes=args.max_llm_bytes,
            use_llm=not args.no_llm,
            chunking_strategy=args.chunking_strategy,
            chunk_size_lines=args.chunk_size_lines,
            format_variant=args.format_variant,
            compression_prompt_extra=args.compression_prompt_extra or "",
        ),
    )
    print(json.dumps(result.to_json(), indent=2, sort_keys=True))
    return 0 if result.success else 1


def _main_autoresearch(argv: list[str]) -> int:
    parser = _build_autoresearch_parser()
    args = parser.parse_args(argv)

    if args.list_benchmarks:
        print(json.dumps(benchmark_table(), indent=2))
        return 0

    client = OpenRouterClient(OpenRouterConfig.from_env(model=args.model))
    research_model = args.research_model or os.environ.get("OPENROUTER_RESEARCH_MODEL", DEFAULT_RESEARCH_MODEL)
    research_client = OpenRouterClient(OpenRouterConfig.from_env(model=research_model))
    if not client.available:
        print("OPENROUTER_API_KEY is not set; autoresearch will use deterministic fallbacks and lossless-only compression where needed.")

    options = RunOptions(
        runs_root=Path(args.work_dir),
        max_iterations=args.max_iterations,
        workers=args.workers,
        max_llm_bytes=args.max_llm_bytes,
        verify=not args.no_verify,
        install=not args.no_install,
        timeout=args.timeout,
        setup_timeout=args.setup_timeout,
        continue_on_baseline_fail=args.continue_on_baseline_fail,
        max_candidates=args.max_candidates,
        model_candidates=_parse_models(args.models, args.model),
        program_path=Path(args.program) if args.program else None,
    )

    if args.benchmarks:
        results = run_benchmarks(client=client, research_client=research_client, options=options, repo_workers=args.repo_workers)
        summary = summarize_benchmarks(results)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["success_count"] == summary["repo_count"] else 1

    if not args.target:
        parser.error("target is required unless --benchmarks or --list-benchmarks is used")

    result = run_target(args.target, client=client, research_client=research_client, options=options)
    print(json.dumps(result.to_json(), indent=2, sort_keys=True))
    return 0 if result.success else 1


def _build_direct_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-compress",
        description="Compress or decompress codebases. Autoresearch is a separate command: llm-compress autoresearch ...",
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Local repo/file, public git URL, compressed.jsonl, run dir, or prior target when using -d.",
    )
    parser.add_argument(
        "-d",
        "--decompress",
        action="store_true",
        help="Decompress an artifact. If target is a prior repo/URL, resolves its latest artifact from .llm-compress/index.json.",
    )
    parser.add_argument("--list-benchmarks", action="store_true", help="Print the built-in benchmark repos as JSON and exit.")
    parser.add_argument("-o", "--output", help="Output directory for -d. Defaults beside the artifact.")
    parser.add_argument("--work-dir", default=".llm-compress/runs", help="Run directory root.")
    parser.add_argument("--model", help="OpenRouter model. Defaults to OPENROUTER_MODEL or openrouter/auto.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel file LLM workers.")
    parser.add_argument("--max-llm-bytes", type=int, default=20_000, help="Max file size to send to LLM.")
    parser.add_argument("--no-llm", action="store_true", help="Force lossless-only compression.")
    parser.add_argument("--format-variant", choices=FORMAT_VARIANTS, default="component_v1")
    parser.add_argument("--chunking-strategy", choices=CHUNKING_STRATEGIES, default="file")
    parser.add_argument("--chunk-size-lines", type=int, default=120)
    parser.add_argument("--compression-prompt-extra", default="")
    return parser


def _build_autoresearch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-compress autoresearch",
        description="Run LLM-driven autoresearch over compression strategies. This is opt-in and separate from normal compression.",
    )
    parser.add_argument("target", nargs="?", help="Local repo/file or public git URL.")
    parser.add_argument("--benchmarks", action="store_true", help="Run the built-in seven-repo global autoresearch benchmark set.")
    parser.add_argument("--list-benchmarks", action="store_true", help="Print the built-in benchmark repos as JSON and exit.")
    parser.add_argument("--work-dir", default=".llm-compress/runs", help="Run directory root.")
    parser.add_argument("--model", help="OpenRouter worker model. Defaults to OPENROUTER_MODEL or openrouter/auto.")
    parser.add_argument(
        "--research-model",
        help=f"OpenRouter model for autoresearch planning. Defaults to OPENROUTER_RESEARCH_MODEL or {DEFAULT_RESEARCH_MODEL}.",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated worker model candidates the research controller may choose from. Defaults to OPENROUTER_MODELS plus --model.",
    )
    parser.add_argument("--max-iterations", type=int, default=3, help="Autoresearch iterations per repo.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel file LLM workers.")
    parser.add_argument("--repo-workers", type=int, default=1, help="Deprecated in global benchmark mode; repos run sequentially so lessons carry forward.")
    parser.add_argument("--max-candidates", type=int, default=3, help="Max competing decompression candidates per iteration.")
    parser.add_argument("--program", help="Path to autoresearch program.md instructions. Defaults to ./program.md if present.")
    parser.add_argument("--max-llm-bytes", type=int, default=20_000, help="Max file size to send to LLM.")
    parser.add_argument("--timeout", type=int, default=180, help="Per check command timeout seconds.")
    parser.add_argument("--setup-timeout", type=int, default=600, help="Per setup command timeout seconds.")
    parser.add_argument("--no-verify", action="store_true", help="Skip baseline/candidate tests, lints, and builds.")
    parser.add_argument("--no-install", action="store_true", help="Do not run auto install/setup commands before verification.")
    parser.add_argument(
        "--continue-on-baseline-fail",
        action="store_true",
        help="Compress even if the original repo does not pass detected verification.",
    )
    return parser


def _parse_models(raw: str | None, primary: str | None) -> list[str] | None:
    parts: list[str] = []
    env_raw = os.environ.get("OPENROUTER_MODELS")
    for source in (primary, raw, env_raw):
        if not source:
            continue
        for item in source.split(","):
            model = item.strip()
            if model and model not in parts:
                parts.append(model)
    return parts or None


def _root_help() -> str:
    return """usage:
  llm-compress TARGET [options]                      Compress once; no autoresearch
  llm-compress -d ARTIFACT_OR_TARGET [-o DIR]        Decompress
  llm-compress compress TARGET [options]             Explicit compress-once command
  llm-compress decompress ARTIFACT_OR_TARGET         Explicit decompress command
  llm-compress autoresearch TARGET [options]         Run per-repo autoresearch
  llm-compress autoresearch --benchmarks [options]   Run global benchmark autoresearch
  llm-compress benchmarks [options]                  Alias for autoresearch --benchmarks

Normal compression is intentionally independent of autoresearch. Use
`llm-compress autoresearch ...` only when you want the research loop.
"""
