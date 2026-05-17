# LLM Compression

Compress a file or codebase into recoverable component descriptions and restore it later. LLM-compressed components are decompressed with an LLM; lossless records restore exactly. Autoresearch is available as a separate opt-in command for improving compression strategies, but it is not required for normal compression/decompression.

## Core idea

For a given file or codebase, reduce chars at the cost of inference while maintaining identical function.

Component compression format:

```text
<-Name:name|Input:inputs or none|Return:exact behavior/code-shaped spec|Path:path|Order:order+indent->
```

`Order` is file order plus indent level, where `a` is top-level. Compression is only valid when expansion is recoverable: keep exact names, literals, selectors, globs, protocol shapes, formulas, and side effects. Prefer code-shaped specs over prose; omit only formatting/comments that do not affect behavior.

## What is built

This repo now provides a Python CLI named `llm-compress`.

Normal compression can:

- accept a local repo/file or public git URL;
- clone/copy the target into an isolated run directory;
- compress source files with OpenRouter-backed LLM calls;
- keep tests, lockfiles, package metadata, generated-risk files, binaries, and oversized files lossless;
- write a `compressed.jsonl` artifact under `.llm-compress/runs/`;
- decompress an artifact back into a restored tree, using an LLM only for files that were stored as LLM components.

The separate `llm-compress autoresearch ...` command can additionally:

- detect and run project verification commands:
  - Python: pytest, ruff if configured, wheel build;
  - Node: package-manager install, lint/build/test scripts;
  - Go: `go test`, `go vet`, `go build`;
  - Rust: `cargo test`, `cargo clippy` when available, `cargo build`;
- decompress multiple competing restored repo candidates;
- use failed verification output for optional diagnostic LLM repair;
- invoke an autoresearch LLM to adjust prompts, model, chunking, format variant, candidates, repair, and size thresholds between iterations;
- run global benchmark learning across the built-in seven-repo suite.

The artifact is JSONL (`compressed.jsonl`). LLM-compressed files store component lines; exact fallback files store deterministic gzip+base64 bytes.

## Quickstart

Run the built-in benchmark suite with Claude Sonnet 4.6 as the research/controller model, Gemini Flash Lite as the worker model, and two autoresearch iterations per repo:

```bash
PYTHONPATH=src python3 -m llm_compress autoresearch --benchmarks --research-model anthropic/claude-sonnet-4.6 --model google/gemini-3.1-flash-lite --models google/gemini-3.1-flash-lite --max-iterations 2
```

Set `OPENROUTER_API_KEY` first, or install the CLI and use `llm-compress` instead of `PYTHONPATH=src python3 -m llm_compress`.

## Install/use locally

```bash
python3 -m pip install -e .
```

Set your OpenRouter key for LLM compression/decompression:

```bash
export OPENROUTER_API_KEY=...
# optional worker model; default is openrouter/auto
export OPENROUTER_MODEL=openrouter/auto
# optional research/controller model; default is anthropic/claude-sonnet-4.6
export OPENROUTER_RESEARCH_MODEL=anthropic/claude-sonnet-4.6
# optional worker candidates the autoresearch LLM can choose among
export OPENROUTER_MODELS=openrouter/auto,provider/model-name
```

Compress once without autoresearch:

```bash
llm-compress ./some-repo
llm-compress https://github.com/pallets/itsdangerous.git
# explicit equivalent
llm-compress compress ./some-repo
```

Run the opt-in compress → decompress candidates → verify → autoresearch loop:

```bash
llm-compress autoresearch ./some-repo
llm-compress autoresearch https://github.com/pallets/itsdangerous.git
```

Decompress an artifact or the latest artifact remembered for a target:

```bash
llm-compress -d .llm-compress/runs/<run>/compressed.jsonl -o restored-repo
llm-compress -d .llm-compress/runs/<run>/final/compressed.jsonl -o restored-repo  # autoresearch final artifact
llm-compress decompress .llm-compress/runs/<run>/compressed.jsonl -o restored-repo
llm-compress -d ./some-repo -o restored-repo
llm-compress -d https://github.com/pallets/itsdangerous.git -o restored-repo
```

If `OPENROUTER_API_KEY` is absent, compression falls back to lossless-only artifacts. Those artifacts decompress without a key. Decompressing an artifact that already contains LLM components still requires `OPENROUTER_API_KEY`.

Console output is human-readable by default; compression and autoresearch runs also write `summary.json` in the run directory. Add `--json` to print the full machine-readable summary to stdout.

Useful normal-compression flags:

```bash
llm-compress ./repo --workers 8
llm-compress ./repo --max-llm-bytes 40000
llm-compress ./repo --no-llm
llm-compress ./repo --format-variant component_literal_heavy
llm-compress ./repo --chunking-strategy line_chunks --chunk-size-lines 80
llm-compress ./repo --json
```

Useful autoresearch flags:

```bash
llm-compress autoresearch ./repo --max-iterations 5
llm-compress autoresearch ./repo --workers 8
llm-compress autoresearch ./repo --max-candidates 3
llm-compress autoresearch ./repo --research-model anthropic/claude-sonnet-4.6
llm-compress autoresearch ./repo --models openrouter/auto,provider/model-name
llm-compress autoresearch ./repo --program ./program.md
llm-compress autoresearch ./repo --max-llm-bytes 40000
llm-compress autoresearch ./repo --no-install       # skip auto dependency install
llm-compress autoresearch ./repo --no-verify        # compress/decompress candidates only
llm-compress autoresearch ./repo --continue-on-baseline-fail
llm-compress autoresearch ./repo --json
```

General model/iteration syntax:

```bash
llm-compress autoresearch TARGET --research-model anthropic/claude-sonnet-4.6 --model provider/worker-model --models provider/worker-model --max-iterations N
llm-compress autoresearch --benchmarks --research-model anthropic/claude-sonnet-4.6 --model provider/worker-model --models provider/worker-model --max-iterations N
```

`--research-model` selects the OpenRouter model used by the autoresearch/global-research controller. `--model` selects the default worker model for compression/decompression/repair. `--models` is the comma-separated worker candidate set the autoresearch controller may choose from. `--max-iterations` is per target; in benchmark mode it applies to each built-in repo.

## Built-in seven-repo benchmark set

For opt-in autoresearch benchmarking, the CLI includes seven public GitHub repos of varied language/size:

| Size | Repo | Type |
| --- | --- | --- |
| small | `https://github.com/ai/nanoid.git` | JavaScript/TypeScript package |
| small | `https://github.com/pallets/itsdangerous.git` | Python library |
| small | `https://github.com/gorilla/mux.git` | Go router |
| small | `https://github.com/dtolnay/itoa.git` | Rust crate |
| medium | `https://github.com/pallets/click.git` | Python CLI library |
| medium | `https://github.com/expressjs/express.git` | Node web framework |
| large | `https://github.com/lodash/lodash.git` | JavaScript utility library |

List them:

```bash
llm-compress --list-benchmarks
```

Run them with global autoresearch learning enabled:

```bash
llm-compress autoresearch --benchmarks --workers 4
# alias
llm-compress benchmarks --workers 4
```

Because each repo's outcome seeds the strategy for the next repo, benchmark repos run sequentially. `--repo-workers` is ignored in global benchmark mode.

## Autoresearch loop

The MVP now uses an LLM-driven autoresearch controller inspired by Karpathy's `autoresearch`: a human-editable `program.md` defines the research organization, the harness runs one bounded experiment at a time, the project verification result is the loss metric, and the research LLM proposes the next experiment from the observed logs.

For each target:

1. Run baseline project verification on the original copied/cloned source.
2. Ask the autoresearch LLM for an experiment plan. The plan can change:
   - OpenRouter worker model candidate;
   - compression prompt addendum;
   - decompression prompt addendum;
   - compression format variant;
   - whole-file vs line-chunk compression and chunk size;
   - number of competing decompression candidates;
   - whether failed tests/lints/builds should drive diagnostic LLM repair;
   - max bytes eligible for LLM compression, which autoresearch may raise but not lower below the CLI default;
   - sampling temperature;
   - temporary `code_patch` unified diffs against the llm-compression tool codebase for isolated experiments.
3. Compress the repo with that plan. Tests/config/locks/binaries and user-threshold oversized files remain deterministic lossless records, but autoresearch cannot add path-specific source-file lossless overrides.
4. Decompress `candidate_count` restored repos. Each candidate is independently verified.
5. If raw verification fails and `repair_enabled` is true, an LLM diagnostic repair pass receives failed verification output, compressed components, and current reconstructed file content, writes full repaired files, and the candidate is reverified for diagnosis only.
6. Competing candidates are scored and accepted using the unrepaired decompression result. A repaired candidate never counts as success; autoresearch must find a fresh decompression that passes without repair.
7. If no raw candidate passes, the research LLM receives the previous plan, compression stats, raw candidate scores, failed output tails, and repair diagnostics such as changed paths and repaired reverification results, then proposes the next experiment without opting source files out of compression.
8. Repeat until success or `--max-iterations`.

The primary loss metric is raw-decompression verification failure, scored as granular failed test count when the tool output exposes it and otherwise as fallback failure units for failed setup/lint/build/test commands. Tests/lints/builds that pass on baseline should pass after decompression before any repair. The secondary metric is compression ratio. The research controller receives a directory tree and selected source files from this repository; when needed, it can include a `code_patch` unified diff. The patch is applied only to an isolated copy for that experiment's compression/decompression step and then discarded; the main harness still performs scoring and verification.

## Global benchmark learning

`llm-compress autoresearch --benchmarks` runs a suite-level meta-research loop, not seven isolated loops:

1. A global research LLM proposes an initial seed strategy for the whole benchmark suite.
2. Repo 1 runs the normal per-repo autoresearch loop using that seed.
3. The repo result is summarized: success/failure, final plan, compression ratio, verification counts, and failed output tail.
4. The global research LLM updates global lessons and a new seed plan.
5. Repo 2 starts from the updated global seed.
6. This repeats through all seven repos.

Global state is written to and automatically reloaded from:

```text
.llm-compress/benchmark-global-research.json
```

This makes benchmark lessons persist across separate `llm-compress autoresearch --benchmarks` invocations. A later benchmark run seeds the global strategy and per-repo researcher context with the previous run's lessons and recent repo history. Global lessons are constrained to language-level or tool-level patterns; exact repo names, package names, file paths, and file names are filtered out so one repo's local fix does not leak into another.

Path-specific source-file `lossless_overrides` are disabled. Global learning carries general lessons via prompts, model choice, format variant, chunking, candidate count, repair behavior, size thresholds, temperature, and optional temporary tool-code patches.

## Development validation

This project intentionally uses only the Python standard library at runtime.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
```

## Prompt references

The compression/decompression prompts live in `PROMPTS.md` and are embedded in `src/llm_compress/prompts.py`.
