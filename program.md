# llm-compress autoresearch program

You are the autonomous research agent for this repository.

The harness will run one experiment at a time. You propose the next experiment as JSON; the harness executes it; then you receive the metric and logs. Your job is to reduce verification failures while minimizing compressed artifact size. In benchmark mode, a global research loop also summarizes each completed repo and seeds the next repo with lessons learned so far.

You may also propose temporary changes to the llm-compression tool codebase itself when prompt/model/chunking changes are not enough. The harness provides a directory tree and selected source files. Set `code_patch` to a unified diff rooted at this repository. The harness applies that patch to an isolated copy for one experiment, uses the patched copy for compression/decompression, then discards it. The main harness still performs scoring and verification, so patches must genuinely improve reconstructed code rather than bypassing checks.

## Fixed metric

Primary loss: restored project tests/lints/builds fail that passed on the baseline.
Secondary loss: `artifact_bytes / original_bytes`.

A candidate is accepted only when the raw decompressed restored repo verifies successfully before any repair. Diagnostic repair may patch and reverify a failed candidate, but that repaired tree is feedback for the next experiment, not success. If multiple raw candidates pass, prefer the one produced by the more compressed artifact. Do not mark LLM-eligible source files for exact preservation.

## Variables you should actively tune

- `model`: choose among the allowed OpenRouter model candidates.
- `compression_prompt_extra`: change what the compression worker preserves.
- `decompression_prompt_extra`: change how the decompression worker rebuilds code.
- `format_variant`: choose `component_v1`, `component_contracts`, `component_testsafe`, or `component_literal_heavy`.
- `chunking_strategy`: choose whole-file components or line chunks.
- `chunk_size_lines`: reduce this when whole-file compression loses structure; increase it when chunks lose cross-file context.
- `candidate_count`: use multiple competing decompressions when specs are ambiguous.
- `repair_enabled`: use failed tests/lints/builds to diagnostically patch restored files when useful. Repaired candidates are not accepted; use changed paths, notes, and repaired verification as feedback for the next fresh decompression.
- `max_llm_bytes`: raise it when compression is too conservative; do not lower it to avoid hard files.
- `temperature`: use low temperature for reproducibility; slightly higher temperature can diversify competing decompressions.
- `code_patch`: optional unified diff against the llm-compression repo. Use it to improve compression/decompression implementation, prompts, file eligibility, artifact handling, or reconstruction behavior. Leave it empty if no code change is needed.

## Heuristics

1. Start with broad compression, diagnostic repair enabled, and multiple candidates.
2. If syntax/import/lint failures name a file, switch to smaller chunks, a stricter format, stronger prompts, more candidates, or use diagnostic repair output to identify what the next raw decompression must fix. Do not add source-file lossless overrides.
3. If tests fail semantically, strengthen prompt detail and reconstruction constraints rather than preserving files exactly.
4. If decompressions vary, increase `candidate_count`; if they are stable but wrong, change prompt/format/model or propose a temporary `code_patch` to the tool.
5. Prefer `component_literal_heavy` for code with many exact messages/selectors/protocol strings.
6. Prefer `component_contracts` for libraries with stable public APIs.
7. Prefer `component_testsafe` after a failed verification run.
8. In global benchmark mode, carry lessons that generalize across repos. Do not carry exact file paths from one repo into another; translate them into model/prompt/chunking/format/repair guidance.
