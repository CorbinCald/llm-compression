from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .compressor import CompressionOptions, Compressor
from .decompressor import Decompressor
from .openrouter import OpenRouterClient, OpenRouterConfig


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m llm_compress.iteration_runner CONFIG_JSON", file=sys.stderr)
        return 2
    config_path = Path(args[0])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_path = Path(config["output_json"])

    plan = config["research_plan"]
    client = OpenRouterClient(
        OpenRouterConfig.from_env(
            model=str(plan.get("model") or "openrouter/auto"),
            temperature=float(plan.get("temperature", 0.1)),
        )
    )
    source_root = Path(config["source_root"])
    artifact_path = Path(config["artifact_path"])
    iteration_dir = Path(config["iteration_dir"])
    workers = int(config.get("workers", 4))
    candidate_count = max(1, int(plan.get("candidate_count", 1)))

    compression = Compressor(client).compress_repo(
        source_root,
        artifact_path,
        options=CompressionOptions(
            max_llm_bytes=int(plan.get("max_llm_bytes", 20_000)),
            workers=workers,
            use_llm=client.available,
            chunking_strategy=str(plan.get("chunking_strategy") or "file"),
            chunk_size_lines=int(plan.get("chunk_size_lines", 120)),
            format_variant=str(plan.get("format_variant") or "component_v1"),
            compression_prompt_extra=str(plan.get("compression_prompt_extra") or ""),
        ),
        target_label=str(config.get("target_label") or "target"),
    )

    decompressions: list[dict[str, Any]] = []
    for candidate_index in range(candidate_count):
        restore_dir = iteration_dir / f"candidate-{candidate_index}"
        decompression = Decompressor(
            client,
            prompt_extra=str(plan.get("decompression_prompt_extra") or ""),
            candidate_index=candidate_index,
            candidate_count=candidate_count,
        ).decompress_artifact(
            artifact_path,
            restore_dir,
            workers=workers,
            clean=True,
        )
        decompressions.append(
            {
                "index": candidate_index,
                "restore_dir": str(restore_dir),
                "decompression": decompression.to_json(),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "compression": compression.to_json(),
                "decompressions": decompressions,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
