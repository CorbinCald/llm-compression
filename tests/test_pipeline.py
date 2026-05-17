import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from llm_compress.compressor import CompressionSummary
from llm_compress.decompressor import DecompressionSummary
from llm_compress.pipeline import (
    CandidateResult,
    IterationResult,
    RunOptions,
    _evaluate_candidate,
    _history_entry,
)
from llm_compress.repair import RepairSummary
from llm_compress.research import ResearchPlan
from llm_compress.verify import CommandResult, VerificationCommand, VerificationPlan, VerificationReport


def _verification_report(root: Path, *, ok: bool) -> VerificationReport:
    command = VerificationCommand(name="unit-test", command="pytest")
    result = CommandResult(
        name=command.name,
        command=command.command,
        kind=command.kind,
        returncode=0 if ok else 1,
        seconds=0.01,
        output="passed" if ok else "failed in ./a.py",
    )
    return VerificationReport(
        root=root,
        plan=VerificationPlan(checks=[command]),
        setup_results=[],
        check_results=[result],
    )


def _decompression_summary(root: Path) -> DecompressionSummary:
    return DecompressionSummary(
        artifact_path=root / "compressed.jsonl",
        output_dir=root / "candidate",
        file_count=1,
        llm_file_count=1,
        lossless_file_count=0,
        hash_matches=0,
        hash_mismatches=["./a.py"],
        errors={},
    )


class _FakeClient:
    available = True


def _compression_summary(root: Path) -> CompressionSummary:
    return CompressionSummary(
        artifact_path=root / "compressed.jsonl",
        original_bytes=100,
        artifact_bytes=50,
        file_count=1,
        llm_file_count=1,
        lossless_file_count=0,
        llm_paths=["./a.py"],
        fallback_paths=[],
        errors={},
        chunking_strategy="file",
        format_variant="component_v1",
    )


class PipelineRepairTests(unittest.TestCase):
    def test_repaired_candidate_is_diagnostic_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = _verification_report(root, ok=False)
            repaired = _verification_report(root, ok=True)
            (root / "candidate").mkdir()

            with (
                patch("llm_compress.pipeline.run_verification", side_effect=[failed, repaired]) as run_verification,
                patch(
                    "llm_compress.pipeline.repair_restored_candidate",
                    return_value=RepairSummary(
                        attempted=True,
                        repaired_paths=["./a.py"],
                        notes="fixed diagnostic candidate",
                    ),
                ),
                redirect_stdout(io.StringIO()),
            ):
                candidate = _evaluate_candidate(
                    candidate_index=0,
                    restore_dir=root / "candidate",
                    decompression=_decompression_summary(root),
                    compression=_compression_summary(root),
                    artifact_path=root / "compressed.jsonl",
                    verification_plan=VerificationPlan(checks=[VerificationCommand(name="unit-test", command="pytest")]),
                    run_options=RunOptions(verify=True),
                    research_plan=ResearchPlan(repair_enabled=True),
                    iteration_client=_FakeClient(),
                    baseline=None,
                )

        self.assertEqual(run_verification.call_count, 2)
        self.assertEqual(run_verification.call_args_list[1].args[0].name, "candidate-repair")
        self.assertIs(candidate.verification, failed)
        self.assertIs(candidate.repaired_verification, repaired)
        self.assertFalse(candidate.success)

    def test_history_includes_repair_diagnostics_for_next_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = _verification_report(root, ok=False)
            repaired = _verification_report(root, ok=True)
            candidate = CandidateResult(
                index=0,
                restore_dir=root / "candidate",
                decompression=_decompression_summary(root),
                verification=failed,
                repair=RepairSummary(
                    attempted=True,
                    repaired_paths=["./a.py"],
                    notes="fixed diagnostic candidate",
                ),
                repaired_verification=repaired,
                success=False,
                score=1.0,
            )
            iteration = IterationResult(
                index=0,
                artifact_path=root / "compressed.jsonl",
                restore_dir=root / "candidate",
                compression=_compression_summary(root),
                decompression=candidate.decompression,
                verification=failed,
                success=False,
                lossless_overrides=[],
                plan=ResearchPlan(),
                candidates=[candidate],
            )

            history = _history_entry(iteration)

        self.assertEqual(history["best_candidate"]["verification_ok"], False)
        self.assertEqual(history["best_candidate"]["failure_units"], 1)
        self.assertEqual(history["repair_diagnostics"][0]["repair"]["repaired_paths"], ["./a.py"])
        self.assertEqual(history["repair_diagnostics"][0]["repaired_verification_ok"], True)
        self.assertFalse(history["repair_diagnostics"][0]["accepted"])


if __name__ == "__main__":
    unittest.main()
