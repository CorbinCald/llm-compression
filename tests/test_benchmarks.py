import unittest

from llm_compress.benchmarks import BenchmarkFailure, BenchmarkRepo, BenchmarkSuiteResult, summarize_benchmarks
from llm_compress.global_research import GlobalResearchState


class BenchmarkSummaryTests(unittest.TestCase):
    def test_summary_counts_failures_in_repo_count(self):
        suite = BenchmarkSuiteResult(
            results=[],
            failures=[
                BenchmarkFailure(
                    repo=BenchmarkRepo("x", "https://example.com/x.git", "Python", "small", "unit"),
                    error="boom",
                )
            ],
            global_history=[{"repo": {"name": "x"}, "success": False}],
            final_global_state=GlobalResearchState(),
        )
        summary = summarize_benchmarks(suite)
        self.assertEqual(summary["repo_count"], 1)
        self.assertEqual(summary["success_count"], 0)
        self.assertTrue(summary["global_learning"])
        self.assertEqual(summary["rows"][0]["stopped_reason"], "boom")


if __name__ == "__main__":
    unittest.main()
