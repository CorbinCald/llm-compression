import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from llm_compress import cli
from llm_compress.cli import DEFAULT_RESEARCH_MODEL, _build_autoresearch_parser, _build_direct_parser, _root_help


class CliTests(unittest.TestCase):
    def test_root_help_separates_autoresearch(self):
        help_text = _root_help()
        self.assertIn("Compress once; no autoresearch", help_text)
        self.assertIn("llm-compress autoresearch TARGET", help_text)

    def test_direct_parser_has_no_autoresearch_iterations(self):
        parser = _build_direct_parser()
        args = parser.parse_args(["./repo", "--no-llm"])
        self.assertEqual(args.target, "./repo")
        self.assertTrue(args.no_llm)
        self.assertFalse(hasattr(args, "max_iterations"))

    def test_autoresearch_parser_owns_iterations(self):
        parser = _build_autoresearch_parser()
        args = parser.parse_args(["./repo", "--max-iterations", "5"])
        self.assertEqual(args.target, "./repo")
        self.assertEqual(args.max_iterations, 5)

    def test_autoresearch_parser_accepts_research_model(self):
        parser = _build_autoresearch_parser()
        args = parser.parse_args(["./repo", "--research-model", DEFAULT_RESEARCH_MODEL])
        self.assertEqual(args.research_model, DEFAULT_RESEARCH_MODEL)

    def test_autoresearch_uses_separate_research_client(self):
        captured = {}

        class Result:
            success = True

            @staticmethod
            def to_json():
                return {"success": True}

        def fake_run_target(target, *, client, research_client, options):
            captured["target"] = target
            captured["worker_model"] = client.model
            captured["research_model"] = research_client.model
            captured["models"] = options.model_candidates
            return Result()

        with patch.object(cli, "run_target", fake_run_target), redirect_stdout(io.StringIO()):
            code = cli._main_autoresearch(
                [
                    "./repo",
                    "--model",
                    "provider/worker",
                    "--research-model",
                    "provider/research",
                    "--models",
                    "provider/worker",
                    "--no-verify",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(captured["worker_model"], "provider/worker")
        self.assertEqual(captured["research_model"], "provider/research")
        self.assertEqual(captured["models"], ["provider/worker"])


if __name__ == "__main__":
    unittest.main()
