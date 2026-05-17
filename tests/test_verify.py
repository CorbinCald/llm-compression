import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm_compress.verify import (
    CommandResult,
    VerificationCommand,
    VerificationPlan,
    _parse_test_summary_counts,
    discover_verification_plan,
    run_verification,
)


class VerificationDiscoveryTests(unittest.TestCase):
    def test_parses_pytest_granular_failure_count(self):
        failed, total = _parse_test_summary_counts("=== 2 failed, 3 passed, 1 skipped in 0.12s ===")
        self.assertEqual(failed, 2)
        self.assertEqual(total, 6)

    def test_parses_jest_granular_failure_count(self):
        failed, total = _parse_test_summary_counts("Tests:       4 failed, 10 passed, 14 total")
        self.assertEqual(failed, 4)
        self.assertEqual(total, 14)

    def test_parses_unittest_failures_and_errors(self):
        output = "Ran 9 tests in 0.1s\n\nFAILED (failures=2, errors=1)"
        failed, total = _parse_test_summary_counts(output)
        self.assertEqual(failed, 3)
        self.assertEqual(total, 9)

    def test_failed_test_command_with_zero_parsed_failures_has_fallback_unit(self):
        result = CommandResult(
            name="python-test",
            command="pytest",
            kind="check",
            returncode=1,
            seconds=0.01,
            output="0 failed",
            failed_test_count=0,
        )
        self.assertEqual(result.failure_units, 1)

    def test_discovers_node_scripts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package.json").write_text(
                json.dumps({"scripts": {"test": "echo test", "lint": "echo lint", "build": "echo build"}}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"LLM_COMPRESS_NODE_RUNNER": "npm"}):
                plan = discover_verification_plan(root, install=False)
            commands = [command.command for command in plan.checks]
            self.assertEqual(commands, ["npm run lint", "npm run build", "npm test"])

    def test_uses_node_lts_runner_when_host_node_is_too_new(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package.json").write_text(json.dumps({"scripts": {"test": "echo test"}}), encoding="utf-8")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("llm_compress.verify._node_major_version", return_value=25),
                patch("llm_compress.verify.shutil.which", side_effect=lambda name: "/usr/bin/npx" if name == "npx" else None),
            ):
                plan = discover_verification_plan(root, install=True)
            self.assertEqual(plan.setup[0].command, "npx -y -p node@22 -p npm@10 npm install")
            self.assertEqual(plan.checks[0].command, "npx -y -p node@22 -p npm@10 npm test")

    def test_python_dependency_groups_and_isolated_build(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text(
                """
[project]
name = "unit"
version = "0.1.0"

[dependency-groups]
tests = ["freezegun", "pytest"]

[build-system]
requires = ["flit_core<4"]
build-backend = "flit_core.buildapi"
""".strip(),
                encoding="utf-8",
            )
            (root / "tests").mkdir()

            plan = discover_verification_plan(root, install=True)
            setup_command = next(command.command for command in plan.setup if command.name == "python-install")
            build_command = next(command.command for command in plan.checks if command.name == "python-build")

            self.assertIn("-m pip install --group tests", setup_command)
            self.assertEqual(build_command, "./.llm-compress-venv/bin/python -m build --wheel")

    def test_stages_verification_when_root_has_hidden_path_parts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / ".hidden" / "repo"
            root.mkdir(parents=True)
            plan = VerificationPlan(
                checks=[
                    VerificationCommand(
                        name="cwd-visible",
                        command=(
                            f"{sys.executable} -c \"import pathlib, sys; "
                            "sys.exit(any(part.startswith('.') for part in pathlib.Path.cwd().parts))\""
                        ),
                    )
                ]
            )
            report = run_verification(root, plan, print_progress=False)
            self.assertTrue(report.ok, report.all_output())

    def test_discovers_go_commands(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "go.mod").write_text("module example.com/x\n", encoding="utf-8")
            plan = discover_verification_plan(root, install=False)
            self.assertEqual(
                [command.command for command in plan.checks],
                ["go test ./...", "go vet ./...", "go build ./..."],
            )


if __name__ == "__main__":
    unittest.main()
