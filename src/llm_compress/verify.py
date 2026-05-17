from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .console import fail, format_duration, info, ok, print_output_tail, warn


@dataclass(frozen=True)
class VerificationCommand:
    name: str
    command: str
    kind: str = "check"
    timeout: int | None = None


@dataclass
class CommandResult:
    name: str
    command: str
    kind: str
    returncode: int
    seconds: float
    output: str
    timed_out: bool = False
    failed_test_count: int | None = None
    total_test_count: int | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def failure_units(self) -> int:
        if self.failed_test_count is not None:
            if self.failed_test_count > 0 or self.ok:
                return self.failed_test_count
            return 1
        return 0 if self.ok else 1

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "kind": self.kind,
            "returncode": self.returncode,
            "seconds": round(self.seconds, 3),
            "timed_out": self.timed_out,
            "ok": self.ok,
            "failed_test_count": self.failed_test_count,
            "total_test_count": self.total_test_count,
            "failure_units": self.failure_units,
            "output": self.output[-12_000:],
        }


@dataclass
class VerificationPlan:
    setup: list[VerificationCommand] = field(default_factory=list)
    checks: list[VerificationCommand] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "setup": [cmd.__dict__ for cmd in self.setup],
            "checks": [cmd.__dict__ for cmd in self.checks],
        }


@dataclass
class VerificationReport:
    root: Path
    plan: VerificationPlan
    setup_results: list[CommandResult]
    check_results: list[CommandResult]

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.setup_results + self.check_results)

    @property
    def check_pass_count(self) -> int:
        return sum(1 for result in self.check_results if result.ok)

    @property
    def check_count(self) -> int:
        return len(self.check_results)

    @property
    def failed_test_count(self) -> int:
        return sum(result.failed_test_count or 0 for result in self.check_results)

    @property
    def total_test_count(self) -> int | None:
        totals = [result.total_test_count for result in self.check_results if result.total_test_count is not None]
        return sum(totals) if totals else None

    @property
    def failure_units(self) -> int:
        return sum(result.failure_units for result in self.setup_results + self.check_results)

    def all_output(self) -> str:
        return "\n".join(result.output for result in self.setup_results + self.check_results)

    def to_json(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "ok": self.ok,
            "check_pass_count": self.check_pass_count,
            "check_count": self.check_count,
            "failed_test_count": self.failed_test_count,
            "total_test_count": self.total_test_count,
            "failure_units": self.failure_units,
            "plan": self.plan.to_json(),
            "setup_results": [result.to_json() for result in self.setup_results],
            "check_results": [result.to_json() for result in self.check_results],
        }


def discover_verification_plan(root: Path, *, install: bool = True) -> VerificationPlan:
    root = root.resolve()
    setup: list[VerificationCommand] = []
    checks: list[VerificationCommand] = []

    package_json = root / "package.json"
    if package_json.exists():
        package_manager = _node_package_manager(root)
        if install:
            setup.append(
                VerificationCommand(
                    name="node-install",
                    command=_node_install_command(root, package_manager),
                    kind="setup",
                )
            )
        scripts = _package_scripts(package_json)
        for script in ("lint", "build", "test"):
            if script in scripts:
                checks.append(
                    VerificationCommand(
                        name=f"node-{script}",
                        command=_node_run_command(package_manager, script),
                        kind="check",
                    )
                )

    if _is_python_project(root):
        host_python = shlex.quote(sys.executable)
        py = "./.llm-compress-venv/bin/python" if install else host_python
        if install:
            setup.append(
                VerificationCommand(
                    name="python-install",
                    command=_python_install_command(root, host_python=host_python, py=py),
                    kind="setup",
                )
            )
        if _has_python_tests(root):
            checks.append(
                VerificationCommand(name="python-test", command=f"{py} -m pytest", kind="check")
            )
        if _has_ruff_config(root):
            checks.append(
                VerificationCommand(name="python-lint", command=f"{py} -m ruff check .", kind="check")
            )
        checks.append(
            VerificationCommand(
                name="python-build",
                command=f"{py} -m build --wheel",
                kind="check",
            )
        )
    elif any(root.rglob("*.py")):
        checks.append(
            VerificationCommand(
                name="python-compile", command=f"{shlex.quote(sys.executable)} -m compileall -q .", kind="check"
            )
        )

    if (root / "go.mod").exists():
        if install:
            setup.append(VerificationCommand(name="go-download", command="go mod download", kind="setup"))
        checks.extend(
            [
                VerificationCommand(name="go-test", command="go test ./...", kind="check"),
                VerificationCommand(name="go-vet", command="go vet ./...", kind="check"),
                VerificationCommand(name="go-build", command="go build ./...", kind="check"),
            ]
        )

    if (root / "Cargo.toml").exists():
        checks.append(VerificationCommand(name="rust-test", command="cargo test --all", kind="check"))
        if shutil.which("cargo") and _command_available("cargo clippy --version", cwd=root):
            checks.append(
                VerificationCommand(
                    name="rust-lint",
                    command="cargo clippy --all-targets -- -D warnings",
                    kind="check",
                )
            )
        checks.append(VerificationCommand(name="rust-build", command="cargo build --all", kind="check"))

    return VerificationPlan(setup=setup, checks=_dedupe_commands(checks))


def run_verification(
    root: Path,
    plan: VerificationPlan,
    *,
    timeout: int = 180,
    setup_timeout: int = 600,
    run_setup: bool = True,
    print_progress: bool = True,
) -> VerificationReport:
    setup_results: list[CommandResult] = []
    check_results: list[CommandResult] = []

    root = root.resolve()
    with _verification_execution_root(root, plan) as execution_root:
        if print_progress and not plan.setup and not plan.checks:
            warn("No verification commands to run.")

        if run_setup:
            for index, command in enumerate(plan.setup, start=1):
                if print_progress:
                    info(f"[setup {index}/{len(plan.setup)}] {command.name}", indent=2)
                    info(f"$ {command.command}", indent=4)
                result = run_command(execution_root, command, default_timeout=setup_timeout)
                setup_results.append(result)
                if print_progress:
                    if result.ok:
                        ok(f"setup {command.name} passed ({format_duration(result.seconds)})")
                    else:
                        fail(f"setup {command.name} failed (exit {result.returncode}, {format_duration(result.seconds)})")
                        print_output_tail(result.output)
                if not result.ok:
                    return VerificationReport(root, plan, setup_results, check_results)

        for index, command in enumerate(plan.checks, start=1):
            if print_progress:
                info(f"[check {index}/{len(plan.checks)}] {command.name}", indent=2)
                info(f"$ {command.command}", indent=4)
            result = run_command(execution_root, command, default_timeout=timeout)
            check_results.append(result)
            if print_progress:
                if result.ok:
                    ok(f"check {command.name} passed ({format_duration(result.seconds)})")
                else:
                    fail(f"check {command.name} failed (exit {result.returncode}, {format_duration(result.seconds)})")
                    print_output_tail(result.output)

    return VerificationReport(root, plan, setup_results, check_results)


@contextmanager
def _verification_execution_root(root: Path, plan: VerificationPlan):
    if not _path_has_hidden_part(root):
        yield root
        return

    with TemporaryDirectory(prefix="llm-compress-verify-") as tmp:
        staged = Path(tmp) / "repo"
        shutil.copytree(root, staged, symlinks=True, ignore=_verification_copy_ignore(plan))
        yield staged


def _path_has_hidden_part(path: Path) -> bool:
    return any(part.startswith(".") and part not in {".", ".."} for part in path.parts)


def _verification_copy_ignore(plan: VerificationPlan):
    setup_names = {command.name for command in plan.setup}
    dependency_dirs = set()
    if "node-install" in setup_names:
        dependency_dirs.add("node_modules")
    if "python-install" in setup_names:
        dependency_dirs.update({".llm-compress-venv", ".venv", "venv", "env", ".tox", ".nox"})
    dependency_dirs.update({".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in dependency_dirs or name.endswith(".egg-info")}

    return ignore


def run_command(root: Path, command: VerificationCommand, *, default_timeout: int) -> CommandResult:
    env = os.environ.copy()
    env.setdefault("CI", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    timeout = command.timeout or default_timeout
    start = time.time()
    try:
        completed = subprocess.run(
            command.command,
            cwd=root,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=env,
        )
        output = completed.stdout or ""
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = _to_text(exc.stdout)
        stderr = _to_text(exc.stderr)
        output = stdout + stderr
        returncode = 124
        timed_out = True
    failed_tests, total_tests = _test_counts_from_output(command, output, returncode=returncode, timed_out=timed_out)
    return CommandResult(
        name=command.name,
        command=command.command,
        kind=command.kind,
        returncode=returncode,
        seconds=time.time() - start,
        output=output[-50_000:],
        timed_out=timed_out,
        failed_test_count=failed_tests,
        total_test_count=total_tests,
    )


def _python_install_command(root: Path, *, host_python: str, py: str) -> str:
    commands = [
        "rm -rf .llm-compress-venv",
        f"{host_python} -m venv .llm-compress-venv",
        f"{py} -m pip install -U pip setuptools wheel >/dev/null",
        f"{py} -m pip install -e .",
        *_python_test_dependency_install_commands(root, py),
        f"{py} -m pip install pytest build ruff >/dev/null",
    ]
    return " && ".join(commands)


def _python_test_dependency_install_commands(root: Path, py: str) -> list[str]:
    commands: list[str] = []
    extras = _pyproject_section_keys(root, "project.optional-dependencies")
    for extra in ("test", "tests"):
        if extra in extras:
            commands.append(f"{py} -m pip install -e '.[{extra}]'")

    groups = _pyproject_section_keys(root, "dependency-groups")
    for group in ("test", "tests"):
        if group in groups:
            commands.append(f"{py} -m pip install --group {group}")

    if not commands:
        commands.extend(
            [
                f"({py} -m pip install -e '.[test]' || true)",
                f"({py} -m pip install -e '.[tests]' || true)",
            ]
        )
    return commands


def _pyproject_section_keys(root: Path, section: str) -> set[str]:
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return set()
    try:
        text = pyproject.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return set()

    keys: set[str] = set()
    in_section = False
    section_header = f"[{section}]"
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line == section_header
            continue
        if not in_section:
            continue
        match = re.match(r"[\"']?([A-Za-z0-9_-]+)[\"']?\s*=", line)
        if match:
            keys.add(match.group(1))
    return keys


def _node_package_manager(root: Path) -> str:
    if (root / "pnpm-lock.yaml").exists() and shutil.which("pnpm"):
        return "pnpm"
    if (root / "yarn.lock").exists() and shutil.which("yarn"):
        return "yarn"
    if (root / "bun.lockb").exists() and shutil.which("bun"):
        return "bun"
    return "npm"


def _node_install_command(root: Path, package_manager: str) -> str:
    runner = _node_runner(package_manager)
    if package_manager == "pnpm":
        return f"{runner} install --frozen-lockfile"
    if package_manager == "yarn":
        return f"{runner} install --frozen-lockfile"
    if package_manager == "bun":
        return f"{runner} install --frozen-lockfile"
    return f"{runner} ci" if (root / "package-lock.json").exists() else f"{runner} install"


def _node_run_command(package_manager: str, script: str) -> str:
    runner = _node_runner(package_manager)
    if package_manager == "npm":
        return f"{runner} test" if script == "test" else f"{runner} run {script}"
    return f"{runner} {script}"


def _node_runner(package_manager: str) -> str:
    override = os.environ.get("LLM_COMPRESS_NODE_RUNNER")
    if override:
        return override
    if package_manager == "npm" and _node_major_version() > 22 and shutil.which("npx"):
        return "npx -y -p node@22 -p npm@10 npm"
    return package_manager


def _node_major_version() -> int:
    try:
        completed = subprocess.run(
            ["node", "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    match = re.match(r"v?(\d+)", completed.stdout.strip())
    return int(match.group(1)) if match else 0


def _package_scripts(package_json: Path) -> dict[str, str]:
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    scripts = data.get("scripts", {})
    return scripts if isinstance(scripts, dict) else {}


def _is_python_project(root: Path) -> bool:
    return any((root / name).exists() for name in ("pyproject.toml", "setup.py", "setup.cfg"))


def _has_python_tests(root: Path) -> bool:
    return any((root / name).exists() for name in ("tests", "test")) or any(root.glob("test_*.py"))


def _has_ruff_config(root: Path) -> bool:
    if (root / "ruff.toml").exists() or (root / ".ruff.toml").exists():
        return True
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return False
    try:
        return "[tool.ruff" in pyproject.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False


def _dedupe_commands(commands: list[VerificationCommand]) -> list[VerificationCommand]:
    seen: set[str] = set()
    result: list[VerificationCommand] = []
    for command in commands:
        key = command.command
        if key not in seen:
            seen.add(key)
            result.append(command)
    return result


def _command_available(command: str, *, cwd: Path) -> bool:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _test_counts_from_output(
    command: VerificationCommand,
    output: str,
    *,
    returncode: int,
    timed_out: bool,
) -> tuple[int | None, int | None]:
    if not _is_test_command(command):
        return None, None

    failed, total = _parse_test_summary_counts(output)
    if failed is None:
        failed = 0 if returncode == 0 and not timed_out else 1
    return failed, total


def _is_test_command(command: VerificationCommand) -> bool:
    text = f"{command.name} {command.command}".lower()
    return "test" in text or "pytest" in text or "cargo test" in text or "go test" in text


def _parse_test_summary_counts(output: str) -> tuple[int | None, int | None]:
    text = output.replace("\r", "")
    lower = text.lower()

    # pytest/unittest: "2 failed, 3 passed ..." or "FAILED (failures=2, errors=1)".
    unittest_match = re.search(r"failed \(([^)]*)\)", lower)
    if unittest_match:
        details = unittest_match.group(1)
        failures = _named_count(details, "failures") + _named_count(details, "errors")
        ran = re.search(r"ran (\d+) tests?", lower)
        return failures, int(ran.group(1)) if ran else None

    # Go test does not print a final granular count; count failed test headings.
    go_failures = re.findall(r"(?m)^--- fail: ", lower)
    if go_failures:
        return len(go_failures), None

    # Rust cargo can print one result line per test binary; sum all of them.
    rust_failures = 0
    rust_total = 0
    rust_seen = False
    for line in lower.splitlines():
        if "test result:" not in line:
            continue
        counts = _counts_from_summary_line(line)
        if counts:
            rust_seen = True
            rust_failures += counts[0]
            rust_total += counts[1] or 0
    if rust_seen:
        return rust_failures, rust_total or None

    # Jest: "Tests: 2 failed, 3 passed, 5 total".
    for line in reversed(lower.splitlines()[-80:]):
        if "tests:" in line:
            counts = _counts_from_summary_line(line)
            if counts:
                return counts

    # Generic summaries from pytest/vitest/mocha and similar tools.
    for line in reversed(lower.splitlines()[-80:]):
        if not any(word in line for word in ("failed", "failure", "failing", "error")):
            continue
        if not any(word in line for word in ("passed", "passing", "total", "test", "tests")):
            continue
        counts = _counts_from_summary_line(line)
        if counts:
            return counts

    return None, None


def _counts_from_summary_line(line: str) -> tuple[int, int | None] | None:
    counts: dict[str, int] = {}
    labels = "failed|failing|failures|failure|errors|error|passed|passing|skipped|ignored|pending|total|tests|test"
    for match in re.finditer(rf"(\d+)\s+({labels})\b", line):
        label = match.group(2)
        counts[label] = counts.get(label, 0) + int(match.group(1))

    failed = 0
    seen_failure_label = False
    for label, value in counts.items():
        if label in {"failed", "failing", "failures", "failure", "errors", "error"}:
            failed += value
            seen_failure_label = True
    if not seen_failure_label:
        return None

    total = None
    for label, value in counts.items():
        if label in {"total", "tests", "test"}:
            total = value
            break
    if total is None:
        passed = sum(counts.get(label, 0) for label in ("passed", "passing"))
        skipped = sum(counts.get(label, 0) for label in ("skipped", "ignored", "pending"))
        if passed or skipped or failed:
            total = passed + skipped + failed
    return failed, total


def _named_count(text: str, name: str) -> int:
    match = re.search(rf"{re.escape(name)}=(\d+)", text)
    return int(match.group(1)) if match else 0


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
