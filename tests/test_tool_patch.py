import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_compress.tool_patch import apply_code_patch, build_tool_repo_context, validate_code_patch


class ToolPatchTests(unittest.TestCase):
    def test_build_context_exposes_tree_and_selected_files(self):
        context = build_tool_repo_context(max_chars=2000, max_tree_entries=20)
        self.assertTrue(context["patch_capability"]["enabled"])
        self.assertTrue(context["tree"])
        self.assertTrue(context["selected_files"])

    def test_validate_rejects_unsafe_paths(self):
        patch = """diff --git a/../bad.py b/../bad.py
--- a/../bad.py
+++ b/../bad.py
@@ -1 +1 @@
-a
+b
"""
        with self.assertRaises(ValueError):
            validate_code_patch(patch)

    def test_apply_code_patch_to_temp_repo(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            target = root / "src" / "x.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            patch = """diff --git a/src/x.py b/src/x.py
--- a/src/x.py
+++ b/src/x.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""
            apply_code_patch(root, patch)
            self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 2\n")

    def test_apply_code_patch_accepts_relative_repo_root(self):
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            target = root / "src" / "x.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            patch = """diff --git a/src/x.py b/src/x.py
--- a/src/x.py
+++ b/src/x.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""
            apply_code_patch(root.relative_to(Path.cwd()), patch)
            self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 2\n")


if __name__ == "__main__":
    unittest.main()
