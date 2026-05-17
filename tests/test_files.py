import unittest

from llm_compress.files import is_test_path, should_use_llm


class FileClassificationTests(unittest.TestCase):
    def test_test_paths_are_detected(self):
        self.assertTrue(is_test_path("./tests/test_app.py"))
        self.assertTrue(is_test_path("./src/foo.test.ts"))
        self.assertTrue(is_test_path("./pkg/router_test.go"))
        self.assertFalse(is_test_path("./src/app.py"))

    def test_critical_config_is_lossless(self):
        use_llm, reason = should_use_llm("./package.json", b'{"scripts":{}}', max_llm_bytes=20_000)
        self.assertFalse(use_llm)
        self.assertEqual(reason, "critical_config")

    def test_source_file_can_use_llm(self):
        use_llm, reason = should_use_llm("./src/app.py", b"def f():\n    return 1\n", max_llm_bytes=20_000)
        self.assertTrue(use_llm)
        self.assertEqual(reason, "source_llm")


if __name__ == "__main__":
    unittest.main()
