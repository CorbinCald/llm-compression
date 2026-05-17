import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_compress.artifact import read_artifact
from llm_compress.compressor import CompressionOptions, Compressor


class FakeClient:
    available = True
    model = "fake/model"

    def chat(self, *, system: str, user: str, max_completion_tokens: int | None = None) -> str:
        return "<-Name:app|Input:none|Return:def f returns 1|Path:./src/app.py|Order:1a->"


class CompressorTests(unittest.TestCase):
    def test_lossless_overrides_are_ignored_for_source_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            source = root / "src" / "app.py"
            artifact = Path(temp) / "compressed.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text("def f():\n    return 1\n", encoding="utf-8")

            Compressor(FakeClient()).compress_repo(
                root,
                artifact,
                options=CompressionOptions(lossless_overrides={"./src/app.py"}),
                target_label="unit",
            )

            _, records = read_artifact(artifact)
            self.assertEqual(records[0]["path"], "./src/app.py")
            self.assertEqual(records[0]["mode"], "llm")


if __name__ == "__main__":
    unittest.main()
