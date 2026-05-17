import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_compress.artifact import encode_lossless, sha256_bytes, write_artifact
from llm_compress.decompressor import Decompressor


class DecompressorTests(unittest.TestCase):
    def test_decompress_lossless_artifact(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data = b"print('ok')\n"
            artifact = tmp_path / "compressed.jsonl"
            write_artifact(
                artifact,
                {"target": "unit"},
                [
                    {"type": "empty_dir", "path": "./empty"},
                    {
                        "type": "file",
                        "path": "./src/app.py",
                        "mode": "lossless",
                        "reason": "unit",
                        "size": len(data),
                        "sha256": sha256_bytes(data),
                        "data": encode_lossless(data),
                    },
                ],
            )

            out = tmp_path / "out"
            summary = Decompressor(None).decompress_artifact(artifact, out)
            self.assertTrue(summary.ok)
            self.assertEqual((out / "src/app.py").read_bytes(), data)
            self.assertTrue((out / "empty").is_dir())


if __name__ == "__main__":
    unittest.main()
