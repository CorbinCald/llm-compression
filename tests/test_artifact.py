import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_compress.artifact import decode_lossless, encode_lossless, read_artifact, write_artifact


class ArtifactTests(unittest.TestCase):
    def test_lossless_codec_round_trips_bytes(self):
        data = b"hello\nworld\x00\xff"
        self.assertEqual(decode_lossless(encode_lossless(data)), data)

    def test_artifact_round_trip(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "compressed.jsonl"
            write_artifact(artifact, {"target": "x"}, [{"type": "empty_dir", "path": "./a"}])
            meta, records = read_artifact(artifact)
            self.assertEqual(meta["format"], "llm-compress-jsonl")
            self.assertEqual(meta["target"], "x")
            self.assertEqual(records, [{"type": "empty_dir", "path": "./a"}])


if __name__ == "__main__":
    unittest.main()
