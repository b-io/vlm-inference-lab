import unittest
from vlm_inference_lab.embeddings.utils import EmbeddingUtils


class TestEmbeddingUtils(unittest.TestCase):
    def test_cosine_similarity(self):
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        self.assertAlmostEqual(EmbeddingUtils.cosine_similarity(v1, v2), 1.0)

        v3 = [0.0, 1.0]
        self.assertAlmostEqual(EmbeddingUtils.cosine_similarity(v1, v3), 0.0)

        v4 = [-1.0, 0.0]
        self.assertAlmostEqual(EmbeddingUtils.cosine_similarity(v1, v4), -1.0)

    def test_norm(self):
        self.assertEqual(EmbeddingUtils.norm([3.0, 4.0]), 5.0)


if __name__ == "__main__":
    unittest.main()
