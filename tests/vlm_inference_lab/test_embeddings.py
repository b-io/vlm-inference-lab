import unittest
from vlm_inference_lab.embeddings.demo import generate_mock_embedding
from vlm_inference_lab.embeddings.utils import EmbeddingUtils

class TestEmbeddings(unittest.TestCase):
    def test_unit_length(self):
        dim = 64
        vec = generate_mock_embedding(dim)
        self.assertEqual(len(vec), dim)
        norm = sum(x*x for x in vec)**0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_cosine_similarity_identity(self):
        vec = [1.0, 0.0, 0.0]
        sim = EmbeddingUtils.cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        sim = EmbeddingUtils.cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 0.0, places=5)

if __name__ == "__main__":
    unittest.main()
