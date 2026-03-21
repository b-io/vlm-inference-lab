import math
from typing import List

class EmbeddingUtils:
    """Utilities for working with embeddings (cosine similarity, normalization)."""
    
    @staticmethod
    def dot_product(v1: List[float], v2: List[float]) -> float:
        return sum(x * y for x, y in zip(v1, v2))

    @staticmethod
    def norm(v: List[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    @classmethod
    def cosine_similarity(cls, v1: List[float], v2: List[float]) -> float:
        n1 = cls.norm(v1)
        n2 = cls.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return cls.dot_product(v1, v2) / (n1 * n2)

def demo():
    v_text = [0.1, 0.5, -0.2]
    v_img_1 = [0.12, 0.48, -0.18]  # Similar
    v_img_2 = [-0.5, 0.1, 0.8]    # Dissimilar
    
    sim1 = EmbeddingUtils.cosine_similarity(v_text, v_img_1)
    sim2 = EmbeddingUtils.cosine_similarity(v_text, v_img_2)
    
    print(f"Cosine Similarity (Text vs Img 1): {sim1:.4f}")
    print(f"Cosine Similarity (Text vs Img 2): {sim2:.4f}")

if __name__ == "__main__":
    demo()
