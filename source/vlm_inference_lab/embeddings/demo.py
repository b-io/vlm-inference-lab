import random
from typing import List, Tuple
from vlm_inference_lab.embeddings.utils import EmbeddingUtils


def generate_mock_embedding(dim: int = 128) -> List[float]:
    """Generates a random unit-length embedding vector."""
    # Create a vector with random values between -1 and 1
    vec = [random.uniform(-1, 1) for _ in range(dim)]
    # Calculate the Euclidean norm of the vector
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        # Return a zero vector if the norm is zero
        return [0.0] * dim
    # Normalize the vector to unit length
    return [x / norm for x in vec]


def run_vlm_demo():
    """Simulates image-text alignment using mock embeddings to illustrate a dual-encoder architecture."""
    dim = 128
    # Set seed for reproducibility
    random.seed(42)

    # 1. Generate a 'text' embedding for a query
    query_text = "a cat sitting on a mat"
    text_embedding = generate_mock_embedding(dim)

    # 2. Generate several 'image' embeddings
    # In a real VLM, these would come from a Vision Encoder (e.g., ViT)
    images = [("image_dog.jpg", generate_mock_embedding(dim)),
            # Simulate a close match by adding small noise to the text embedding
            ("image_cat.jpg", [x + random.uniform(-0.05, 0.05) for x in text_embedding]),
            ("image_car.jpg", generate_mock_embedding(dim)), ("image_forest.jpg", generate_mock_embedding(dim)), ]

    print(f"--- VLM Cross-Modal Alignment Demo ---")
    print(f"Concept: Dual-Encoder Retrieval (CLIP-style)")
    print(f"Query Text: '{query_text}'\n")
    print(f"{'Image Candidate':20} | {'Cosine Similarity':18} | {'Match'}")
    print("-" * 60)

    results = []
    for name, img_emb in images:
        # Re-normalize just in case noise was added
        norm = sum(x * x for x in img_emb) ** 0.5
        img_emb = [x / norm for x in img_emb]

        # Calculate cosine similarity between text and image embeddings
        similarity = EmbeddingUtils.cosine_similarity(text_embedding, img_emb)
        is_match = "YES" if similarity > 0.9 else "NO"
        results.append((name, similarity))
        print(f"{name:20} | {similarity:18.4f} | {is_match}")

    # Identify the best matching candidate
    best_match = max(results, key=lambda x: x[1])
    print(f"\nBest match: {best_match[0]} (Similarity: {best_match[1]:.4f})")
    print("\nNote: In generative VLMs (like LLaVA), visual tokens are often ")
    print("concatenated with text tokens rather than compared via cosine similarity.")


if __name__ == "__main__":
    run_vlm_demo()
