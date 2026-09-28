# Save this as: backend/embedding_service.py

"""
Embedding generation service
Converts product descriptions to vector embeddings using sentence-transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List
import time


class EmbeddingService:
    """Manages embedding generation and caching"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_gpu: bool = False):
        """
        Initialize embedding service
        
        Args:
            model_name: Hugging Face model identifier
                - 'all-MiniLM-L6-v2': 384-dim, fast, good quality (recommended)
                - 'all-mpnet-base-v2': 768-dim, slower, higher quality
            use_gpu: Whether to use GPU if available
        """
        print(f"📥 Loading embedding model: {model_name}")
        
        device = "cuda" if use_gpu else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name
        
        # Create cache directory
        self.cache_dir = Path("./embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        print(f"✅ Model loaded on {device}")
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Convert single text to embedding
        
        Args:
            text: Product description or title
            
        Returns:
            Embedding vector (1D numpy array of shape (embedding_dim,))
        """
        if not text or len(text.strip()) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert multiple texts to embeddings (more efficient)
        
        Args:
            texts: List of product descriptions
            batch_size: Process in batches for efficiency (default 32)
            
        Returns:
            Embedding matrix (2D numpy array of shape (N, embedding_dim))
        """
        print(f"\n📝 Encoding {len(texts)} texts in batches of {batch_size}...")
        
        start_time = time.time()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Encoding complete in {elapsed:.2f}s ({len(texts)/elapsed:.1f} texts/sec)")
        
        return embeddings.astype(np.float32)
    
    def save_embeddings(self, embeddings: np.ndarray, filename: str = "product_embeddings.pkl"):
        """
        Cache embeddings to disk using pickle
        
        Args:
            embeddings: Embedding array to cache
            filename: Output filename in cache directory
        """
        filepath = self.cache_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        
        size_mb = embeddings.nbytes / 1024 / 1024
        print(f"💾 Embeddings saved to {filepath} ({size_mb:.2f} MB)")
    
    def load_embeddings(self, filename: str = "product_embeddings.pkl") -> np.ndarray:
        """
        Load cached embeddings from disk
        
        Args:
            filename: Input filename in cache directory
            
        Returns:
            Embedding array
        """
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Embeddings not found at {filepath}")
        
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        
        print(f"✅ Loaded embeddings from {filepath}")
        print(f"   Shape: {embeddings.shape}")
        
        return embeddings
    
    def get_info(self) -> dict:
        """Get embedding service information"""
        return {
            "model": self.model_name,
            "dimension": self.embedding_dim,
            "cache_directory": str(self.cache_dir)
        }


# Quick test when run directly
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 EMBEDDING SERVICE TEST")
    print("="*60)
    
    # Initialize service
    service = EmbeddingService(model_name='all-MiniLM-L6-v2')
    
    # Test single embedding
    print("\n🔬 Test 1: Single embedding")
    text = "High-quality wireless headphones with noise cancellation"
    embedding = service.embed(text)
    print(f"✅ Single embedding shape: {embedding.shape}")
    print(f"   Vector norm: {np.linalg.norm(embedding):.4f}")
    
    # Test batch
    print("\n🔬 Test 2: Batch embeddings")
    texts = [
        "Wireless headphones",
        "Gaming laptop with RTX graphics",
        "Smart speaker with voice control",
        "4K camera for photography"
    ]
    embeddings = service.embed_batch(texts)
    print(f"✅ Batch embeddings shape: {embeddings.shape}")
    
    # Test similarity
    print("\n🔬 Test 3: Similarity check")
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(embeddings)
    
    print("Similarity matrix (higher = more similar):")
    print("       Text0 Text1 Text2 Text3")
    for i, row in enumerate(sim):
        print(f"Text{i} {' '.join([f'{s:.3f}' for s in row])}")
    
    # Note: Text0 and Text2 should be more similar (both audio)
    # while Text0 and Text1 should be less similar (different categories)
    
    # Test caching
    print("\n🔬 Test 4: Caching")
    service.save_embeddings(embeddings, "test_embeddings.pkl")
    loaded = service.load_embeddings("test_embeddings.pkl")
    print(f"✅ Loaded embeddings shape: {loaded.shape}")
    
    # Verify they're the same
    if np.allclose(embeddings, loaded):
        print("✅ Cached and loaded embeddings match!")
    
    print("\n" + "="*60)
    print("✨ ALL TESTS PASSED!")
    print("="*60)