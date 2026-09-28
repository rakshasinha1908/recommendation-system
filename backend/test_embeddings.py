"""
Quick test of sentence embeddings
Run this to understand how embeddings work
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download model (first time only)
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Test sentences
sentences = [
    "High-quality wireless headphones with noise cancellation",
    "Premium earbuds with active noise canceling",
    "Gaming laptop with RTX graphics card",
    "Best earphones for music lovers"
]

print("\nEncoding sentences...")
embeddings = model.encode(sentences)

print(f"\nGenerated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")
print(f"Shape: {embeddings.shape}")

# Similarity analysis
sim = cosine_similarity(embeddings)

print("\nSimilarity Matrix:")
print("Higher = more similar\n")

print("   " + " ".join([f"Sent{i}" for i in range(len(sentences))]))

for i, row in enumerate(sim):
    print(f"Sent{i} {' '.join([f'{s:.2f}' for s in row])}")

print("\nNotice:")
print("Sentences 0 & 1: Similar products")
print("Sentences 0 & 2: Different products")
print("This is why embeddings work!")
