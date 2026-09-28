"""
Benchmark vector search performance
Measure search latency and throughput
"""

import time
import numpy as np
from vector_store import MilvusVectorStore
from embedding_service import EmbeddingService

def benchmark_search():
    """Run search performance benchmarks"""
    print("\n" + "="*60)
    print("⚡ VECTOR SEARCH BENCHMARKING")
    print("="*60)
    
    # Initialize
    embedding_service = EmbeddingService()
    vector_store = MilvusVectorStore()
    
    # Generate random queries
    queries = [
        "wireless headphones",
        "gaming laptop",
        "smart home device",
        "camera for photography",
        "fitness tracker"
    ]
    
    print(f"\n🔎 Running {len(queries)} search queries...\n")
    
    times = []
    
    for query in queries:
        # Generate embedding
        embedding = embedding_service.embed(query)
        
        # Time the search
        start = time.time()
        results = vector_store.search(embedding, top_k=5)
        elapsed = time.time() - start
        
        times.append(elapsed)
        
        print(f"   Query: '{query}' → {elapsed*1000:.2f}ms ({len(results)} results)")
    
    # Statistics
    print("\n📊 Performance Statistics:")
    print(f"   Total searches: {len(times)}")
    print(f"   Min latency: {min(times)*1000:.2f}ms")
    print(f"   Max latency: {max(times)*1000:.2f}ms")
    print(f"   Avg latency: {np.mean(times)*1000:.2f}ms")
    print(f"   Median latency: {np.median(times)*1000:.2f}ms")
    print(f"   Throughput: {1/np.mean(times):.1f} searches/sec")
    
    print("\n" + "="*60)
    print("✅ BENCHMARK COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    benchmark_search()