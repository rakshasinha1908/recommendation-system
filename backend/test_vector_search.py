"""
Test vector similarity search
Demonstrates how to find similar products
"""

import time
from vector_store import MilvusVectorStore
from embedding_service import EmbeddingService
from models import Item, SessionLocal

def test_search():
    """Test similarity search"""
    print("\n" + "="*60)
    print("🔍 TESTING VECTOR SIMILARITY SEARCH")
    print("="*60)
    
    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = MilvusVectorStore()
    
    # Get database session
    db = SessionLocal()
    
    # Test with different queries
    test_queries = [
        "wireless headphones with noise cancellation",
        "gaming laptop high performance",
        "smart home speaker with voice control"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        print("-" * 60)
        
        # Generate embedding for query
        query_embedding = embedding_service.embed(query)
        
        # Search in Milvus
        start_time = time.time()
        results = vector_store.search(query_embedding, top_k=5)
        search_time = time.time() - start_time
        
        print(f"⏱️  Search time: {search_time*1000:.2f}ms")
        print(f"\nTop 5 similar products:")
        
        for rank, (item_id, distance) in enumerate(results, 1):
            item = db.query(Item).filter(Item.id == item_id).first()
            
            # Convert L2 distance to similarity score (0-1, higher = more similar)
            similarity = 1 / (1 + distance)
            
            print(f"\n   {rank}. [{item.id}] {item.title}")
            print(f"      Category: {item.category}")
            print(f"      Price: ${item.price}")
            print(f"      Similarity: {similarity:.3f}")
            print(f"      Distance: {distance:.4f}")
    
    db.close()
    
    print("\n" + "="*60)
    print("✅ VECTOR SEARCH TEST COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    test_search()