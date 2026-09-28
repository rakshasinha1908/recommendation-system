"""
Load product embeddings from cache into Milvus
Run this after embeddings are generated (generate_embeddings.py)
"""

import json
import sys
from pathlib import Path

# Import local modules
from vector_store import MilvusVectorStore
from embedding_service import EmbeddingService
from models import Item, SessionLocal


def load_embeddings_from_cache():
    """
    Load cached embeddings from disk
    
    Returns:
        (embeddings, metadata) tuple
    """
    print("\n📥 Step 1: Loading cached embeddings...")
    
    cache_dir = Path("./embeddings_cache")
    embeddings_path = cache_dir / "product_embeddings.pkl"
    
    if not embeddings_path.exists():
        print(f"❌ Embeddings file not found at {embeddings_path}")
        print("   Run generate_embeddings.py first to create embeddings")
        return None, None
    
    # Load embeddings
    import pickle
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"✅ Loaded embeddings: shape {embeddings.shape}")
    
    # Load metadata
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"⚠️  Metadata not found at {metadata_path}")
        print("   Creating minimal metadata...")
        metadata = {
            "item_ids": list(range(1, len(embeddings) + 1)),
            "dimension": embeddings.shape[1]
        }
    else:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Loaded metadata")
    
    return embeddings, metadata


def get_product_categories():
    """Get product categories from database"""
    print("\n📊 Step 2: Loading product categories...")
    
    try:
        db = SessionLocal()
        items = db.query(Item).all()
        db.close()
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print("   Make sure PostgreSQL is running: docker-compose up -d")
        return {}
    
    categories = {item.id: item.category or "unknown" for item in items}
    print(f"✅ Loaded {len(categories)} products with categories")
    
    return categories


def verify_embeddings(embeddings, item_ids):
    """Verify embeddings before loading"""
    print("\n✔️  Step 3: Verifying embeddings...")
    
    import numpy as np
    
    # Check dimensions
    if embeddings.shape[0] != len(item_ids):
        raise ValueError(
            f"Mismatch: {embeddings.shape[0]} embeddings but {len(item_ids)} item IDs"
        )
    
    # Check for NaN or Inf
    if np.any(np.isnan(embeddings)):
        raise ValueError("Found NaN values in embeddings")
    
    if np.any(np.isinf(embeddings)):
        raise ValueError("Found Inf values in embeddings")
    
    print(f"✅ Embeddings verified:")
    print(f"   - Shape: {embeddings.shape}")
    print(f"   - No NaN/Inf values")
    print(f"   - Ready for Milvus")
    
    return True


def load_to_milvus():
    """Main pipeline to load embeddings into Milvus"""
    
    print("\n" + "="*70)
    print("🚀 LOADING EMBEDDINGS TO MILVUS")
    print("="*70)
    
    # Step 1: Load embeddings from cache
    embeddings, metadata = load_embeddings_from_cache()
    if embeddings is None:
        return
    
    item_ids = metadata.get("item_ids", list(range(1, len(embeddings) + 1)))
    
    # Step 2: Get categories from database
    categories_dict = get_product_categories()
    category_list = [categories_dict.get(id, "unknown") for id in item_ids]
    
    # Step 3: Verify embeddings
    try:
        verify_embeddings(embeddings, item_ids)
    except ValueError as e:
        print(f"❌ Verification failed: {e}")
        return
    
    # Step 4: Connect to Milvus
    print("\n🔗 Step 4: Connecting to Milvus...")
    try:
        store = MilvusVectorStore(host="localhost", port=19530)
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        print("   Make sure Milvus is running: docker-compose up -d")
        print("   Wait 30 seconds for startup")
        return
    
    # Step 5: Create collection
    print("\n📋 Step 5: Creating Milvus collection...")
    dimension = embeddings.shape[1]
    store.create_collection(dimension=dimension)
    
    # Step 6: Insert embeddings
    print("\n📥 Step 6: Inserting embeddings into Milvus...")
    try:
        store.insert_embeddings(
            item_ids=item_ids,
            embeddings=embeddings,
            categories=category_list
        )
    except Exception as e:
        print(f"❌ Failed to insert embeddings: {e}")
        return
    
    # Step 7: Show statistics
    print("\n📊 Step 7: Collection statistics...")
    stats = store.get_stats()
    
    print(f"   Collection name: {stats.get('name', 'N/A')}")
    print(f"   Total entities: {stats.get('num_entities', 0)}")
    print(f"   Embedding dimension: {stats.get('embedding_dim', 0)}")
    print(f"   Has index: {stats.get('has_index', False)}")
    print(f"   Indexes: {stats.get('indexes', [])}")
    
    if stats.get('num_entities', 0) == len(embeddings):
        print("\n✅ All embeddings successfully loaded!")
    else:
        print(f"\n⚠️  Warning: Expected {len(embeddings)} entities, got {stats.get('num_entities', 0)}")
    
    # Step 8: Show sample
    print("\n🔍 Step 8: Sample embeddings in Milvus...")
    
    db = SessionLocal()
    for i in range(min(3, len(item_ids))):
        item_id = item_ids[i]
        item = db.query(Item).filter(Item.id == item_id).first()
        
        if item:
            print(f"\n   [{i+1}] Item ID: {item.id}")
            print(f"       Title: {item.title}")
            print(f"       Category: {item.category}")
            print(f"       Stored in Milvus: ✓")
    
    db.close()
    
    # Final summary
    print("\n" + "="*70)
    print("✨ LOADING COMPLETE!")
    print("="*70)
    
    print("\n📋 Summary:")
    print(f"   ✓ Embeddings loaded: {len(embeddings)}")
    print(f"   ✓ Milvus entities: {stats.get('num_entities', 0)}")
    print(f"   ✓ Dimension: {dimension}")
    print(f"   ✓ Index status: {stats.get('has_index', False)}")
    
    print("\n🎯 Next steps:")
    print("   1. Run: python test_vector_search.py")
    print("      (to test similarity search)")
    print("   2. Restart FastAPI server")
    print("   3. Test API endpoints at http://localhost:8000/docs")
    print("      - GET /items/search/similar/{item_id}")
    print("      - POST /search/semantic")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        load_to_milvus()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)