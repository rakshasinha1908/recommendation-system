"""
Generate embeddings for all products in database
Run this once to create embeddings for entire product catalog
"""

from embedding_service import EmbeddingService
from models import Item, get_db
import numpy as np
import json
from pathlib import Path

def generate_product_embeddings():
    """Generate embeddings for all products"""
    
    print("\n" + "="*60)
    print("🚀 PRODUCT EMBEDDING PIPELINE")
    print("="*60)
    
    # Initialize embedding service
    service = EmbeddingService()
    
    # Get all products from database
    print("\n📦 Loading products from database...")
    db = SessionLocal()
    items = db.query(Item).all()
    db.close()
    
    if not items:
        print("❌ No products found in database")
        return
    
    print(f"✅ Loaded {len(items)} products")
    
    # Combine title + description for better embeddings
    # (more context = better semantic representation)
    texts = []
    item_ids = []
    
    for item in items:
        # Combine title and description for richer context
        text = f"{item.title}. {item.description}" if item.description else item.title
        texts.append(text)
        item_ids.append(item.id)
    
    # Generate embeddings
    print("\n🔧 Generating embeddings...")
    embeddings = service.embed_batch(texts)
    
    print(f"\n✅ Generated {len(embeddings)} embeddings")
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Memory usage: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    # Create embedding metadata
    embedding_data = {
        "item_ids": item_ids,
        "embeddings": embeddings,
        "metadata": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "num_items": len(items),
            "product_count": len(items)
        }
    }
    
    # Save embeddings
    service.save_embeddings(embeddings)
    
    # Save metadata
    metadata_path = Path("./embeddings_cache/metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump({
            "item_ids": item_ids,
            "model": embedding_data["metadata"]["model"],
            "dimension": embedding_data["metadata"]["dimension"],
            "num_items": len(item_ids)
        }, f, indent=2)
    
    print(f"📄 Metadata saved to {metadata_path}")
    
    # Show sample embeddings
    print("\n📊 Sample embeddings:")
    for i in range(min(3, len(items))):
        item = items[i]
        embedding = embeddings[i]
        print(f"\n   Product #{item.id}: {item.title}")
        print(f"   Price: ${item.price}")
        print(f"   Embedding: {embedding[:10]}... (showing first 10 of {len(embedding)} dims)")
        print(f"   L2 Norm: {np.linalg.norm(embedding):.4f}")
    
    print("\n" + "="*60)
    print("✨ EMBEDDING GENERATION COMPLETE!")
    print("="*60)
    print(f"\n✅ {len(embeddings)} embeddings ready for Milvus")
    print("   Next: Run Week2_3_milvus_setup.py to load into Milvus")


if __name__ == "__main__":
    from models import SessionLocal
    generate_product_embeddings() 