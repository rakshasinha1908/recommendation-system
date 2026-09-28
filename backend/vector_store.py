"""
Vector store service - interface with Milvus database
Handles inserting and searching embeddings
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np
from typing import List, Tuple
import time

class MilvusVectorStore:
    """Interface to Milvus vector database"""
    
    def __init__(self, host: str = "localhost", port: int = 19530):
        """
        Connect to Milvus
        
        Args:
            host: Milvus server hostname
            port: Milvus server port
        """
        self.host = host
        self.port = port
        self.collection_name = "products"
        
        print(f"🔗 Connecting to Milvus at {host}:{port}...")
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=host,
            port=port,
            db_name="default"
        )
        
        print("✅ Connected to Milvus")
    
    def create_collection(self, dimension: int = 384):
        """
        Create collection for product embeddings
        
        Args:
            dimension: Embedding dimension (384 for our model)
        """
        print(f"\n📋 Creating collection '{self.collection_name}'...")
        
        # Check if collection already exists
        try:
            collection = Collection(self.collection_name)
            print(f"✅ Collection '{self.collection_name}' already exists")
            return collection
        except:
            pass
        
        # Define schema
        fields = [
            FieldSchema(
                name="item_id",
                dtype=DataType.INT64,
                is_primary=True,
                description="Product ID from PostgreSQL"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dimension,
                description="Product description embedding"
            ),
            FieldSchema(
                name="category",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Product category"
            )
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Product embeddings for semantic search"
        )
        
        # Create collection
        collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        print(f"✅ Collection '{self.collection_name}' created")
        return collection
    
    def insert_embeddings(self, item_ids: List[int], embeddings: np.ndarray, categories: List[str] = None):
        """
        Insert product embeddings into Milvus
        
        Args:
            item_ids: Product IDs from PostgreSQL
            embeddings: Embedding vectors (N x dimension)
            categories: Product categories
        """
        print(f"\n📥 Inserting {len(item_ids)} embeddings into Milvus...")
        
        collection = Collection(self.collection_name)
        
        # Prepare data
        if categories is None:
            categories = ["unknown"] * len(item_ids)
        
        data = [
            item_ids,
            embeddings.tolist(),  # Convert numpy array to list
            categories
        ]
        
        # Insert data
        start_time = time.time()
        result = collection.insert(data)
        insert_time = time.time() - start_time
        
        print(f"✅ Inserted {len(item_ids)} embeddings in {insert_time:.2f}s")
        print(f"   Insertion IDs: {len(result.primary_keys)}")
        
        # Create index for fast search
        self.create_index()
    
    def create_index(self):
        """Create index for fast vector search"""
        print("\n🔧 Creating search index...")
        
        collection = Collection(self.collection_name)
        
        # Check if index already exists
        if len(collection.indexes) > 0:
            print("✅ Index already exists")
            return
        
        # Create IVF_FLAT index (good balance of speed/accuracy)
        index_params = {
            "metric_type": "L2",  # Euclidean distance
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}  # Number of clusters
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        print("✅ Index created successfully")
    
    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for similar products
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (item_id, distance) tuples, sorted by similarity
        """
        collection = Collection(self.collection_name)
        
        # Load collection into memory for search
        collection.load()
        
        # Search
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["item_id", "category"]
        )
        
        # Extract results
        hits = results[0]
        results_list = []
        
        for hit in hits:
            item_id = hit.entity.get("item_id")
            distance = hit.distance
            results_list.append((item_id, distance))
        
        return results_list
    
    def get_stats(self) -> dict:
        """Get collection statistics"""
        collection = Collection(self.collection_name)
        
        return {
            "name": collection.name,
            "num_entities": collection.num_entities,
            "primary_field": "item_id",
            "embedding_dim": 384,
            "indexed": len(collection.indexes) > 0
        }


# Test script
if __name__ == "__main__":
    # Connect and create collection
    store = MilvusVectorStore(host="localhost", port=19530)
    store.create_collection(dimension=384)
    
    # Show stats
    stats = store.get_stats()
    print("\n📊 Collection Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")