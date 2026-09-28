from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from embedding_service import EmbeddingService
from vector_store import MilvusVectorStore
from models import engine, Base, Item  # ADD THIS

load_dotenv()

Base.metadata.create_all(bind=engine)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(
    title="Recommendation Engine API",
    description="E-Commerce Recommendation System",
    version="0.1.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_service = EmbeddingService()
vector_store = MilvusVectorStore()

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Recommendation Engine API",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": os.getenv("ENVIRONMENT", "development")
    }
    
@app.get("/items/search/similar/{item_id}")
def find_similar_items(item_id: int, top_k: int = 5, db: Session = Depends(get_db)):
    """
    Find products similar to a given product using semantic similarity
    
    Args:
        item_id: Product ID to find similar products for
        top_k: Number of similar products to return
        
    Returns:
        List of similar products with similarity scores
    """
    # Get the product
    item = db.query(Item).filter(Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Generate embedding for this product
    product_text = f"{item.title}. {item.description}" if item.description else item.title
    embedding = embedding_service.embed(product_text)
    
    # Search for similar products in Milvus
    results = vector_store.search(embedding, top_k=top_k)
    
    # Format results
    similar_items = []
    for result_item_id, distance in results:
        if result_item_id == item_id:  # Skip the product itself
            continue
        
        similar_item = db.query(Item).filter(Item.id == result_item_id).first()
        if similar_item:
            # Convert distance to similarity (0-1)
            similarity_score = 1 / (1 + distance)
            
            similar_items.append({
                "id": similar_item.id,
                "title": similar_item.title,
                "price": similar_item.price,
                "category": similar_item.category,
                "similarity_score": round(similarity_score, 3),
                "distance": round(distance, 4)
            })
    
    return {
        "product": {
            "id": item.id,
            "title": item.title,
            "price": item.price
        },
        "similar_products": similar_items[:top_k]
    }


@app.post("/search/semantic")
def semantic_search(query: str, top_k: int = 5, db: Session = Depends(get_db)):
    """
    Search for products by natural language query
    
    Example:
    {
        "query": "fast laptop for programming",
        "top_k": 5
    }
    """
    # Generate embedding for query
    embedding = embedding_service.embed(query)
    
    # Search in Milvus
    results = vector_store.search(embedding, top_k=top_k)
    
    # Format results
    found_items = []
    for item_id, distance in results:
        item = db.query(Item).filter(Item.id == item_id).first()
        if item:
            similarity_score = 1 / (1 + distance)
            found_items.append({
                "id": item.id,
                "title": item.title,
                "description": item.description,
                "price": item.price,
                "category": item.category,
                "similarity_score": round(similarity_score, 3)
            })
    
    return {
        "query": query,
        "results": found_items
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)