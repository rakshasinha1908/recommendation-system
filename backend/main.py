from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import logging
from contextlib import contextmanager

# === IMPORTS ===
from models import Base, engine, SessionLocal, User, Item, Interaction, Recommendation
from schemas import (
    UserCreate, UserResponse, UserUpdate,
    ItemResponse, ItemCreate,
    InteractionCreate, InteractionResponse,
    RecommendationListResponse, SearchResponse
)
from embedding_service import EmbeddingService
from vector_store import MilvusVectorStore
from recommendation_engine import RecommendationEngine

# === SETUP ===
app = FastAPI(title="E-Commerce Recommendation Engine", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "localhost", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === DATABASE & SERVICES ===

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize services at module level
embedding_service = EmbeddingService()
vector_store = MilvusVectorStore()

# Create tables
Base.metadata.create_all(bind=engine)

# Initialize recommendation engine (with a dummy session for startup)
db_startup = SessionLocal()
try:
    recommendation_engine = RecommendationEngine(embedding_service, vector_store, db_startup)
    logger.info("✅ RecommendationEngine initialized")
finally:
    db_startup.close()

# ============================================
# ROOT & HEALTH
# ============================================

@app.get("/")
def root():
    return {
        "app": "E-Commerce Recommendation Engine",
        "version": "1.0.0",
        "week": "5 - Hybrid Recommendations",
        "status": "running"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

# ============================================
# USER ENDPOINTS
# ============================================

@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    # Check if user already exists
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = User(username=user.username, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"✅ User created: {user.username} ({user.email})")
    return db_user

@app.get("/users", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db)):
    """List all users"""
    users = db.query(User).all()
    return users

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user by ID"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/users/{user_id}", response_model=UserResponse)
def update_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
    """Update user"""
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.username:
        db_user.username = user.username
    if user.email:
        # Check email not already used
        existing = db.query(User).filter(User.email == user.email, User.id != user_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already in use")
        db_user.email = user.email
    
    db.commit()
    db.refresh(db_user)
    logger.info(f"✅ User updated: {user_id}")
    return db_user

@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Delete user"""
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(db_user)
    db.commit()
    logger.info(f"✅ User deleted: {user_id}")
    return {"message": f"User {user_id} deleted"}

# ============================================
# ITEM ENDPOINTS
# ============================================

@app.get("/items", response_model=list[ItemResponse])
def list_items(db: Session = Depends(get_db)):
    """List all items"""
    items = db.query(Item).all()
    return items

@app.get("/items/{item_id}", response_model=ItemResponse)
def get_item(item_id: int, db: Session = Depends(get_db)):
    """Get item by ID"""
    item = db.query(Item).filter(Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.get("/items/category/{category}", response_model=list[ItemResponse])
def get_items_by_category(category: str, db: Session = Depends(get_db)):
    """Get items by category"""
    items = db.query(Item).filter(Item.category == category).all()
    if not items:
        raise HTTPException(status_code=404, detail=f"No items in category '{category}'")
    return items

@app.post("/items", response_model=ItemResponse)
def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    """Create a new item"""
    db_item = Item(
        title=item.title,
        description=item.description,
        price=item.price,
        category=item.category
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    logger.info(f"✅ Item created: {item.title}")
    return db_item

# ============================================
# SEARCH ENDPOINTS
# ============================================

@app.get("/items/search/similar/{item_id}", response_model=SearchResponse)
def search_similar_items(item_id: int, top_k: int = Query(5, ge=1, le=20), db: Session = Depends(get_db)):
    """Search for items similar to a given item (vector similarity)"""
    item = db.query(Item).filter(Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Generate embedding for the item
    text = f"{item.title}. {item.description}" if item.description else item.title
    embedding = embedding_service.embed(text)
    
    # Search vector store
    results = vector_store.search(embedding, top_k=top_k)
    
    similar_items = []
    for found_item_id, distance in results:
        if found_item_id != item_id:  # Exclude the query item itself
            found_item = db.query(Item).filter(Item.id == found_item_id).first()
            if found_item:
                similarity_score = 1 / (1 + distance)
                similar_items.append({
                    "item_id": found_item.id,
                    "title": found_item.title,
                    "price": found_item.price,
                    "category": found_item.category,
                    "similarity_score": round(similarity_score, 3)
                })
    
    return {
        "query_item_id": item_id,
        "query_item_title": item.title,
        "total_results": len(similar_items),
        "items": similar_items
    }

@app.post("/search/semantic", response_model=SearchResponse)
def semantic_search(query: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=20), db: Session = Depends(get_db)):
    """Semantic search using NLP embeddings"""
    # Generate embedding for query
    embedding = embedding_service.embed(query)
    
    # Search vector store
    results = vector_store.search(embedding, top_k=top_k)
    
    items = []
    for item_id, distance in results:
        item = db.query(Item).filter(Item.id == item_id).first()
        if item:
            similarity_score = 1 / (1 + distance)
            items.append({
                "item_id": item.id,
                "title": item.title,
                "price": item.price,
                "category": item.category,
                "similarity_score": round(similarity_score, 3)
            })
    
    return {
        "query": query,
        "total_results": len(items),
        "items": items
    }

# ============================================
# INTERACTION ENDPOINTS
# ============================================

@app.post("/interactions", response_model=InteractionResponse)
def create_interaction(interaction: InteractionCreate, db: Session = Depends(get_db)):
    """Create/log a user-item interaction (view, click, purchase, or rating)"""
    # Validate user and item exist
    user = db.query(User).filter(User.id == interaction.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    item = db.query(Item).filter(Item.id == interaction.item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Create interaction
    db_interaction = Interaction(
        user_id=interaction.user_id,
        item_id=interaction.item_id,
        interaction_type=interaction.interaction_type,
        rating=interaction.rating
    )
    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)
    
    # Invalidate user's recommendation cache
    recommendation_engine.invalidate_user_cache(interaction.user_id)
    
    logger.info(f"✅ Interaction logged: user={interaction.user_id}, item={interaction.item_id}, type={interaction.interaction_type}")
    return db_interaction

@app.get("/users/{user_id}/interactions", response_model=list[InteractionResponse])
def get_user_interactions(user_id: int, db: Session = Depends(get_db)):
    """Get all interactions for a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    interactions = db.query(Interaction).filter(Interaction.user_id == user_id).all()
    return interactions

@app.get("/users/{user_id}/ratings")
def get_user_ratings(user_id: int, db: Session = Depends(get_db)):
    """Get all ratings for a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    ratings = db.query(Interaction).filter(
        Interaction.user_id == user_id,
        Interaction.rating.isnot(None)
    ).all()
    
    return {
        "user_id": user_id,
        "total_ratings": len(ratings),
        "ratings": [
            {
                "item_id": r.item_id,
                "rating": r.rating,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None
            }
            for r in ratings
        ]
    }

# ============================================
# RECOMMENDATION ENDPOINTS (WEEK 5)
# ============================================

@app.get("/users/{user_id}/recommendations", response_model=RecommendationListResponse)
def get_hybrid_recommendations(
    user_id: int, 
    top_k: int = Query(5, ge=1, le=20),
    content_weight: float = Query(0.6, ge=0.0, le=1.0),
    collab_weight: float = Query(0.4, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    """
    Get hybrid recommendations (combines content-based + collaborative filtering)
    
    Weights:
    - content_weight: importance of content-based filtering (default 0.6)
    - collab_weight: importance of collaborative filtering (default 0.4)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Normalize weights
    total_weight = content_weight + collab_weight
    if total_weight == 0:
        raise HTTPException(status_code=400, detail="Weights cannot both be 0")
    
    content_weight = content_weight / total_weight
    collab_weight = collab_weight / total_weight
    
    recommendations, breakdown = recommendation_engine.get_hybrid_recommendations(
        user_id, top_k, content_weight, collab_weight, db
    )
    
    return {
        "user_id": user_id,
        "user_username": user.username,
        "algorithm": "hybrid",
        "total_recommendations": len(recommendations),
        "recommendations": recommendations,
        "breakdown": breakdown
    }

@app.get("/users/{user_id}/recommendations/content", response_model=RecommendationListResponse)
def get_content_based_recommendations(
    user_id: int, 
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """
    Get content-based recommendations
    
    Algorithm: Finds items similar to ones the user previously liked
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    recommendations = recommendation_engine.get_content_based_recommendations(user_id, top_k, db)
    
    return {
        "user_id": user_id,
        "user_username": user.username,
        "algorithm": "content_based",
        "total_recommendations": len(recommendations),
        "recommendations": recommendations,
        "breakdown": {
            "method": "Content-based filtering",
            "description": "Items similar to previously liked items"
        }
    }

@app.get("/users/{user_id}/recommendations/collab", response_model=RecommendationListResponse)
def get_collab_recommendations(
    user_id: int, 
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """
    Get collaborative filtering recommendations
    
    Algorithm: Recommends items liked by similar users
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    recommendations = recommendation_engine.get_collaborative_recommendations(user_id, top_k, db)
    
    return {
        "user_id": user_id,
        "user_username": user.username,
        "algorithm": "collaborative",
        "total_recommendations": len(recommendations),
        "recommendations": recommendations,
        "breakdown": {
            "method": "Collaborative filtering",
            "description": "Items liked by similar users"
        }
    }

# ============================================
# STATS ENDPOINTS
# ============================================

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get database statistics"""
    user_count = db.query(User).count()
    item_count = db.query(Item).count()
    interaction_count = db.query(Interaction).count()
    
    return {
        "users": user_count,
        "items": item_count,
        "interactions": interaction_count,
        "avg_interactions_per_user": interaction_count / user_count if user_count > 0 else 0
    }

@app.get("/stats/categories")
def get_category_stats(db: Session = Depends(get_db)):
    """Get stats by category"""
    items = db.query(Item).all()
    categories = {}
    
    for item in items:
        if item.category not in categories:
            categories[item.category] = {
                "count": 0,
                "avg_price": 0,
                "items": []
            }
        categories[item.category]["count"] += 1
        categories[item.category]["items"].append({
            "id": item.id,
            "title": item.title,
            "price": item.price
        })
    
    # Calculate averages
    for cat in categories:
        prices = [item["price"] for item in categories[cat]["items"]]
        categories[cat]["avg_price"] = round(sum(prices) / len(prices), 2) if prices else 0
    
    return categories

# ============================================
# STARTUP LOG
# ============================================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 E-Commerce Recommendation Engine - Week 5")
    logger.info("=" * 60)
    logger.info("✅ FastAPI app initialized")
    logger.info("✅ Database initialized")
    logger.info("✅ Embedding service ready")
    logger.info("✅ Milvus vector store ready")
    logger.info("✅ Recommendation engine ready")
    logger.info("📊 Available endpoints:")
    logger.info("   - User: POST/GET /users, GET/PUT/DELETE /users/{id}")
    logger.info("   - Item: GET /items, GET /items/{id}, GET /items/category/{cat}")
    logger.info("   - Search: GET /items/search/similar/{id}, POST /search/semantic")
    logger.info("   - Interaction: POST /interactions, GET /users/{id}/interactions")
    logger.info("   🆕 Recommendations (Week 5):")
    logger.info("   - GET /users/{id}/recommendations (HYBRID)")
    logger.info("   - GET /users/{id}/recommendations/content (CONTENT-BASED)")
    logger.info("   - GET /users/{id}/recommendations/collab (COLLABORATIVE)")
    logger.info("   - Stats: GET /stats, GET /stats/categories")
    logger.info("=" * 60)
    logger.info("📍 API Docs: http://localhost:8000/docs")
    logger.info("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)