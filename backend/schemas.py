from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

# ============================================
# USER SCHEMAS
# ============================================

class UserCreate(BaseModel):
    username: str
    email: EmailStr

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    
    class Config:
        from_attributes = True

# ============================================
# ITEM SCHEMAS
# ============================================

class ItemCreate(BaseModel):
    title: str
    description: Optional[str] = None
    price: float
    category: str

class ItemResponse(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    price: float
    category: str
    
    class Config:
        from_attributes = True

# ============================================
# INTERACTION SCHEMAS
# ============================================

class InteractionCreate(BaseModel):
    user_id: int
    item_id: int
    interaction_type: str  # 'view', 'click', 'purchase', 'rating'
    rating: Optional[int] = None  # 1-5 if interaction_type == 'rating'

class InteractionResponse(BaseModel):
    id: int
    user_id: int
    item_id: int
    interaction_type: str
    rating: Optional[int] = None
    timestamp: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# ============================================
# RECOMMENDATION SCHEMAS (NEW - WEEK 5)
# ============================================

class RecommendationItem(BaseModel):
    """Individual recommendation item (hybrid)"""
    item_id: int
    title: str
    price: float
    category: str
    description: Optional[str] = None
    hybrid_score: Optional[float] = None
    content_score: Optional[float] = None
    collab_score: Optional[float] = None
    score: Optional[float] = None  # For content/collab only
    method: Optional[str] = None  # For content/collab only

class RecommendationBreakdown(BaseModel):
    """Breakdown of recommendation sources"""
    total_recommendations: Optional[int] = None
    content_weight: Optional[float] = None
    collab_weight: Optional[float] = None
    sources: Optional[Dict[str, int]] = None
    method: Optional[str] = None
    description: Optional[str] = None

class RecommendationListResponse(BaseModel):
    """Full recommendation response with metadata"""
    user_id: int
    user_username: str
    algorithm: str  # 'hybrid', 'content_based', 'collaborative'
    total_recommendations: int
    recommendations: List[RecommendationItem]
    breakdown: Dict[str, Any]

# ============================================
# SEARCH SCHEMAS
# ============================================

class SearchItem(BaseModel):
    item_id: int
    title: str
    price: float
    category: str
    similarity_score: float

class SearchResponse(BaseModel):
    query: Optional[str] = None
    query_item_id: Optional[int] = None
    query_item_title: Optional[str] = None
    total_results: int
    items: List[SearchItem]