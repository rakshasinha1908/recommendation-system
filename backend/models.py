from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/recommendation_db")

# Create engine with pool_pre_ping for connection verification
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connection before using
    echo=False,  # Set to True to see SQL queries
    connect_args={"connect_timeout": 5}  # 5 second timeout
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """Represents a user in the recommendation system"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"


class Item(Base):
    """Represents a product/item in the e-commerce catalog"""
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True, nullable=False)
    description = Column(Text)
    category = Column(String(100), index=True)
    price = Column(Float)
    image_url = Column(String(500))
    rating_avg = Column(Float, default=0.0)  # Average rating
    rating_count = Column(Integer, default=0)  # Number of ratings
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    interactions = relationship("Interaction", back_populates="item", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Item(id={self.id}, title={self.title}, price=${self.price})>"


class Interaction(Base):
    """Represents user-item interactions: views, clicks, ratings, purchases"""
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    item_id = Column(Integer, ForeignKey("items.id"), index=True, nullable=False)
    interaction_type = Column(String(50), nullable=False)  # 'view', 'click', 'rate', 'purchase'
    rating = Column(Float, nullable=True)  # 1-5 stars (only for 'rate' type)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Unique constraint: one rating per user-item pair
    __table_args__ = (
        UniqueConstraint('user_id', 'item_id', 'interaction_type', name='unique_user_item_interaction'),
    )
    
    # Relationships
    user = relationship("User", back_populates="interactions")
    item = relationship("Item", back_populates="interactions")
    
    def __repr__(self):
        return f"<Interaction(user_id={self.user_id}, item_id={self.item_id}, type={self.interaction_type})>"


class Recommendation(Base):
    """Stores generated recommendations for users"""
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    recommended_item_id = Column(Integer, ForeignKey("items.id"), nullable=False)
    score = Column(Float)  # Confidence score 0-1
    method = Column(String(50))  # 'collaborative', 'content-based', 'hybrid'
    explanation = Column(Text)  # Why was this recommended?
    generated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="recommendations")
    
    def __repr__(self):
        return f"<Recommendation(user_id={self.user_id}, item_id={self.recommended_item_id}, score={self.score})>"


# ✅ FIXED: Only create tables when explicitly called, not on import
def init_db():
    """
    Initialize database tables
    Call this explicitly when needed, not on module import
    """
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created/verified")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        raise


def get_db():
    """Database session dependency for FastAPI routes"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# For backward compatibility - only create tables if explicitly imported and called
if __name__ == "__main__":
    init_db()