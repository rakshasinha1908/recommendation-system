"""
Data loading and preprocessing for Amazon reviews dataset
This script downloads product data and loads it into PostgreSQL
"""

import pandas as pd
import numpy as np
import requests
from sqlalchemy.orm import Session
from models import Item, User, get_db
import os

# For this MVP, we'll use a simpler approach:
# Create a sample CSV with e-commerce products

SAMPLE_DATA = """id,title,description,category,price,image_url
1,Samsung 55" 4K Smart TV,Crystal UHD display with AI upscaling 4K resolution,Electronics,599.99,https://via.placeholder.com/300x200?text=Samsung+TV
2,Sony WH-1000XM5 Headphones,Industry leading noise cancellation wireless headphones,Electronics,399.99,https://via.placeholder.com/300x200?text=Sony+Headphones
3,Apple AirPods Pro,Premium true wireless earbuds with active noise cancellation,Electronics,249.99,https://via.placeholder.com/300x200?text=AirPods
4,Canon EOS R6 Camera,Full-frame mirrorless camera with 20MP sensor,Electronics,2499.99,https://via.placeholder.com/300x200?text=Canon+Camera
5,Logitech MX Master 3S,Advanced wireless mouse for professionals,Electronics,99.99,https://via.placeholder.com/300x200?text=Logitech+Mouse
6,Dell XPS 13 Laptop,Ultra-portable 13 inch laptop with Intel i7,Electronics,1299.99,https://via.placeholder.com/300x200?text=Dell+Laptop
7,Amazon Echo Dot,Smart speaker with Alexa voice control,Smart Home,49.99,https://via.placeholder.com/300x200?text=Echo+Dot
8,Philips Hue Smart Bulbs,Color adjustable LED smart bulbs 16M colors,Smart Home,199.99,https://via.placeholder.com/300x200?text=Hue+Bulbs
9,TP-Link WiFi 6 Router,Ultra-fast mesh WiFi 6 router,Networking,299.99,https://via.placeholder.com/300x200?text=WiFi+Router
10,Google Nest Mini,Compact smart speaker with Google Assistant,Smart Home,39.99,https://via.placeholder.com/300x200?text=Nest+Mini"""

def load_sample_data():
    """Load initial sample product data into PostgreSQL"""
    from io import StringIO
    
    # Read CSV
    df = pd.read_csv(StringIO(SAMPLE_DATA))
    
    # Get database session
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        # Clear existing items
        db.query(Item).delete()
        
        # Insert new items
        for _, row in df.iterrows():
            item = Item(
                id=int(row['id']),
                title=row['title'],
                description=row['description'],
                category=row['category'],
                price=float(row['price']),
                image_url=row['image_url']
            )
            db.add(item)
        
        db.commit()
        print(f"✅ Loaded {len(df)} sample products into PostgreSQL")
        return len(df)
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        db.rollback()
    finally:
        db.close()


def create_sample_users():
    """Create sample users for testing"""
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        # Clear existing users
        db.query(User).delete()
        
        sample_users = [
            User(username="alice", email="alice@example.com"),
            User(username="bob", email="bob@example.com"),
            User(username="charlie", email="charlie@example.com"),
        ]
        
        for user in sample_users:
            db.add(user)
        
        db.commit()
        print(f"✅ Created {len(sample_users)} sample users")
        return len(sample_users)
    
    except Exception as e:
        print(f"❌ Error creating users: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    print("🚀 Loading sample data...")
    load_sample_data()
    create_sample_users()
    print("✅ Data pipeline complete!")