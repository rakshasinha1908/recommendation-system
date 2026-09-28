import logging
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import redis
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Hybrid recommendation engine with 3 algorithms:
    1. Content-Based: Items similar to ones user liked
    2. Collaborative: Items liked by similar users
    3. Hybrid: Combination of both (customizable weights)
    """
    
    def __init__(self, embedding_service, vector_store, db_session):
        """Initialize with services"""
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.db = db_session
        
        # Try to connect to Redis for caching (optional)
        try:
            self.redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                db=0, 
                decode_responses=True
            )
            self.redis_client.ping()
            self.use_cache = True
            logger.info("✅ Redis cache connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {str(e)}. Running without cache.")
            self.use_cache = False
    
    # ============================================================
    # ALGORITHM 1: CONTENT-BASED FILTERING
    # ============================================================
    
    def get_content_based_recommendations(
        self, 
        user_id: int, 
        top_k: int = 5, 
        db=None
    ) -> List[Dict]:
        """
        CONTENT-BASED FILTERING ALGORITHM
        
        How it works:
        1. Find all items user rated >= 4 (liked items)
        2. Generate embedding for each liked item
        3. Average all embeddings together
        4. Search Milvus for items similar to this average
        5. Exclude items user already saw
        6. Return top K with similarity scores
        
        Best for: Item discovery, new users, semantic similarity
        
        Args:
            user_id: Which user to recommend for
            top_k: How many recommendations to return (default 5)
            db: Database session
            
        Returns:
            List of recommendations with scores
        """
        from models import Interaction, Item
        
        db = db or self.db
        
        try:
            # STEP 1: Get all items user liked (rated >= 4)
            logger.info(f"[CONTENT] Finding liked items for user {user_id}")
            liked_interactions = db.query(Interaction).filter(
                Interaction.user_id == user_id,
                Interaction.rating >= 4
            ).all()
            
            # If user hasn't rated anything, return empty
            if not liked_interactions:
                logger.info(f"[CONTENT] User {user_id} has no highly-rated items")
                return []
            
            logger.info(f"[CONTENT] User {user_id} has {len(liked_interactions)} liked items")
            
            # STEP 2: Generate embeddings for each liked item
            logger.info(f"[CONTENT] Generating embeddings for liked items")
            liked_embeddings = []
            for interaction in liked_interactions:
                item = db.query(Item).filter(Item.id == interaction.item_id).first()
                if item:
                    # Create text description of item
                    text = f"{item.title}. {item.description}" if item.description else item.title
                    # Generate embedding using SentenceTransformer
                    embedding = self.embedding_service.embed(text)
                    liked_embeddings.append(embedding)
            
            if not liked_embeddings:
                return []
            
            # STEP 3: Average the embeddings
            # This creates a "user preference vector" based on what they liked
            logger.info(f"[CONTENT] Averaging {len(liked_embeddings)} embeddings")
            avg_embedding = np.mean(liked_embeddings, axis=0)
            
            # STEP 4: Search Milvus for similar items
            # We get extra results to filter out items user already saw
            logger.info(f"[CONTENT] Searching Milvus for similar items")
            results = self.vector_store.search(avg_embedding, top_k=top_k + 10)
            
            # STEP 5: Get list of items user already interacted with (to exclude)
            user_items = db.query(Interaction.item_id).filter(
                Interaction.user_id == user_id
            ).all()
            user_item_ids = set([item[0] for item in user_items])
            
            # STEP 6: Format and filter results
            recommendations = []
            for item_id, distance in results:
                # Skip items user already saw
                if item_id not in user_item_ids:
                    item = db.query(Item).filter(Item.id == item_id).first()
                    if item:
                        # Convert distance to similarity score (0-1)
                        similarity_score = 1 / (1 + distance)
                        recommendations.append({
                            "item_id": item.id,
                            "title": item.title,
                            "price": item.price,
                            "category": item.category,
                            "description": item.description,
                            "score": round(similarity_score, 3),
                            "method": "content_based"
                        })
                
                # Stop once we have enough
                if len(recommendations) >= top_k:
                    break
            
            logger.info(f"[CONTENT] Returning {len(recommendations)} recommendations for user {user_id}")
            return recommendations
        
        except Exception as e:
            logger.error(f"[CONTENT] Error: {str(e)}")
            return []
    
    # ============================================================
    # ALGORITHM 2: COLLABORATIVE FILTERING
    # ============================================================
    
    def get_collaborative_recommendations(
        self, 
        user_id: int, 
        top_k: int = 5, 
        db=None
    ) -> List[Dict]:
        """
        COLLABORATIVE FILTERING ALGORITHM
        
        How it works:
        1. Build a matrix: users × items (rows=users, cols=items, values=ratings)
        2. For each user, calculate cosine similarity with current user
        3. Find top 5 most similar users
        4. Get items they rated highly (>= 4)
        5. Aggregate scores from similar users
        6. Exclude items current user already saw
        7. Return top K
        
        Best for: Serendipity, community-driven, cold-start user with collab data
        
        Args:
            user_id: Which user to recommend for
            top_k: How many recommendations to return (default 5)
            db: Database session
            
        Returns:
            List of recommendations with aggregated scores
        """
        from models import User, Item, Interaction
        
        db = db or self.db
        
        try:
            # STEP 1: Get all users and items
            logger.info(f"[COLLAB] Building recommendation matrix")
            all_users = db.query(User).all()
            all_items = db.query(Item).all()
            
            # Check if we have enough data
            if len(all_users) < 2 or not all_items:
                logger.info(f"[COLLAB] Not enough users ({len(all_users)}) or items ({len(all_items)})")
                return []
            
            logger.info(f"[COLLAB] Building matrix: {len(all_users)} users × {len(all_items)} items")
            
            # STEP 2: Build user-item matrix
            # Rows = users, Columns = items, Values = ratings (or 1 if no rating)
            user_indices = {u.id: i for i, u in enumerate(all_users)}
            item_indices = {it.id: i for i, it in enumerate(all_items)}
            
            # Initialize empty matrix
            matrix = np.zeros((len(all_users), len(all_items)))
            
            # Fill matrix with ratings
            all_interactions = db.query(Interaction).all()
            for interaction in all_interactions:
                if interaction.user_id in user_indices and interaction.item_id in item_indices:
                    user_idx = user_indices[interaction.user_id]
                    item_idx = item_indices[interaction.item_id]
                    # Use rating if available, else 1 for interaction
                    matrix[user_idx, item_idx] = interaction.rating or 1
            
            logger.info(f"[COLLAB] Matrix built: {np.count_nonzero(matrix)} non-zero entries")
            
            # Check if target user exists
            if user_id not in user_indices:
                logger.warning(f"[COLLAB] User {user_id} not in matrix")
                return []
            
            # STEP 3: Calculate user similarity
            # Get the target user's row from matrix
            user_idx = user_indices[user_id]
            user_vector = matrix[user_idx].reshape(1, -1)
            
            logger.info(f"[COLLAB] Calculating similarities for user {user_id}")
            
            # Compare with all other users using cosine similarity
            similarities = []
            for i, other_user in enumerate(all_users):
                if i != user_idx:  # Don't compare with self
                    other_vector = matrix[i].reshape(1, -1)
                    # Cosine similarity = dot product / (norm1 × norm2)
                    # Returns value between -1 and 1 (higher = more similar)
                    similarity = cosine_similarity(user_vector, other_vector)[0, 0]
                    similarities.append((other_user.id, similarity))
            
            if not similarities:
                return []
            
            # STEP 4: Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"[COLLAB] Top similar users: {similarities[:3]}")
            
            # Get top 5 most similar users
            top_similar_users = similarities[:min(5, len(similarities))]
            similar_user_ids = [u[0] for u in top_similar_users]
            
            # STEP 5: Get items liked by similar users (rating >= 4)
            logger.info(f"[COLLAB] Finding items liked by similar users")
            similar_user_items = db.query(Interaction).filter(
                Interaction.user_id.in_(similar_user_ids),
                Interaction.rating >= 4
            ).all()
            
            # STEP 6: Get items current user already saw
            user_items = db.query(Interaction.item_id).filter(
                Interaction.user_id == user_id
            ).all()
            user_item_ids = set([item[0] for item in user_items])
            
            # STEP 7: Aggregate scores
            # Score = sum of ratings from similar users / number of similar users
            item_scores = {}
            for interaction in similar_user_items:
                if interaction.item_id not in user_item_ids:  # Exclude already-seen
                    if interaction.item_id not in item_scores:
                        item_scores[interaction.item_id] = 0
                    item_scores[interaction.item_id] += interaction.rating or 1
            
            if not item_scores:
                logger.info(f"[COLLAB] No recommendations found")
                return []
            
            # Sort by score (highest first)
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Format results
            recommendations = []
            for item_id, score in sorted_items[:top_k]:
                item = db.query(Item).filter(Item.id == item_id).first()
                if item:
                    recommendations.append({
                        "item_id": item.id,
                        "title": item.title,
                        "price": item.price,
                        "category": item.category,
                        "description": item.description,
                        "score": round(score / len(similar_user_ids), 3),
                        "method": "collaborative"
                    })
            
            logger.info(f"[COLLAB] Returning {len(recommendations)} recommendations for user {user_id}")
            return recommendations
        
        except Exception as e:
            logger.error(f"[COLLAB] Error: {str(e)}")
            return []
    
    # ============================================================
    # ALGORITHM 3: HYBRID (COMBINES BOTH)
    # ============================================================
    
    def get_hybrid_recommendations(
        self, 
        user_id: int, 
        top_k: int = 5, 
        content_weight: float = 0.6, 
        collab_weight: float = 0.4, 
        db=None
    ) -> Tuple[List[Dict], Dict]:
        """
        HYBRID ALGORITHM (RECOMMENDED!)
        
        How it works:
        1. Get content-based recommendations
        2. Get collaborative recommendations
        3. Combine both lists
        4. Normalize scores to 0-1 range
        5. Calculate hybrid score = (0.6 × content_score) + (0.4 × collab_score)
        6. Sort by hybrid score
        7. Return top K
        8. Cache result in Redis for 1 hour (optional)
        
        Why hybrid?
        - Combines strengths of both approaches
        - Better coverage than either alone
        - Customizable weights (adjust for your use case)
        - Faster (caches both recommendations)
        
        Args:
            user_id: Which user to recommend for
            top_k: How many recommendations to return (default 5)
            content_weight: How much to weight content (default 0.6 = 60%)
            collab_weight: How much to weight collaborative (default 0.4 = 40%)
            db: Database session
            
        Returns:
            Tuple of (recommendations list, breakdown dict)
        """
        from models import Item
        
        db = db or self.db
        
        logger.info(f"[HYBRID] Getting recommendations for user {user_id} with weights {content_weight}/{collab_weight}")
        
        # STEP 0: Check cache first (optional optimization)
        if self.use_cache:
            cache_key = f"recommendations:hybrid:{user_id}"
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    logger.info(f"[HYBRID] Cache HIT for user {user_id}")
                    data = json.loads(cached)
                    return data['recommendations'], data['breakdown']
            except Exception as e:
                logger.warning(f"[HYBRID] Cache read error: {str(e)}")
        
        # STEP 1: Get content-based recommendations
        content_recs = self.get_content_based_recommendations(user_id, top_k * 2, db)
        
        # STEP 2: Get collaborative recommendations
        collab_recs = self.get_collaborative_recommendations(user_id, top_k * 2, db)
        
        logger.info(f"[HYBRID] Got {len(content_recs)} content + {len(collab_recs)} collab recs")
        
        # STEP 3: Combine both lists into a single dictionary
        # Key = item_id, Value = scores from both methods
        combined_scores = {}
        
        # Add content-based scores
        for rec in content_recs:
            item_id = rec["item_id"]
            combined_scores[item_id] = {
                "item": rec,
                "content_score": rec["score"],
                "collab_score": 0,
                "hybrid_score": 0
            }
        
        # Add collaborative scores
        for rec in collab_recs:
            item_id = rec["item_id"]
            if item_id not in combined_scores:
                combined_scores[item_id] = {
                    "item": rec,
                    "content_score": 0,
                    "collab_score": rec["score"],
                    "hybrid_score": 0
                }
            else:
                combined_scores[item_id]["collab_score"] = rec["score"]
        
        # STEP 4: Normalize scores and calculate hybrid score
        # Normalize: divide by max to get 0-1 range
        max_content = max([s["content_score"] for s in combined_scores.values()]) or 1
        max_collab = max([s["collab_score"] for s in combined_scores.values()]) or 1
        
        for item_id, data in combined_scores.items():
            # Normalize scores to 0-1
            norm_content = data["content_score"] / max_content if max_content > 0 else 0
            norm_collab = data["collab_score"] / max_collab if max_collab > 0 else 0
            
            # Calculate weighted hybrid score
            hybrid = (norm_content * content_weight + norm_collab * collab_weight)
            data["hybrid_score"] = round(hybrid, 3)
        
        # STEP 5: Sort by hybrid score (highest first)
        sorted_items = sorted(combined_scores.items(), 
                            key=lambda x: x[1]["hybrid_score"], 
                            reverse=True)
        
        # STEP 6: Format final recommendations
        recommendations = []
        for item_id, data in sorted_items[:top_k]:
            item = db.query(Item).filter(Item.id == item_id).first()
            if item:
                recommendations.append({
                    "item_id": item.id,
                    "title": item.title,
                    "price": item.price,
                    "category": item.category,
                    "description": item.description,
                    "hybrid_score": data["hybrid_score"],
                    "content_score": round(data["content_score"], 3),
                    "collab_score": round(data["collab_score"], 3)
                })
        
        # STEP 7: Create breakdown metadata
        breakdown = {
            "total_recommendations": len(recommendations),
            "content_weight": content_weight,
            "collab_weight": collab_weight,
            "sources": {
                "content_based": len(content_recs),
                "collaborative": len(collab_recs)
            }
        }
        
        # STEP 8: Cache result for 1 hour (optional)
        if self.use_cache:
            try:
                cache_data = {
                    "recommendations": recommendations,
                    "breakdown": breakdown,
                    "timestamp": datetime.utcnow().isoformat()
                }
                self.redis_client.setex(
                    f"recommendations:hybrid:{user_id}",
                    3600,  # 1 hour in seconds
                    json.dumps(cache_data)
                )
                logger.info(f"[HYBRID] Cached results for user {user_id}")
            except Exception as e:
                logger.warning(f"[HYBRID] Cache write error: {str(e)}")
        
        logger.info(f"[HYBRID] Returning {len(recommendations)} recommendations for user {user_id}")
        return recommendations, breakdown
    
    # ============================================================
    # CACHE MANAGEMENT
    # ============================================================
    
    def invalidate_user_cache(self, user_id: int):
        """Clear cached recommendations for a user"""
        if not self.use_cache:
            return
    
        try:
            keys_to_delete = [
                f"recommendations:hybrid:{user_id}",
            ]
        
            for key in keys_to_delete:
                try:
                    self.redis_client.delete(key)
                    logger.debug(f"[CACHE] Deleted key: {key}")
                except redis.ConnectionError as e:
                    logger.warning(f"[CACHE] Redis connection error: {str(e)}")
                except Exception as e:
                    logger.warning(f"[CACHE] Failed to delete {key}: {str(e)}")
        
            logger.info(f"[CACHE] Invalidated cache for user {user_id}")
    
        except Exception as e:
            logger.warning(f"[CACHE] Cache invalidation failed: {str(e)}")