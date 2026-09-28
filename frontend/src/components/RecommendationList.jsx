import React from 'react';
import ItemCard from './ItemCard';

export default function RecommendationList({ recommendations, algorithm, onRate, onView, loading }) {
  if (loading) {
    return (
      <div className="text-center py-10">
        <p className="text-lg text-gray-600">Loading recommendations...</p>
      </div>
    );
  }

  if (!recommendations || recommendations.length === 0) {
    return (
      <div className="text-center py-10">
        <p className="text-lg text-gray-600">No recommendations available yet.</p>
        <p className="text-sm text-gray-500 mt-2">Rate some items to get personalized recommendations!</p>
      </div>
    );
  }

  const algorithmLabel = {
    hybrid: '🔀 Hybrid (60% Content + 40% Collab)',
    content_based: '📚 Content-Based',
    collaborative: '👥 Collaborative',
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">{algorithmLabel[algorithm]}</h2>
        <p className="text-gray-600">{recommendations.length} recommendations for you</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {recommendations.map((rec) => (
          <div key={rec.item_id} className="relative">
            <ItemCard
              item={rec}
              onRate={onRate}
              onView={onView}
            />
            {/* Score Badge */}
            <div className="absolute top-4 right-4 bg-green-500 text-white px-3 py-1 rounded-full text-sm font-bold">
              {rec.hybrid_score ? `${(rec.hybrid_score * 100).toFixed(0)}%` : `${(rec.score * 100).toFixed(0)}%`}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}