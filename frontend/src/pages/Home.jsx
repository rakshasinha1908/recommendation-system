import React from 'react';
import { itemAPI, recommendationAPI, interactionAPI } from '../api/client';
import ItemCard from '../components/ItemCard';
import RecommendationList from '../components/RecommendationList';
import SearchBar from '../components/SearchBar';

export default function Home() {
  const [items, setItems] = React.useState([]);
  const [recommendations, setRecommendations] = React.useState([]);
  const [userId, setUserId] = React.useState(4); // Alice (default for testing)
  const [algorithm, setAlgorithm] = React.useState('hybrid');
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');

  // Fetch all items on mount
  React.useEffect(() => {
    const fetchItems = async () => {
      try {
        const response = await itemAPI.getAll();
        setItems(response.data);
      } catch (err) {
        setError('Failed to load items');
        console.error(err);
      }
    };
    fetchItems();
  }, []);

  // Fetch recommendations
  const fetchRecommendations = async () => {
    if (!userId) return;
    
    setLoading(true);
    try {
      let response;
      if (algorithm === 'hybrid') {
        response = await recommendationAPI.getHybrid(userId, 5);
      } else if (algorithm === 'content_based') {
        response = await recommendationAPI.getContentBased(userId, 5);
      } else {
        response = await recommendationAPI.getCollaborative(userId, 5);
      }
      setRecommendations(response.data.recommendations);
    } catch (err) {
      setError('Failed to load recommendations');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    fetchRecommendations();
  }, [userId, algorithm]);

  // Handle rating an item
  const handleRate = async (itemId, rating) => {
    try {
      await interactionAPI.create({
        user_id: userId,
        item_id: itemId,
        interaction_type: 'rating',
        rating: rating,
      });
      // Refresh recommendations after rating
      fetchRecommendations();
      alert('Rating saved! Recommendations updated.');
    } catch (err) {
      console.error('Failed to save rating:', err);
      alert('Failed to save rating');
    }
  };

  const handleSearch = async (query) => {
    setLoading(true);
    try {
      const response = await itemAPI.semanticSearch(query, 5);
      setItems(response.data.items);
    } catch (err) {
      setError('Search failed');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-800">🛍️ Smart Recommendations</h1>
          <p className="text-gray-600 mt-1">Discover products tailored for you</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-10">
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            {error}
          </div>
        )}

        {/* Search Section */}
        <section className="mb-10">
          <SearchBar onSearch={handleSearch} />
        </section>

        {/* Algorithm Selector */}
        <section className="mb-10">
          <h2 className="text-xl font-bold text-gray-800 mb-4">Choose Algorithm:</h2>
          <div className="flex gap-4">
            {['hybrid', 'content_based', 'collaborative'].map((algo) => (
              <button
                key={algo}
                onClick={() => setAlgorithm(algo)}
                className={`px-4 py-2 rounded-lg font-semibold transition ${
                  algorithm === algo
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                }`}
              >
                {algo === 'hybrid' && '🔀 Hybrid'}
                {algo === 'content_based' && '📚 Content-Based'}
                {algo === 'collaborative' && '👥 Collaborative'}
              </button>
            ))}
          </div>
        </section>

        {/* Recommendations */}
        <section className="mb-10">
          <RecommendationList
            recommendations={recommendations}
            algorithm={algorithm}
            onRate={handleRate}
            loading={loading}
          />
        </section>

        {/* All Items Section */}
        <section>
          <h2 className="text-2xl font-bold text-gray-800 mb-6">All Items</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {items.map((item) => (
              <ItemCard
                key={item.id}
                item={item}
                onRate={handleRate}
              />
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}