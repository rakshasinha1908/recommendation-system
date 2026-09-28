import React from 'react';

export default function ItemCard({ item, onRate, onView }) {
  const [rating, setRating] = React.useState(0);
  const [showRating, setShowRating] = React.useState(false);

  const handleRate = (stars) => {
    setRating(stars);
    onRate && onRate(item.id, stars);
    setShowRating(false);
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow">
      {/* Item Image Placeholder */}
      <div className="w-full h-48 bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center">
        <div className="text-white text-center">
          <p className="text-4xl">📦</p>
          <p className="text-sm mt-2">{item.category}</p>
        </div>
      </div>

      {/* Item Details */}
      <div className="p-4">
        <h3 className="text-lg font-bold text-gray-800 truncate">{item.title}</h3>
        <p className="text-gray-600 text-sm mt-1 line-clamp-2">{item.description}</p>

        {/* Price */}
        <p className="text-2xl font-bold text-green-600 mt-3">${item.price}</p>

        {/* Rating Section */}
        <div className="mt-4">
          {!showRating ? (
            <>
              {rating > 0 && <p className="text-sm text-gray-600">Your rating: {'⭐'.repeat(rating)}</p>}
              <button
                onClick={() => setShowRating(true)}
                className="mt-2 w-full bg-blue-500 hover:bg-blue-600 text-white py-2 rounded-lg text-sm font-semibold transition"
              >
                {rating > 0 ? 'Change Rating' : 'Rate This'}
              </button>
            </>
          ) : (
            <div className="flex gap-2 justify-center">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  onClick={() => handleRate(star)}
                  className="text-2xl hover:scale-125 transition transform"
                >
                  {star <= rating ? '⭐' : '☆'}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <button
          onClick={() => onView && onView(item.id)}
          className="mt-3 w-full bg-gray-200 hover:bg-gray-300 text-gray-800 py-2 rounded-lg text-sm font-semibold transition"
        >
          View Details
        </button>
      </div>
    </div>
  );
}