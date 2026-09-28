import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// User API
export const userAPI = {
  getAll: () => apiClient.get('/users'),
  getById: (id) => apiClient.get(`/users/${id}`),
  create: (data) => apiClient.post('/users', data),
  update: (id, data) => apiClient.put(`/users/${id}`, data),
  delete: (id) => apiClient.delete(`/users/${id}`),
};

// Item API
export const itemAPI = {
  getAll: () => apiClient.get('/items'),
  getById: (id) => apiClient.get(`/items/${id}`),
  getByCategory: (category) => apiClient.get(`/items/category/${category}`),
  create: (data) => apiClient.post('/items', data),
  searchSimilar: (id, topK = 5) => apiClient.get(`/items/search/similar/${id}?top_k=${topK}`),
  semanticSearch: (query, topK = 5) => apiClient.post('/search/semantic', null, { params: { query, top_k: topK } }),
};

// Interaction API
export const interactionAPI = {
  create: (data) => apiClient.post('/interactions', data),
  getByUser: (userId) => apiClient.get(`/users/${userId}/interactions`),
  getRatings: (userId) => apiClient.get(`/users/${userId}/ratings`),
};

// Recommendation API
export const recommendationAPI = {
  getHybrid: (userId, topK = 5, contentWeight = 0.6, collabWeight = 0.4) =>
    apiClient.get(`/users/${userId}/recommendations`, {
      params: { top_k: topK, content_weight: contentWeight, collab_weight: collabWeight },
    }),
  getContentBased: (userId, topK = 5) =>
    apiClient.get(`/users/${userId}/recommendations/content?top_k=${topK}`),
  getCollaborative: (userId, topK = 5) =>
    apiClient.get(`/users/${userId}/recommendations/collab?top_k=${topK}`),
};

// Stats API
export const statsAPI = {
  getStats: () => apiClient.get('/stats'),
  getCategoryStats: () => apiClient.get('/stats/categories'),
};

export default apiClient;