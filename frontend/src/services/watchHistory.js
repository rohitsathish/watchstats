import api from '../utils/api';

/**
 * Watch History Service
 *
 * This service handles API calls to the watch history endpoints
 * on our FastAPI backend, which in turn manages interactions with Trakt.
 */

/**
 * Fetch watch history data from the backend
 * 
 * @param {Object} params - Query parameters
 * @param {number} [params.limit=100] - Maximum number of records to return
 * @param {number} [params.offset=0] - Number of records to skip
 * @param {string} [params.media_type] - Filter by media type (movie/episode)
 * @param {string} [params.start_date] - Filter by start date (YYYY-MM-DD)
 * @param {string} [params.end_date] - Filter by end date (YYYY-MM-DD)
 * @returns {Promise<{data: Array, count: number}>} - Watch history data and count
 */
export const getWatchHistory = async (params = {}) => {
  try {
    const response = await api.get('/api/watch/history', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching watch history:', error);
    throw error;
  }
};

/**
 * Get count of watch history items
 * 
 * @returns {Promise<{count: number}>} - Object with count property
 */
export const getWatchHistoryCount = async () => {
  try {
    const response = await api.get('/api/watch/count');
    return response.data;
  } catch (error) {
    console.error('Error fetching watch history count:', error);
    throw error;
  }
};

/**
 * Trigger watch history refresh in the backend
 * 
 * @param {boolean} [force=false] - Whether to force a full refresh
 * @returns {Promise<{status: string, message: string, initial_load: boolean}>} - Response with status and message
 */
export const refreshWatchHistory = async (force = false) => {
  try {
    const response = await api.post('/api/watch/refresh', null, {
      params: { force }
    });
    return response.data;
  } catch (error) {
    console.error('Error refreshing watch history:', error);
    throw error;
  }
};

export default {
  getWatchHistory,
  getWatchHistoryCount,
  refreshWatchHistory
};