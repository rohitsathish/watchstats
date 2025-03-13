import axios from 'axios'
import Cookies from 'js-cookie'

// Base API URL - use environment variable if available, fallback to explicit URL during development
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
console.log('API Base URL:', API_BASE_URL)

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  // Ensure credentials (cookies) are sent with requests
  withCredentials: true
})

// Add request interceptor for request logging and modifications if needed
api.interceptors.request.use(
  config => {
    // We're using HTTP-only cookies for authentication, 
    // so we don't need to set the Authorization header
    // The backend will extract the JWT from cookies
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// Add response interceptor for better error handling
api.interceptors.response.use(
  response => response,
  error => {
    // Log the error for debugging
    console.error('API Error:', error.message, error.response || 'No response');
    
    // Add connection info to the error
    if (!error.response && error.message.includes('Network Error')) {
      error.isConnectionError = true;
    }
    
    // Handle authentication errors
    if (error.response && error.response.status === 401) {
      console.warn('Authentication error detected');
      // You can trigger auth refresh here or handle it in the component
    }
    
    return Promise.reject(error);
  }
);

// API Health check - only call this on demand, not on an interval
export const checkHealth = async () => {
  try {
    const response = await api.get('/health')
    return { status: 'ok', data: response.data }
  } catch (error) {
    console.error('Health check failed:', error)
    return {
      status: 'error',
      error: error.isConnectionError ? 'Connection failed' : 'Service unavailable'
    }
  }
}

// Auth functions
export const getLoginUrl = async () => {
  try {
    const response = await api.get('/auth/login')
    if (!response.data?.authorize_url) {
      throw new Error('Invalid login URL response from server')
    }
    return response.data.authorize_url
  } catch (error) {
    console.error('Error getting login URL:', error)
    throw error
  }
}

// Get current user info
export const getUserInfo = async () => {
  try {
    const response = await api.get('/auth/user')
    return response.data
  } catch (error) {
    console.error('Error getting user info:', error)
    throw error
  }
}

// Logout user
export const logout = async () => {
  try {
    await api.post('/auth/logout')
  } catch (error) {
    console.error('Error during logout:', error)
    throw error
  }
}

// Exchange authorization code for token
export const exchangeCodeForToken = async (code) => {
  try {
    const response = await api.get(`/auth/token?code=${code}`)
    return response.data
  } catch (error) {
    console.error('Error exchanging code for token:', error)
    throw error
  }
}

export default api