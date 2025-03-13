/**
 * Authentication service for Trakt.tv integration
 * 
 * Handles user authentication, login, logout, and token management
 */
import axios from 'axios';
import api from '../utils/api';

/**
 * Get the Trakt authorization URL for login
 * @returns {Promise<string>} Authorization URL
 */
const getLoginUrl = async () => {
  try {
    const response = await api.get('/auth/login');
    if (!response.data?.authorize_url) {
      throw new Error('Invalid login URL response from server');
    }
    return response.data.authorize_url;
  } catch (error) {
    console.error('Error getting login URL:', error);
    throw error;
  }
};

/**
 * Exchange authorization code for token
 * @param {string} code Authorization code from Trakt OAuth
 * @returns {Promise<Object>} Authentication result
 */
const exchangeCode = async (code) => {
  try {
    const response = await api.get(`/auth/token?code=${code}`);
    return response.data;
  } catch (error) {
    console.error('Error exchanging code for token:', error);
    throw error;
  }
};

/**
 * Get current user info
 * @returns {Promise<Object>} User information
 */
const getUserInfo = async () => {
  try {
    const response = await api.get('/auth/user');
    return response.data;
  } catch (error) {
    console.error('Error getting user info:', error);
    throw error;
  }
};

/**
 * Logout user
 * @returns {Promise<Object>} Logout result
 */
const logout = async () => {
  try {
    const response = await api.post('/auth/logout');
    return response.data;
  } catch (error) {
    console.error('Error during logout:', error);
    throw error;
  }
};

export const authService = {
  getLoginUrl,
  exchangeCode,
  getUserInfo,
  logout
};