import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { useToast } from '@chakra-ui/react';
import { getWatchHistory, getWatchHistoryCount, refreshWatchHistory } from '../services/watchHistory';
import { useAuth } from './AuthContext';

// Create the context
const WatchHistoryContext = createContext();

// Custom hook for using the context
export const useWatchHistory = () => {
  const context = useContext(WatchHistoryContext);
  if (!context) {
    throw new Error('useWatchHistory must be used within a WatchHistoryProvider');
  }
  return context;
};

// Provider component
export const WatchHistoryProvider = ({ children }) => {
  const { isAuthenticated } = useAuth();
  const toast = useToast();
  
  // State
  const [historyData, setHistoryData] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [hasNewData, setHasNewData] = useState(false);
  const [error, setError] = useState(null);
  const [isInitializing, setIsInitializing] = useState(true);
  
  // Pagination state
  const [pagination, setPagination] = useState({
    limit: 50,
    offset: 0,
  });
  
  // Filters state
  const [filters, setFilters] = useState({
    media_type: null,
    start_date: null,
    end_date: null,
  });

  // Refs to manage polling
  const pollIntervalRef = useRef(null);
  const pollTimeoutRef = useRef(null);
  const initialLoadAttemptedRef = useRef(false);
  const refreshAttemptedRef = useRef(false);

  // Get watch history count
  const fetchHistoryCount = useCallback(async () => {
    if (!isAuthenticated) return 0;

    try {
      const { count } = await getWatchHistoryCount();
      setTotalCount(count);
      return count;
    } catch (error) {
      console.error('Error fetching history count:', error);
      // Downgrade from error to warning to avoid panic on initial load
      if (isInitializing) {
        console.warn('Initial history count check failed, will retry');
        return 0;
      } else {
        setError('Failed to fetch watch history count');
        return 0;
      }
    }
  }, [isAuthenticated, isInitializing]);

  // Fetch watch history data
  const fetchHistory = useCallback(async (options = {}) => {
    if (!isAuthenticated) return;

    setIsLoading(true);
    setError(null);
    
    try {
      const params = {
        ...pagination,
        ...filters,
        ...options,
      };
      
      const response = await getWatchHistory(params);
      setHistoryData(response.data);
      setTotalCount(response.count || response.data.length);
      setHasNewData(false);
    } catch (error) {
      console.error('Error fetching watch history:', error);
      
      // Only show error toast if not in initialization phase
      if (!isInitializing) {
        setError('Failed to fetch watch history');
        toast({
          title: 'Error fetching watch history',
          description: error.response?.data?.detail || error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    } finally {
      setIsLoading(false);
      if (isInitializing) setIsInitializing(false);
    }
  }, [isAuthenticated, pagination, filters, toast, isInitializing]);

  // Clear all polling timers and intervals
  const clearAllPolling = useCallback(() => {
    // Clear any existing polling interval
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    
    // Clear any existing timeout
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
  }, []);

  // Refresh watch history
  const refresh = useCallback(async (force = false) => {
    if (!isAuthenticated) return;

    // Avoid redundant refresh attempts if we already have one in progress
    if (isRefreshing && !force) return;
    
    // Track that we attempted a refresh
    refreshAttemptedRef.current = true;

    // Clear any existing polling
    clearAllPolling();
    
    setIsRefreshing(true);
    setError(null);
    
    try {
      const response = await refreshWatchHistory(force);
      
      // Only show toast if it's a manual refresh or initial load
      if (!isInitializing || force) {
        toast({
          title: 'Refresh started',
          description: response.message,
          status: 'info',
          duration: 5000,
          isClosable: true,
        });
      }
      
      // If this is an initial load, show a different message
      if (response.initial_load) {
        // Only show toast if it's not part of silent initialization
        if (!isInitializing || force) {
          toast({
            title: 'Initial data load',
            description: 'Your watch history is being fetched for the first time. This may take a few minutes.',
            status: 'info',
            duration: 8000,
            isClosable: true,
          });
        }
      }
      
      // Poll for changes periodically if this is an initial load
      if (response.initial_load) {
        let pollAttempts = 0;
        const maxPollAttempts = 60; // Maximum number of polling attempts (5 minutes with 5 second intervals)
        
        pollIntervalRef.current = setInterval(async () => {
          pollAttempts++;
          const count = await fetchHistoryCount();
          
          if (count > 0) {
            // Data is available, stop polling and fetch it
            clearAllPolling();
            await fetchHistory();
            setIsRefreshing(false);
            toast({
              title: 'Watch history loaded',
              description: `${count} items have been loaded.`,
              status: 'success',
              duration: 5000,
              isClosable: true,
            });
          } else if (pollAttempts >= maxPollAttempts) {
            // We've reached the maximum number of attempts
            clearAllPolling();
            setIsRefreshing(false);
            toast({
              title: 'Watch history fetch timeout',
              description: 'The data loading process is taking longer than expected. Please try refreshing manually.',
              status: 'warning',
              duration: 8000,
              isClosable: true,
            });
          }
        }, 5000); // Check every 5 seconds
        
        // Set a timeout to clear the interval after 5 minutes max
        pollTimeoutRef.current = setTimeout(() => {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
            setIsRefreshing(false);
          }
        }, 5 * 60 * 1000); // 5 minutes
      } else {
        // For normal refreshes, wait a bit and then fetch the data
        setTimeout(async () => {
          await fetchHistory();
          setHasNewData(true);
          setIsRefreshing(false);
        }, 3000);
      }
    } catch (error) {
      console.error('Error refreshing watch history:', error);
      setIsRefreshing(false);
      
      // Only show error if it's a manual refresh (not during initialization)
      if (!isInitializing || force) {
        setError('Failed to refresh watch history');
        toast({
          title: 'Error refreshing watch history',
          description: error.response?.data?.detail || error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    }
  }, [isAuthenticated, fetchHistory, fetchHistoryCount, toast, clearAllPolling, isRefreshing, isInitializing]);

  // Handle pagination change
  const changePage = useCallback((newOffset) => {
    setPagination(prev => ({
      ...prev,
      offset: newOffset
    }));
  }, []);

  // Handle limit change
  const changeLimit = useCallback((newLimit) => {
    setPagination(prev => ({
      ...prev,
      limit: newLimit,
      offset: 0 // Reset offset when changing limit
    }));
  }, []);

  // Handle filter change
  const updateFilters = useCallback((newFilters) => {
    setFilters(prev => ({
      ...prev,
      ...newFilters
    }));
    // Reset pagination when filters change
    setPagination(prev => ({
      ...prev,
      offset: 0
    }));
  }, []);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      clearAllPolling();
    };
  }, [clearAllPolling]);

  // Initialize history data when authenticated
  useEffect(() => {
    if (isAuthenticated && !initialLoadAttemptedRef.current) {
      initialLoadAttemptedRef.current = true;
      setIsInitializing(true);
      
      const initializeHistory = async () => {
        try {
          const count = await fetchHistoryCount();
          if (count === 0) {
            // If no history data, trigger a refresh but only once
            if (!refreshAttemptedRef.current) {
              refresh(false);
            }
          } else {
            // If there is history data, fetch it
            fetchHistory();
          }
        } catch (error) {
          console.error('Error during initialization:', error);
        } finally {
          setIsInitializing(false);
        }
      };
      
      initializeHistory();
    } else if (!isAuthenticated) {
      // Reset the initialization flags when logged out
      initialLoadAttemptedRef.current = false;
      refreshAttemptedRef.current = false;
      setIsInitializing(true);
      // Clear data when logging out
      setHistoryData([]);
      setTotalCount(0);
      setError(null);
    }
  }, [isAuthenticated, fetchHistoryCount, fetchHistory, refresh]);

  // Refetch when pagination or filters change
  useEffect(() => {
    if (isAuthenticated && !isLoading && !isRefreshing && totalCount > 0 && !isInitializing) {
      fetchHistory();
    }
  }, [isAuthenticated, pagination, filters, fetchHistory, isLoading, isRefreshing, totalCount, isInitializing]);

  const value = {
    historyData,
    totalCount,
    isLoading,
    isRefreshing,
    hasNewData,
    error,
    isInitializing,
    pagination: { ...pagination },
    filters: { ...filters },
    refresh,
    changePage,
    changeLimit,
    updateFilters,
  };

  return (
    <WatchHistoryContext.Provider value={value}>
      {children}
    </WatchHistoryContext.Provider>
  );
};

export default WatchHistoryContext;