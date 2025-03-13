import React, { createContext, useState, useEffect, useContext, useRef, useCallback } from 'react';
import { useQueryClient, useMutation } from '@tanstack/react-query';
import { authService } from '../services/authService';
import { sseService } from '../services/sseService';

// Create the auth context
const AuthContext = createContext(null);

/**
 * Custom hook to use the auth context
 * @returns {Object} Auth context value
 */
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === null) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

/**
 * Auth provider component
 * @param {Object} props - Component props
 * @param {React.ReactNode} props.children - Child components
 */
export const AuthProvider = ({ children }) => {
  const queryClient = useQueryClient();
  const sseInitializedRef = useRef(false);
  const isProcessingAuthRef = useRef(false); // Prevent concurrent auth operations

  // Auth state
  const [authState, setAuthState] = useState({
    isAuthenticated: false,
    user: null,
    loading: true,
    error: null
  });

  // Connection status state
  const [connectionStatus, setConnectionStatus] = useState({
    connected: false,
    connecting: false,
    error: null,
    lastHeartbeat: null
  });

  // Handle SSE auth updates - wrapped in useCallback to avoid recreation on every render
  const handleAuthUpdate = useCallback((data) => {
    if (isProcessingAuthRef.current) return; // Skip if already processing auth

    console.log('Auth update received:', data);

    // Compare with current state to avoid unnecessary updates
    if (data.authenticated) {
      // Only update if authentication state would change or user data is different
      setAuthState(prevState => {
        // If already authenticated with same user, no need to update
        if (prevState.isAuthenticated &&
            prevState.user?.username === data.user?.username) {
          return prevState;
        }

        return {
          isAuthenticated: true,
          user: data.user || prevState.user,
          loading: false,
          error: null
        };
      });

      // Invalidate user query to refetch if needed - but don't trigger a new refresh
      queryClient.setQueryData(['user'], data.user || null);
    } else {
      // Only update if currently authenticated
      setAuthState(prevState => {
        if (!prevState.isAuthenticated) {
          return prevState;
        }

        return {
          isAuthenticated: false,
          user: null,
          loading: false,
          error: data.error || null
        };
      });

      // Clear cached user data
      queryClient.setQueryData(['user'], null);
    }
  }, [queryClient]);

  // Handle SSE connection status changes - wrapped in useCallback
  const handleConnectionStatus = useCallback((status) => {
    setConnectionStatus(status);
  }, []);

  // Handle SSE connection event
  const handleConnection = useCallback((data) => {
    // The initial connection event tells us if we're authenticated
    if (data.authenticated && data.user) {
      setAuthState({
        isAuthenticated: true,
        user: data.user,
        loading: false,
        error: null
      });
    }
  }, []);

  // Clean reset function to properly clean up state
  const resetState = useCallback(() => {
    // Notify sseService that auth state changed - it will automatically reconnect
    // with the updated cookies (which will now not include the auth cookie)
    sseService.updateConnection();

    // Reset auth state
    setAuthState({
      isAuthenticated: false,
      user: null,
      loading: false,
      error: null
    });

    // Don't reset the connection status - keep it as is

    // Clear all auth related queries
    queryClient.removeQueries(['user']);
  }, [queryClient]);

  // Initialize auth state and connect SSE - only once on mount
  useEffect(() => {
    // Skip if SSE is already initialized to prevent multiple connections
    if (sseInitializedRef.current) {
      return;
    }

    const initializeAuth = async () => {
      try {
        // Try to get user info
        const response = await authService.getUserInfo();

        // Check if the user is authenticated based on the response
        if (response && response.authenticated === true && response.username) {
          setAuthState({
            isAuthenticated: true,
            user: response,
            loading: false,
            error: null
          });
          console.log('Auth initialized: User is authenticated', response);
        } else {
          // Not authenticated
          setAuthState({
            isAuthenticated: false,
            user: null,
            loading: false,
            error: null
          });
          console.log('Auth initialized: User is not authenticated');
        }
      } catch (error) {
        // Error or not authenticated
        console.error('Auth initialization error:', error);
        setAuthState({
          isAuthenticated: false,
          user: null,
          loading: false,
          error: error.message || 'Failed to initialize authentication'
        });
      } finally {
        // Connect SSE after auth check if not already connected
        if (!sseInitializedRef.current) {
          sseService.connect({
            onConnection: handleConnection,
            onAuthUpdate: handleAuthUpdate,
            onStatusChange: handleConnectionStatus,
            onHeartbeat: (data) => {
              // Update last heartbeat in connection status
              setConnectionStatus(prev => ({
                ...prev,
                lastHeartbeat: data.timestamp
              }));
            },
            onDisconnect: () => {
              setConnectionStatus(prev => ({
                ...prev,
                connected: false,
                connecting: false
              }));
            },
            onError: (error) => {
              setConnectionStatus(prev => ({
                ...prev,
                connected: false,
                connecting: false,
                error
              }));
            }
          });
          sseInitializedRef.current = true;
        }
      }
    };

    initializeAuth();

    // Cleanup on unmount
    return () => {
      sseService.disconnect();
      sseInitializedRef.current = false;
    };
  }, [handleAuthUpdate, handleConnection, handleConnectionStatus]);

  // Code exchange mutation
  const exchangeCodeMutation = useMutation({
    mutationFn: (code) => authService.exchangeCode(code),
    onSuccess: (data) => {
      if (data.authenticated && data.user) {
        setAuthState({
          isAuthenticated: true,
          user: data.user,
          loading: false,
          error: null
        });

        // Notify SSE service to reconnect with new auth cookies
        sseService.updateConnection();

        // Update cached user data
        queryClient.setQueryData(['user'], data.user);
      } else {
        setAuthState({
          isAuthenticated: false,
          user: null,
          loading: false,
          error: data.error || 'Authentication failed'
        });
      }
    },
    onError: (error) => {
      setAuthState({
        isAuthenticated: false,
        user: null,
        loading: false,
        error: error.message || 'Failed to exchange code for token'
      });
    }
  });

  // Logout mutation
  const logoutMutation = useMutation({
    mutationFn: () => authService.logout(),
    onSuccess: () => {
      // Use the clean reset function on logout
      resetState();
    },
    onError: (error) => {
      setAuthState(prev => ({
        ...prev,
        loading: false,
        error: error.message || 'Failed to logout'
      }));
    }
  });

  // Initialize login process
  const handleLogin = useCallback(async () => {
    if (isProcessingAuthRef.current) return; // Prevent concurrent login attempts
    isProcessingAuthRef.current = true;

    try {
      const loginUrl = await authService.getLoginUrl();
      window.location.href = loginUrl;
    } catch (error) {
      setAuthState(prev => ({
        ...prev,
        error: error.message || 'Failed to start login process'
      }));
      isProcessingAuthRef.current = false;
    }
  }, []);

  // Handle authorization code exchange
  const handleCodeExchange = useCallback(async (code) => {
    if (isProcessingAuthRef.current) return false; // Prevent concurrent operations
    isProcessingAuthRef.current = true;

    setAuthState(prev => ({ ...prev, loading: true, error: null }));
    try {
      await exchangeCodeMutation.mutateAsync(code);
      isProcessingAuthRef.current = false;
      return true;
    } catch (error) {
      isProcessingAuthRef.current = false;
      return false;
    }
  }, [exchangeCodeMutation]);

  // Handle logout
  const handleLogout = useCallback(async () => {
    if (isProcessingAuthRef.current) return; // Prevent concurrent logout attempts
    isProcessingAuthRef.current = true;

    setAuthState(prev => ({ ...prev, loading: true }));
    try {
      await logoutMutation.mutateAsync();
    } finally {
      isProcessingAuthRef.current = false;
    }
  }, [logoutMutation]);

  // Manually refresh auth state
  const refreshAuthState = useCallback(async () => {
    if (isProcessingAuthRef.current) return false; // Skip if already processing auth
    isProcessingAuthRef.current = true;

    setAuthState(prev => ({ ...prev, loading: true, error: null }));
    try {
      const response = await authService.getUserInfo();

      // Check if the user is authenticated based on the response
      if (response && response.authenticated === true && response.username) {
        setAuthState({
          isAuthenticated: true,
          user: response,
          loading: false,
          error: null
        });
        isProcessingAuthRef.current = false;
        return true;
      } else {
        // Not authenticated
        setAuthState({
          isAuthenticated: false,
          user: null,
          loading: false,
          error: null
        });
        isProcessingAuthRef.current = false;
        return false;
      }
    } catch (error) {
      console.error('Error refreshing auth state:', error);
      setAuthState({
        isAuthenticated: false,
        user: null,
        loading: false,
        error: error.message || 'Failed to refresh authentication'
      });
      isProcessingAuthRef.current = false;
      return false;
    }
  }, []);

  // Manually reconnect SSE
  const reconnectSSE = useCallback(() => {
    // Ensure SSE is disconnected before reconnecting
    sseService.disconnect();
    sseInitializedRef.current = false;

    // Reconnect
    sseService.connect({
      onConnection: handleConnection,
      onAuthUpdate: handleAuthUpdate,
      onStatusChange: handleConnectionStatus
    });
    sseInitializedRef.current = true;
  }, [handleConnection, handleAuthUpdate, handleConnectionStatus]);

  // Context value - memoize to prevent unnecessary re-renders
  const contextValue = React.useMemo(() => ({
    ...authState,
    connectionStatus,
    handleLogin,
    handleLogout,
    handleCodeExchange,
    refreshAuthState,
    reconnectSSE
  }), [
    authState,
    connectionStatus,
    handleLogin,
    handleLogout,
    handleCodeExchange,
    refreshAuthState,
    reconnectSSE
  ]);

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};
