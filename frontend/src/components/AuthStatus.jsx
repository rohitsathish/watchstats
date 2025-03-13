import React, { useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Text,
  Button,
  Flex,
  Badge,
  Spinner,
  useColorModeValue,
  useToast,
  Avatar,
  VStack,
  HStack,
  Tooltip,
  Icon
} from '@chakra-ui/react';
import { useAuth } from '../context/AuthContext';
import { MdRefresh, MdWifiOff } from 'react-icons/md';

/**
 * Component that displays authentication status and provides login/logout functionality
 */
const AuthStatus = () => {
  // Get authentication state and methods from context
  const {
    isAuthenticated,
    user,
    loading,
    error,
    connectionStatus,
    handleLogin,
    handleLogout,
    refreshAuthState,
    reconnectSSE
  } = useAuth();

  const toast = useToast();

  // Track previous auth state to detect changes with refs to prevent re-renders
  const prevAuthRef = useRef({
    isAuthenticated: false,
    username: null, // Track by username for display purposes
    connectionStatus: { connected: false }
  });

  // Track if initial render has happened
  const initialRenderRef = useRef(true);

  // Track if a notification is currently showing
  const notificationActiveRef = useRef(false);

  // UI theme colors
  const bgColor = useColorModeValue('gray.100', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // Handle connection status changes - throttled to avoid update loops
  useEffect(() => {
    // Skip initial render to prevent notification on page load
    if (initialRenderRef.current) {
      // Still update the ref on initial render
      prevAuthRef.current.connectionStatus = connectionStatus;
      return;
    }

    // Don't show another notification if one is already active
    if (notificationActiveRef.current) return;

    // Check if connection status actually changed
    const wasConnected = prevAuthRef.current.connectionStatus.connected;
    const isConnected = connectionStatus.connected;

    // Only show notifications if connection status changed
    if (wasConnected !== isConnected) {
      if (wasConnected && !isConnected) {
        // Connection lost
        notificationActiveRef.current = true;
        toast({
          title: 'Connection Lost',
          description: 'Connection to server lost. Trying to reconnect...',
          status: 'warning',
          duration: 5000,
          isClosable: true,
          onCloseComplete: () => {
            notificationActiveRef.current = false;
          }
        });
      }
    }

    // Update previous connection status
    prevAuthRef.current.connectionStatus = connectionStatus;
  }, [connectionStatus, toast]);

  // Refresh auth state when connection is restored - separate effect to avoid loops
  useEffect(() => {
    if (!initialRenderRef.current &&
        !prevAuthRef.current.connectionStatus.connected &&
        connectionStatus.connected) {
      // Use setTimeout to avoid React update loops
      const timeoutId = setTimeout(() => {
        refreshAuthState();
      }, 1000);

      return () => clearTimeout(timeoutId);
    }
  }, [connectionStatus.connected, refreshAuthState]);

  // Show toast when auth state changes
  useEffect(() => {
    // Skip initial render to prevent notification on page load
    if (initialRenderRef.current) {
      initialRenderRef.current = false;
      // Still update the ref on initial render
      prevAuthRef.current.isAuthenticated = isAuthenticated;
      prevAuthRef.current.username = user?.username;
      return;
    }

    // Don't show another notification if one is already active
    if (notificationActiveRef.current) return;

    // Check if auth state actually changed (including user identity)
    const wasAuthenticated = prevAuthRef.current.isAuthenticated;
    const prevUsername = prevAuthRef.current.username;
    const currentUsername = user?.username;

    // Login state change
    if (!wasAuthenticated && isAuthenticated) {
      notificationActiveRef.current = true;
      toast({
        title: 'Login Successful',
        description: `You have successfully connected with Trakt${user?.username ? ` as ${user.username}` : ''}`,
        status: 'success',
        duration: 5000,
        isClosable: true,
        onCloseComplete: () => {
          notificationActiveRef.current = false;
        }
      });
    }
    // Logout state change
    else if (wasAuthenticated && !isAuthenticated) {
      notificationActiveRef.current = true;
      toast({
        title: 'Logged Out',
        description: 'You have been logged out',
        status: 'info',
        duration: 5000,
        isClosable: true,
        onCloseComplete: () => {
          notificationActiveRef.current = false;
        }
      });
    }
    // User identity change while still authenticated
    else if (isAuthenticated &&
            prevUsername !== currentUsername &&
            prevUsername !== null &&
            currentUsername !== null) {
      notificationActiveRef.current = true;
      toast({
        title: 'User Changed',
        description: `Authenticated as ${currentUsername}`,
        status: 'info',
        duration: 5000,
        isClosable: true,
        onCloseComplete: () => {
          notificationActiveRef.current = false;
        }
      });
    }

    // Update previous auth state
    prevAuthRef.current.isAuthenticated = isAuthenticated;
    prevAuthRef.current.username = currentUsername;
  }, [isAuthenticated, user, toast]);

  // Handle manual refresh - memoize to prevent recreating on each render
  const handleRefresh = useCallback(() => {
    if (!connectionStatus.connected) {
      reconnectSSE();
      toast({
        title: 'Reconnecting',
        description: 'Attempting to reconnect to server...',
        status: 'info',
        duration: 2000,
        isClosable: true,
      });
    } else {
      refreshAuthState();
      toast({
        title: 'Refreshing',
        description: 'Checking authentication status...',
        status: 'info',
        duration: 2000,
        isClosable: true,
      });
    }
  }, [connectionStatus.connected, reconnectSSE, refreshAuthState, toast]);

  return (
    <Box
      p={4}
      borderRadius="md"
      bg={bgColor}
      borderWidth="1px"
      borderColor={borderColor}
      width="100%"
      maxW="350px"
    >
      <VStack spacing={4} align="stretch">
        <Flex width="100%" justify="space-between" align="center">
          <Text fontWeight="bold" fontSize="lg">Authentication Status</Text>
          <HStack spacing={2}>
            {!connectionStatus.connected && (
              <Tooltip label="Connection lost">
                <span>
                  <Icon
                    as={MdWifiOff}
                    color="orange.500"
                    boxSize={5}
                  />
                </span>
              </Tooltip>
            )}
            <Tooltip label={connectionStatus.connected ? "Refresh auth status" : "Reconnect"}>
              <Button
                size="xs"
                onClick={handleRefresh}
                isLoading={loading || connectionStatus.connecting}
                variant="ghost"
                aria-label="Refresh authentication status"
              >
                <Icon as={MdRefresh} />
              </Button>
            </Tooltip>
          </HStack>
        </Flex>

        {loading ? (
          <Flex align="center" justify="center" py={4}>
            <Spinner size="sm" mr={2} />
            <Text>Authenticating...</Text>
          </Flex>
        ) : isAuthenticated ? (
          <VStack spacing={3} align="stretch">
            <Badge colorScheme="green" alignSelf="flex-start">Logged In</Badge>

            {user && (
              <>
                <HStack spacing={3} align="center">
                  <Avatar
                    size="md"
                    src={user.avatar || undefined}
                    name={user.name || user.username}
                  />
                  <VStack align="start" spacing={0}>
                    <Text fontWeight="bold">{user.name || user.username}</Text>
                    <Text fontSize="sm" color="gray.500">@{user.username}</Text>
                  </VStack>
                </HStack>

                {user.uuid && (
                  <Text fontSize="xs" color="gray.500">
                    ID: {user.uuid.substring(0, 8)}...
                  </Text>
                )}

                {user.timezone && (
                  <Text fontSize="xs" color="gray.500">
                    Timezone: {user.timezone}
                  </Text>
                )}
              </>
            )}

            <Button
              size="sm"
              colorScheme="red"
              variant="outline"
              onClick={handleLogout}
              isLoading={loading}
              mt={2}
              isDisabled={!connectionStatus.connected}
            >
              Logout
            </Button>
          </VStack>
        ) : (
          <VStack spacing={3} align="stretch">
            <Badge colorScheme="red" alignSelf="flex-start">Logged Out</Badge>

            <Text fontSize="sm">
              Connect with your Trakt.tv account to access your media information.
            </Text>

            {error && (
              <Text fontSize="xs" color="red.500">
                {error}
              </Text>
            )}

            <Button
              size="sm"
              colorScheme="blue"
              onClick={handleLogin}
              isLoading={loading}
              isDisabled={!connectionStatus.connected}
            >
              Login with Trakt
            </Button>
          </VStack>
        )}
      </VStack>
    </Box>
  );
};

export default AuthStatus;