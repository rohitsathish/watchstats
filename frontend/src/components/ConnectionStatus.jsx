import React, { useCallback, useRef } from 'react';
import {
  Box,
  Text,
  HStack,
  Icon,
  Tooltip,
  useColorModeValue,
  Button,
} from '@chakra-ui/react';
import { MdWifi, MdWifiOff, MdSync } from 'react-icons/md';
import { useAuth } from '../context/AuthContext';

/**
 * Component to display real-time SSE connection status
 */
const ConnectionStatus = () => {
  const { connectionStatus, reconnectSSE } = useAuth();
  const { connected, connecting, error, lastHeartbeat } = connectionStatus;
  
  // Use ref to track if reconnection is in progress
  const reconnectingRef = useRef(false);
  
  // Calculate time since last heartbeat
  const timeSinceHeartbeat = lastHeartbeat
    ? Math.floor((Date.now() / 1000 - lastHeartbeat))
    : null;
  
  // Theme colors
  const bgColor = useColorModeValue('gray.50', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  // Handle reconnect click with useCallback to prevent unnecessary re-renders
  const handleReconnect = useCallback(() => {
    if (reconnectingRef.current) return; // Prevent multiple reconnection attempts
    
    reconnectingRef.current = true;
    reconnectSSE();
    
    // Reset the flag after a delay to allow reconnection attempts again
    setTimeout(() => {
      reconnectingRef.current = false;
    }, 3000);
  }, [reconnectSSE]);
  
  return (
    <Box
      position="fixed"
      bottom="4"
      right="4"
      zIndex="10"
      p="2"
      borderRadius="md"
      bg={bgColor}
      borderWidth="1px"
      borderColor={borderColor}
      boxShadow="sm"
      maxW="250px"
    >
      <HStack spacing={2}>
        {connected ? (
          <Tooltip label="Connected to server">
            <span>
              <Icon as={MdWifi} color="green.500" boxSize={5} />
            </span>
          </Tooltip>
        ) : connecting ? (
          <Tooltip label="Connecting...">
            <span>
              <Icon as={MdSync} color="blue.500" boxSize={5} />
            </span>
          </Tooltip>
        ) : (
          <Tooltip label="Disconnected">
            <span>
              <Icon as={MdWifiOff} color="orange.500" boxSize={5} />
            </span>
          </Tooltip>
        )}
        
        <Text fontSize="sm" fontWeight="medium" flex="1">
          {connected
            ? `Connected${timeSinceHeartbeat ? ` (${timeSinceHeartbeat}s ago)` : ''}`
            : connecting
              ? 'Connecting...'
              : 'Disconnected'
          }
        </Text>
        
        {!connected && !connecting && (
          <Button
            size="xs"
            onClick={handleReconnect}
            colorScheme="blue"
            isDisabled={reconnectingRef.current}
            leftIcon={<Icon as={MdSync} />}
          >
            Reconnect
          </Button>
        )}
      </HStack>
      
      {error && (
        <Text fontSize="xs" color="red.500" mt="1">
          {error.message || String(error)}
        </Text>
      )}
    </Box>
  );
};

export default ConnectionStatus;