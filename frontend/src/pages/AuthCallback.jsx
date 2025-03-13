import React, { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Box, Spinner, VStack, Text, useToast } from '@chakra-ui/react';
import { useAuth } from '../context/AuthContext';

/**
 * Component that handles OAuth callback after Trakt.tv authentication
 */
const AuthCallback = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { handleCodeExchange } = useAuth();
  const toast = useToast();

  useEffect(() => {
    const handleAuthCallback = async () => {
      const params = new URLSearchParams(location.search);
      const code = params.get('code');
      const authStatus = params.get('auth');

      // Log parameters (truncate code for security)
      console.log('Auth callback params:', { 
        code: code ? `${code.substring(0, 5)}...` : 'none', 
        authStatus
      });

      try {
        // Handle backend redirect flow
        if (authStatus) {
          if (authStatus === 'success') {
            // No need to manually set UUID cookie anymore - backend sets HTTP-only JWT cookie
            console.log('Authentication successful via backend redirect');
            navigate('/');
            return;
          }
          
          // Handle backend auth errors
          const reason = params.get('reason') || 'unknown';
          throw new Error(reason === 'code_reused' 
            ? 'Authorization code has already been used'
            : reason === 'no_uuid'
              ? 'Could not retrieve UUID from Trakt'
              : 'Failed to authenticate with Trakt'
          );
        }

        // Handle direct Trakt.tv callback flow
        if (code) {
          console.log('Direct callback from Trakt, exchanging code for token');
          await handleCodeExchange(code);
          navigate('/');
          return;
        }

        // No auth information received
        throw new Error('No authorization information received');

      } catch (error) {
        console.error('Authentication error:', error);
        
        // Only show error toast if it's a real error, not a navigation event
        if (error.message !== 'Navigation aborted') {
          toast({
            title: 'Authentication Error',
            description: error.message || 'Failed to authenticate with Trakt',
            status: 'error',
            duration: 5000,
            isClosable: true,
          });
        }
        
        navigate('/');
      }
    };

    handleAuthCallback();
  }, [location.search, navigate, toast, handleCodeExchange]);

  return (
    <Box p={8}>
      <VStack spacing={4}>
        <Spinner size="xl" />
        <Text>Processing authentication...</Text>
        <Text fontSize="sm" color="gray.500">Please wait while we connect your account</Text>
      </VStack>
    </Box>
  );
};

export default AuthCallback;