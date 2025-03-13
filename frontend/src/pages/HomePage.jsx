import React from 'react';
import { Box, Heading, Text, VStack, Container } from '@chakra-ui/react';
import AuthStatus from '../components/AuthStatus';

const HomePage = () => {
  return (
    <Container maxW="container.md" py={8}>
      <VStack spacing={8} align="stretch">
        <Box textAlign="center">
          <Heading as="h1" size="xl" mb={4}>
            Trakt.tv Authentication Demo
          </Heading>
          <Text fontSize="lg" color="gray.600">
            A simple demonstration of OAuth authentication with Trakt.tv
          </Text>
        </Box>

        <Box display="flex" justifyContent="center" py={4}>
          <AuthStatus />
        </Box>

        <Box>
          <Heading as="h2" size="md" mb={4}>
            How it works
          </Heading>
          <Text>
            This demo shows how to implement OAuth authentication with Trakt.tv.
            Click the "Login with Trakt" button to start the authentication flow.
            After authenticating, you'll be redirected back to this page with your
            authentication status updated.
          </Text>
        </Box>
      </VStack>
    </Container>
  );
};

export default HomePage;
