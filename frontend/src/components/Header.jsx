import React from 'react';
import { 
  Flex, 
  Heading, 
  Spacer, 
  Button, 
  HStack, 
  Text, 
  useColorModeValue, 
  Link,
  Icon,
  Tooltip,
  Box 
} from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { MdWifiOff } from 'react-icons/md';
import { FaHistory } from 'react-icons/fa';

/**
 * Header component with authentication controls
 */
function Header() {
  // Get auth state and methods from context
  const { 
    isAuthenticated, 
    user, 
    loading, 
    handleLogout,
    handleLogin,
    connectionStatus 
  } = useAuth();

  // Theme colors
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const activeLinkColor = useColorModeValue('blue.500', 'blue.300');

  return (
    <Flex
      as="header"
      width="full"
      align="center"
      p={4}
      bg={bgColor}
      borderBottomWidth="1px"
      borderColor={borderColor}
      boxShadow="sm"
    >
      <Heading size="md">
        <Link as={RouterLink} to="/" _hover={{ textDecoration: 'none' }}>
          Auth Demo
        </Link>
      </Heading>
      
      <HStack ml={8} spacing={4}>
        <Link as={RouterLink} to="/" _hover={{ color: activeLinkColor }} px={2}>
          Home
        </Link>
        {isAuthenticated && (
          <Link 
            as={RouterLink} 
            to="/watch-history" 
            _hover={{ color: activeLinkColor }} 
            display="flex" 
            alignItems="center" 
            px={2}
          >
            <Icon as={FaHistory} mr={1} />
            Watch History
          </Link>
        )}
      </HStack>
      
      <Spacer />
      
      <HStack spacing={4}>
        {!connectionStatus.connected && (
          <Tooltip label="Connection lost">
            <Icon as={MdWifiOff} color="orange.500" />
          </Tooltip>
        )}
        
        {isAuthenticated && user && (
          <Text fontSize="sm" display={{ base: 'none', md: 'block' }}>
            Logged in as: <strong>{user.username}</strong>
          </Text>
        )}
        {!isAuthenticated ? (
          <Button 
            colorScheme="blue" 
            onClick={handleLogin} 
            isLoading={loading}
            isDisabled={!connectionStatus.connected}
            size={{ base: 'sm', md: 'md' }}
          >
            Login with Trakt
          </Button>
        ) : (
          <Button 
            variant="outline" 
            colorScheme="red" 
            onClick={handleLogout} 
            isLoading={loading}
            isDisabled={!connectionStatus.connected}
            size={{ base: 'sm', md: 'md' }}
          >
            Logout
          </Button>
        )}
      </HStack>
    </Flex>
  );
}

export default Header;
