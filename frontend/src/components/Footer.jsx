import React from 'react'
import { Box, Text, Flex, Badge, useColorModeValue, Tooltip } from '@chakra-ui/react'

//#%% Footer Component
function Footer({ apiStatus }) {
  const textColor = useColorModeValue('gray.500', 'gray.400')
  const bgColor = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.700')
  
  // Determine badge color based on API status
  const badgeColor = apiStatus === 'Connected' ? 'green' : 
                    apiStatus === 'Checking...' ? 'yellow' : 'red'
  
  return (
    <Box 
      as="footer" 
      width="full"
      py={4}
      borderTopWidth="1px"
      borderColor={borderColor}
      bg={bgColor}
    >
      <Flex 
        direction={{ base: 'column', md: 'row' }} 
        align="center" 
        justify="space-between"
        px={4}
        gap={2}
      >
        <Text fontSize="sm" color={textColor}>
          &copy; {new Date().getFullYear()} Auth Demo
        </Text>
        
        <Tooltip label={`API is ${apiStatus.toLowerCase()}`}>
          <Badge colorScheme={badgeColor} variant="subtle">
            API: {apiStatus}
          </Badge>
        </Tooltip>
      </Flex>
    </Box>
  )
}

export default Footer
