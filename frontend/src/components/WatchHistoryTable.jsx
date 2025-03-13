import React from 'react';
import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Flex,
  Text,
  Skeleton,
  Icon,
  Link,
  Tooltip
} from '@chakra-ui/react';
import { FaFilm, FaTv, FaExternalLinkAlt } from 'react-icons/fa';
import { format } from 'date-fns';

const WatchHistoryTable = ({ historyData, isLoading, emptyMessage }) => {
  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return format(date, 'MMM d, yyyy h:mm a');
    } catch (e) {
      return dateString;
    }
  };

  // Generate skeleton rows for loading state
  const renderSkeletons = () => {
    return Array.from({ length: 10 }).map((_, index) => (
      <Tr key={`skeleton-${index}`}>
        <Td><Skeleton height="20px" width="40px" /></Td>
        <Td><Skeleton height="20px" /></Td>
        <Td><Skeleton height="20px" width="100px" /></Td>
        <Td><Skeleton height="20px" width="80px" /></Td>
        <Td><Skeleton height="20px" width="150px" /></Td>
      </Tr>
    ));
  };

  // Render empty state
  if (!isLoading && (!historyData || historyData.length === 0)) {
    return (
      <Box textAlign="center" py={10} px={6}>
        <Text fontSize="lg" color="gray.500">
          {emptyMessage || "No watch history data found."}
        </Text>
      </Box>
    );
  }

  return (
    <Box overflowX="auto">
      <Table variant="simple" size="sm">
        <Thead>
          <Tr>
            <Th>Type</Th>
            <Th>Title</Th>
            <Th>Episode</Th>
            <Th>Runtime</Th>
            <Th>Watched At</Th>
          </Tr>
        </Thead>
        <Tbody>
          {isLoading ? (
            renderSkeletons()
          ) : (
            historyData.map((item) => (
              <Tr key={item.event_id}>
                <Td>
                  <Badge
                    colorScheme={item.media_type === 'movie' ? 'blue' : 'purple'}
                    display="flex"
                    alignItems="center"
                  >
                    <Icon 
                      as={item.media_type === 'movie' ? FaFilm : FaTv} 
                      mr={1} 
                    />
                    {item.media_type === 'movie' ? 'Movie' : 'TV'}
                  </Badge>
                </Td>
                <Td>
                  <Flex alignItems="center">
                    <Text fontWeight="medium">{item.title}</Text>
                    {item.trakt_url && (
                      <Tooltip label="Open in Trakt">
                        <Link 
                          href={item.trakt_url} 
                          isExternal 
                          ml={2}
                          color="gray.500"
                          _hover={{ color: 'blue.500' }}
                        >
                          <Icon as={FaExternalLinkAlt} boxSize={3} />
                        </Link>
                      </Tooltip>
                    )}
                  </Flex>
                </Td>
                <Td>
                  {item.media_type === 'episode' ? (
                    <Flex direction="column">
                      <Text>
                        S{String(item.season_num).padStart(2, '0')}E
                        {String(item.ep_num).padStart(2, '0')}
                      </Text>
                      <Text fontSize="xs" color="gray.500" noOfLines={1}>
                        {item.ep_title}
                      </Text>
                    </Flex>
                  ) : (
                    '-'
                  )}
                </Td>
                <Td>
                  {item.runtime ? `${item.runtime} min` : '-'}
                </Td>
                <Td>
                  {formatDate(item.watched_at)}
                </Td>
              </Tr>
            ))
          )}
        </Tbody>
      </Table>
    </Box>
  );
};

export default WatchHistoryTable;