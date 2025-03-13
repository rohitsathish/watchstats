import React from 'react';
import {
  Box,
  Container,
  Heading,
  Flex,
  Spacer,
  Text,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatGroup,
  Alert,
  AlertIcon,
  AlertDescription,
  Divider,
  Progress,
  Select,
  HStack,
  Button
} from '@chakra-ui/react';
import { useWatchHistory } from '../context/WatchHistoryContext';
import { useAuth } from '../context/AuthContext';
import WatchHistoryTable from '../components/WatchHistoryTable';
import RefreshButton from '../components/RefreshButton';
import Pagination from '../components/Pagination';

const WatchHistoryPage = () => {
  const { isAuthenticated, user } = useAuth();
  const {
    historyData,
    totalCount,
    isLoading,
    isRefreshing,
    hasNewData,
    error,
    isInitializing,
    refresh,
    pagination,
    changePage,
    changeLimit
  } = useWatchHistory();

  // Calculate watch stats
  const movieCount = historyData?.filter(item => item.media_type === 'movie').length || 0;
  const episodeCount = historyData?.filter(item => item.media_type === 'episode').length || 0;

  // Calculate current page
  const currentPage = Math.floor(pagination.offset / pagination.limit) + 1;

  // Handle page change
  const handlePageChange = (page) => {
    const newOffset = (page - 1) * pagination.limit;
    changePage(newOffset);
  };

  // Handle force refresh
  const handleForceRefresh = () => {
    refresh(true);
  };

  if (!isAuthenticated) {
    return (
      <Container maxW="container.xl" py={6}>
        <Alert status="warning">
          <AlertIcon />
          <AlertDescription>
            Please log in to view your watch history.
          </AlertDescription>
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxW="container.xl" py={6}>
      <Flex alignItems="center" mb={6}>
        <Heading size="lg">Watch History</Heading>
        <Spacer />
        <RefreshButton
          onClick={() => refresh(false)}
          isRefreshing={isRefreshing}
          hasNewData={hasNewData}
          tooltip="Refresh watch history data"
        />
      </Flex>

      {error && !isInitializing && (
        <Alert status="error" mb={4}>
          <AlertIcon />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {(isRefreshing || isInitializing) && (
        <Box mb={6}>
          <Text mb={2}>
            {isInitializing 
              ? "Loading watch history data..." 
              : "Refreshing watch history..."}
          </Text>
          <Progress isIndeterminate size="xs" colorScheme="blue" mb={2} />
          {isInitializing && (
            <Text fontSize="sm" color="gray.600">
              This may take a few moments while we fetch your data
            </Text>
          )}
        </Box>
      )}

      {!isInitializing && totalCount === 0 && !isRefreshing && (
        <Box mb={6}>
          <Alert status="info" mb={4}>
            <AlertIcon />
            <AlertDescription>
              No watch history found. If you're new, we need to fetch your data from Trakt.
            </AlertDescription>
          </Alert>
          <Button 
            colorScheme="blue" 
            onClick={handleForceRefresh} 
            isLoading={isRefreshing}
          >
            Get My Watch History
          </Button>
        </Box>
      )}

      {(!isInitializing && totalCount > 0) && (
        <>
          <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6} mb={6}>
            <StatGroup
              p={4}
              borderWidth="1px"
              borderRadius="lg"
              boxShadow="sm"
              bg="white"
            >
              <Stat>
                <StatLabel>Total Items</StatLabel>
                <StatNumber>{totalCount}</StatNumber>
              </Stat>
            </StatGroup>
            <StatGroup
              p={4}
              borderWidth="1px"
              borderRadius="lg"
              boxShadow="sm"
              bg="white"
            >
              <Stat>
                <StatLabel>Movies</StatLabel>
                <StatNumber>{movieCount}</StatNumber>
              </Stat>
            </StatGroup>
            <StatGroup
              p={4}
              borderWidth="1px"
              borderRadius="lg"
              boxShadow="sm"
              bg="white"
            >
              <Stat>
                <StatLabel>TV Episodes</StatLabel>
                <StatNumber>{episodeCount}</StatNumber>
              </Stat>
            </StatGroup>
          </SimpleGrid>

          <Box mb={4}>
            <Divider mb={4} />
            <Flex align="center" mb={4}>
              <Text>Recent History</Text>
              <Spacer />
              <HStack>
                <Text fontSize="sm">Items per page:</Text>
                <Select
                  size="sm"
                  width="80px"
                  value={pagination.limit}
                  onChange={(e) => changeLimit(Number(e.target.value))}
                >
                  <option value={10}>10</option>
                  <option value={25}>25</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                </Select>
              </HStack>
            </Flex>
          </Box>

          <Box
            borderWidth="1px"
            borderRadius="lg"
            overflow="hidden"
            bg="white"
          >
            <WatchHistoryTable
              historyData={historyData}
              isLoading={isLoading}
              emptyMessage={
                isRefreshing
                  ? "Loading your watch history..."
                  : "No watch history found. Click 'Refresh' to fetch your data."
              }
            />
          </Box>

          {/* Pagination controls */}
          {totalCount > 0 && (
            <Pagination
              totalItems={totalCount}
              currentPage={currentPage}
              pageSize={pagination.limit}
              onPageChange={handlePageChange}
            />
          )}
        </>
      )}
    </Container>
  );
};

export default WatchHistoryPage;