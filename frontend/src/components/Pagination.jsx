import React from 'react';
import { 
  Flex, 
  Button, 
  HStack, 
  Text,
  IconButton,
  ButtonGroup
} from '@chakra-ui/react';
import { 
  ChevronLeftIcon, 
  ChevronRightIcon, 
  ArrowLeftIcon, 
  ArrowRightIcon 
} from '@chakra-ui/icons';

const Pagination = ({ 
  totalItems = 0,
  currentPage = 1, 
  pageSize = 10, 
  onPageChange = () => {},
  showTotalItems = true
}) => {
  // Calculate total number of pages
  const totalPages = Math.ceil(totalItems / pageSize);
  
  // Generate page numbers to display
  const getPageNumbers = () => {
    // Always show first page, last page, current page, and pages around current page
    const delta = 1; // Number of pages to show before and after current page
    const pages = [];
    
    // Start with page 1
    pages.push(1);
    
    // Calculate range around current page
    const rangeStart = Math.max(2, currentPage - delta);
    const rangeEnd = Math.min(totalPages - 1, currentPage + delta);
    
    // Add ellipsis after page 1 if needed
    if (rangeStart > 2) {
      pages.push('...');
    }
    
    // Add pages around current page
    for (let i = rangeStart; i <= rangeEnd; i++) {
      pages.push(i);
    }
    
    // Add ellipsis before last page if needed
    if (rangeEnd < totalPages - 1) {
      pages.push('...');
    }
    
    // Add last page if there's more than 1 page
    if (totalPages > 1) {
      pages.push(totalPages);
    }
    
    return pages;
  };
  
  // Handle page change
  const handlePageChange = (page) => {
    if (page >= 1 && page <= totalPages) {
      onPageChange(page);
    }
  };
  
  // No pagination needed if there's only 1 page or no items
  if (totalPages <= 1) {
    return showTotalItems ? (
      <Flex justify="flex-end" py={4}>
        <Text fontSize="sm">{totalItems} items</Text>
      </Flex>
    ) : null;
  }

  const pageNumbers = getPageNumbers();

  return (
    <Flex justify="space-between" align="center" mt={4} py={2}>
      <HStack spacing={1}>
        {/* Previous page buttons */}
        <IconButton
          icon={<ArrowLeftIcon />}
          size="sm"
          variant="ghost"
          isDisabled={currentPage === 1}
          onClick={() => handlePageChange(1)}
          aria-label="First page"
        />
        <IconButton
          icon={<ChevronLeftIcon />}
          size="sm"
          variant="ghost"
          isDisabled={currentPage === 1}
          onClick={() => handlePageChange(currentPage - 1)}
          aria-label="Previous page"
        />
      </HStack>

      {/* Page number buttons */}
      <ButtonGroup spacing={1} variant="outline" size="sm">
        {pageNumbers.map((page, index) => 
          page === '...' ? (
            <Text key={`ellipsis-${index}`} px={2}>...</Text>
          ) : (
            <Button
              key={page}
              colorScheme={currentPage === page ? 'blue' : 'gray'}
              variant={currentPage === page ? 'solid' : 'outline'}
              onClick={() => handlePageChange(page)}
            >
              {page}
            </Button>
          )
        )}
      </ButtonGroup>

      <HStack spacing={1}>
        {/* Next page buttons */}
        <IconButton
          icon={<ChevronRightIcon />}
          size="sm"
          variant="ghost"
          isDisabled={currentPage === totalPages}
          onClick={() => handlePageChange(currentPage + 1)}
          aria-label="Next page"
        />
        <IconButton
          icon={<ArrowRightIcon />}
          size="sm"
          variant="ghost"
          isDisabled={currentPage === totalPages}
          onClick={() => handlePageChange(totalPages)}
          aria-label="Last page"
        />
      </HStack>
    </Flex>
  );
};

export default Pagination;