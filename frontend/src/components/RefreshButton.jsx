import React from 'react';
import { Button, Icon, Tooltip } from '@chakra-ui/react';
import { FaSyncAlt } from 'react-icons/fa';

const RefreshButton = ({ 
  onClick, 
  isRefreshing = false, 
  hasNewData = false, 
  size = 'sm',
  tooltip = 'Refresh data'
}) => {
  return (
    <Tooltip label={tooltip}>
      <Button
        size={size}
        colorScheme={hasNewData ? "teal" : "gray"}
        leftIcon={<Icon as={FaSyncAlt} />}
        isLoading={isRefreshing}
        loadingText="Refreshing"
        onClick={onClick}
        variant={hasNewData ? "solid" : "outline"}
      >
        {hasNewData ? "New Data Available" : "Refresh"}
      </Button>
    </Tooltip>
  );
};

export default RefreshButton;