/**
 * Server-Sent Events (SSE) service for real-time updates
 * 
 * This service manages the connection to the SSE endpoint and handles events.
 */

// Singleton instance
let eventSource = null;
let connectionStatus = {
  connected: false,
  connecting: false,
  error: null,
  lastHeartbeat: null
};

// Event handlers
let eventHandlers = {
  onConnection: null,
  onAuthUpdate: null,
  onHeartbeat: null,
  onDisconnect: null,
  onError: null,
  onStatusChange: null  // Add this line
};

// Reconnection settings
const RECONNECT_INTERVAL = 3000; // Start with 3 seconds
const MAX_RECONNECT_INTERVAL = 30000; // Max 30 seconds
let reconnectTimeout = null;
let reconnectInterval = RECONNECT_INTERVAL;
let reconnectAttempts = 0;
let isReconnecting = false;

// Heartbeat monitoring
let heartbeatMonitorInterval = null;

/**
 * Connect to the SSE endpoint
 * 
 * @param {Object} handlers - Event handlers
 * @param {function} handlers.onConnection - Called when connection is established
 * @param {function} handlers.onAuthUpdate - Called on auth state change
 * @param {function} handlers.onHeartbeat - Called on heartbeat
 * @param {function} handlers.onDisconnect - Called when disconnected
 * @param {function} handlers.onError - Called on connection error
 */
const connect = (handlers = {}) => {
  // Update handlers
  eventHandlers = { ...eventHandlers, ...handlers };

  // Don't connect if already connected or connecting
  if (eventSource && (eventSource.readyState === 0 || eventSource.readyState === 1)) {
    console.log('SSE connection already exists');
    return;
  }

  try {
    // Update status
    updateStatus({
      connecting: true,
      connected: false
    });

    // Create EventSource - Updated to use the new events/stream endpoint
    const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
    eventSource = new EventSource(`${apiBaseUrl}/events/stream`, { 
      withCredentials: true 
    });

    // Connection opened
    eventSource.addEventListener('open', (event) => {
      console.log('SSE connection opened');
      reconnectAttempts = 0;
      reconnectInterval = RECONNECT_INTERVAL;
    });

    // Listen for connection event
    eventSource.addEventListener('connection', (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('SSE connection event:', data);
        
        // Update connection status
        updateStatus({
          connected: true,
          connecting: false,
          error: null
        });
        
        if (eventHandlers.onConnection) {
          eventHandlers.onConnection(data);
        }
      } catch (error) {
        console.error('Error parsing connection event:', error);
      }
    });

    // Auth update event
    eventSource.addEventListener('auth_update', (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('SSE auth update event:', data);
        
        if (eventHandlers.onAuthUpdate) {
          eventHandlers.onAuthUpdate(data);
        }
      } catch (error) {
        console.error('Error parsing auth_update event:', error);
      }
    });

    // Heartbeat event
    eventSource.addEventListener('heartbeat', (event) => {
      try {
        const data = JSON.parse(event.data);
        connectionStatus.lastHeartbeat = data.timestamp;
        
        if (eventHandlers.onHeartbeat) {
          eventHandlers.onHeartbeat(data);
        }
      } catch (error) {
        console.error('Error parsing heartbeat event:', error);
      }
    });

    // Error handling
    eventSource.addEventListener('error', (event) => {
      console.error('SSE connection error:', event);
      
      // Update status
      const error = new Error('Connection error');
      updateStatus({
        connected: false,
        connecting: false,
        error: error
      });
      
      if (eventHandlers.onError) {
        eventHandlers.onError(error);
      }
      
      // Close and attempt to reconnect
      disconnect();
      handleReconnect();
    });

    // Start heartbeat monitoring
    if (heartbeatMonitorInterval) {
      clearInterval(heartbeatMonitorInterval);
    }
    
    heartbeatMonitorInterval = setInterval(() => {
      // Check if we've received a heartbeat recently (within 30 seconds)
      if (connectionStatus.lastHeartbeat) {
        const now = Date.now();
        const lastHeartbeat = new Date(connectionStatus.lastHeartbeat).getTime();
        const timeSinceHeartbeat = now - lastHeartbeat;
        
        // If no heartbeat for more than 30 seconds, reconnect
        if (timeSinceHeartbeat > 30000) {
          console.warn(`No heartbeat received for ${Math.round(timeSinceHeartbeat / 1000)}s, reconnecting...`);
          updateConnection();
        }
      }
    }, 10000); // Check every 10 seconds

    return true;
  } catch (error) {
    console.error('Failed to establish SSE connection:', error);
    
    // Update status
    updateStatus({
      connected: false,
      connecting: false,
      error: error
    });
    
    if (eventHandlers.onError) {
      eventHandlers.onError(error);
    }
    
    // Attempt to reconnect
    handleReconnect();
    return false;
  }
};

/**
 * Handle reconnection with exponential backoff
 */
const handleReconnect = () => {
  if (isReconnecting) return;
  
  isReconnecting = true;
  
  if (reconnectTimeout) {
    clearTimeout(reconnectTimeout);
  }
  
  // Calculate backoff time (with jitter)
  const jitter = Math.random() * 1000;
  const backoff = Math.min(reconnectInterval * Math.pow(1.5, reconnectAttempts), MAX_RECONNECT_INTERVAL) + jitter;
  
  console.log(`SSE reconnecting in ${Math.round(backoff / 1000)}s (attempt ${reconnectAttempts + 1})`);
  
  reconnectTimeout = setTimeout(() => {
    reconnectAttempts++;
    isReconnecting = false;
    connect(eventHandlers);
  }, backoff);
};

/**
 * Disconnect from the SSE endpoint
 */
const disconnect = () => {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  
  // Clear any pending reconnect
  if (reconnectTimeout) {
    clearTimeout(reconnectTimeout);
    reconnectTimeout = null;
  }
  
  // Clear heartbeat monitor
  if (heartbeatMonitorInterval) {
    clearInterval(heartbeatMonitorInterval);
    heartbeatMonitorInterval = null;
  }
  
  // Update status
  updateStatus({ connected: false, connecting: false });
  
  if (eventHandlers.onDisconnect) {
    eventHandlers.onDisconnect();
  }
};

/**
 * Update the connection (reconnect)
 * Use this when auth state changes
 */
const updateConnection = () => {
  disconnect();
  
  // Reset reconnect parameters
  reconnectAttempts = 0;
  reconnectInterval = RECONNECT_INTERVAL;
  
  // Reconnect immediately
  connect(eventHandlers);
};

/**
 * Get the current connection status
 */
const getStatus = () => {
  return { ...connectionStatus };
};

/**
 * Update the connection status
 * @param {Object} newStatus - New status properties to update
 */
const updateStatus = (newStatus) => {
  connectionStatus = { ...connectionStatus, ...newStatus };
  
  if (eventHandlers.onStatusChange) {
    eventHandlers.onStatusChange(connectionStatus);
  }
};

export const sseService = {
  connect,
  disconnect,
  updateConnection,
  getStatus,
  updateStatus
};