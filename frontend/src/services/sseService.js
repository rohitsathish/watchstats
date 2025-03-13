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
  onError: null
};

// Reconnection settings
const RECONNECT_INTERVAL = 3000; // Start with 3 seconds
const MAX_RECONNECT_INTERVAL = 30000; // Max 30 seconds
let reconnectTimeout = null;
let reconnectInterval = RECONNECT_INTERVAL;
let reconnectAttempts = 0;
let isReconnecting = false;

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
    connectionStatus.connecting = true;
    connectionStatus.connected = false;
    
    if (handlers.onStatusChange) {
      handlers.onStatusChange(connectionStatus);
    }

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
        connectionStatus.connected = true;
        connectionStatus.connecting = false;
        connectionStatus.error = null;
        
        if (eventHandlers.onConnection) {
          eventHandlers.onConnection(data);
        }
        
        if (handlers.onStatusChange) {
          handlers.onStatusChange(connectionStatus);
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
      connectionStatus.connected = false;
      connectionStatus.connecting = false;
      connectionStatus.error = new Error('Connection error');
      
      if (eventHandlers.onError) {
        eventHandlers.onError(connectionStatus.error);
      }
      
      if (handlers.onStatusChange) {
        handlers.onStatusChange(connectionStatus);
      }
      
      // Close and attempt to reconnect
      disconnect();
      handleReconnect();
    });

    return true;
  } catch (error) {
    console.error('Failed to establish SSE connection:', error);
    
    // Update status
    connectionStatus.connected = false;
    connectionStatus.connecting = false;
    connectionStatus.error = error;
    
    if (eventHandlers.onError) {
      eventHandlers.onError(error);
    }
    
    if (handlers.onStatusChange) {
      handlers.onStatusChange(connectionStatus);
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
  
  // Update status
  connectionStatus.connected = false;
  connectionStatus.connecting = false;
  
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

export const sseService = {
  connect,
  disconnect,
  updateConnection,
  getStatus
};