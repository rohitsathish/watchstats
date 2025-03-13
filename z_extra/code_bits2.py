# %%

websocket.py

# %% Socket.IO integration for authentication
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Set

import socketio

# Setup logging
logger = logging.getLogger(__name__)

# Create a Socket.IO server
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",  # Update with your allowed origins in production
    ping_interval=25,  # Send ping every 25 seconds
    ping_timeout=60,  # Wait 60 seconds for pong before disconnect
    logger=True,
    engineio_logger=False,  # Set to True only for debugging
)

# User tracking - keep track of user sessions by UUID
# Format: {uuid: [session_id1, session_id2, ...]}
user_sessions: Dict[str, list] = {}

# Session tracking - keep track of session to UUID mapping
# Format: {session_id: uuid}
session_uuids: Dict[str, str] = {}

# Anonymous sessions - sessions without authenticated UUID
# Format: [session_id1, session_id2, ...]
anonymous_sessions: Set[str] = set()

# Keep track of recently processed auth events to deduplicate
# Format: {uuid: last_event_timestamp}
recent_auth_events: Dict[str, int] = {}

# Deduplication time window in seconds
DEDUP_WINDOW_SECONDS = 5

from app.core.auth import get_user_credentials, get_uuid_from_jwt


# %% Socket event handlers
@sio.event
async def connect(sid: str, environ: Dict, auth: Optional[Dict] = None) -> None:
    """
    Handle new socket connections

    Args:
        sid: Socket session ID
        environ: WSGI environment dictionary
        auth: Authentication data
    """
    try:
        # Get uuid from JWT cookie
        uuid = None

        # Extract cookie header from environ
        cookie_header = environ.get("HTTP_COOKIE", "")

        # Simple cookie parser to extract JWT token
        if cookie_header:
            from http.cookies import SimpleCookie

            cookie = SimpleCookie()
            cookie.load(cookie_header)

            # Create mock request object with cookies for get_uuid_from_jwt
            from fastapi import Request

            mock_request = Request(
                scope={
                    "type": "http",
                    "headers": [],
                    "cookies": {k: v.value for k, v in cookie.items()},
                }
            )

            # Extract UUID from JWT cookie
            uuid = get_uuid_from_jwt(mock_request)

        if uuid:
            # Verify the UUID exists in database through cache
            user_creds = await get_user_credentials(uuid)
            if user_creds:
                # Register authenticated session
                await register_user_session(uuid, sid)

                # Send connection confirmation with auth status and user info
                await sio.emit(
                    "connection",
                    {
                        "status": "connected",
                        "authenticated": True,
                        "user": user_creds["user"],
                        "timestamp": int(datetime.now().timestamp()),
                    },
                    room=sid,
                )

                logger.info(
                    f"Authenticated connection established for user {uuid} (session {sid})"
                )
                return
            else:
                logger.debug(f"UUID {uuid} found in cookie but not in database/cache")
                # Fall through to anonymous connection

        # Handle anonymous connection - allow this for first-time users
        # Add to anonymous sessions set
        anonymous_sessions.add(sid)

        # Send connection confirmation for anonymous user
        await sio.emit(
            "connection",
            {
                "status": "connected",
                "authenticated": False,
                "timestamp": int(datetime.now().timestamp()),
            },
            room=sid,
        )

        logger.info(f"Anonymous connection established (session {sid})")

    except Exception as e:
        logger.error(f"Error handling connection for session {sid}: {str(e)}")
        # Try to send error to client
        try:
            await sio.emit(
                "error",
                {
                    "message": "Connection error occurred",
                    "timestamp": int(datetime.now().timestamp()),
                },
                room=sid,
            )
        except Exception:
            pass


# %% Socket disconnect handler
@sio.event
async def disconnect(sid: str) -> None:
    """
    Handle socket disconnections

    Args:
        sid: Socket session ID
    """
    try:
        # Get user info before cleaning up
        uuid = session_uuids.get(sid)

        # Unregister the session
        await unregister_session(sid)

        if uuid:
            logger.info(f"Connection closed for session {sid} (user {uuid})")
        else:
            logger.info(f"Connection closed for session {sid}")

    except Exception as e:
        logger.error(f"Error handling disconnection for session {sid}: {str(e)}")


# %% Socket error handler
@sio.event
async def error(sid: str, error: Dict) -> None:
    """
    Handle socket errors

    Args:
        sid: Socket session ID
        error: Error information
    """
    uuid = session_uuids.get(sid)
    if uuid:
        logger.error(f"Socket error for session {sid} (user {uuid}): {error}")
    else:
        logger.error(f"Socket error for session {sid}: {error}")


# %% User session management
async def register_user_session(uuid: str, sid: str) -> None:
    """
    Register a new socket session for a user by UUID

    Args:
        uuid: User UUID
        sid: Socket session ID
    """
    # If this was an anonymous session before, remove it from anonymous tracking
    if sid in anonymous_sessions:
        anonymous_sessions.remove(sid)

    if uuid not in user_sessions:
        user_sessions[uuid] = []

    # Only add if not already present to avoid duplicates
    if sid not in user_sessions[uuid]:
        user_sessions[uuid].append(sid)

    session_uuids[sid] = uuid
    logger.info(f"Registered session {sid} for user {uuid}")


# %% Session cleanup
async def unregister_session(sid: str) -> None:
    """
    Unregister a socket session

    Args:
        sid: Socket session ID to remove
    """
    # Check if this is an anonymous session
    if sid in anonymous_sessions:
        anonymous_sessions.remove(sid)
        logger.debug(f"Removed anonymous session {sid}")
        return

    # Handle authenticated session
    uuid = session_uuids.get(sid)
    if uuid:
        if sid in user_sessions.get(uuid, []):
            user_sessions[uuid].remove(sid)

        # If no more sessions for user, clean up
        if uuid in user_sessions and not user_sessions[uuid]:
            del user_sessions[uuid]
            # Also clean up from recent events tracking
            if uuid in recent_auth_events:
                del recent_auth_events[uuid]

        # Remove from session map
        del session_uuids[sid]
        logger.info(f"Unregistered session {sid} for user {uuid}")


# %% Send message to user
async def send_to_user(uuid: str, event: str, data: Dict[str, Any]) -> None:
    """
    Send a message to all sessions of a specific user with deduplication

    Args:
        uuid: Target user UUID
        event: Name of the event
        data: Data payload to send
    """
    # Skip if user has no active sessions
    if uuid not in user_sessions:
        logger.debug(f"No active sessions for user {uuid}")
        return

    # Add timestamp if not present
    if "timestamp" not in data:
        data["timestamp"] = int(datetime.now().timestamp())

    # Check for duplicate events for this user
    if event == "auth_update":
        current_time = data["timestamp"]
        last_event_time = recent_auth_events.get(uuid, 0)

        # Skip if we've sent a similar event recently
        if current_time - last_event_time < DEDUP_WINDOW_SECONDS:
            logger.debug(
                f"Skipping duplicate auth_update for user {uuid} within {DEDUP_WINDOW_SECONDS}s window"
            )
            return

        # Update last event time for this user
        recent_auth_events[uuid] = current_time

    # Send to all user sessions
    failed_sids = []
    for sid in user_sessions[uuid]:
        try:
            await sio.emit(event, data, room=sid)
            logger.debug(f"Sent {event} to user {uuid} (session {sid})")
        except Exception as e:
            logger.error(f"Error sending to user {uuid} (session {sid}): {str(e)}")
            failed_sids.append(sid)

    # Clean up any failed sessions
    for sid in failed_sids:
        await unregister_session(sid)


# %% Broadcast heartbeat to all clients
async def broadcast_heartbeat() -> None:
    """Send a heartbeat message to all connected clients"""
    timestamp = int(datetime.now().timestamp())
    data = {"timestamp": timestamp}

    # Get list of all current sessions
    all_sids = list(session_uuids.keys()) + list(anonymous_sessions)

    # Send heartbeat to each session
    failed_sids = []
    for sid in all_sids:
        try:
            await sio.emit("heartbeat", data, room=sid)
        except Exception as e:
            logger.error(f"Error sending heartbeat to session {sid}: {str(e)}")
            failed_sids.append(sid)

    # Clean up any failed sessions
    for sid in failed_sids:
        await unregister_session(sid)

    # Log only if we have active connections and after cleanup
    if session_uuids or anonymous_sessions:
        logger.debug(
            f"Sent heartbeat to {len(session_uuids) + len(anonymous_sessions)} clients"
        )

# %%
socketservice.js

import { io } from 'socket.io-client';
import { authService } from './authService';

/**
 * Socket service for real-time communication with the backend
 */
class SocketService {
  /**
   * Initialize the socket service
   */
  constructor() {
    this.socket = null;
    this.connectionStatus = {
      connected: false,
      connecting: false,
      error: null,
      lastHeartbeat: null
    };
    this.reconnectAttempts = 0;
    this.MAX_RECONNECT_ATTEMPTS = 5;
    this.connectionUpdateTimeout = null;

    // Track event handlers to avoid duplicates
    this.authUpdateCallback = null;
    this.statusChangeCallback = null;

    // Track processed events to deduplicate
    this.lastProcessedEventTimestamp = null;
    this.lastProcessedEventContent = null;
  }

  /**
   * Connect to the socket server
   * @param {Function} onAuthUpdate - Callback for auth updates
   * @param {Function} onStatusChange - Callback for connection status changes
   * @returns {void}
   */
  connect(onAuthUpdate, onStatusChange) {
    // Store callbacks
    this.authUpdateCallback = onAuthUpdate;
    this.statusChangeCallback = onStatusChange;

    // If socket exists and is connected, just update callbacks without reconnecting
    if (this.socket && this.socket.connected) {
      console.log('Socket already connected, updating callbacks only');
      return;
    }

    // If socket exists but isn't connected, clean it up first
    if (this.socket) {
      // Only close the connection but preserve state
      this.closeConnection();
    }

    const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

    try {
      this.connectionStatus.connecting = true;

      // Debounce connection status updates
      this.debounceConnectionStatusUpdate();

      // Initialize socket with withCredentials to send cookies
      this.socket = io(baseURL, {
        path: '/ws/socket.io',  // This matches our FastAPI mount path
        reconnection: true,
        reconnectionAttempts: this.MAX_RECONNECT_ATTEMPTS,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        timeout: 10000,
        withCredentials: true,  // Important for sending cookies
        transports: ['websocket', 'polling']  // Try WebSocket first, fall back to polling
      });

      // Set up connection event handlers
      this.setupEventListeners();

    } catch (error) {
      console.error('Error initializing socket:', error);
      this.connectionStatus = {
        connected: false,
        connecting: false,
        error: `Failed to initialize: ${error.message}`,
        lastHeartbeat: null
      };

      this.debounceConnectionStatusUpdate();
    }
  }

  /**
   * Debounce connection status updates to avoid UI flicker
   * @private
   */
  debounceConnectionStatusUpdate() {
    // Clear any pending update
    if (this.connectionUpdateTimeout) {
      clearTimeout(this.connectionUpdateTimeout);
    }

    // Schedule update after a short delay
    this.connectionUpdateTimeout = setTimeout(() => {
      if (this.statusChangeCallback) {
        this.statusChangeCallback({...this.connectionStatus});
      }
      this.connectionUpdateTimeout = null;
    }, 50);
  }

  /**
   * Set up event listeners for the socket
   * @private
   */
  setupEventListeners() {
    if (!this.socket) return;

    // Remove any existing listeners first to prevent duplicates
    this.socket.removeAllListeners();

    // Set up connection event handlers
    this.socket.on('connect', () => {
      console.log('Socket connected');
      this.connectionStatus = {
        connected: true,
        connecting: false,
        error: null,
        lastHeartbeat: Date.now()
      };
      this.reconnectAttempts = 0;
      this.debounceConnectionStatusUpdate();
    });

    this.socket.on('disconnect', (reason) => {
      console.log(`Socket disconnected: ${reason}`);
      this.connectionStatus = {
        connected: false,
        connecting: reason === 'io server disconnect' ? false : true,
        error: `Disconnected: ${reason}`,
        lastHeartbeat: this.connectionStatus.lastHeartbeat
      };
      this.debounceConnectionStatusUpdate();
    });

    this.socket.on('connect_error', (error) => {
      console.error('Socket connection error:', error);
      this.reconnectAttempts++;
      this.connectionStatus = {
        connected: false,
        connecting: this.reconnectAttempts < this.MAX_RECONNECT_ATTEMPTS,
        error: `Connection error: ${error.message}`,
        lastHeartbeat: this.connectionStatus.lastHeartbeat
      };
      this.debounceConnectionStatusUpdate();
    });

    // Set up auth event listeners
    this.socket.on('auth_update', (data) => {
      // Skip processing if we don't have a callback
      if (!this.authUpdateCallback) return;

      // Implement more robust deduplication based on both timestamp and content
      if (this.shouldSkipDuplicateEvent(data)) {
        console.log('Ignoring duplicate auth update event');
        return;
      }

      console.log('Received auth update:', data);
      this.connectionStatus.lastHeartbeat = Date.now();
      this.debounceConnectionStatusUpdate();

      // Store reference of this event for deduplication
      this.updateLastProcessedEvent(data);

      // Invoke the callback
      this.authUpdateCallback(data);
    });

    this.socket.on('heartbeat', (data) => {
      console.log('Received heartbeat');
      this.connectionStatus.lastHeartbeat = Date.now();
      this.debounceConnectionStatusUpdate();
    });

    // Custom connection event from our server
    this.socket.on('connection', (data) => {
      console.log('Received connection confirmation:', data);
      this.connectionStatus.lastHeartbeat = Date.now();
      this.debounceConnectionStatusUpdate();
    });
  }

  /**
   * Check if an event is a duplicate that should be skipped
   * @param {Object} data - Event data to check
   * @returns {boolean} True if event should be skipped
   * @private
   */
  shouldSkipDuplicateEvent(data) {
    // If no previous event, this isn't a duplicate
    if (!this.lastProcessedEventTimestamp || !this.lastProcessedEventContent) {
      return false;
    }

    // Check timestamp
    if (data.timestamp && data.timestamp <= this.lastProcessedEventTimestamp) {
      // If timestamps match, also check content to avoid skipping different events with same timestamp
      if (JSON.stringify(data) === this.lastProcessedEventContent) {
        return true;
      }
    }

    return false;
  }

  /**
   * Update reference of last processed event for deduplication
   * @param {Object} data - Event data that was processed
   * @private
   */
  updateLastProcessedEvent(data) {
    if (data.timestamp) {
      this.lastProcessedEventTimestamp = data.timestamp;
      this.lastProcessedEventContent = JSON.stringify(data);
    }
  }

  /**
   * Close the socket connection but preserve state
   * @private
   */
  closeConnection() {
    if (this.socket) {
      console.log('Closing socket connection');

      // Remove all event listeners
      this.socket.removeAllListeners();

      // Close the connection
      this.socket.disconnect();
      this.socket = null;

      // Update connection status but preserve some state
      this.connectionStatus.connected = false;
      this.connectionStatus.connecting = false;

      this.debounceConnectionStatusUpdate();
      // Don't reset other state to maintain a smoother reconnection
    }
  }

  /**
   * Disconnect from the socket server and reset all state
   */
  disconnect() {
    if (this.socket) {
      console.log('Disconnecting socket and resetting state');

      // Remove all event listeners
      this.socket.removeAllListeners();

      // Close the connection
      this.socket.disconnect();
      this.socket = null;

      // Reset state
      this.connectionStatus = {
        connected: false,
        connecting: false,
        error: null,
        lastHeartbeat: null
      };

      this.debounceConnectionStatusUpdate();

      // Reset event tracking
      this.lastProcessedEventTimestamp = null;
      this.lastProcessedEventContent = null;
    }
  }

  /**
   * Check if socket is currently connected
   * @returns {boolean} Connection status
   */
  isConnected() {
    return this.socket && this.socket.connected;
  }

  /**
   * Manually reconnect to the socket server
   * @param {Function} onAuthUpdate - Callback for auth updates
   * @param {Function} onStatusChange - Callback for connection status changes
   */
  reconnect(onAuthUpdate, onStatusChange) {
    // Only close the connection but preserve state for reconnection
    this.closeConnection();
    this.reconnectAttempts = 0;
    this.connect(onAuthUpdate, onStatusChange);
  }

  /**
   * Update the connection after auth state changes
   * Simply reconnects to refresh the socket connection with updated auth cookies
   */
  updateConnection() {
    console.log('Updating socket connection after auth state change');

    if (this.socket && this.authUpdateCallback && this.statusChangeCallback) {
      // Reconnect to ensure cookies are sent with the updated auth state
      this.reconnect(this.authUpdateCallback, this.statusChangeCallback);
    }
  }
}

// Export a singleton instance
export const socketService = new SocketService();

# %%

old_db.py

# %%
"""
Database connection and query module.

This module provides optimized database operations with connection pooling
and prepared statements for the application.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
from dotenv import load_dotenv

# Import cache module
from app.db.cache import (
    cache_user,
    get_cached_user,
    invalidate_user_cache,
    get_cache_stats,
)

# Load environment variables from .env file
load_dotenv()
logger = logging.getLogger(__name__)

# Load database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

# Global connection pool
_pool = None


# %%
async def init_db() -> None:
    """Initialize database connection pool."""
    global _pool
    try:
        # Create optimized connection pool
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=10,  # Ensure minimum connections ready
            max_size=20,  # Limit maximum connections
            max_queries=50000,  # Maximum queries per connection
            max_inactive_connection_lifetime=300.0,  # 5 minutes
            command_timeout=60.0,  # 60 second query timeout
            statement_cache_size=100,  # Cache prepared statements
        )
        logger.info("Database pool initialized successfully")

        # Initialize prepared statements
        await _prepare_statements()

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


# %%
async def _prepare_statements() -> None:
    """Pre-prepare commonly used SQL statements for performance."""
    try:
        async with get_connection() as conn:
            # Prepare statement for user lookups
            await conn.prepare(
                """
                SELECT 
                    uuid, user_id, access_token, access_token_expires_at, 
                    refresh_token, username, name, slug, avatar_url, timezone
                FROM users 
                WHERE uuid = $1
            """
            )

            # Prepare statement for user insertion/update
            await conn.prepare(
                """
                INSERT INTO users (
                    uuid, user_id, access_token, access_token_expires_at, 
                    refresh_token, username, name, slug, avatar_url, timezone,
                    last_db_update
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP)
                ON CONFLICT (uuid) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    access_token = EXCLUDED.access_token,
                    access_token_expires_at = EXCLUDED.access_token_expires_at,
                    refresh_token = EXCLUDED.refresh_token,
                    username = EXCLUDED.username,
                    name = EXCLUDED.name,
                    slug = EXCLUDED.slug,
                    avatar_url = EXCLUDED.avatar_url,
                    timezone = EXCLUDED.timezone,
                    last_db_update = CURRENT_TIMESTAMP
            """
            )

            logger.info("Database prepared statements initialized")
    except Exception as e:
        logger.error(f"Failed to prepare statements: {e}")


# %%
async def close_db() -> None:
    """Close database connection pool."""
    global _pool
    try:
        # Close database pool
        if _pool:
            await _pool.close()
            _pool = None
            logger.info("Database pool closed")

    except Exception as e:
        logger.error(f"Error during database shutdown: {e}")


# %%
async def get_pool() -> asyncpg.Pool:
    """Get or create the database connection pool with optimized settings."""
    global _pool
    if _pool is None:
        await init_db()
    return _pool


# %%
@asynccontextmanager
async def get_connection():
    """
    Context manager for getting a database connection with error handling.

    Yields:
        asyncpg.Connection: Database connection
    """
    pool = await get_pool()
    start_time = time.time()
    async with pool.acquire() as connection:
        try:
            yield connection
            query_time = time.time() - start_time
            if query_time > 1.0:  # Log slow queries (>1s)
                logger.warning(f"Slow database operation: {query_time:.2f}s")
        except asyncpg.PostgresError as e:
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during database operation: {e}")
            raise


# %%
async def test_connection() -> Dict[str, Any]:
    """
    Test database connection and return status info with detailed cache statistics.

    Returns:
        Dict[str, Any]: Connection information including version, connection pool and cache stats
    """
    try:
        async with get_connection() as conn:
            version = await conn.fetchval("SELECT version()")

            # Get pool statistics
            pool = await get_pool()
            pool_stats = {
                "min_size": pool._minsize,
                "max_size": pool._maxsize,
                "size": len(pool._holders),
                "free": pool._queue.qsize(),
            }

            # Get cache statistics from cache module
            cache_stats = await get_cache_stats()

            return {
                "status": "connected",
                "version": version,
                "pool": pool_stats,
                "cache": cache_stats,
            }
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return {"status": "failed", "error": str(e)}


# %%
async def get_user_by_uuid(uuid: str) -> Optional[Dict[str, Any]]:
    """
    Get user details from database by UUID with optimized query.

    Args:
        uuid: User's UUID from Trakt

    Returns:
        Optional[Dict[str, Any]]: User data or None if not found
    """
    if not uuid:
        return None

    try:
        logger.debug(f"Database lookup: Querying user with UUID {uuid}")
        start_time = time.time()

        async with get_connection() as conn:
            # Use the prepared statement for user lookup
            row = await conn.fetchrow(
                """
                SELECT 
                    uuid, user_id, access_token, access_token_expires_at,
                    refresh_token, username, name, slug, avatar_url, timezone
                FROM users 
                WHERE uuid = $1
                """,
                uuid,
            )

            query_time = time.time() - start_time

            if not row:
                logger.debug(
                    f"Database lookup: No user found with UUID {uuid} (took {query_time:.3f}s)"
                )
                return None

            # Convert row to dict
            user_data = dict(row)

            # Check if token is expired
            if (
                user_data["access_token_expires_at"]
                and datetime.now() > user_data["access_token_expires_at"]
            ):
                logger.info(f"Token expired for user: {uuid}")
                return None

            logger.debug(
                f"Database lookup: Retrieved user {uuid} ({user_data.get('username')}) in {query_time:.3f}s"
            )

            # Import the Trakt headers building function from auth module
            from app.core.auth import create_trakt_headers

            # Build and return the complete user credentials object
            headers = create_trakt_headers(user_data["access_token"])

            user_info = {
                "username": user_data["username"],
                "name": user_data["name"],
                "slug": user_data["slug"],
                "uuid": user_data["uuid"],
                "avatar": user_data["avatar_url"],
                "timezone": user_data["timezone"],
            }

            return {
                "uuid": user_data["uuid"],
                "access_token": user_data["access_token"],
                "refresh_token": user_data["refresh_token"],
                "headers": headers,
                "expires_at": user_data["access_token_expires_at"],
                "user": user_info,
            }

    except Exception as e:
        logger.error(f"Error getting user {uuid} from database: {e}")
        return None


# %%
async def upsert_user(
    uuid: str,
    user_id: str,
    access_token: str,
    access_token_expires_at: datetime,
    refresh_token: str,
    username: Optional[str] = None,
    name: Optional[str] = None,
    slug: Optional[str] = None,
    avatar_url: Optional[str] = None,
    timezone: Optional[str] = None,
) -> bool:
    """
    Insert or update a user in the database with optimized query.

    Args:
        uuid: User UUID from Trakt
        user_id: User ID from Trakt
        access_token: OAuth access token
        access_token_expires_at: Token expiration timestamp
        refresh_token: OAuth refresh token
        username: Trakt username
        name: User's display name
        slug: User's slug
        avatar_url: Avatar image URL
        timezone: User's timezone

    Returns:
        bool: True if operation was successful, False otherwise
    """
    if not uuid or not user_id or not access_token:
        logger.error("Missing required fields for user upsert")
        return False

    try:
        async with get_connection() as conn:
            # Use optimized upsert with transaction
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO users (
                        uuid, user_id, access_token, access_token_expires_at, 
                        refresh_token, username, name, slug, avatar_url, timezone,
                        last_db_update
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP)
                    ON CONFLICT (uuid) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        access_token = EXCLUDED.access_token,
                        access_token_expires_at = EXCLUDED.access_token_expires_at,
                        refresh_token = EXCLUDED.refresh_token,
                        username = EXCLUDED.username,
                        name = EXCLUDED.name,
                        slug = EXCLUDED.slug,
                        avatar_url = EXCLUDED.avatar_url,
                        timezone = EXCLUDED.timezone,
                        last_db_update = CURRENT_TIMESTAMP
                    """,
                    uuid,
                    user_id,
                    access_token,
                    access_token_expires_at,
                    refresh_token,
                    username,
                    name,
                    slug,
                    avatar_url,
                    timezone,
                )

            logger.info(f"User {uuid} upserted successfully")

            # Clear user from cache to ensure fresh data on next read
            invalidate_user_cache(uuid)

            return True

    except Exception as e:
        logger.error(f"Error upserting user {uuid}: {e}", exc_info=True)
        return False


# %%
async def delete_user(uuid: str) -> bool:
    """
    Delete a user from the database.

    Args:
        uuid: User's UUID

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    if not uuid:
        return False

    try:
        async with get_connection() as conn:
            result = await conn.execute("DELETE FROM users WHERE uuid = $1", uuid)
            # Check if any rows were affected
            if result and "DELETE" in result:
                logger.info(f"User {uuid} deleted successfully")

                # Clear user from cache
                invalidate_user_cache(uuid)

                return True
            else:
                logger.info(f"No user found with UUID {uuid} to delete")
                return False
    except Exception as e:
        logger.error(f"Error deleting user {uuid}: {e}")
        return False


# %%
async def get_cached_user_from_db(uuid: str) -> Optional[Dict[str, Any]]:
    """
    Get user from cache or database with TTL-based expiry.

    Args:
        uuid: User's UUID

    Returns:
        Optional[Dict[str, Any]]: User credentials or None if not found
    """
    if not uuid:
        return None

    # Try to get from cache first
    user_data = await get_cached_user(uuid)
    if user_data is not None:
        return user_data

    # Not in cache, get from database
    logger.info(f"Cache miss: Fetching user {uuid} from database")
    user_data = await get_user_by_uuid(uuid)

    # Store in cache if found
    if user_data:
        cache_user(uuid, user_data)

    return user_data


#%%

old schema.sql

-- User table for authentication and profile data
CREATE TABLE IF NOT EXISTS users (
    uuid VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    access_token VARCHAR NOT NULL,
    access_token_expires_at TIMESTAMP,
    refresh_token VARCHAR NOT NULL,
    username VARCHAR,
    name VARCHAR,
    slug VARCHAR,
    avatar_url VARCHAR,
    timezone VARCHAR,
    last_db_update TIMESTAMP DEFAULT NOW()
);

-- Create index on username for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- TMDB Media table for media data from TMDB
CREATE TABLE IF NOT EXISTS tmdb_media (
    show_tmdb_id INTEGER PRIMARY KEY,
    language VARCHAR,
    country VARCHAR,
    genres TEXT[],  -- Changed from JSONB to TEXT[]
    keywords TEXT[],  -- Changed from JSONB to TEXT[]
    certification VARCHAR,
    networks TEXT[],  -- Changed from JSONB to TEXT[]
    collection VARCHAR,
    poster_url VARCHAR,
    overview TEXT,
    runtime INTEGER,
    released TIMESTAMP,
    last_air_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);


-- IMDB Media table for media data from IMDB
CREATE TABLE IF NOT EXISTS imdb_media (
    show_imdb_id VARCHAR PRIMARY KEY,
    genres TEXT[],  -- Changed from JSONB to TEXT[]
    country VARCHAR,
    language VARCHAR,
    poster_url VARCHAR,
    certification VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Trakt Media table for media data from Trakt
CREATE TABLE IF NOT EXISTS trakt_media (
    trakt_url VARCHAR PRIMARY KEY,
    trakt_id BIGINT,
    media_type VARCHAR,
    title VARCHAR,
    season_num INTEGER,
    ep_num INTEGER,
    ep_title VARCHAR,
    ep_overview TEXT,
    overview TEXT,
    status VARCHAR,
    runtime INTEGER,
    released TIMESTAMP,
    show_released TIMESTAMP,
    total_episodes INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User Watch History table
CREATE TABLE IF NOT EXISTS user_watch_history (
    event_id BIGINT PRIMARY KEY,
    uuid VARCHAR REFERENCES users(uuid),
    media_type VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    trakt_url VARCHAR REFERENCES trakt_media(trakt_url),
    watched_at TIMESTAMP NOT NULL,
    season_num INTEGER,
    ep_num INTEGER,
    ep_title VARCHAR,
    runtime INTEGER,
    show_trakt_id BIGINT,
    show_imdb_id VARCHAR REFERENCES imdb_media(show_imdb_id),
    show_tmdb_id INTEGER REFERENCES tmdb_media(show_tmdb_id),
    user_rating INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_watch_history_uuid ON user_watch_history(uuid);
CREATE INDEX IF NOT EXISTS idx_watch_history_watched_at ON user_watch_history(watched_at);
CREATE INDEX IF NOT EXISTS idx_watch_history_title ON user_watch_history(title);
CREATE INDEX IF NOT EXISTS idx_watch_history_show_trakt_id ON user_watch_history(show_trakt_id);

-- User Ratings table
CREATE TABLE IF NOT EXISTS user_ratings (
    uuid VARCHAR REFERENCES users(uuid),
    rated_at TIMESTAMP NOT NULL,
    rating INTEGER NOT NULL,
    media_type VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    trakt_url VARCHAR REFERENCES trakt_media(trakt_url),
    show_trakt_id BIGINT,
    show_imdb_id VARCHAR REFERENCES imdb_media(show_imdb_id),
    show_tmdb_id INTEGER REFERENCES tmdb_media(show_tmdb_id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (uuid, trakt_url)
);

-- Create trigger to update timestamp on record changes
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for all relevant tables
CREATE TRIGGER update_users_timestamp
BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_tmdb_media_timestamp
BEFORE UPDATE ON tmdb_media
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_imdb_media_timestamp
BEFORE UPDATE ON imdb_media
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_trakt_media_timestamp
BEFORE UPDATE ON trakt_media
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_watch_history_timestamp
BEFORE UPDATE ON user_watch_history
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_user_ratings_timestamp
BEFORE UPDATE ON user_ratings
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

#%%

old cache.py

# %%
"""
Cache module for optimized data storage.

This module provides functions for efficient caching of frequently accessed data
using diskcache to reduce API calls and database queries.
"""
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Callable

import diskcache

# Set up logging
logger = logging.getLogger(__name__)

# Create cache directory if it doesn't exist
CACHE_DIR = Path("assets/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the cache with optimized settings
cache = diskcache.Cache(
    directory=str(CACHE_DIR),
    size_limit=int(1e9),  # 1GB cache size limit
    cull_limit=10,  # Cull 10% of entries when size limit is reached
    statistics=True,  # Enable hit/miss statistics
    tag_index=True,  # Enable tag-based eviction
    eviction_policy="least-recently-used",
)

# Set default TTL for different cache types (in seconds)
DEFAULT_TTLs = {
    "user": 86400 * 7,  # 7 days for user data
    "api": 86400,       # 1 day for API responses
    "processed": 86400, # 1 day for processed data
    "short": 3600,      # 1 hour for short-term data
    "db": 43200,        # 12 hours for database results
}

# %%
def cache_key(prefix: str, *args) -> str:
    """
    Generate a standardized cache key from prefix and arguments.
    
    Args:
        prefix: Key prefix
        *args: Additional key components
        
    Returns:
        str: Formatted cache key
    """
    components = [str(prefix)]
    for arg in args:
        if arg is not None:
            components.append(str(arg))
    return ":".join(components)


# %%
def cached(
    prefix: str, ttl: int = None, cache_type: str = "api", ignore_args: bool = False
) -> Callable:
    """
    Decorator for caching function results.
    
    Args:
        prefix: Cache key prefix
        ttl: Time-to-live in seconds (uses default TTL for cache_type if None)
        cache_type: Type of cache for default TTL selection
        ignore_args: Whether to ignore function arguments in cache key
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Determine TTL
            actual_ttl = ttl if ttl is not None else DEFAULT_TTLs.get(cache_type, 3600)
            
            # Generate cache key
            if ignore_args:
                key = cache_key(prefix, func.__name__)
            else:
                # Filter out AsyncClient instances and non-serializable objects
                filtered_args = [
                    arg for arg in args 
                    if not str(type(arg)).find("AsyncClient") > 0
                ]
                
                # Filter kwargs that can be part of cache key
                filtered_kwargs = {
                    k: v for k, v in kwargs.items() 
                    if k not in ["client", "headers"] and isinstance(v, (str, int, float, bool, type(None)))
                }
                
                # Create key using filtered arguments
                key = cache_key(
                    prefix, 
                    func.__name__,
                    *filtered_args,
                    **{k: filtered_kwargs[k] for k in sorted(filtered_kwargs)}
                )
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                logger.debug(f"Cache hit: {key}")
                return result
            
            # Not in cache, call function
            logger.debug(f"Cache miss: {key}")
            result = await func(*args, **kwargs)
            
            # Store result in cache
            if result is not None:
                cache.set(key, result, expire=actual_ttl)
                logger.debug(f"Cached result for {key} with TTL {actual_ttl}s")
                
            return result
        return wrapper
    return decorator


# %%
def invalidate_cache(pattern: str = None) -> int:
    """
    Invalidate cache entries matching a pattern.
    
    Args:
        pattern: Pattern to match cache keys (None to clear entire cache)
        
    Returns:
        int: Number of invalidated entries
    """
    if pattern is None:
        count = len(cache)
        cache.clear()
        logger.info(f"Cleared entire cache ({count} entries)")
        return count
    
    # Find keys matching pattern
    count = 0
    for key in list(cache):
        if isinstance(key, str) and pattern in key:
            cache.delete(key)
            count += 1
    
    logger.info(f"Invalidated {count} cache entries matching '{pattern}'")
    return count


# %%
def cache_user(uuid: str, user_data: Dict[str, Any], ttl: int = None) -> None:
    """
    Cache user data.
    
    Args:
        uuid: User UUID
        user_data: User data to cache
        ttl: Cache TTL in seconds
    """
    key = cache_key("user", uuid)
    actual_ttl = ttl if ttl is not None else DEFAULT_TTLs["user"]
    cache.set(key, user_data, expire=actual_ttl)
    logger.debug(f"Cached user data for {uuid} with TTL {actual_ttl}s")


# %%
async def get_cached_user(uuid: str) -> Optional[Dict[str, Any]]:
    """
    Get cached user data.
    
    Args:
        uuid: User UUID
        
    Returns:
        Optional[Dict[str, Any]]: Cached user data or None
    """
    key = cache_key("user", uuid)
    result = cache.get(key)
    if result is not None:
        logger.debug(f"Cache hit: user data for {uuid}")
        
        # Check if token is expired
        expires_at = result.get("expires_at")
        if expires_at and datetime.now() > expires_at:
            logger.info(f"Cached token expired for user {uuid}")
            cache.delete(key)
            return None
            
        return result
    
    logger.debug(f"Cache miss: user data for {uuid}")
    return None


# %%
def invalidate_user_cache(uuid: str) -> None:
    """
    Invalidate cached user data.
    
    Args:
        uuid: User UUID
    """
    key = cache_key("user", uuid)
    cache.delete(key)
    logger.debug(f"Invalidated cache for user {uuid}")


# %%
async def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dict[str, Any]: Cache statistics
    """
    stats = cache.stats()
    
    # Count entries by type
    entry_counts = {"user": 0, "api": 0, "processed": 0, "db": 0, "other": 0}
    entries_by_prefix = {}
    
    # Sample up to 1000 keys for statistics
    sample_size = min(1000, len(cache))
    sampled_keys = list(cache)[:sample_size]
    for key in sampled_keys:
        if isinstance(key, str):
            prefix = key.split(":")[0] if ":" in key else "other"
            
            # Count by broad category
            if prefix == "user":
                entry_counts["user"] += 1
            elif prefix in ["trakt", "tmdb", "omdb"]:
                entry_counts["api"] += 1
            elif prefix in ["watch_history", "processed"]:
                entry_counts["processed"] += 1
            elif prefix == "db":
                entry_counts["db"] += 1
            else:
                entry_counts["other"] += 1
                
            # Count by specific prefix
            if prefix not in entries_by_prefix:
                entries_by_prefix[prefix] = 0
            entries_by_prefix[prefix] += 1
    
    # Extrapolate counts if we sampled
    if sample_size < len(cache):
        scaling_factor = len(cache) / sample_size
        for key in entry_counts:
            entry_counts[key] = int(entry_counts[key] * scaling_factor)
        for key in entries_by_prefix:
            entries_by_prefix[key] = int(entries_by_prefix[key] * scaling_factor)
    
    return {
        "total_entries": len(cache),
        "size": cache.volume(),
        "hits": stats.get("hits", 0),
        "misses": stats.get("misses", 0),
        "hit_rate": stats.get("hit_rate", 0),
        "entries_by_type": entry_counts,
        "entries_by_prefix": entries_by_prefix,
    }