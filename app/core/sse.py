"""
Server-Sent Events (SSE) implementation for real-time updates.

This module provides SSE functionality for sending real-time updates to clients,
replacing the previous WebSocket implementation.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

from fastapi import Request, Response
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask

# Setup logging
logger = logging.getLogger(__name__)


# Global state for tracking active client connections
class SSEState:
    """Global state for SSE connections"""

    # User tracking - keep track of user connections by UUID
    # Format: {uuid: [connection_id1, connection_id2, ...]}
    user_connections: Dict[str, List[str]] = {}

    # Connection tracking - keep track of connection to UUID mapping
    # Format: {connection_id: uuid}
    connection_uuids: Dict[str, str] = {}

    # Anonymous connections - connections without authenticated UUID
    # Format: set(connection_id1, connection_id2, ...)
    anonymous_connections: Set[str] = set()

    # Connection objects for cleanup
    # Format: {connection_id: SSEConnection}
    active_connections: Dict[str, "SSEConnection"] = {}

    # Deduplication tracking
    # Format: {uuid: last_event_timestamp}
    recent_events: Dict[str, int] = {}

    # Counter for generating unique connection IDs
    connection_counter: int = 0

    @classmethod
    def generate_connection_id(cls) -> str:
        """Generate a unique connection ID"""
        cls.connection_counter += 1
        return f"conn_{cls.connection_counter}_{int(datetime.now().timestamp())}"


@dataclass
class SSEConnection:
    """Represents an active SSE connection"""

    connection_id: str
    request: Request
    is_authenticated: bool
    user_uuid: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None

    # Queue for sending messages to this specific connection
    # Use default_factory to create a new asyncio.Queue for each instance
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Flag to indicate if the connection is active
    is_active: bool = True

    async def enqueue_message(self, event: str, data: Dict[str, Any]):
        """Add a message to this connection's queue"""
        await self.queue.put({"event": event, "data": data})


# Deduplication window in seconds
DEDUP_WINDOW_SECONDS = 5


async def register_connection(
    request: Request,
    uuid: Optional[str] = None,
    user_info: Optional[Dict[str, Any]] = None,
) -> SSEConnection:
    """
    Register a new SSE connection.

    Args:
        request: The FastAPI request object
        uuid: Optional user UUID for authenticated connections
        user_info: Optional user information to include in connection events

    Returns:
        SSEConnection: The created connection object
    """
    # Generate a unique connection ID
    connection_id = SSEState.generate_connection_id()

    # Create the connection object
    connection = SSEConnection(
        connection_id=connection_id,
        request=request,
        is_authenticated=bool(uuid),
        user_uuid=uuid,
        user_info=user_info,
    )

    # Store the connection
    SSEState.active_connections[connection_id] = connection

    # Track the connection appropriately
    if uuid:
        # Authenticated connection
        if uuid not in SSEState.user_connections:
            SSEState.user_connections[uuid] = []

        SSEState.user_connections[uuid].append(connection_id)
        SSEState.connection_uuids[connection_id] = uuid
        logger.info(
            f"Registered authenticated SSE connection {connection_id} for user {uuid}"
        )
    else:
        # Anonymous connection
        SSEState.anonymous_connections.add(connection_id)
        logger.info(f"Registered anonymous SSE connection {connection_id}")

    return connection


async def unregister_connection(connection_id: str) -> None:
    """
    Unregister an SSE connection.

    Args:
        connection_id: The connection ID to unregister
    """
    # Get the connection
    connection = SSEState.active_connections.get(connection_id)
    if not connection:
        logger.warning(
            f"Attempted to unregister non-existent connection {connection_id}"
        )
        return

    # Mark the connection as inactive
    connection.is_active = False

    try:
        # Remove from active connections
        if connection_id in SSEState.active_connections:
            del SSEState.active_connections[connection_id]

        # Check if this is an anonymous connection
        if connection_id in SSEState.anonymous_connections:
            SSEState.anonymous_connections.remove(connection_id)
            logger.debug(f"Removed anonymous SSE connection {connection_id}")
            return

        # Handle authenticated connection
        uuid = SSEState.connection_uuids.get(connection_id)
        if uuid:
            if connection_id in SSEState.user_connections.get(uuid, []):
                SSEState.user_connections[uuid].remove(connection_id)

            # If no more connections for user, clean up
            if (
                uuid in SSEState.user_connections
                and not SSEState.user_connections[uuid]
            ):
                del SSEState.user_connections[uuid]
                # Also clean up from recent events tracking
                if uuid in SSEState.recent_events:
                    del SSEState.recent_events[uuid]

            # Remove from connection map
            if connection_id in SSEState.connection_uuids:
                del SSEState.connection_uuids[connection_id]

            logger.info(f"Unregistered SSE connection {connection_id} for user {uuid}")
    except Exception as e:
        logger.error(f"Error unregistering SSE connection {connection_id}: {e}")


async def send_to_user(uuid: str, event: str, data: Dict[str, Any]) -> None:
    """
    Send a message to all connections of a specific user with deduplication.

    Args:
        uuid: Target user UUID
        event: Name of the event
        data: Data payload to send
    """
    # Skip if user has no active connections
    if uuid not in SSEState.user_connections:
        logger.debug(f"No active connections for user {uuid}")
        return

    # Add timestamp if not present
    if "timestamp" not in data:
        data["timestamp"] = int(datetime.now().timestamp())

    # Check for duplicate events for authentication updates
    if event == "auth_update":
        current_time = data["timestamp"]
        last_event_time = SSEState.recent_events.get(uuid, 0)

        # Skip if we've sent a similar event recently
        if current_time - last_event_time < DEDUP_WINDOW_SECONDS:
            logger.debug(
                f"Skipping duplicate auth_update for user {uuid} within {DEDUP_WINDOW_SECONDS}s window"
            )
            return

        # Update last event time for this user
        SSEState.recent_events[uuid] = current_time

    # Send to all user connections
    failed_connections = []
    for connection_id in SSEState.user_connections.get(uuid, []):
        try:
            connection = SSEState.active_connections.get(connection_id)
            if connection and connection.is_active:
                await connection.enqueue_message(event, data)
                logger.debug(
                    f"Enqueued {event} to user {uuid} (connection {connection_id})"
                )
            else:
                logger.debug(
                    f"Connection {connection_id} for user {uuid} is no longer active"
                )
                failed_connections.append(connection_id)
        except Exception as e:
            logger.error(
                f"Error sending to user {uuid} (connection {connection_id}): {e}"
            )
            failed_connections.append(connection_id)

    # Clean up any failed connections
    for connection_id in failed_connections:
        await unregister_connection(connection_id)


async def broadcast_event(event: str, data: Dict[str, Any]) -> None:
    """
    Broadcast an event to all connected clients.

    Args:
        event: Name of the event
        data: Data payload to send
    """
    # Add timestamp if not present
    if "timestamp" not in data:
        data["timestamp"] = int(datetime.now().timestamp())

    # Get list of all current connections - FIXED: use set.union() to combine different connection types
    # instead of the | operator which might not be supported between different collection types
    authenticated_connections = set(SSEState.connection_uuids.keys())
    all_connection_ids = set.union(
        authenticated_connections, SSEState.anonymous_connections
    )

    # Send event to each connection
    failed_connections = []
    for connection_id in all_connection_ids:
        try:
            connection = SSEState.active_connections.get(connection_id)
            if connection and connection.is_active:
                await connection.enqueue_message(event, data)
            else:
                logger.debug(f"Connection {connection_id} is no longer active")
                failed_connections.append(connection_id)
        except Exception as e:
            logger.error(f"Error broadcasting to connection {connection_id}: {e}")
            failed_connections.append(connection_id)

    # Clean up any failed connections
    for connection_id in failed_connections:
        await unregister_connection(connection_id)

    if all_connection_ids:
        logger.debug(f"Broadcast {event} to {len(all_connection_ids)} connections")


async def broadcast_heartbeat() -> None:
    """Send a heartbeat message to all connected clients"""
    timestamp = int(datetime.now().timestamp())
    data = {"timestamp": timestamp}

    # Use the broadcast function to send the heartbeat
    await broadcast_event("heartbeat", data)


async def start_heartbeat_task():
    """Start periodic heartbeat task"""
    while True:
        try:
            await broadcast_heartbeat()
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}", exc_info=True)

        # Wait for interval
        await asyncio.sleep(25)  # Send heartbeat every 25 seconds


async def event_generator(
    connection: SSEConnection,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate SSE events for a specific connection.

    Args:
        connection: The SSE connection

    Yields:
        Dict: Event message to be sent as SSE
    """
    # Send initial connection event
    connection_data = {
        "status": "connected",
        "authenticated": connection.is_authenticated,
        "timestamp": int(datetime.now().timestamp()),
    }

    # Add user info for authenticated connections
    if connection.is_authenticated and connection.user_info:
        connection_data["user"] = connection.user_info

    # Yield the initial connection event
    yield {"event": "connection", "data": json.dumps(connection_data)}

    # Set a watchdog to clean up the connection if it becomes stale
    # We'll use a task that will cancel itself when the connection is closed
    connection_id = connection.connection_id

    try:
        # Process messages from the queue
        while connection.is_active:
            try:
                # Use wait_for with a timeout to allow for connection cleanup
                message = await asyncio.wait_for(connection.queue.get(), timeout=60)

                if message:
                    yield {
                        "event": message["event"],
                        "data": json.dumps(message["data"]),
                    }

                    # Mark item as processed
                    connection.queue.task_done()
            except asyncio.TimeoutError:
                # No message for a while, send a keep-alive comment
                yield {"comment": f"keep-alive {datetime.now().isoformat()}"}
            except Exception as e:
                logger.error(
                    f"Error processing message for connection {connection_id}: {e}"
                )
                # If there's an error, send a comment with the error type but keep the connection
                yield {"comment": f"error: {type(e).__name__}"}
    except Exception as e:
        logger.error(f"Connection {connection_id} stream error: {e}")
    finally:
        # Ensure connection is unregistered when the generator stops
        await unregister_connection(connection_id)
        logger.info(f"Connection {connection_id} stream closed")


# This function is maintained for backward compatibility
# But the main SSE endpoint should now be handled by the events.py module
async def sse_endpoint(
    request: Request, user_uuid: Optional[str] = None, user_info: Optional[Dict] = None
) -> Response:
    """
    SSE endpoint handler for FastAPI.

    Args:
        request: FastAPI request object
        user_uuid: Optional user UUID for authenticated connections
        user_info: Optional user information

    Returns:
        Response: The SSE response
    """
    # Register the connection
    connection = await register_connection(request, user_uuid, user_info)

    # Create the event source response
    return EventSourceResponse(
        event_generator(connection),
        media_type="text/event-stream",
        background=BackgroundTask(unregister_connection, connection.connection_id),
    )


async def send_auth_update(
    uuid: str, is_authenticated: bool, user_info: Optional[Dict] = None
) -> None:
    """
    Send an authentication update to a specific user.

    Args:
        uuid: User UUID
        is_authenticated: Authentication status
        user_info: Optional user information
    """
    data = {
        "authenticated": is_authenticated,
        "timestamp": int(datetime.now().timestamp()),
    }

    if user_info:
        data["user"] = user_info

    await send_to_user(uuid, "auth_update", data)
