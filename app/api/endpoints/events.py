"""
Server-Sent Events (SSE) endpoint.

This module defines the API routes for SSE connections, providing real-time
updates to the frontend via a streaming response.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from app.core.auth import decode_jwt_token, get_user_credentials, get_uuid_from_jwt
from app.core.sse import (
    SSEConnection,
    register_connection,
    start_heartbeat_task,
    unregister_connection,
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Asynchronous generator for SSE events with proper format for StreamingResponse
async def sse_generator(connection: SSEConnection):
    """
    Generate SSE events formatted for HTTP streaming response.

    Args:
        connection: The SSE connection object

    Yields:
        str: Properly formatted SSE message
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

    # Format the initial message in SSE format
    from json import dumps

    yield f"event: connection\ndata: {dumps(connection_data)}\n\n"

    connection_id = connection.connection_id

    try:
        # Process messages from the queue
        while connection.is_active:
            try:
                # Use wait_for with a timeout to allow for connection cleanup
                message = await asyncio.wait_for(connection.queue.get(), timeout=60)

                if message:
                    # Format the SSE message correctly
                    event_name = message["event"]
                    event_data = message["data"]
                    yield f"event: {event_name}\ndata: {dumps(event_data)}\n\n"

                    # Mark item as processed
                    connection.queue.task_done()
            except asyncio.TimeoutError:
                # No message for a while, send a keep-alive comment
                # Send as a comment to avoid triggering event handlers
                yield f": keep-alive {datetime.now().isoformat()}\n\n"
            except Exception as e:
                logger.error(
                    f"Error processing message for connection {connection_id}: {e}"
                )
                # If there's an error, send a comment with the error type but keep the connection
                yield f": error: {type(e).__name__}\n\n"
    except Exception as e:
        logger.error(f"Connection {connection_id} stream error: {e}")
    finally:
        # Ensure connection is unregistered when the generator stops
        await unregister_connection(connection_id)
        logger.info(f"Connection {connection_id} stream closed")


@router.get("/stream")
async def stream_events(request: Request):
    """
    Stream events to the client using Server-Sent Events (SSE).

    This endpoint establishes a long-lived connection for sending real-time
    updates about authentication state, data changes, etc.

    The connection automatically authenticates if valid JWT credentials are provided.
    """
    # Try to get user UUID from JWT cookie or Authorization header
    uuid = get_uuid_from_jwt(request)

    # If no UUID from cookie, try Authorization header
    if not uuid:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            payload = decode_jwt_token(token)
            if payload:
                uuid = payload.get("sub")  # "sub" holds the UUID

    # Get user info if authenticated
    user_info = None
    if uuid:
        credentials = await get_user_credentials(uuid)
        if credentials and "user" in credentials:
            user_info = credentials["user"]
            logger.debug(
                f"Authenticated SSE connection for user: {user_info.get('username')}"
            )

    # Register connection
    connection = await register_connection(request, uuid=uuid, user_info=user_info)

    # Create streaming response with proper headers
    return StreamingResponse(
        sse_generator(connection),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in Nginx
        },
        background=BackgroundTask(unregister_connection, connection.connection_id),
    )


# Heartbeat task needs to be started when the application starts
# This will be imported and started in main.py
async def start_sse_heartbeat():
    """Start the SSE heartbeat task."""
    await start_heartbeat_task()
