"""
Watch history API endpoints.

This module defines the API routes for retrieving and managing user watch history
from Trakt. All business logic is delegated to service layer.
"""

import logging
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.core.auth import get_authenticated_user
from app.services.watch_service import (
    format_history_for_response,
    get_history_count,
    get_watch_history_data,
    refresh_watch_history,
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.get("/history")
async def watch_history(
    request: Request,
    start_date: Optional[str] = Query(
        None, description="Start date for filtering (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = Query(
        None, description="End date for filtering (YYYY-MM-DD)"
    ),
    media_type: Optional[str] = Query(
        None, description="Filter by media type (movie/episode)"
    ),
    limit: int = Query(100, description="Number of items to return"),
    offset: int = Query(0, description="Number of items to skip"),
    user_data: Dict = Depends(get_authenticated_user),
):
    """
    Get user's watch history from the database.

    This endpoint retrieves the user's watch history, optionally filtered by date range
    and media type. If no history exists in the database, it will trigger a full history pull.
    """
    if not user_data:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        # Prepare filters
        filters = {}
        if media_type:
            filters["media_type"] = media_type

        if start_date:
            filters["start_date"] = start_date

        if end_date:
            filters["end_date"] = end_date

        # Get watch history from database via service layer
        # The service will handle appropriate refresh if needed
        history_data = await get_watch_history_data(
            trakt_id=user_data["user"]["slug"],
            uuid=user_data["uuid"],
            headers=user_data["headers"],
            force_refresh=False,  # Let the service determine if refresh is needed
            limit=limit,
            offset=offset,
            filters=filters,
        )

        # Format data for response
        formatted_data = format_history_for_response(history_data)

        return JSONResponse(
            content={"data": formatted_data, "count": len(formatted_data)}
        )

    except Exception as e:
        logger.error(f"Error retrieving watch history: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve watch history: {str(e)}"
        )


@router.post("/refresh")
async def refresh_history_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    force: bool = Query(
        False, description="Force complete refresh even if recent data exists"
    ),
    user_data: Dict = Depends(get_authenticated_user),
):
    """
    Refresh the user's watch history from Trakt.

    This endpoint triggers a background task to fetch the latest watch history
    from Trakt and update the database. If force=True, it performs a full history pull
    regardless of existing data. Otherwise, it performs an incremental update.
    """
    if not user_data:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        # Get current history count to determine if this is initial load or update
        count = await get_history_count(user_data["uuid"])

        # Start background task to refresh data using service layer
        background_tasks.add_task(
            refresh_watch_history,
            uuid=user_data["uuid"],
            trakt_id=user_data["user"]["slug"],
            headers=user_data["headers"],
            force=force,
        )

        if count == 0:
            message = "Initial watch history fetch has started"
        else:
            message = "Watch history refresh has started"

        if force:
            message = "Full " + message.lower()

        return {
            "status": "refresh_started",
            "message": message,
            "initial_load": count == 0,
        }

    except Exception as e:
        logger.error(f"Error starting watch history refresh: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to start refresh: {str(e)}"
        )


@router.get("/count")
async def history_count_endpoint(
    request: Request, user_data: Dict = Depends(get_authenticated_user)
):
    """
    Get the count of watch history items for the current user.

    This endpoint is useful for frontend to check if initial data loading is needed.
    """
    if not user_data:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        count = await get_history_count(user_data["uuid"])
        return {"count": count}

    except Exception as e:
        logger.error(f"Error getting watch history count: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get count: {str(e)}")
