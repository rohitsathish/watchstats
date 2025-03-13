"""
Trakt API integration module.

This module provides functions for interacting with the Trakt.tv API,
handling rate limiting and error cases properly.
"""

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx
from fastapi import HTTPException

# Set up logging
logger = logging.getLogger(__name__)

# Trakt API base URL
TRAKT_API_BASE_URL = "https://api.trakt.tv"

# Rate limiting parameters
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds
MAX_RETRY_DELAY = 60  # maximum seconds to wait between retries


async def api_call_wrapper(
    method: str,
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
    retry_codes: List[int] = None,
    success_codes: List[int] = None,
) -> Tuple[Any, Dict[str, str]]:
    """
    Generic API call wrapper with retry logic, timeout handling, and error management.

    Args:
        method: HTTP method ('GET', 'POST', etc.)
        url: API endpoint URL
        headers: Request headers
        params: Optional query parameters
        json_data: Optional JSON body data
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        retry_codes: HTTP status codes that should trigger a retry, defaults to [429, 500, 502, 503, 504]
        success_codes: HTTP status codes considered successful, defaults to [200, 201, 204]

    Returns:
        tuple[Any, Dict[str, str]]: (API response data, response headers)

    Raises:
        HTTPException: When the request fails after all retries or with a non-retryable error code
    """
    method = method.upper()

    # Set default retry and success codes if not provided
    if retry_codes is None:
        retry_codes = [429, 500, 502, 503, 504]  # Common temporary error codes

    if success_codes is None:
        success_codes = [200, 201, 204]  # Common success codes

    # Track the attempt number
    attempts = 0

    # Prepare request kwargs
    request_kwargs = {
        "headers": headers,
        "params": params,
        "follow_redirects": True,
        "timeout": timeout,
    }

    # Add JSON data if provided
    if json_data is not None:
        request_kwargs["json"] = json_data

    while True:
        attempts += 1

        try:
            # Create a new client for each attempt to avoid connection pooling issues
            async with httpx.AsyncClient(http2=True) as client:
                start_time = time.time()

                # Make the request
                response = await getattr(client, method.lower())(url, **request_kwargs)

                request_time = time.time() - start_time
                logger.debug(
                    f"{method} request to {url} completed in {request_time:.2f}s (status: {response.status_code})"
                )

                # Check if we should retry based on status code
                if response.status_code in retry_codes and attempts <= max_retries:
                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        retry_delay * (2 ** (attempts - 1)) * (0.5 + random.random()),
                        MAX_RETRY_DELAY,
                    )

                    # For rate limiting, use server-provided retry-after if available
                    if (
                        response.status_code == 429
                        and "Retry-After" in response.headers
                    ):
                        try:
                            delay = float(response.headers["Retry-After"])
                        except (ValueError, TypeError):
                            # If parsing fails, stick with the calculated delay
                            pass

                    logger.warning(
                        f"{method} request to {url} returned {response.status_code}, "
                        f"retrying in {delay:.2f}s (attempt {attempts}/{max_retries})"
                    )

                    # Wait before retrying
                    await asyncio.sleep(delay)
                    continue

                # If we got here, we're not retrying

                # Check if the response is successful
                if response.status_code in success_codes:
                    # Parse response data
                    try:
                        data = response.json() if response.content else None
                    except ValueError:
                        # If JSON parsing fails, return the raw content
                        data = response.content

                    # Return the data and headers
                    return data, dict(response.headers)

                # If we've reached this point, it's an error response
                error_detail = f"API error: {response.status_code}"

                # Try to parse response for more info
                try:
                    error_data = response.json() if response.content else {}
                    if isinstance(error_data, dict) and error_data.get("error"):
                        error_detail = f"{error_detail} - {error_data.get('error')}"
                except Exception:
                    # Fall back to text content if JSON parsing fails
                    error_content = response.text[:100] if response.content else ""
                    if error_content:
                        error_detail = f"{error_detail} - {error_content}"

                logger.error(f"{method} request to {url} failed: {error_detail}")
                raise HTTPException(
                    status_code=response.status_code, detail=error_detail
                )

        except httpx.RequestError as e:
            # Network-level errors (DNS failure, connection refused, etc.)
            if attempts <= max_retries:
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    retry_delay * (2 ** (attempts - 1)) * (0.5 + random.random()),
                    MAX_RETRY_DELAY,
                )

                logger.warning(
                    f"{method} request to {url} failed with network error: {str(e)}, "
                    f"retrying in {delay:.2f}s (attempt {attempts}/{max_retries})"
                )

                # Wait before retrying
                await asyncio.sleep(delay)
                continue

            # If we've reached max retries, raise the exception
            logger.error(
                f"{method} request to {url} failed after {max_retries} attempts: {str(e)}"
            )
            raise HTTPException(
                status_code=503, detail=f"Service unavailable: {str(e)}"
            )

        except Exception as e:
            # Unexpected errors
            logger.error(
                f"Unexpected error in {method} request to {url}: {str(e)}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )


async def get_user_profile(headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Fetch the user's profile information from Trakt API.

    Args:
        headers: API request headers with access token

    Returns:
        Dict[str, Any]: User profile information
    """
    url = f"{TRAKT_API_BASE_URL}/users/settings"

    data, _ = await api_call_wrapper("GET", url, headers)
    return data


async def fetch_complete_watch_history(
    headers: Dict[str, str],
    trakt_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch complete watch history for a user with pagination handling.

    Args:
        headers: API request headers
        trakt_id: Trakt user ID
        start_date: Optional start date (ISO format)
        end_date: Optional end date (ISO format)

    Returns:
        List[Dict[str, Any]]: Complete watch history
    """
    start_time = time.time()

    # Build base URL
    url = f"{TRAKT_API_BASE_URL}/users/{trakt_id}/history"

    # Build query parameters
    params = {"extended": "full", "limit": 100, "page": 1}

    # Add date filters if provided
    if start_date:
        params["start_at"] = start_date
    if end_date:
        params["end_at"] = end_date

    # Make the initial request to get pagination info
    first_page_data, headers_info = await api_call_wrapper(
        "GET", url, headers, params=params
    )

    total_pages = int(headers_info.get("X-Pagination-Page-Count", "1"))

    all_history = first_page_data

    # Fetch remaining pages in parallel with concurrency limit
    if total_pages > 1:
        # Create batches to avoid overwhelming the API
        batch_size = 5
        for batch_start in range(2, total_pages + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, total_pages)

            tasks = []
            for page in range(batch_start, batch_end + 1):
                # Create a copy of params with updated page number
                page_params = params.copy()
                page_params["page"] = page

                tasks.append(api_call_wrapper("GET", url, headers, params=page_params))

            batch_results = await asyncio.gather(*tasks)

            # Extract data from results (ignore headers)
            for data, _ in batch_results:
                all_history.extend(data)

    logger.info(
        f"Fetched complete watch history for {trakt_id}: "
        f"{len(all_history)} items from {total_pages} pages in "
        f"{time.time() - start_time:.2f}s"
    )

    return all_history


async def fetch_show_details(
    headers: Dict[str, str],
    slug: str,
) -> Dict[str, Any]:
    """
    Fetch details for a TV show.

    Args:
        headers: API request headers
        slug: Show slug

    Returns:
        Dict[str, Any]: Show details
    """
    url = f"{TRAKT_API_BASE_URL}/shows/{slug}"
    params = {"extended": "full"}

    data, _ = await api_call_wrapper("GET", url, headers, params=params)
    return data


async def fetch_movie_details(
    headers: Dict[str, str],
    slug: str,
) -> Dict[str, Any]:
    """
    Fetch details for a movie.

    Args:
        headers: API request headers
        slug: Movie slug

    Returns:
        Dict[str, Any]: Movie details
    """
    url = f"{TRAKT_API_BASE_URL}/movies/{slug}"
    params = {"extended": "full"}

    data, _ = await api_call_wrapper("GET", url, headers, params=params)
    return data


async def fetch_episode_details(
    headers: Dict[str, str],
    slug: str,
    season: int,
    episode: int,
) -> Dict[str, Any]:
    """
    Fetch details for a TV episode.

    Args:
        headers: API request headers
        slug: Show slug
        season: Season number
        episode: Episode number

    Returns:
        Dict[str, Any]: Episode details
    """
    url = f"{TRAKT_API_BASE_URL}/shows/{slug}/seasons/{season}/episodes/{episode}"
    params = {"extended": "full"}

    data, _ = await api_call_wrapper("GET", url, headers, params=params)
    return data


async def fetch_user_stats(
    headers: Dict[str, str],
    trakt_id: str,
) -> Dict[str, Any]:
    """
    Fetch stats for a user.

    Args:
        headers: API request headers
        trakt_id: Trakt user ID

    Returns:
        Dict[str, Any]: User stats
    """
    url = f"{TRAKT_API_BASE_URL}/users/{trakt_id}/stats"

    data, _ = await api_call_wrapper("GET", url, headers)
    return data


async def fetch_user_ratings(
    headers: Dict[str, str],
    trakt_id: str,
) -> List[Dict[str, Any]]:
    """
    Fetch ratings for a user.

    Args:
        headers: API request headers
        trakt_id: Trakt user ID

    Returns:
        List[Dict[str, Any]]: User ratings
    """
    url = f"{TRAKT_API_BASE_URL}/users/{trakt_id}/ratings"

    data, _ = await api_call_wrapper("GET", url, headers)
    return data


# Legacy function to maintain backwards compatibility
async def _fetch_trakt_data(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    retries: int = 0,
) -> tuple[Any, Dict[str, str]]:
    """
    Legacy helper function to fetch data from Trakt API with retry logic.
    This function is maintained for backwards compatibility.
    Use api_call_wrapper for new code.

    Args:
        client: HTTP client
        url: API endpoint URL
        headers: Request headers
        retries: Current retry count

    Returns:
        tuple[Any, Dict[str, str]]: (API response data, response headers)
    """
    logger.warning(
        "Using deprecated _fetch_trakt_data, consider migrating to api_call_wrapper"
    )

    # Extract the HTTP method (default to GET)
    method = "GET"

    # Call the new api_call_wrapper function
    data, headers_dict = await api_call_wrapper(
        method=method,
        url=url,
        headers=headers,
        max_retries=MAX_RETRIES - retries,  # Adjust remaining retries
    )

    return data, headers_dict
