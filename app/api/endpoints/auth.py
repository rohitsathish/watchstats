"""
Authentication endpoints for Trakt.tv OAuth integration.

This module provides endpoints for the authentication flow:
- /login to start the OAuth process
- /callback to handle the OAuth callback
- /token to exchange the authorization code for tokens
- /user to get the current user information
- /logout to end the user session
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional
from urllib.parse import urlencode

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.status import HTTP_401_UNAUTHORIZED

from app.core.auth import (
    create_jwt_token,
    create_trakt_headers,
    get_user_credentials,
    get_uuid_from_jwt,
)
from app.core.config import (
    ALLOWED_ORIGINS,
    FRONTEND_URL,
    JWT_ALGORITHM,
    JWT_COOKIE_NAME,
    JWT_EXPIRE_MINUTES,
    JWT_SECRET_KEY,
    TRAKT_CLIENT_ID,
    TRAKT_CLIENT_SECRET,
    TRAKT_REDIRECT_URI,
)
from app.core.sse import send_auth_update
from app.core.trakt_api import get_user_profile
from app.db.db import delete_user, get_cached_user_from_db, upsert_user

# Initialize router
router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize OAuth with Trakt - explicitly pass credentials from config
oauth = OAuth()
oauth.register(
    name="trakt",
    client_id=TRAKT_CLIENT_ID,
    client_secret=TRAKT_CLIENT_SECRET,
    authorize_url="https://trakt.tv/oauth/authorize",
    access_token_url="https://api.trakt.tv/oauth/token",
    api_base_url="https://api.trakt.tv",
)


@router.get("/login")
async def login(request: Request):
    """
    Start the OAuth login flow by redirecting to Trakt.tv

    Returns:
        RedirectResponse: Redirect to Trakt.tv authorization URL
    """
    # Generate a random state value to prevent CSRF
    state = secrets.token_hex(16)
    request.session["oauth_state"] = state

    # Get the frontend URL for the redirect
    frontend_url = request.headers.get("Origin", FRONTEND_URL)
    if frontend_url not in ALLOWED_ORIGINS:
        frontend_url = FRONTEND_URL

    # Store the redirect URL in the session
    request.session["redirect_url"] = f"{frontend_url}/auth/callback"

    try:
        # Use the configured redirect URI from config.py instead of dynamically building it
        # This ensures it exactly matches what's registered with Trakt
        redirect_uri = TRAKT_REDIRECT_URI

        params = {
            "response_type": "code",
            "client_id": TRAKT_CLIENT_ID,  # Use the ID directly from config
            "redirect_uri": redirect_uri,
            "state": state,
        }
        authorize_url = f"{oauth.trakt.authorize_url}?{urlencode(params)}"

        # Return the URL to the frontend instead of redirecting
        return {"authorize_url": authorize_url}
    except Exception as e:
        logger.error(f"Error starting OAuth flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/callback")
async def auth_callback(request: Request):
    """
    Handle the OAuth callback from Trakt.tv

    Args:
        request: The request object containing the OAuth code

    Returns:
        RedirectResponse: Redirect to the frontend with auth result
    """
    # Get the code and state from query parameters
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    # Validate state to prevent CSRF
    if not code or not state:
        logger.warning("Missing code or state in callback")
        return RedirectResponse(url=f"{FRONTEND_URL}?error=missing_params")

    session_state = request.session.get("oauth_state")
    if not session_state or state != session_state:
        logger.warning(f"Invalid state: expected {session_state}, got {state}")
        return RedirectResponse(url=f"{FRONTEND_URL}?error=invalid_state")

    try:
        # Get the redirect URL from the session or use default
        redirect_url = request.session.get(
            "redirect_url", f"{FRONTEND_URL}/auth/callback"
        )

        # Build the redirect URL with the authorization code
        return RedirectResponse(url=f"{redirect_url}?code={code}")
    except Exception as e:
        logger.error(f"Error handling callback: {e}")
        return RedirectResponse(url=f"{FRONTEND_URL}?error={str(e)}")


@router.get("/token")
async def exchange_token(request: Request, code: str):
    """
    Exchange the authorization code for an access token

    Args:
        request: The request object
        code: The authorization code from Trakt.tv

    Returns:
        JSONResponse: Authentication result with user information
    """
    try:
        # Use the configured redirect URI from config.py for token exchange
        # Must match what was used in the authorization request
        redirect_uri = TRAKT_REDIRECT_URI

        token = await oauth.trakt.fetch_access_token(
            code=code, redirect_uri=redirect_uri
        )

        # Get user profile from Trakt
        access_token = token.get("access_token")
        if not access_token:
            logger.error("Missing access token in OAuth response")
            return JSONResponse(
                {"authenticated": False, "error": "Missing access token"}
            )

        # Use the access token to get user info
        headers = create_trakt_headers(access_token)
        user_profile = await get_user_profile(headers)

        # Extract user details
        user_id = str(user_profile.get("user", {}).get("ids", {}).get("slug", ""))
        username = user_profile.get("user", {}).get("username", "")
        name = user_profile.get("user", {}).get("name", "")

        # Generate a UUID for the user
        # We use the Trakt user ID as the UUID for simplicity
        uuid = user_id

        # Calculate token expiration
        token_expires_in = token.get("expires_in", 7200)  # Default to 2 hours
        expires_at = datetime.now() + timedelta(seconds=token_expires_in)

        # Create a JWT token for the user
        jwt_token = create_jwt_token(uuid=uuid, expires_minutes=JWT_EXPIRE_MINUTES)

        # Store user in database with tokens
        await upsert_user(
            uuid=uuid,
            user_id=user_id,
            access_token=access_token,
            access_token_expires_at=expires_at,
            refresh_token=token.get("refresh_token", ""),
            username=username,
            name=name,
            slug=user_id,  # Using slug as the user_id for now
            avatar_url=None,  # Can add avatar logic later
            timezone=None,  # Can add timezone logic later
        )

        # Create success response
        response_data = {
            "authenticated": True,
            "user": {
                "uuid": uuid,
                "username": username,
                "name": name,
                "slug": user_id,
            },
        }

        # Set the JWT token in a cookie
        response = JSONResponse(response_data)
        response.set_cookie(
            key=JWT_COOKIE_NAME,
            value=jwt_token,
            httponly=True,
            samesite="lax",  # Needed for OAuth redirect flow
            secure=request.url.scheme == "https",
            max_age=JWT_EXPIRE_MINUTES * 60,  # Convert to seconds
            path="/",
        )

        # Send auth update via SSE
        await send_auth_update(
            uuid=uuid, is_authenticated=True, user_info=response_data["user"]
        )

        return response
    except Exception as e:
        logger.error(f"Error exchanging token: {e}")
        return JSONResponse(
            {"authenticated": False, "error": str(e)},
            status_code=500,
        )


@router.get("/user")
async def get_user_info(request: Request):
    """
    Get information about the currently authenticated user

    Args:
        request: The request object

    Returns:
        dict: User information if authenticated
    """
    try:
        user_uuid = get_uuid_from_jwt(request)
        if user_uuid:
            # Get user from database
            user = await get_cached_user_from_db(user_uuid)

            if user and "user" in user:
                user_info = user["user"]
                return {
                    "authenticated": True,
                    "username": user_info.get("username"),
                    "name": user_info.get("name"),
                    "slug": user_info.get("slug"),
                    "avatar": user_info.get("avatar"),
                }

        return {"authenticated": False}
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        return {"authenticated": False, "error": str(e)}


@router.post("/logout")
async def logout(request: Request):
    """
    Log out the current user by clearing cookies and tokens

    Args:
        request: The request object

    Returns:
        JSONResponse: Logout result
    """
    try:
        # Get the user's UUID before clearing cookies
        user_uuid = get_uuid_from_jwt(request)

        # Create response to clear cookies
        response = JSONResponse({"success": True})
        response.delete_cookie(
            key=JWT_COOKIE_NAME,
            path="/",
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
        )

        # Send auth update via SSE if we had an authenticated user
        if user_uuid:
            # Only send an update if we had an authenticated user
            await send_auth_update(
                uuid=user_uuid, is_authenticated=False, user_info=None
            )

            logger.info(f"User {user_uuid} logged out successfully")

        return response
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500,
        )
