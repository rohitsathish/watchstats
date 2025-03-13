import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import httpx
import jwt
from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import (
    JWT_ALGORITHM,
    JWT_COOKIE_NAME,
    JWT_EXPIRE_MINUTES,
    JWT_SECRET_KEY,
    TRAKT_CLIENT_ID,
    TRAKT_CLIENT_SECRET,
    TRAKT_REDIRECT_URI,
)

# Import database functions only
from app.db.db import (
    delete_user,
    get_cached_user_from_db,
    get_user_by_uuid,
    upsert_user,
)

# Setup logging
logger = logging.getLogger(__name__)

# Security scheme for protected endpoints
security = HTTPBearer()


def create_trakt_headers(access_token: str) -> Dict[str, str]:
    """Create headers for Trakt API requests"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
        "trakt-api-version": "2",
        "trakt-api-key": TRAKT_CLIENT_ID,
    }


# JWT token functions
def create_jwt_token(uuid: str) -> str:
    """Create a JWT token with the user's UUID and expiration time"""
    expiration = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": uuid, "exp": expiration, "iat": datetime.utcnow()}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str) -> Optional[Dict]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.PyJWTError as e:
        logger.error(f"JWT decode error: {str(e)}")
        return None


def set_auth_cookie(response: Response, uuid: str) -> None:
    """Set JWT authentication cookie in response"""
    token = create_jwt_token(uuid)
    secure = not (
        os.getenv("DEBUG", "").lower() == "true"
    )  # In development, we might need non-secure cookies
    logger.info(f"Setting auth cookie for UUID: {uuid}, secure={secure}")
    response.set_cookie(
        key=JWT_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=secure,  # Set to False for HTTP in development
        samesite="lax",
        max_age=JWT_EXPIRE_MINUTES * 60,
        path="/",
    )


def get_uuid_from_jwt(request: Request) -> Optional[str]:
    """Extract and validate UUID from JWT cookie"""
    token = request.cookies.get(JWT_COOKIE_NAME)
    if not token:
        return None
    payload = decode_jwt_token(token)
    if not payload:
        return None
    return payload.get("sub")  # "sub" holds the UUID


def clear_auth_cookie(response: Response) -> None:
    """Clear the authentication cookie"""
    logger.info(f"Clearing auth cookie")
    response.delete_cookie(
        key=JWT_COOKIE_NAME,
        path="/",
        secure=not (os.getenv("DEBUG", "").lower() == "true"),  # Match set_auth_cookie
        httponly=True,
    )


# Setup OAuth with Trakt configuration
oauth = OAuth()
oauth.register(
    name="trakt",
    client_id=TRAKT_CLIENT_ID,
    client_secret=TRAKT_CLIENT_SECRET,
    access_token_url="https://api.trakt.tv/oauth/token",
    authorize_url="https://trakt.tv/oauth/authorize",
    api_base_url="https://api.trakt.tv",
    client_kwargs={
        "scope": "public",
        "token_endpoint_auth_method": "client_secret_post",
    },
)


async def get_authorization_url(request: Request) -> str:
    """Generate the Trakt authorization URL for the frontend to redirect to"""
    try:
        redirect_uri = TRAKT_REDIRECT_URI
        logger.info(f"Generating authorization URL with redirect URI: {redirect_uri}")
        # Manually construct the authorization URL
        authorize_url = f"https://trakt.tv/oauth/authorize?response_type=code&client_id={TRAKT_CLIENT_ID}&redirect_uri={redirect_uri}"
        logger.info(f"Generated authorization URL: {authorize_url}")
        return authorize_url
    except Exception as e:
        logger.error(f"Error generating authorization URL: {str(e)}")
        raise


async def get_trakt_user_info(headers: Dict[str, str]) -> Dict:
    """Get user information from Trakt API using provided headers"""
    try:
        async with httpx.AsyncClient() as client:
            # Make the request to get user settings (includes more detailed info)
            response = await client.get(
                "https://api.trakt.tv/users/settings", headers=headers
            )
            if response.status_code != 200:
                logger.error(
                    f"Failed to get user settings: {response.status_code} - {response.text}"
                )
                return {}
            # Parse the response
            settings = response.json()
            # Extract the required fields
            user_data = {
                "username": settings.get("user", {}).get("username"),
                "name": settings.get("user", {}).get("name"),
                "slug": settings.get("user", {}).get("ids", {}).get("slug"),
                "uuid": settings.get("user", {}).get("ids", {}).get("uuid"),
                "avatar": settings.get("user", {})
                .get("images", {})
                .get("avatar", {})
                .get("full"),
                "timezone": settings.get("account", {}).get("timezone"),
            }

            logger.info(f"Retrieved user settings for: {user_data.get('username')}")
            return user_data
    except Exception as e:
        logger.error(f"Error getting user info: {str(e)}")
        return {}


async def exchange_code_for_token(
    request: Request, code: str, response: Optional[Response] = None
) -> Dict:
    """Exchange the authorization code for an access token"""
    try:
        logger.info(f"Exchanging code for token: {code[:5]}...")
        # Prepare the token request
        token_url = "https://api.trakt.tv/oauth/token"
        data = {
            "code": code,
            "client_id": TRAKT_CLIENT_ID,
            "client_secret": TRAKT_CLIENT_SECRET,
            "redirect_uri": TRAKT_REDIRECT_URI,
            "grant_type": "authorization_code",
        }
        headers = {"Content-Type": "application/json"}
        logger.info(f"Token request from exchange_code_for_token")
        # Make the token request
        async with httpx.AsyncClient() as client:
            resp = await client.post(token_url, json=data, headers=headers)
            if resp.status_code != 200:
                logger.error(
                    f"Token request failed with status {resp.status_code}: {resp.text}"
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to exchange code for token: {resp.text}",
                )
            token = resp.json()

        if not token or "access_token" not in token:
            logger.error("No token received from Trakt")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to exchange code for token",
            )

        # Create headers for future API requests
        api_headers = create_trakt_headers(token["access_token"])

        # Get user information from Trakt API using the headers
        user_info = await get_trakt_user_info(api_headers)

        if user_info:
            token["user"] = user_info
            logger.info(f"Retrieved user info: {user_info.get('username', 'unknown')}")

            # Use the uuid as the user ID
            uuid = user_info.get("uuid")
            if not uuid:
                logger.error("No UUID found in user info")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No UUID found in user info",
                )

            # Calculate token expiration time
            expires_at = datetime.now() + timedelta(seconds=token.get("expires_in", 0))
            token["expires_at"] = expires_at

            # Store user data in the database
            user_id = user_info.get("username", "unknown")
            success = await upsert_user(
                uuid=uuid,
                user_id=user_id,
                access_token=token["access_token"],
                access_token_expires_at=expires_at,
                refresh_token=token.get("refresh_token", ""),
                username=user_info.get("username"),
                name=user_info.get("name"),
                slug=user_info.get("slug"),
                avatar_url=user_info.get("avatar"),
                timezone=user_info.get("timezone"),
            )

            if not success:
                logger.error(f"Failed to store user {uuid} in database")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to store user data",
                )

            # Set JWT cookie if response object provided
            if response:
                set_auth_cookie(response, uuid)

            logger.info(f"Token exchange successful for user UUID: {uuid}")
            return token
        else:
            logger.error("Failed to retrieve user info")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to retrieve user info",
            )
    except Exception as e:
        logger.error(f"Error exchanging code for token: {str(e)}")
        raise


async def get_current_user_creds(request: Request) -> Optional[Dict]:
    """Get current user's credentials from the database"""
    # Try to get UUID from JWT cookie
    uuid = get_uuid_from_jwt(request)
    if not uuid:
        logger.debug("No UUID found in JWT cookie")
        return None

    # Get user from database
    return await get_cached_user_from_db(uuid)


async def get_current_user(
    request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """
    Dependency for protected routes that require authentication via bearer token.
    Returns user data or raises 401 exception if not authenticated.
    """
    token = credentials.credentials
    payload = decode_jwt_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    uuid = payload.get("sub")
    if not uuid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_data = await get_cached_user_from_db(uuid)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_data


async def get_authenticated_user(request: Request) -> Dict:
    """
    Dependency function to get authenticated user credentials via JWT cookie.
    Raises HTTPException if not authenticated.

    Usage:
        @router.get("/protected")
        async def protected_endpoint(user_data: Dict = Depends(get_authenticated_user)):
            # Use user_data['headers'] for API requests
            # Use user_data['user'] for user info
            pass
    """
    user_data = await get_current_user_creds(request)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )
    return user_data


async def get_user_credentials(uuid: str) -> Optional[Dict]:
    """Get user credentials from the database"""
    if not uuid:
        return None

    # Get user from database
    return await get_cached_user_from_db(uuid)


async def clear_user_credentials(uuid: str) -> None:
    """Clear user credentials from database"""
    if not uuid:
        return

    # Delete from database
    success = await delete_user(uuid)
    if success:
        logger.info(f"User {uuid} deleted from database")
    else:
        logger.warning(f"Failed to delete user {uuid} from database")
