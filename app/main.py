"""
FastAPI application main module.
This module initializes the FastAPI application, sets up middleware,
and defines startup/shutdown events for the API server.
"""

import asyncio
import logging
from datetime import datetime

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED

from app.api.endpoints import auth, events, watch_history
from app.core.auth import decode_jwt_token, get_user_credentials, get_uuid_from_jwt
from app.core.config import ALLOWED_ORIGINS, JWT_COOKIE_NAME, JWT_SECRET_KEY
from app.core.sse import start_heartbeat_task
from app.db.db import close_db, init_db, test_connection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Watchstats API",
    description="API for watchstats with Trakt.tv integration",
    version="1.0.0",
)


# Create AuthMiddleware to validate JWT cookies and inject user credentials
class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate JWT cookies and inject user credentials"""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Public paths that don't require authentication
        public_paths = [
            "/auth/login",
            "/auth/callback",
            "/auth/token",
            "/auth/user",
            "/health",
            "/",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/events/stream",  # SSE stream endpoint is public (auth happens after connection)
        ]

        # Skip auth middleware for non-API routes and public endpoints
        path = request.url.path
        if any(path.startswith(p) for p in public_paths) or not path.startswith(
            "/api/"
        ):
            return await call_next(request)

        # Debug: Log request details
        logger.debug(f"Auth middleware processing request to {path}")
        logger.debug(f"Cookies present: {list(request.cookies.keys())}")
        logger.debug(f"JWT cookie name being checked: {JWT_COOKIE_NAME}")

        # First try to get UUID from JWT cookie
        uuid = get_uuid_from_jwt(request)
        if uuid:
            logger.debug(f"Found UUID from cookie: {uuid}")
        else:
            logger.debug(f"No UUID found in cookie")

        # If no UUID from cookie, try to get from Authorization header
        if not uuid:
            auth_header = request.headers.get("Authorization")
            logger.debug(
                f"Authorization header: {auth_header if auth_header else 'None'}"
            )

            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.replace("Bearer ", "")
                payload = decode_jwt_token(token)
                if payload:
                    uuid = payload.get("sub")  # "sub" holds the UUID
                    logger.debug(f"Found UUID from Authorization header: {uuid}")
                else:
                    logger.debug("Token decode failed from Authorization header")

        if uuid:
            # Get user credentials from database
            credentials = await get_user_credentials(uuid)
            if credentials:
                logger.debug(f"User credentials found for UUID: {uuid}")
                # Inject user credentials into request state for handlers
                request.state.user = credentials
                return await call_next(request)
            else:
                logger.warning(f"No user credentials found for UUID: {uuid}")
        else:
            logger.warning(f"No UUID found in request cookies or headers")

        # Return 401 Unauthorized for API routes without valid credentials
        logger.warning(f"Authentication failed for request to {path}")
        return Response(
            status_code=HTTP_401_UNAUTHORIZED,
            content='{"detail": "Authentication required"}',
            media_type="application/json",
        )


# Add middlewares
# Add SessionMiddleware for Authlib OAuth
app.add_middleware(SessionMiddleware, secret_key=JWT_SECRET_KEY)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom auth middleware for API routes
app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(watch_history.router, prefix="/api/watch", tags=["Watch History"])
app.include_router(events.router, prefix="/events", tags=["Events"])


# Heartbeat task
async def heartbeat_task():
    """Start the SSE heartbeat task"""
    await start_heartbeat_task()


# Startup event
@app.on_event("startup")
async def startup_event():
    """Execute startup events"""
    logger.info("Starting watchstats service...")

    # Initialize the database with SQLite WAL mode
    try:
        await init_db()
        db_status = await test_connection()
        logger.info(f"Database initialized: {db_status['status']}")
        if db_status["status"] == "connected":
            logger.info(
                f"SQLite {db_status['version']} connected in {db_status['query_time_ms']}ms"
            )
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        # Continue even if DB fails - it might recover later

    # Start heartbeat task
    asyncio.create_task(heartbeat_task())
    logger.info("Watchstats service startup complete")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Execute shutdown events"""
    logger.info("Stopping watchstats service...")

    # Close database connections
    try:
        await close_db()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")

    logger.info("Watchstats service shutdown complete")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "status": "ok",
        "message": "Watchstats API",
        "docs": "/docs",
        "version": "1.0.0",
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        db_status = await test_connection()

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": db_status,
            "version": "1.0.0",
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
