import os
import secrets
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# %% Configuration settings
# API Keys and Credentials - Required for Trakt OAuth
TRAKT_CLIENT_ID = os.getenv(
    "TRAKT_CLIENT_ID",
    "b8f321f93f6bc1d18d08e6d90fd65c2f43ff39801caee2bc76561827d51dfe19",
)
TRAKT_CLIENT_SECRET = os.getenv(
    "TRAKT_CLIENT_SECRET",
    "4e65579de5aa8890afd6bbac80a91c1634ff2743066b05df8a4424b92554ff9b",
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Authentication
JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", secrets.token_urlsafe(64)
)  # Generate a secure random key if not provided
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(
    os.getenv("JWT_EXPIRATION_MINUTES", 1440)
)  # 24 hours default
JWT_COOKIE_NAME = os.getenv("JWT_COOKIE_NAME", "watchstats_auth")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "10080")
)  # 7 days default

# Trakt API
TRAKT_API_VERSION = os.getenv("TRAKT_API_VERSION", "2")
TRAKT_REDIRECT_URI = os.getenv(
    "TRAKT_REDIRECT_URI", "http://localhost:3000/auth/callback"
)

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")

# CORS Settings
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# Trakt API Base URL
TRAKT_API_BASE_URL = "https://api.trakt.tv"

# Database Settings
# Create database directory if it doesn't exist
DB_DIR = Path("assets/db")
DB_DIR.mkdir(parents=True, exist_ok=True)

# SQLite database path
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(DB_DIR / "media.db"))

# API Keys for external services
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "652a87adc664247119bad869af0728b3")
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "fadeb875")

# TMDB API settings
TMDB_API_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/original"

# Data Fetch Settings
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))  # seconds
