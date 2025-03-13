"""
Database interface module with SQLite3 and WAL journaling mode.

This module provides optimized asynchronous database operations using aiosqlite in WAL mode
for improved concurrency and performance.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiosqlite
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
logger = logging.getLogger(__name__)

# Create database directory if it doesn't exist
DB_DIR = Path("assets/db")
DB_DIR.mkdir(parents=True, exist_ok=True)

# Database path
DB_PATH = os.getenv("SQLITE_DB_PATH", str(DB_DIR / "media.db"))

# Connection pool configuration
POOL_SIZE = 20  # Adjust based on expected concurrent requests
_pool = None
_pool_lock = asyncio.Lock()
_prepared_statements = {}  # Cache for prepared statements


# Initialize the connection pool
async def _init_pool():
    """Initialize the database connection pool."""
    global _pool
    if _pool is not None:
        return _pool

    async with _pool_lock:
        if _pool is not None:
            return _pool

        _pool = SQLiteConnectionPool(DB_PATH, POOL_SIZE)
        await _pool.initialize()

        return _pool


class SQLiteConnectionPool:
    """Pool of SQLite connections for concurrent access."""

    def __init__(self, database_path: str, max_connections: int = POOL_SIZE):
        self.database_path = database_path
        self.max_connections = max_connections
        self._pool = asyncio.Queue(max_connections)
        self._active_connections = 0
        self._initialization_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize the connection pool with connections."""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return

            for _ in range(self.max_connections):
                conn = await self._create_connection()
                await self._pool.put(conn)

            self._initialized = True
            logger.info(
                f"Connection pool initialized with {self.max_connections} connections"
            )

    async def _create_connection(self):
        """Create a new optimized database connection."""
        conn = await aiosqlite.connect(self.database_path)

        # Enable row factory for dict-like rows
        conn.row_factory = self._dict_factory

        # Set optimal PRAGMAs for each connection
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA temp_store=MEMORY")
        await conn.execute("PRAGMA cache_size=-6000")  # ~6MB cache

        self._active_connections += 1
        return conn

    @staticmethod
    def _dict_factory(cursor, row):
        """Convert SQLite row to dictionary."""
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self._initialized:
            await self.initialize()

        try:
            # Try to get a connection from the pool
            conn = await asyncio.wait_for(self._pool.get(), timeout=10.0)
            return conn
        except asyncio.TimeoutError:
            # If pool is empty and we haven't reached max, create a new one
            if self._active_connections < self.max_connections:
                return await self._create_connection()
            # Otherwise raise an error
            raise Exception("Connection pool exhausted")

    async def release(self, conn):
        """Return a connection to the pool."""
        # If the connection is still open, return it to the pool
        try:
            await conn.execute("SELECT 1")  # Quick test if connection is good
            await self._pool.put(conn)
        except Exception:
            # If connection is bad, close it and create a new one
            await conn.close()
            self._active_connections -= 1
            conn = await self._create_connection()
            await self._pool.put(conn)

    async def close_all(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()
            self._active_connections -= 1

        assert (
            self._active_connections == 0
        ), "Some connections were not properly returned to the pool"
        self._initialized = False
        logger.info("All database connections closed")


# Connection context manager for use with async/await
@asynccontextmanager
async def get_db():
    """Get a database connection from the pool."""
    pool = await _init_pool()
    conn = await pool.acquire()
    try:
        yield conn
    finally:
        await pool.release(conn)


async def optimize_db():
    """Run optimization tasks on the database."""
    async with get_db() as db:
        # Run the SQLite optimizer (3.38.0+)
        try:
            await db.execute("PRAGMA optimize")
        except aiosqlite.OperationalError:
            # Older SQLite version
            pass

        # Run ANALYZE to update statistics
        await db.execute("ANALYZE")

        # Check for index usage
        await db.execute("PRAGMA analysis_limit=1000")
        await db.execute("PRAGMA optimize")

        # Compact the database if needed
        await db.execute("PRAGMA incremental_vacuum")

        logger.info("Database optimized successfully")


async def init_db() -> None:
    """Initialize SQLite database with tables if they don't exist."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)

        # Initialize pool
        pool = await _init_pool()

        # Create database if it doesn't exist and apply schema
        async with get_db() as db:
            # Set optimal PRAGMAs for performance and concurrency
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA temp_store=MEMORY")
            await db.execute("PRAGMA mmap_size=30000000")  # ~30MB memory mapping
            await db.execute("PRAGMA cache_size=-6000")  # ~6MB cache
            await db.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
            await db.execute("PRAGMA foreign_keys=ON")

            # Apply schema if needed
            if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) < 5000:
                with open(os.path.join(os.path.dirname(__file__), "schema.sql")) as f:
                    schema_script = f.read()
                await db.executescript(schema_script)
                await db.commit()
                logger.info("Database schema initialized")

        # Verify and optimize the database
        await optimize_db()
        logger.info(f"Database initialized at {DB_PATH}")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_db() -> None:
    """Close database connections."""
    global _pool
    if _pool:
        await _pool.close_all()
        _pool = None
    logger.info("Database connections closed")


# Helper functions for common database operations
async def execute(query: str, params: tuple = (), *, commit: bool = True) -> int:
    """Execute a query and return the lastrowid."""
    async with get_db() as db:
        cursor = await db.execute(query, params)
        if commit:
            await db.commit()
        return cursor.lastrowid


async def execute_many(
    query: str, params_list: List[tuple], *, commit: bool = True
) -> None:
    """Execute a query with multiple parameter sets."""
    async with get_db() as db:
        await db.executemany(query, params_list)
        if commit:
            await db.commit()


async def fetch_one(query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    """Fetch a single record as a dictionary."""
    async with get_db() as db:
        cursor = await db.execute(query, params)
        row = await cursor.fetchone()
        return dict(row) if row else None


async def fetch_all(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Fetch all records as a list of dictionaries."""
    async with get_db() as db:
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


@asynccontextmanager
async def transaction():
    """Context manager for executing multiple operations in a transaction."""
    async with get_db() as db:
        await db.execute("BEGIN TRANSACTION")
        try:
            yield db
            await db.commit()
        except Exception:
            await db.rollback()
            raise


# Function to run maintenance tasks periodically
async def run_maintenance():
    """Run database maintenance tasks. Call this periodically (e.g., daily)."""
    async with get_db() as db:
        # Get database stats
        cursor = await db.execute("PRAGMA page_count")
        page_count = (await cursor.fetchone())[0]

        cursor = await db.execute("PRAGMA page_size")
        page_size = (await cursor.fetchone())[0]

        db_size_mb = (page_count * page_size) / (1024 * 1024)

        # Only run VACUUM if the database is not too large
        # VACUUM can be expensive on large databases
        if db_size_mb < 1000:  # Less than 1GB
            await db.execute("VACUUM")

        # Always run incremental optimization
        await optimize_db()

        return {
            "size_mb": round(db_size_mb, 2),
            "page_count": page_count,
            "page_size": page_size,
            "wal_mode": True,
        }


async def test_connection() -> Dict[str, Any]:
    """
    Test database connection and return status info.

    Returns:
        Dict[str, Any]: Connection information including version and stats
    """
    try:
        start_time = time.time()
        async with get_db() as conn:
            cursor = await conn.execute("SELECT sqlite_version()")
            version = (await cursor.fetchone())[0]

            # Get database stats
            cursor = await conn.execute("PRAGMA page_size")
            page_size = (await cursor.fetchone())[0]

            cursor = await conn.execute("PRAGMA page_count")
            page_count = (await cursor.fetchone())[0]

            size_mb = (page_size * page_count) / (1024 * 1024)

            # Get table counts
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = await cursor.fetchall()

            table_counts = {}
            for table in tables:
                table_name = table[0]
                if not table_name.startswith("sqlite_"):
                    cursor = await conn.execute(
                        f"SELECT COUNT(*) as count FROM {table_name}"
                    )
                    count = (await cursor.fetchone())[0]
                    table_counts[table_name] = count

            query_time = time.time() - start_time

            cursor = await conn.execute("PRAGMA journal_mode")
            journal_mode = (await cursor.fetchone())[0]

            return {
                "status": "connected",
                "version": version,
                "size_mb": round(size_mb, 2),
                "query_time_ms": round(query_time * 1000, 2),
                "table_counts": table_counts,
                "wal_mode": journal_mode == "wal",
                "type": "sqlite3",
                "architecture": "aiosqlite with connection pooling and WAL mode",
            }
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return {"status": "failed", "error": str(e), "type": "sqlite3"}


# User methods
async def get_user_by_uuid(uuid: str) -> Optional[Dict[str, Any]]:
    """
    Get user details from database by UUID.

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

        async with get_db() as conn:
            cursor = await conn.execute(
                """
                SELECT 
                    uuid, user_id, access_token, access_token_expires_at,
                    refresh_token, username, name, slug, avatar_url, timezone
                FROM users 
                WHERE uuid = ?
                """,
                (uuid,),
            )
            row = await cursor.fetchone()

            query_time = time.time() - start_time

            if not row:
                logger.debug(
                    f"Database lookup: No user found with UUID {uuid} (took {query_time:.3f}s)"
                )
                return None

            # Check if token is expired
            if row["access_token_expires_at"]:
                expires_at = datetime.fromisoformat(
                    row["access_token_expires_at"].replace("Z", "+00:00")
                )
                if datetime.now() > expires_at:
                    logger.info(f"Token expired for user: {uuid}")
                    return None

            logger.debug(
                f"Database lookup: Retrieved user {uuid} ({row['username']}) in {query_time:.3f}s"
            )

            # Import the Trakt headers building function from auth module
            from app.core.auth import create_trakt_headers

            # Build and return the complete user credentials object
            headers = create_trakt_headers(row["access_token"])

            user_info = {
                "username": row["username"],
                "name": row["name"],
                "slug": row["slug"],
                "uuid": row["uuid"],
                "avatar": row["avatar_url"],
                "timezone": row["timezone"],
            }

            return {
                "uuid": row["uuid"],
                "access_token": row["access_token"],
                "refresh_token": row["refresh_token"],
                "headers": headers,
                "expires_at": expires_at if "expires_at" in locals() else None,
                "user": user_info,
            }

    except Exception as e:
        logger.error(f"Error getting user {uuid} from database: {e}")
        return None


async def get_cached_user_from_db(uuid: str) -> Optional[Dict[str, Any]]:
    """
    Get user data directly from the database.

    Args:
        uuid: User's UUID

    Returns:
        Optional[Dict[str, Any]]: User credentials or None if not found
    """
    # Direct database access, no cache layer
    if not uuid:
        return None
    return await get_user_by_uuid(uuid)


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
    Insert or update a user in the database.

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
        async with get_db() as conn:
            await conn.execute(
                """
                INSERT INTO users (
                    uuid, user_id, access_token, access_token_expires_at, 
                    refresh_token, username, name, slug, avatar_url, timezone
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(uuid) DO UPDATE SET
                    user_id = excluded.user_id,
                    access_token = excluded.access_token,
                    access_token_expires_at = excluded.access_token_expires_at,
                    refresh_token = excluded.refresh_token,
                    username = excluded.username,
                    name = excluded.name,
                    slug = excluded.slug,
                    avatar_url = excluded.avatar_url,
                    timezone = excluded.timezone
                """,
                (
                    uuid,
                    user_id,
                    access_token,
                    access_token_expires_at.isoformat(),
                    refresh_token,
                    username,
                    name,
                    slug,
                    avatar_url,
                    timezone,
                ),
            )
            await conn.commit()

            logger.info(f"User {uuid} upserted successfully")
            return True

    except Exception as e:
        logger.error(f"Error upserting user {uuid}: {e}", exc_info=True)
        return False


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
        async with get_db() as conn:
            cursor = await conn.execute("DELETE FROM users WHERE uuid = ?", (uuid,))
            await conn.commit()
            rows_deleted = cursor.rowcount

            if rows_deleted > 0:
                logger.info(f"User {uuid} deleted successfully")
                return True
            else:
                logger.info(f"No user found with UUID {uuid} to delete")
                return False
    except Exception as e:
        logger.error(f"Error deleting user {uuid}: {e}")
        return False


# Media data methods
async def trakt_media_exists(trakt_url: str) -> bool:
    """
    Check if a Trakt media item exists in the database.

    Args:
        trakt_url: Unique Trakt URL for the media

    Returns:
        bool: True if exists, False otherwise
    """
    try:
        async with get_db() as conn:
            cursor = await conn.execute(
                "SELECT 1 FROM trakt_media WHERE trakt_url = ?", (trakt_url,)
            )
            result = await cursor.fetchone()
            return result is not None
    except Exception as e:
        logger.error(f"Error checking trakt media existence: {e}")
        return False


async def upsert_batch_media(table: str, records: List[Dict[str, Any]]) -> int:
    """
    Insert or update multiple records in a given table.

    Args:
        table: Table name
        records: List of record dictionaries

    Returns:
        int: Number of records processed
    """
    if not records:
        return 0

    try:
        async with get_db() as conn:
            # Get column names from the first record
            records = [
                {k: serialize_for_sqlite(v) for k, v in record.items()}
                for record in records
            ]
            columns = list(records[0].keys())

            # Create placeholders for SQL query
            placeholders = ", ".join(["?"] * len(columns))
            column_str = ", ".join(columns)

            # Determine primary key column based on table
            pk_column = ""
            if table == "trakt_media":
                pk_column = "trakt_url"
            elif table == "imdb_media":
                pk_column = "show_imdb_id"
            elif table == "tmdb_media":
                pk_column = "show_tmdb_id"
            else:
                logger.error(f"Unknown table: {table}")
                return 0

            # Create update string (exclude primary key)
            update_str = ", ".join(
                [f"{col} = excluded.{col}" for col in columns if col != pk_column]
            )

            # Construct the SQL query
            query = f"""
            INSERT INTO {table} ({column_str})
            VALUES ({placeholders})
            ON CONFLICT({pk_column}) DO UPDATE SET {update_str}
            """

            # Execute batch insert in a transaction
            await conn.execute("BEGIN TRANSACTION")
            try:
                for record in records:
                    values = [record.get(col) for col in columns]
                    await conn.execute(query, values)
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                raise e

            return len(records)

    except Exception as e:
        logger.error(f"Error in batch upsert for {table}: {e}")
        return 0


async def update_watch_history(
    user_uuid: str, watch_history: List[Dict[str, Any]]
) -> Tuple[int, int]:
    """
    Update user watch history records.

    Args:
        user_uuid: User UUID
        watch_history: List of watch history records

    Returns:
        Tuple[int, int]: (records inserted, records updated)
    """
    if not watch_history:
        return (0, 0)

    inserted = 0
    updated = 0

    try:
        async with transaction() as conn:
            # First get existing records for this user
            cursor = await conn.execute(
                "SELECT event_id FROM user_watch_history WHERE uuid = ?", (user_uuid,)
            )
            existing_records = await cursor.fetchall()

            existing_ids = {record["event_id"] for record in existing_records}

            # Verify foreign key references in bulk
            trakt_urls = {
                item.get("trakt_url") for item in watch_history if item.get("trakt_url")
            }

            # Get valid trakt_urls
            valid_trakt_urls = set()
            if trakt_urls:
                # SQLite has parameter limits, so chunk the queries if needed
                trakt_urls_list = list(trakt_urls)
                for i in range(
                    0, len(trakt_urls_list), 500
                ):  # Process in chunks of 500
                    chunk = trakt_urls_list[i : i + 500]
                    placeholders = ",".join(["?"] * len(chunk))
                    cursor = await conn.execute(
                        f"SELECT trakt_url FROM trakt_media WHERE trakt_url IN ({placeholders})",
                        chunk,
                    )
                    valid_chunk = {row["trakt_url"] for row in await cursor.fetchall()}
                    valid_trakt_urls.update(valid_chunk)

            # Process each record
            for item in watch_history:
                event_id = item.get("event_id")
                watched_at = item.get("watched_at")

                # Convert datetime to string if needed
                if isinstance(watched_at, datetime):
                    watched_at = watched_at.isoformat()

                # Handle foreign keys properly to avoid constraint violations
                trakt_url = item.get("trakt_url")
                if trakt_url and trakt_url not in valid_trakt_urls:
                    trakt_url = None

                # Check if it's an insert or update
                if event_id in existing_ids:
                    # Update only the fields defined in schema.sql
                    await conn.execute(
                        """
                        UPDATE user_watch_history SET
                            trakt_url = ?,
                            watched_at = ?,
                            runtime = ?
                        WHERE event_id = ? AND uuid = ?
                        """,
                        (
                            trakt_url,
                            watched_at,
                            item.get("runtime"),
                            event_id,
                            user_uuid,
                        ),
                    )
                    updated += 1
                else:
                    # Insert only the fields defined in schema.sql
                    await conn.execute(
                        """
                        INSERT INTO user_watch_history (
                            event_id, uuid, trakt_url, watched_at, runtime
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            event_id,
                            user_uuid,
                            trakt_url,
                            watched_at,
                            item.get("runtime"),
                        ),
                    )
                    inserted += 1

            return (inserted, updated)

    except Exception as e:
        logger.error(f"Error updating watch history: {e}", exc_info=True)
        return (0, 0)


async def get_user_watch_history(
    user_uuid: str,
    limit: int = None,
    offset: int = 0,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve user watch history with optional filters.

    Args:
        user_uuid: User's UUID
        limit: Maximum number of records to return
        offset: Number of records to skip
        filters: Dictionary of filter conditions

    Returns:
        List[Dict[str, Any]]: List of watch history records
    """
    try:
        query = """
            SELECT
                uwh.event_id, uwh.uuid, uwh.watched_at, uwh.runtime,
                tm.trakt_url, tm.media_type, tm.title, tm.overview, tm.ep_overview, 
                tm.season_num, tm.ep_num, tm.ep_title, tm.status, tm.released, 
                tm.show_released, tm.total_episodes, tm.show_trakt_id, 
                tm.show_imdb_id, tm.show_tmdb_id,
                tmdb.genres as tmdb_genres, tmdb.poster_url as tmdb_poster_url,
                imdb.genres as imdb_genres
            FROM user_watch_history uwh
            LEFT JOIN trakt_media tm ON uwh.trakt_url = tm.trakt_url
            LEFT JOIN tmdb_media tmdb ON tm.show_tmdb_id = tmdb.show_tmdb_id
            LEFT JOIN imdb_media imdb ON tm.show_imdb_id = imdb.show_imdb_id
            WHERE uwh.uuid = ?
        """

        params = [user_uuid]

        # Add filters if provided
        if filters:
            for key, value in filters.items():
                if key == "media_type":
                    query += " AND tm.media_type = ?"
                    params.append(value)
                elif key == "title":
                    query += " AND tm.title LIKE ?"
                    params.append(f"%{value}%")
                elif key == "start_date":
                    query += " AND uwh.watched_at >= ?"
                    params.append(value)
                elif key == "end_date":
                    query += " AND uwh.watched_at <= ?"
                    params.append(value)
                # Add more filter conditions as needed

        # Add ordering
        query += " ORDER BY uwh.watched_at DESC"

        # Add pagination
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        if offset:
            query += " OFFSET ?"
            params.append(offset)

        async with get_db() as conn:
            cursor = await conn.execute(query, params)
            results = await cursor.fetchall()

            # Handle JSON arrays stored as strings
            for row in results:
                for key in ["tmdb_genres", "imdb_genres"]:
                    if row.get(key) and isinstance(row[key], str):
                        try:
                            row[key] = json.loads(row[key])
                        except:
                            row[key] = []

            return results

    except Exception as e:
        logger.error(f"Error getting user watch history: {e}")
        return []


async def get_user_last_updated(user_uuid: str) -> Optional[datetime]:
    """
    Get the last update timestamp for a user.

    Args:
        user_uuid: User UUID

    Returns:
        Optional[datetime]: Last update timestamp or None if never updated
    """
    try:
        async with get_db() as conn:
            cursor = await conn.execute(
                "SELECT last_db_update FROM users WHERE uuid = ?", (user_uuid,)
            )
            result = await cursor.fetchone()

            if result and result.get("last_db_update"):
                try:
                    # Convert ISO string to datetime
                    return datetime.fromisoformat(
                        result["last_db_update"].replace("Z", "+00:00")
                    )
                except ValueError:
                    return None
            return None

    except Exception as e:
        logger.error(f"Error getting last updated timestamp: {e}")
        return None


# Helper to convert data for database storage
def serialize_for_sqlite(data: Any) -> Any:
    """
    Convert Python data types to SQLite-compatible types.

    Args:
        data: Value to serialize

    Returns:
        Serialized value for SQLite storage
    """
    if isinstance(data, (list, dict)):
        return json.dumps(data)
    elif isinstance(data, datetime):
        return data.isoformat()
    return data
