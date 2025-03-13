"""
Watch history service module.

This module handles fetching, transforming and storing watch history data
from Trakt API to the database using polars for efficient data processing.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl

from app.core.trakt_api import (
    fetch_complete_watch_history,
    fetch_episode_details,
    fetch_movie_details,
    fetch_show_details,
)
from app.db.db import (
    execute,
    fetch_all,
    get_user_last_updated,
    get_user_watch_history,
    transaction,
    update_watch_history,
    upsert_batch_media,
)

# Set up logging
logger = logging.getLogger(__name__)


async def get_watch_history_data(
    trakt_id: str,
    uuid: str,
    headers: Dict[str, str],
    force_refresh: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
    filters: Optional[Dict[str, Any]] = None,
) -> Union[pl.DataFrame, List[Dict[str, Any]]]:
    """
    Get watch history data either from database or Trakt API.

    Args:
        trakt_id: Trakt user ID
        uuid: User UUID
        headers: API request headers
        force_refresh: Whether to force refresh from API
        limit: Maximum number of records to return
        offset: Number of records to skip
        filters: Dictionary of filter conditions

    Returns:
        Union[pl.DataFrame, List[Dict[str, Any]]]: Watch history data
    """
    start_time = time.time()

    # If not forcing refresh, get from database
    if not force_refresh:
        db_filters = {}
        if filters:
            # Map endpoint filters to database filters
            if "media_type" in filters:
                db_filters["media_type"] = filters["media_type"]
            if "start_date" in filters:
                db_filters["start_date"] = filters["start_date"]
            if "end_date" in filters:
                db_filters["end_date"] = filters["end_date"]

        db_data = await get_user_watch_history(
            user_uuid=uuid, limit=limit, offset=offset, filters=db_filters
        )

        if db_data and len(db_data) > 0:
            logger.info(
                f"Retrieved watch history for {trakt_id} from database ({len(db_data)} items)"
            )
            return db_data

    # If no data in database or forcing refresh, trigger a refresh and then get from DB
    await refresh_watch_history(uuid, trakt_id, headers, force=force_refresh)

    # Fetch the refreshed data from database
    db_filters = {}
    if filters:
        if "media_type" in filters:
            db_filters["media_type"] = filters["media_type"]
        if "start_date" in filters:
            db_filters["start_date"] = filters["start_date"]
        if "end_date" in filters:
            db_filters["end_date"] = filters["end_date"]

    db_data = await get_user_watch_history(
        user_uuid=uuid, limit=limit, offset=offset, filters=db_filters
    )

    logger.info(
        f"Retrieved refreshed watch history for {trakt_id} from database ({len(db_data)} items)"
    )
    return db_data


def transform_watch_history(history_data: List[Dict[str, Any]]) -> pl.DataFrame:
    """
    Transform raw Trakt API watch history data into a structured polars DataFrame.

    Args:
        history_data: Raw API response data

    Returns:
        pl.DataFrame: Transformed data
    """
    if not history_data:
        return pl.DataFrame()

    # Extract data from API response
    records = []

    for item in history_data:
        # Extract base watch history record data
        watch_history_record = {
            "event_id": item.get("id"),
            "watched_at": item.get("watched_at"),
            "trakt_url": None,  # Will be constructed later
            "runtime": None,  # Will be populated from movie/episode data
        }

        # Extract media data for trakt_media table
        trakt_media_record = {
            "media_type": item.get("type"),
            "trakt_url": None,  # Will be constructed later
            "title": None,
            "ep_title": None,
            "season_num": None,
            "ep_num": None,
            "ep_num_abs": None,
            "total_episodes": None,
            "status": None,
            "ep_overview": None,
            "overview": None,
            "runtime": None,
            "released": None,
            "show_released": None,
            "genres": [],
            "country": None,
            "show_trakt_id": None,
            "show_imdb_id": None,
            "show_tmdb_id": None,
        }

        # Extract show/movie specific data
        if item.get("type") == "movie":
            movie = item.get("movie", {})
            trakt_media_record.update(
                {
                    "title": movie.get("title"),
                    "show_trakt_id": movie.get("ids", {}).get("trakt"),
                    "show_imdb_id": movie.get("ids", {}).get("imdb"),
                    "show_tmdb_id": movie.get("ids", {}).get("tmdb"),
                    "runtime": movie.get("runtime"),
                    "released": movie.get("released"),
                    "country": movie.get("country"),
                    "genres": movie.get("genres", []),
                    "overview": movie.get("overview"),
                    "status": movie.get("status"),
                }
            )

            # Update watch history runtime from movie data
            watch_history_record["runtime"] = movie.get("runtime")

            # Construct Trakt URL for movie
            slug = movie.get("ids", {}).get("slug")
            if slug:
                trakt_url = f"https://trakt.tv/movies/{slug}"
                trakt_media_record["trakt_url"] = trakt_url
                watch_history_record["trakt_url"] = trakt_url

        else:
            # Episode data
            show = item.get("show", {})
            episode = item.get("episode", {})
            trakt_media_record.update(
                {
                    "title": show.get("title"),
                    "show_trakt_id": show.get("ids", {}).get("trakt"),
                    "show_imdb_id": show.get("ids", {}).get("imdb"),
                    "show_tmdb_id": show.get("ids", {}).get("tmdb"),
                    "runtime": episode.get("runtime"),
                    "released": episode.get("first_aired"),
                    "country": show.get("country"),
                    "genres": show.get("genres", []),
                    "overview": show.get("overview"),
                    "status": show.get("status"),
                    "season_num": episode.get("season"),
                    "ep_num": episode.get("number"),
                    "ep_title": episode.get("title"),
                    "ep_overview": episode.get("overview"),
                    "total_episodes": show.get("aired_episodes"),
                    "show_released": show.get("first_aired"),
                    "ep_num_abs": episode.get("number_abs"),
                }
            )

            # Update watch history runtime from episode data
            watch_history_record["runtime"] = episode.get("runtime")

            # Construct Trakt URL for episode
            slug = show.get("ids", {}).get("slug")
            season = episode.get("season")
            ep_num = episode.get("number")
            if slug and season is not None and ep_num is not None:
                trakt_url = (
                    f"https://trakt.tv/shows/{slug}/seasons/{season}/episodes/{ep_num}"
                )
                trakt_media_record["trakt_url"] = trakt_url
                watch_history_record["trakt_url"] = trakt_url

        # Add the watch history record with proper runtime and trakt_url
        watch_record = {
            "event_id": watch_history_record["event_id"],
            "watched_at": watch_history_record["watched_at"],
            "trakt_url": watch_history_record["trakt_url"],
            "runtime": watch_history_record["runtime"],
            "media_type": trakt_media_record[
                "media_type"
            ],  # Added for filtering but not stored
            "title": trakt_media_record["title"],  # Added for filtering but not stored
        }

        # Skip records without trakt_url as they can't be properly linked
        if watch_record["trakt_url"]:
            records.append(watch_record)

    # Create polars DataFrame
    df = pl.DataFrame(records)

    # Apply special case adjustments
    if not df.is_empty():
        df = adjust_special_cases(df)

    # Ensure proper types
    df = apply_schema_types(df)

    return df


def adjust_special_cases(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply special case adjustments to the data.

    Args:
        df: Input DataFrame

    Returns:
        pl.DataFrame: Adjusted DataFrame
    """
    if df.is_empty():
        return df

    # Handle "We Got Married" show with special runtime adjustment
    if "title" in df.columns and "runtime" in df.columns and "media_type" in df.columns:
        # Create masks for conditions
        mask_wgm = pl.col("title") == "We Got Married"
        mask_episode = pl.col("media_type") == "episode"

        # Apply runtime adjustments based on episode properties derived from trakt_url
        if "trakt_url" in df.columns:
            # Extract season info from URL when possible
            df = df.with_columns(
                pl.col("trakt_url")
                .str.extract(r"/seasons/(\d+)/episodes/")
                .cast(pl.Int64)
                .alias("_season_num")
            )

            mask_season4 = pl.col("_season_num") == 4

            # For season 4, specific episodes with 2/3 adjustment
            df = df.with_columns(
                pl.when(mask_wgm & mask_episode & mask_season4)
                .then(pl.col("runtime") * 2 / 3)
                .otherwise(pl.col("runtime"))
                .floor()
                .cast(pl.Int64)
                .alias("runtime")
            )

            # For all other seasons with 1/3 adjustment
            df = df.with_columns(
                pl.when(mask_wgm & mask_episode & ~mask_season4)
                .then(pl.col("runtime") / 3)
                .otherwise(pl.col("runtime"))
                .floor()
                .cast(pl.Int64)
                .alias("runtime")
            )

            # Drop the temporary column
            df = df.drop("_season_num")

    return df


def apply_schema_types(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply proper schema types to DataFrame columns.

    Args:
        df: Input DataFrame

    Returns:
        pl.DataFrame: DataFrame with proper types
    """
    if df.is_empty():
        return df

    # Define column types mapping
    type_map = {
        "event_id": pl.Int64,
        "trakt_url": pl.Utf8,
        "watched_at": pl.Datetime,
        "runtime": pl.Int64,
        "media_type": pl.Utf8,  # For filtering only
        "title": pl.Utf8,  # For filtering only
    }

    # Apply types to available columns
    for col_name, dtype in type_map.items():
        if col_name in df.columns:
            # Handle conversion carefully with null values
            try:
                if dtype == pl.Datetime:
                    df = df.with_columns(
                        pl.col(col_name)
                        .str.to_datetime(format=None, strict=False)
                        .alias(col_name)
                    )
                else:
                    df = df.with_columns(
                        pl.col(col_name).cast(dtype, strict=False).alias(col_name)
                    )
            except Exception as e:
                logger.warning(f"Error converting {col_name} to {dtype}: {e}")

    return df


async def store_watch_history_in_db(uuid: str, df: pl.DataFrame) -> bool:
    """
    Store watch history data in the database.

    Args:
        uuid: User UUID
        df: DataFrame with watch history data

    Returns:
        bool: True if successful, False otherwise
    """
    if df.is_empty():
        return False

    try:
        # Begin transaction
        async with transaction() as conn:
            # Extract unique trakt_media records from history data
            trakt_urls = set(
                row["trakt_url"]
                for row in df.iter_rows(named=True)
                if row.get("trakt_url")
            )

            # For each trakt_url, fetch additional media details to populate media tables
            media_records = {}
            imdb_records = {}
            tmdb_records = {}

            for trakt_url in trakt_urls:
                # Extract media type and ID from URL
                media_type = None
                media_slug = None

                if "movies/" in trakt_url:
                    media_type = "movie"
                    media_slug = trakt_url.split("/movies/")[1].split("/")[0]
                elif "shows/" in trakt_url:
                    media_type = "episode"
                    media_slug = trakt_url.split("/shows/")[1].split("/")[0]

                if not media_type or not media_slug:
                    continue

                # Prepare media records for insertion/update
                # For the actual implementation, we'd fetch additional data
                # from Trakt API here if needed

                # For now, extract what we can from the DataFrame
                for row in df.filter(pl.col("trakt_url") == trakt_url).iter_rows(
                    named=True
                ):
                    media_records[trakt_url] = {
                        "trakt_url": trakt_url,
                        "media_type": media_type,
                        "title": row.get("title"),
                        # Other fields would be set from additional API calls if needed
                    }
                    break

            # Insert IMDB and TMDB records first (for foreign key constraints)
            # In a real implementation, we would populate these fields

            # Create the complete watch history records list
            watch_history_records = []
            for row in df.iter_rows(named=True):
                watch_record = {
                    "event_id": row.get("event_id"),
                    "uuid": uuid,
                    "trakt_url": row.get("trakt_url"),
                    "watched_at": row.get("watched_at"),
                    "runtime": row.get("runtime"),
                }
                watch_history_records.append(watch_record)

            # Update watch history in database
            inserted, updated = await update_watch_history(uuid, watch_history_records)
            logger.info(f"Watch history update: {inserted} inserted, {updated} updated")

            # Update last_updated_at timestamp for user
            await conn.execute(
                "UPDATE users SET last_db_update = ? WHERE uuid = ?",
                (datetime.utcnow().isoformat(), uuid),
            )

            return True

    except Exception as e:
        logger.error(f"Error storing watch history: {e}", exc_info=True)
        return False


def format_history_for_response(
    history_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Format watch history data for API response.

    Args:
        history_data: Raw watch history data

    Returns:
        List[Dict[str, Any]]: Formatted history data
    """
    formatted_data = []

    for item in history_data:
        # Deep copy to avoid modifying original
        item_dict = dict(item)

        # Format datetime fields for JSON response
        for field in ["watched_at", "released", "show_released"]:
            if item_dict.get(field) and isinstance(item_dict[field], str):
                try:
                    item_dict[field] = datetime.fromisoformat(
                        item_dict[field].replace("Z", "+00:00")
                    ).isoformat()
                except:
                    pass

        # Parse JSON strings if needed
        for field in ["genres", "tmdb_genres", "imdb_genres"]:
            if item_dict.get(field) and isinstance(item_dict[field], str):
                try:
                    item_dict[field] = json.loads(item_dict[field])
                except:
                    item_dict[field] = []

        formatted_data.append(item_dict)

    return formatted_data


async def get_history_count(uuid: str) -> int:
    """
    Get count of watch history entries for a user.

    Args:
        uuid: User UUID

    Returns:
        int: Count of watch history entries
    """
    try:
        query = "SELECT COUNT(*) as count FROM user_watch_history WHERE uuid = ?"
        result = await fetch_all(query, (uuid,))
        return result[0]["count"] if result else 0
    except Exception as e:
        logger.error(f"Error getting history count: {e}")
        return 0


async def get_most_recent_event_id(uuid: str) -> Optional[int]:
    """
    Get the most recent event_id from user's watch history.

    Args:
        uuid: User UUID

    Returns:
        Optional[int]: Most recent event_id or None if no history
    """
    try:
        query = """
        SELECT event_id 
        FROM user_watch_history 
        WHERE uuid = ? 
        ORDER BY watched_at DESC 
        LIMIT 1
        """
        result = await fetch_all(query, (uuid,))
        return result[0]["event_id"] if result else None
    except Exception as e:
        logger.error(f"Error getting most recent event_id: {e}")
        return None


async def refresh_watch_history(
    uuid: str, trakt_id: str, headers: Dict[str, str], force: bool = False
) -> None:
    """
    Refresh watch history data from Trakt API.

    Implements the logic for full and partial history pull based on
    user's watch history state in the database:
    - If user has no history records: perform full history pull
    - If user has history records: perform partial history pull from last_db_update date

    Args:
        uuid: User UUID
        trakt_id: Trakt user ID/slug
        headers: API request headers
        force: Whether to force a full refresh regardless of existing data
    """
    try:
        logger.info(f"Starting watch history refresh for user {uuid}")

        # Check if this is a full or partial history pull
        history_count = await get_history_count(uuid)
        most_recent_event_id = await get_most_recent_event_id(uuid)

        # Determine if full pull required (force=True or no history)
        full_pull_required = force or history_count == 0

        # Get date for partial pull if needed
        start_date = None
        if not full_pull_required:
            # Get last update time from user record
            last_updated = await get_user_last_updated(uuid)

            if last_updated:
                # Go back 2 days to ensure we don't miss anything
                start_date = (last_updated - timedelta(days=2)).strftime("%Y-%m-%d")
            else:
                # No last_updated date but we have history - do full pull
                full_pull_required = True

        if full_pull_required:
            logger.info(f"Performing full history pull for user {uuid}")
        else:
            logger.info(
                f"Performing partial history pull for user {uuid} from {start_date}"
            )

        # Fetch data from Trakt API
        history_data = await fetch_complete_watch_history(
            headers, trakt_id, start_date=start_date
        )

        if not history_data:
            logger.info(f"No new watch history data for user {uuid}")
            return

        # For partial history, check if we have any new events
        if not full_pull_required and most_recent_event_id:
            if history_data and history_data[0]["id"] == most_recent_event_id:
                logger.info(f"No new events since last update for user {uuid}")

                # Update last_db_update timestamp even though no new data
                await execute(
                    "UPDATE users SET last_db_update = ? WHERE uuid = ?",
                    (datetime.utcnow().isoformat(), uuid),
                    commit=True,
                )
                return

        # Transform API response to DataFrame
        df = transform_watch_history(history_data)

        if not df.is_empty():
            # Store in database
            success = await store_watch_history_in_db(uuid, df)
            if success:
                logger.info(
                    f"Watch history refresh completed for user {uuid}: {df.height} items processed"
                )
            else:
                logger.error(f"Failed to store watch history for user {uuid}")
        else:
            logger.info(f"No valid watch history data to store for user {uuid}")

    except Exception as e:
        logger.error(f"Error in watch history refresh task: {e}", exc_info=True)
