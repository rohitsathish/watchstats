import pandas as pd
import json
import operator
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from sqlalchemy.orm import sessionmaker
import requests as req
import sqlglot
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    func,
    or_,
    and_,
    asc,
    desc,
)

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, scoped_session, joinedload
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql.expression import BinaryExpression, Select
import math


import streamlit as st
from db.models import Base, Users, UserWatchHistory, TraktMedia, IMDBMedia, TMDBMedia

from typing import List, Any, Optional, Union, Tuple, Dict

import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# SQLAlchemy Base
# Base = declarative_base()

# Database Credentials
DB_USER = "postgres"
DB_PASS = "sohyunmina89"
DB_PORT = "5432"
DB_NAME = "ws_media_db"

# Database URL for Synchronous Engine
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@localhost:{DB_PORT}/{DB_NAME}"

# SQLAlchemy Models Mapping
model_map = {
    "users": Users,
    "user_watch_history": UserWatchHistory,
    "trakt_media": TraktMedia,
    "imdb_media": IMDBMedia,
    "tmdb_media": TMDBMedia,
}

# ---------------------------- Engine and Session ---------------------------- #

# Create synchronous engine with optimized connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,  # Increased pool size for better concurrency
    max_overflow=20,  # Allow more connections during high load
    pool_recycle=1800,
    pool_timeout=30,
    pool_pre_ping=True,  # Enable pre-ping to detect stale connections
    echo=False,  # Set to True for SQL query logging
)

# Create synchronous session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

# ----

from contextlib import contextmanager


@contextmanager
def get_session():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def get_cursor_from_session(session):
    """Get the cursor from the underlying connection in a session."""
    connection = session.connection()  # Get the underlying connection
    raw_connection = connection.connection  # Get the raw DB-API connection
    cursor = raw_connection.cursor()  # Get the cursor from the raw connection
    return cursor


def timeit(func):
    """Decorator to measure the execution time of functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            total_time = end_time - start_time
            st.write(f"Function `{func.__name__}` executed in {total_time:.2f} seconds")

    return wrapper


# ----


def get_primary_keys(model: Any) -> List[str]:
    """Retrieve primary key column names for a given model."""
    return [key.name for key in inspect(model).primary_key]


def get_model_columns(model: Any) -> List[str]:
    """Retrieve all column names for a given model."""
    return [column.name for column in inspect(model).columns]


# ----


@timeit
def prepare_dataframe(df: pd.DataFrame, model: Any, uuid: Optional[str] = None) -> pd.DataFrame:
    """
    Prepare DataFrame for insertion/upsert/sync:
    - Add uuid if necessary
    - Replace NaN with None
    - Drop duplicates based on primary keys
    - Select relevant columns
    - Convert data types as necessary
    """
    primary_keys = get_primary_keys(model)

    if hasattr(model, "trakt_uuid") and uuid:
        df["trakt_uuid"] = uuid

    df = df.replace({pd.NA: None, pd.NaT: None}).drop_duplicates(subset=primary_keys)

    insert_cols = [col for col in df.columns if col in get_model_columns(model)]
    df = df[insert_cols]

    # Convert Int64 columns to int (nullable)
    int64_cols = df.select_dtypes(include=["Int64"]).columns
    df[int64_cols] = df[int64_cols].astype("Int32")

    return df


# ----


@timeit
def add_data(
    df: pd.DataFrame,
    uuid: Optional[str],
    model_name: str,
    operation: str = "insert",
) -> None:
    """
    Add data to the specified model/table in the database using psycopg2's execute_values for high performance.

    Supports three operations:
    - insert: Add new data without affecting existing records.
    - upsert: Insert new data and update existing records based on primary keys.
    - sync: Match the table values with the provided DataFrame, inserting/updating records and deleting those not present.

    :param df: pandas DataFrame containing the data to add
    :param uuid: UUID associated with the data (specific to models)
    :param model_name: Name of the model/table to insert data into
    :param operation: Operation type - "insert", "upsert", or "sync"
    """
    model_map = {
        "users": Users,
        "user_watch_history": UserWatchHistory,
        "trakt_media": TraktMedia,
        "imdb_media": IMDBMedia,
        "tmdb_media": TMDBMedia,
    }

    model = model_map.get(model_name)
    if not model:
        logger.error(f"Model '{model_name}' not found in model_map.")
        raise ValueError(f"Model '{model_name}' not found.")

    # Prepare DataFrame
    df_prepared = prepare_dataframe(df, model, uuid)

    # Convert DataFrame to list of tuples
    records = df_prepared.to_dict(orient="records")
    if not records:
        logger.info("No records to process.")
        return

    # Extract column names
    columns = list(df_prepared.columns)

    # Prepare the upsert SQL statement
    primary_keys = get_primary_keys(model)
    non_pk_columns = [col for col in columns if col not in primary_keys]

    # Initial insert statement without ON CONFLICT clause
    insert_stmt = sql.SQL(
        """
        INSERT INTO {table} ({fields})
        VALUES %s
        {on_conflict}
        """
    ).format(
        table=sql.Identifier(model.__tablename__),
        fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
        on_conflict=sql.SQL(""),  # This remains empty in case of a standard insert
    )

    # Now, handle the special cases for 'upsert' and 'sync' separately
    if operation in ["upsert", "sync"]:
        # Building the new upsert statement from scratch
        updates = sql.SQL(", ").join(
            [sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col)) for col in non_pk_columns]
        )

        # Create a new upsert_stmt, without modifying insert_stmt
        insert_stmt = sql.SQL(
            """
            INSERT INTO {table} ({fields})
            VALUES %s
            ON CONFLICT ({pks}) DO UPDATE SET {updates}
            """
        ).format(
            table=sql.Identifier(model.__tablename__),
            fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
            pks=sql.SQL(", ").join(map(sql.Identifier, primary_keys)),
            updates=updates,
        )

    # Establish psycopg2 connection using a context manager
    with get_session() as session:

        cur = get_cursor_from_session(session)
        # try:
        #     with conn.cursor() as cur:
        # Prepare data as list of tuples
        data_tuples = [tuple(record[col] for col in columns) for record in records]

        # Execute the bulk upsert
        execute_values(cur, insert_stmt.as_string(cur), data_tuples, template=None, page_size=1000)

        session.commit()
        st.write(f"Successfully {operation}ed {len(records)} records into '{model_name}'.")

    # except Exception as e:
    #     conn.rollback()
    #     logger.exception("Error during bulk upsert operation.")
    #     raise

    # Handle 'sync' operation specifics
    if operation == "sync":
        try:
            with SessionLocal() as session:
                # Identify records to delete (not present in the provided DataFrame)
                if hasattr(model, "trakt_uuid") and uuid:
                    # Extract primary key values from the DataFrame
                    pk_values = [tuple(x[pk] for pk in primary_keys) for x in records]
                    if len(primary_keys) == 1:
                        pk_column = getattr(model, primary_keys[0])
                        filter_condition = and_(
                            getattr(model, "trakt_uuid") == uuid, ~pk_column.in_([x[0] for x in pk_values])
                        )
                    else:
                        # For composite primary keys
                        from sqlalchemy import tuple_

                        pk_columns = tuple(getattr(model, pk) for pk in primary_keys)
                        filter_condition = and_(
                            getattr(model, "trakt_uuid") == uuid, ~tuple_(pk_columns).in_(pk_values)
                        )

                    # Perform the delete operation
                    deleted = session.query(model).filter(filter_condition).delete(synchronize_session=False)
                    session.commit()
                    logger.info(f"Deleted {deleted} records from '{model_name}' not present in provided data.")

        except Exception as e:
            logger.exception("Error during sync delete operation.")
            raise


@timeit
def read_table_df(
    table_name: str,
    joins: Optional[List[Tuple[str, str, str, str]]] = None,
    filters: Optional[Dict[str, Union[Any, Tuple[str, Any]]]] = None,
    order_by: Optional[Union[str, List[str]]] = None,
    order_desc: bool = False,
    limit: Optional[int] = None,
    ex_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read data from a table with optional joins, filters, ordering, and limits.

    :param table_name: Name of the main table to query
    :param joins: List of tuples specifying joins in the format (left_table, right_table, left_col, right_col)
    :param filters: Dictionary specifying filters where key is "table.column" or "column" and value is the filter value or condition
    :param order_by: Column name or list of column names to order by
    :param order_desc: Boolean indicating if the order should be descending
    :param limit: Maximum number of records to retrieve
    :param ex_columns: List of columns to exclude from the result
    :return: pandas DataFrame containing the queried data
    """
    model_map = {
        "users": Users,
        "user_watch_history": UserWatchHistory,
        "trakt_media": TraktMedia,
        "imdb_media": IMDBMedia,
        "tmdb_media": TMDBMedia,
    }

    main_model = model_map.get(table_name)
    if not main_model:
        logger.error(f"Table '{table_name}' not found in model_map.")
        raise ValueError(f"Table '{table_name}' not found.")

    with get_session() as session:
        try:
            query = session.query(main_model)
            joined_models = {}

            # Apply Joins
            if joins:
                for join in joins:
                    left_table, right_table, left_col, right_col = join
                    left_model = model_map.get(left_table)
                    right_model = model_map.get(right_table)
                    if not left_model or not right_model:
                        logger.error(f"One of the join tables '{left_table}' or '{right_table}' not found.")
                        raise ValueError(f"One of the join tables '{left_table}' or '{right_table}' not found.")

                    # Define the relationship dynamically if not already present
                    # Assuming that relationships are already defined in the models

                    # Perform the join
                    query = query.join(
                        right_model,
                        getattr(left_model, left_col) == getattr(right_model, right_col),
                        isouter=True,  # Use outer join for versatility
                    )
                    joined_models[right_table] = right_model

            # Select Columns
            all_columns = []
            seen_columns = set()
            ex_columns = ex_columns or []

            # Main table columns
            for column in main_model.__table__.columns:
                if column.name not in ex_columns and column.name not in seen_columns:
                    all_columns.append(column)
                    seen_columns.add(column.name)

            # Joined tables columns
            for joined_model in joined_models.values():
                for column in joined_model.__table__.columns:
                    if column.name not in ex_columns and column.name not in seen_columns:
                        all_columns.append(column)
                        seen_columns.add(column.name)

            if all_columns:
                query = query.with_entities(*all_columns)

            # Apply Filters
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if "." in key:
                        tbl, col = key.split(".", 1)
                        model = model_map.get(tbl)
                        if not model:
                            logger.error(f"Filter table '{tbl}' not found.")
                            raise ValueError(f"Filter table '{tbl}' not found.")
                        column = getattr(model, col, None)
                    else:
                        column = getattr(main_model, key, None)
                    if not column:
                        logger.error(f"Filter column '{key}' not found.")
                        raise ValueError(f"Filter column '{key}' not found.")

                    if isinstance(value, tuple) and len(value) == 2:
                        op, val = value
                        if op == "<":
                            condition = column < val
                        elif op == "<=":
                            condition = column <= val
                        elif op == ">":
                            condition = column > val
                        elif op == ">=":
                            condition = column >= val
                        elif op == "==":
                            condition = column == val
                        elif op == "!=":
                            condition = column != val
                        else:
                            logger.error(f"Unsupported filter operator '{op}'.")
                            raise ValueError(f"Unsupported filter operator '{op}'.")
                    elif isinstance(value, list):
                        condition = column.in_(value)
                    else:
                        condition = column == value

                    filter_conditions.append(condition)

                if filter_conditions:
                    query = query.filter(and_(*filter_conditions))

            # Apply Ordering
            if order_by:
                if isinstance(order_by, str):
                    order_by = [order_by]
                order_columns = []
                for col in order_by:
                    if "." in col:
                        tbl, column = col.split(".", 1)
                        model = model_map.get(tbl)
                        if not model:
                            logger.error(f"Order table '{tbl}' not found.")
                            raise ValueError(f"Order table '{tbl}' not found.")
                        column_attr = getattr(model, column, None)
                    else:
                        column_attr = getattr(main_model, col, None)

                    if not column_attr:
                        logger.error(f"Order column '{col}' not found.")
                        raise ValueError(f"Order column '{col}' not found.")

                    if order_desc:
                        order_columns.append(desc(column_attr))
                    else:
                        order_columns.append(asc(column_attr))

                if order_columns:
                    query = query.order_by(*order_columns)

            # Apply Limit
            if limit:
                query = query.limit(limit)

            # Execute Query
            results = query.all()

            # Convert to DataFrame
            df = pd.DataFrame([dict(row._mapping) for row in results])

            return df

        except SQLAlchemyError as e:
            logger.exception("Database read operation failed.")
            raise
        except Exception as e:
            logger.exception("An unexpected error occurred in read_table_df.")
            raise
