import logging
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from sqlalchemy import (
    ARRAY,
    text,
    create_engine,
    func,
    inspect,
    or_,
    and_,
    asc,
    desc,
    exists,
    select,
    tuple_,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import streamlit as st

from db.models import Base, Users, UserWatchHistory, TraktMedia, IMDBMedia, TMDBMedia

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---- Model Map ----

# SQLAlchemy Models Mapping
model_map = {
    "users": Users,
    "user_watch_history": UserWatchHistory,
    "trakt_media": TraktMedia,
    "imdb_media": IMDBMedia,
    "tmdb_media": TMDBMedia,
}

# --- Database Setup ---

# Database Credentials
DB_USER = "postgres"
DB_PASS = "sohyunmina89"
DB_PORT = "5432"
DB_NAME = "ws_media_db"

# Database URL for SQLAlchemy Engine
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@localhost:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy Engine with Optimized Connection Pooling
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


# --- Connection and Session Management ---

# Create SQLAlchemy Session Factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


@contextmanager
def get_session():
    """
    Provide a transactional scope around a series of operations using SQLAlchemy sessions.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        logger.exception("Session rollback due to an error.")
        raise
    finally:
        session.close()


@contextmanager
def get_raw_connection():
    """
    Provide a transactional scope around a series of operations using psycopg2's raw connections.
    """
    conn = engine.raw_connection()
    try:
        yield conn
        conn.commit()
    except:
        conn.rollback()
        logger.exception("Raw connection rollback due to an error.")
        raise
    finally:
        conn.close()


# ---- Decorator ----


def timeit(func):
    """
    Decorator to measure the execution time of functions.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            total_time = end_time - start_time
            logger.info(f"Function `{func.__name__}` executed in {total_time:.2f} seconds.")
            st.toast(f"Function `{func.__name__}` executed in {total_time:.2f} seconds.")

    return wrapper


# ---- Utility Functions ----


def get_primary_keys(model: Any) -> List[str]:
    """
    Retrieve primary key column names for a given SQLAlchemy model.

    :param model: SQLAlchemy model class
    :return: List of primary key column names
    """
    return [key.name for key in inspect(model).primary_key]


def get_model_columns(model: Any) -> List[str]:
    """
    Retrieve all column names for a given SQLAlchemy model.

    :param model: SQLAlchemy model class
    :return: List of column names
    """
    return [column.name for column in inspect(model).columns]


def prepare_dataframe(df: pd.DataFrame, model: Any, uuid: Optional[str] = None) -> pd.DataFrame:
    """
    Prepare DataFrame for insertion/upsert/sync:
    - Add uuid if necessary
    - Replace NaN with None
    - Drop duplicates based on primary keys
    - Select relevant columns
    - Convert data types as necessary
    - Ensure array fields are lists

    :param df: pandas DataFrame to prepare
    :param model: SQLAlchemy model class
    :param uuid: UUID to associate with the data (if applicable)
    :return: Prepared pandas DataFrame
    """
    primary_keys = get_primary_keys(model)

    if hasattr(model, "trakt_uuid") and uuid:
        df["trakt_uuid"] = uuid

    # Replace NaN/NaT with None and drop duplicates based on primary keys
    df = df.replace({pd.NA: None, pd.NaT: None}).drop_duplicates(subset=primary_keys)

    # Select relevant columns
    insert_cols = [col for col in df.columns if col in get_model_columns(model)]
    df = df[insert_cols]

    # Convert Int64 columns to int (nullable)
    int64_cols = df.select_dtypes(include=["Int64"]).columns
    df[int64_cols] = df[int64_cols].astype("Int32")

    # Ensure that list-like columns are actual lists (for ARRAY types)
    inspector = inspect(engine)
    columns_info = inspector.get_columns(model.__tablename__)
    array_columns = [col["name"] for col in columns_info if isinstance(col["type"], ARRAY)]

    for col in array_columns:
        if col in df.columns:

            def convert_to_list(x):
                if isinstance(x, list):
                    return x
                elif isinstance(x, str):
                    try:
                        return ast.literal_eval(x)
                    except:
                        return []
                else:
                    return []

            df[col] = df[col].apply(convert_to_list)

    return df


# ---- Master Functions ----


def test_postgres_connection():
    """
    Test the connection to the PostgreSQL database.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            st.toast("Connection successful")
    except SQLAlchemyError as e:
        st.toast(f"Connection failed: {e}")


def close_all_connections():
    """Dispose all connections in the engine pool."""
    engine.dispose()


def create_schema(drop: bool = True, create: bool = True) -> None:

    # st.toast("Creating schema...")
    """Create or recreate the database schema."""
    # try:
    if drop:

        # inspector = inspect(engine)

        # Drop tables only if they exist

        # if inspector.has_table("imdb_media"):
        #     IMDBMedia.__table__.drop(engine)  # Drop imdb_media

        # if inspector.has_table("tmdb_media"):
        #     TMDBMedia.__table__.drop(engine)  # Drop tmdb_media

        # if inspector.has_table("users"):
        #     Users.__table__.drop(engine)

        # if inspector.has_table("user_watch_history"):
        #     UserWatchHistory.__table__.drop(engine)  # Drop user_watch_history
        # if inspector.has_table("trakt_media"):
        #     TraktMedia.__table__.drop(engine)  # Then drop trakt_media
        # Drop all existing tables
        Base.metadata.drop_all(bind=engine)
        st.toast("Dropped all existing tables.")

    if create:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        st.toast("Successfully created the schema.")

    # Inspect and list all tables and their columns
    inspector = inspect(engine)
    logger.info("Current tables and columns:")
    for table_name in inspector.get_table_names():
        logger.info(f"Table: {table_name}")
        columns = inspector.get_columns(table_name)
        for column in columns:
            logger.info(f"  - {column['name']} ({column['type']})")

    # except SQLAlchemyError as e:
    #     st.toast(f"An error occurred while creating the schema: {str(e)}")


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

    # Build the INSERT statement with optional ON CONFLICT
    insert_stmt = sql.SQL(
        """
        INSERT INTO {table} ({fields})
        VALUES %s
        {on_conflict}
        """
    ).format(
        table=sql.Identifier(model.__tablename__),
        fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
        on_conflict=sql.SQL(""),  # This remains empty for 'insert'
    )

    if operation in ["upsert", "sync"]:
        updates = sql.SQL(", ").join(
            [sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col)) for col in non_pk_columns]
        )
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

    # Establish psycopg2 connection using the raw connection context manager
    with get_raw_connection() as conn:
        try:
            with conn.cursor() as cur:
                # Prepare data as list of tuples
                data_tuples = [tuple(record[col] for col in columns) for record in records]

                # Execute the bulk insert/upsert
                execute_values(cur, insert_stmt.as_string(cur), data_tuples, template=None, page_size=1000)

            logger.info(f"Successfully {operation}ed {len(records)} records into '{model_name}'.")
            st.toast(f"Successfully {operation}ed {len(records)} records into '{model_name}'.")
        except Exception as e:
            logger.exception("Error during bulk upsert operation.")
            st.toast(f"Error during bulk upsert operation: {str(e)}")
            raise

    # Handle 'sync' operation specifics
    if operation == "sync":
        try:
            with get_session() as session:
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
                        pk_columns = tuple(getattr(model, pk) for pk in primary_keys)
                        filter_condition = and_(
                            getattr(model, "trakt_uuid") == uuid, ~tuple_(pk_columns).in_(pk_values)
                        )

                    # Perform the delete operation
                    deleted = session.query(model).filter(filter_condition).delete(synchronize_session=False)
                    session.commit()
                    logger.info(f"Deleted {deleted} records from '{model_name}' not present in provided data.")
                    st.toast(f"Deleted {deleted} records from '{model_name}' not present in provided data.")
        except Exception as e:
            logger.exception("Error during sync delete operation.")
            st.error(f"Error during sync delete operation: {str(e)}")
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
        st.error(f"Table '{table_name}' not found.")
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
                        st.error(f"One of the join tables '{left_table}' or '{right_table}' not found.")
                        raise ValueError(f"One of the join tables '{left_table}' or '{right_table}' not found.")

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
                            st.error(f"Filter table '{tbl}' not found.")
                            raise ValueError(f"Filter table '{tbl}' not found.")
                        column = getattr(model, col, None)
                    else:
                        column = getattr(main_model, key, None)
                    if not column:
                        logger.error(f"Filter column '{key}' not found.")
                        st.error(f"Filter column '{key}' not found.")
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
                            st.error(f"Unsupported filter operator '{op}'.")
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
                            st.error(f"Order table '{tbl}' not found.")
                            raise ValueError(f"Order table '{tbl}' not found.")
                        column_attr = getattr(model, column, None)
                    else:
                        column_attr = getattr(main_model, col, None)

                    if not column_attr:
                        logger.error(f"Order column '{col}' not found.")
                        st.error(f"Order column '{col}' not found.")
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

            logger.info(f"Read {len(df)} records from '{table_name}'.")
            return df

        except SQLAlchemyError as e:
            logger.exception("Database read operation failed.")
            st.error(f"Database read operation failed: {str(e)}")
            raise
        except Exception as e:
            logger.exception("An unexpected error occurred in read_table_df.")
            st.error(f"An unexpected error occurred: {str(e)}")
            raise


def filter_new_data(df: pd.DataFrame, df_col: str, model_name: str, model_col: str) -> List[Any]:
    """
    Filter new data based on existing records in the database.

    :param df: pandas DataFrame containing the data to filter
    :param df_col: DataFrame column to filter on
    :param model_name: Name of the model/table to check against
    :param model_col: Column in the model to compare with
    :return: List of existing values in the database
    """
    model = model_map.get(model_name)
    if not model:
        st.error(f"Model '{model_name}' not found.")
        return []

    # Convert Int64 columns to object
    int64_mask = df.dtypes == "Int64"
    int64_columns = df.columns[int64_mask]
    if len(int64_columns) > 0:
        df[int64_columns] = df[int64_columns].astype(object)

    df = df.replace({pd.NA: None})

    with get_session() as session:
        try:
            query = select(getattr(model, model_col)).where(getattr(model, model_col).in_(df[df_col].tolist()))
            result = session.execute(query)
            existing_values = result.scalars().all()
            logger.info(f"Filtered {len(existing_values)} existing records from '{model_name}'.")
            return existing_values
        except SQLAlchemyError as e:
            logger.exception(f"An error occurred while filtering data: {str(e)}")
            st.error(f"An error occurred while filtering data: {str(e)}")
            return []


def check_value_exists(table_name: str, column_name: str, value: Any) -> bool:
    """
    Efficiently check if a given value exists in a specified column of a table.

    :param table_name: Name of the table to check
    :param column_name: Name of the column to check
    :param value: Value to search for
    :return: True if the value exists, False otherwise
    """
    model = model_map.get(table_name)
    if not model:
        st.error(f"Table '{table_name}' not found in model map.")
        return False

    column = getattr(model, column_name, None)
    if not column:
        st.error(f"Column '{column_name}' not found in table '{table_name}'.")
        return False

    query = select(exists().where(column == value))

    with get_session() as session:
        try:
            result = session.execute(query)
            exists_flag = bool(result.scalar())
            logger.info(f"Value '{value}' exists in '{table_name}.{column_name}': {exists_flag}")
            return exists_flag
        except SQLAlchemyError as e:
            logger.exception(f"An error occurred while checking value existence: {str(e)}")
            st.error(f"An error occurred while checking value existence: {str(e)}")
            return False


def get_column_value(
    table_name: str,
    filter_column: str,
    filter_value: Any,
    target_column: str,
    operation: str = "first",
) -> Optional[Any]:
    """
    Retrieve a specific value from a column based on a filter.

    :param table_name: Name of the table to query
    :param filter_column: Column to apply the filter on
    :param filter_value: Value to filter by
    :param target_column: Column to retrieve the value from
    :param operation: Operation type - "first" or "max"
    :return: Retrieved value or None
    """
    model = model_map.get(table_name)
    if not model:
        st.error(f"Table '{table_name}' not found in model map.")
        return None

    # Verify that the columns exist in the table
    table_columns = inspect(model).c
    if filter_column not in table_columns or target_column not in table_columns:
        st.error(f"Column not found in table {model.__tablename__}")
        return None

    if operation not in ["first", "max"]:
        st.error("Invalid operation. Use 'first' or 'max'.")
        return None

    with get_session() as session:
        try:
            if operation == "first":
                query = (
                    select(getattr(model, target_column)).where(getattr(model, filter_column) == filter_value).limit(1)
                )
                result = session.execute(query)
                value = result.scalar()
                logger.info(
                    f"Retrieved first value from '{table_name}.{target_column}' where '{filter_column}' = '{filter_value}': {value}"
                )
                return value

            elif operation == "max":
                query = select(func.max(getattr(model, target_column))).where(
                    getattr(model, filter_column) == filter_value
                )
                result = session.execute(query)
                value = result.scalar()
                logger.info(
                    f"Retrieved max value from '{table_name}.{target_column}' where '{filter_column}' = '{filter_value}': {value}"
                )
                return value
        except SQLAlchemyError as e:
            logger.exception(f"An error occurred while retrieving column value: {str(e)}")
            st.error(f"An error occurred while retrieving column value: {str(e)}")
            return None
