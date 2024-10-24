import pandas as pd
import json
import operator
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import psycopg2
import requests as req
import sqlglot
from sqlalchemy import (
    ARRAY,
    create_engine,
    inspect,
    select,
    text,
    MetaData,
    Table,
    Column,
    desc,
    asc,
    or_,
    func,
    exists,
    delete,
    insert as sa_insert,
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

# ---------------------------- Configuration ---------------------------- #

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

# ---------------------------- Context Managers ---------------------------- #


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        st.error(f"Session rollback due to: {str(e)}")
        raise
    finally:
        session.close()


# ---------------------------- Helper Functions ---------------------------- #


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


def get_model_columns(model):
    """Retrieve column names from a SQLAlchemy model."""
    return [column.key for column in model.__table__.columns]


def get_primary_keys(model):
    """Retrieve primary key column names from a SQLAlchemy model."""
    return [key.name for key in model.__table__.primary_key.columns]


def _get_column(column_name: str, joined_tables: Dict[str, Any]) -> Column:
    """Helper function to get the column object from the appropriate table."""
    if "." in column_name:
        table_name, col_name = column_name.split(".")
        table = joined_tables.get(table_name)
        if not table:
            raise ValueError(
                f"Table '{table_name}' not found in joined tables. Available tables: {', '.join(joined_tables.keys())}"
            )
    else:
        # Search in all joined tables
        table = next((t for t in joined_tables.values() if hasattr(t, column_name)), None)
        if not table:
            raise ValueError(f"Column '{column_name}' not found in any joined table")
        col_name = column_name

    column = getattr(table, col_name, None)
    if not column:
        available_columns = [c.key for c in table.__table__.columns]
        raise AttributeError(
            f"'{table.__name__}' object has no attribute '{col_name}'. Available columns: {', '.join(available_columns)}"
        )

    return column


def _apply_filter(query, column, value):
    """Helper function to apply the appropriate filter based on the value type."""
    if isinstance(value, bool):
        return query.filter(column.isnot(None) if value else column.is_(None))
    elif isinstance(value, tuple) and len(value) == 2:
        op, val = value
        filter_op = {
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne,
        }.get(op, operator.eq)
        return query.filter(filter_op(column, val))
    elif isinstance(column.type, ARRAY):
        if isinstance(value, list):
            return query.filter(
                or_(*[func.lower(func.array_to_string(column, ",", "")).contains(item.lower()) for item in value])
            )
        else:
            return query.filter(func.lower(func.array_to_string(column, ",", "")).contains(value.lower()))
    elif isinstance(value, list):
        return query.filter(column.in_(value))
    else:
        return query.filter(func.lower(column) == value.lower() if isinstance(value, str) else column == value)


def optimize_query(query: Union[str, Select, Any], params: dict = None):
    """Optimize and execute a given SQL query."""
    if isinstance(query, (sa_insert, pg_insert)):
        compiled = query.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})
    elif isinstance(query, Select):
        compiled = query.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})
    elif isinstance(query, str):
        compiled = query
    else:
        raise TypeError("Unsupported query type")

    sql = str(compiled)

    # Log the original SQL for debugging
    print(f"Original SQL: {sql}")

    try:
        # Parse the SQL using sqlglot
        parsed_sql = sqlglot.parse_one(sql)

        # Optimize the parsed SQL
        optimized_sql = parsed_sql.transform()  # Implement actual optimization logic as needed
        final_sql = optimized_sql.sql()
    except Exception as e:
        print(f"Optimization error: {e}")
        # Fallback to the original SQL if optimization fails
        final_sql = sql

    # Log the optimized SQL for debugging
    print(f"Optimized SQL: {final_sql}")

    # Execute the optimized query synchronously
    with engine.connect() as connection:
        result = connection.execute(text(final_sql), params or {})
        return result


def close_all_connections():
    """Dispose all connections in the engine pool."""
    engine.dispose()


# ---------------------------- Core Functions ---------------------------- #


def create_schema(db_url: str = DATABASE_URL, recreate: bool = False) -> None:
    """Create or recreate the database schema."""
    try:
        if recreate:
            # Drop all existing tables
            Base.metadata.drop_all(bind=engine)
            st.write("Dropped all existing tables.")

        # Create all tables
        Base.metadata.create_all(bind=engine)
        st.write("Successfully created the schema.")

        # Inspect and list all tables and their columns
        inspector = inspect(engine)
        st.write("Current tables and columns:")
        for table_name in inspector.get_table_names():
            st.write(f"Table: {table_name}")
            columns = inspector.get_columns(table_name)
            for column in columns:
                st.write(f"  - {column['name']} ({column['type']})")
    except SQLAlchemyError as e:
        st.error(f"An error occurred while creating the schema: {str(e)}")


import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
import psycopg2
from io import StringIO
import logging
from typing import Any, Dict
import datetime
from psycopg2.extras import execute_values


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 1000  # Adjust based on performance testing
COPY_BUFFER_SIZE = 100000  # Number of rows to buffer before COPY
MAX_RETRIES = 3  # Number of retries for transient failures

import numpy as np


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = engine.raw_connection()
    try:
        yield conn
    finally:
        conn.close()


@timeit
def prepare_dataframe(df: pd.DataFrame, model_name: str, uuid: str) -> pd.DataFrame:
    """Prepare DataFrame for insertion."""
    if model_name in ["user", "user_watch_history"]:
        df["trakt_uuid"] = uuid

    df = df.replace({pd.NA: None})

    # Handle array columns
    array_columns = df.columns[df.applymap(lambda x: isinstance(x, (list, np.ndarray))).any()]
    for col in array_columns:
        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, np.ndarray)) else x)

    return df


@timeit
def add_data(
    df: pd.DataFrame,
    uuid: str,
    model_name: str,
    operation: str = "insert",
) -> None:
    """
    Optimized Add Data Function using COPY and Bulk Operations.
    """
    try:
        # Prepare DataFrame
        df_prepared = prepare_dataframe(df, model_name, uuid)

        if df_prepared.empty:
            logger.info("No records to insert.")
            return

        if operation in ["insert", "upsert"]:
            insert_operation(df_prepared, model_name, operation)
        elif operation == "sync":
            sync_operation(df_prepared, model_name, uuid)
        else:
            raise ValueError(f"Invalid operation: {operation}")

        logger.info(f"{operation.capitalize()} completed for {len(df_prepared)} records in '{model_name}'.")
    except Exception as e:
        logger.exception("An error occurred in add_data function.")
        raise


@timeit
def insert_operation(df: pd.DataFrame, model_name: str, operation: str) -> None:
    """
    Handles 'insert' and 'upsert' operations using COPY for bulk insertion.
    """
    model = model_map.get(model_name)
    if not model:
        raise ValueError(f"Model '{model_name}' not found.")

    table_name = model.__tablename__
    primary_keys = get_primary_keys(model)
    insert_cols = [col for col in df.columns if col in get_model_columns(model)]

    # Create CSV buffer
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t", na_rep="\\N")
    buffer.seek(0)

    # Establish psycopg2 connection for COPY
    conn = engine.raw_connection()
    try:
        with conn.cursor() as cursor:
            if operation == "insert":
                # Escape table and column names
                escaped_table = f'"{table_name}"'
                escaped_cols = ", ".join([f'"{col}"' for col in insert_cols])
                copy_sql = (
                    f"COPY {escaped_table} ({escaped_cols}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\t', NULL '\\N')"
                )
                cursor.copy_expert(copy_sql, buffer)
            elif operation == "upsert":
                # Use temporary table for upsert with escaped table names
                temp_table = f"{table_name}_temp"
                escaped_temp_table = f'"{temp_table}"'
                escaped_main_table = f'"{table_name}"'
                escaped_cols = ", ".join([f'"{col}"' for col in insert_cols])

                # Create temporary table with escaped names
                create_temp_sql = f"""
              CREATE TEMP TABLE {escaped_temp_table} (LIKE {escaped_main_table} INCLUDING ALL);
              """
                cursor.execute(create_temp_sql)

                # COPY data into temporary table
                copy_temp_sql = f"COPY {escaped_temp_table} ({escaped_cols}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\t', NULL '\\N')"
                cursor.copy_expert(copy_temp_sql, buffer)

                # Upsert from temporary table with escaped table and column names
                update_cols = [col for col in insert_cols if col not in primary_keys]
                update_assignments = ", ".join([f'"{col}"=EXCLUDED."{col}"' for col in update_cols])

                upsert_sql = f"""
              INSERT INTO {escaped_main_table} ({escaped_cols})
              SELECT {escaped_cols} FROM {escaped_temp_table}
              ON CONFLICT ({', '.join([f'"{pk}"' for pk in primary_keys])}) DO UPDATE
              SET {update_assignments};
              """
                cursor.execute(upsert_sql)

                # Drop temporary table
                cursor.execute(f"DROP TABLE {escaped_temp_table};")
            else:
                raise ValueError(f"Unsupported operation: {operation}")

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("Error during insert_operation.")
        raise
    finally:
        conn.close()


@timeit
def sync_operation(df: pd.DataFrame, model_name: str, uuid: str) -> None:
    """
    Synchronize the table to match the provided DataFrame:
    - Insert new records
    - Update existing records
    - Delete records not present in the DataFrame
    """
    model = model_map.get(model_name)
    if not model:
        raise ValueError(f"Model '{model_name}' not found.")

    primary_keys = get_primary_keys(model)
    insert_cols = [col for col in df.columns if col in get_model_columns(model)]

    # Create CSV buffer for upsert
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t", na_rep="\\N")
    buffer.seek(0)

    # Establish psycopg2 connection for COPY
    conn = engine.raw_connection()
    try:
        with conn.cursor() as cursor:
            # Create temporary table with escaped names
            temp_table = f"{model.__tablename__}_sync_temp"
            escaped_temp_table = f'"{temp_table}"'
            escaped_main_table = f'"{model.__tablename__}"'
            escaped_cols = ", ".join([f'"{col}"' for col in insert_cols])

            create_temp_sql = f"""
            CREATE TEMP TABLE {escaped_temp_table} (LIKE {escaped_main_table} INCLUDING ALL);
            """
            cursor.execute(create_temp_sql)

            # COPY data into temporary table
            copy_temp_sql = (
                f"COPY {escaped_temp_table} ({escaped_cols}) FROM STDIN WITH (FORMAT csv, DELIMITER '\t', NULL '\\N')"
            )
            cursor.copy_expert(copy_temp_sql, buffer)

            # Upsert from temporary table with escaped table and column names
            update_cols = [col for col in insert_cols if col not in primary_keys]
            update_assignments = ", ".join([f'"{col}"=EXCLUDED."{col}"' for col in update_cols])

            upsert_sql = f"""
            INSERT INTO {escaped_main_table} ({escaped_cols})
            SELECT {escaped_cols} FROM {escaped_temp_table}
            ON CONFLICT ({', '.join([f'"{pk}"' for pk in primary_keys])}) DO UPDATE
            SET {update_assignments};
            """
            cursor.execute(upsert_sql)

            # Delete records not present in the temporary table with escaped names
            delete_condition = " AND ".join([f'main."{pk}" = temp."{pk}"' for pk in primary_keys])
            delete_sql = f"""
            DELETE FROM {escaped_main_table} AS main
            USING {escaped_temp_table} AS temp
            WHERE main."trakt_uuid" = %s
              AND NOT EXISTS (
                SELECT 1 FROM {escaped_temp_table} AS temp_inner
                WHERE {' AND '.join([f'temp_inner."{pk}" = main."{pk}"' for pk in primary_keys])}
              );
            """
            cursor.execute(delete_sql, (uuid,))

            # Drop temporary table
            cursor.execute(f"DROP TABLE {escaped_temp_table};")

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("Error during sync_operation.")
        raise
    finally:
        conn.close()


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

    with session_scope() as session:
        try:
            query = select(getattr(model, model_col)).where(getattr(model, model_col).in_(df[df_col].tolist()))
            result = session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            st.error(f"An error occurred while filtering data: {str(e)}")
            return []


def join_tables(t1: str, t1_col: str, t2: str, t2_col: str) -> pd.DataFrame:
    """
    Join two tables based on specified columns.

    :param t1: Name of the first table
    :param t1_col: Column in the first table to join on
    :param t2: Name of the second table
    :param t2_col: Column in the second table to join on
    :return: Joined DataFrame
    """
    model1 = model_map.get(t1)
    model2 = model_map.get(t2)

    if not model1 or not model2:
        st.error(f"One of the models '{t1}' or '{t2}' does not exist.")
        return pd.DataFrame()

    stmt = select(model1, model2).join(model2, getattr(model1, t1_col) == getattr(model2, t2_col))

    try:
        with session_scope() as session:
            result = session.execute(stmt)
            records = result.fetchall()
            df = pd.DataFrame(records, columns=result.keys())
            # Remove duplicate columns (those ending with _1)
            df = df.loc[:, ~df.columns.str.endswith("_1")]
            return df
    except SQLAlchemyError as e:
        st.error(f"An error occurred during join operation: {str(e)}")
        return pd.DataFrame()


from sqlalchemy import select, asc, desc
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union


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
    Optimized Read Table Function
    """
    main_table = model_map.get(table_name)
    if not main_table:
        st.error(f"Table '{table_name}' not found in model map")
        return pd.DataFrame()

    # Select specific columns if ex_columns is provided
    query = select(main_table)
    joined_tables = {table_name: main_table}

    # Apply joins
    if joins:
        for left_table, right_table, left_col, right_col in joins:
            left_model = joined_tables.get(left_table)
            right_model = model_map.get(right_table)
            if not left_model:
                st.error(f"Left table '{left_table}' not found in joined tables")
                return pd.DataFrame()
            if not right_model:
                st.error(f"Right table '{right_table}' not found in model map")
                return pd.DataFrame()
            join_condition = getattr(left_model, left_col) == getattr(right_model, right_col)
            query = query.outerjoin(right_model, join_condition)
            joined_tables[right_table] = right_model

    # Select only necessary columns
    all_columns = []
    seen_columns = set()
    ex_columns = ex_columns or []
    for table in joined_tables.values():
        for c in table.__table__.columns:
            if c.name not in ex_columns and c.name not in seen_columns:
                all_columns.append(c)
                seen_columns.add(c.name)
    if all_columns:
        query = query.with_only_columns(*all_columns)

    # Apply filters
    if filters:
        for key, value in filters.items():
            try:
                column = _get_column(key, joined_tables)
                query = _apply_filter(query, column, value)
            except (ValueError, AttributeError) as e:
                st.error(str(e))
                return pd.DataFrame()

    # Apply ordering
    if order_by:
        order_columns = [order_by] if isinstance(order_by, str) else order_by
        for col in order_columns:
            try:
                column = _get_column(col, joined_tables)
                query = query.order_by(desc(column) if order_desc else asc(column))
            except (ValueError, AttributeError) as e:
                st.error(str(e))
                return pd.DataFrame()

    # Apply limit
    if limit is not None:
        query = query.limit(limit)

    # Execute the query
    with session_scope() as session:
        try:
            result = session.execute(query)
            records = result.fetchall()
            df = pd.DataFrame(records, columns=result.keys())
            return df
        except SQLAlchemyError as e:
            st.error(f"An error occurred while reading the table: {str(e)}")
            return pd.DataFrame()


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
        st.error(f"Table '{table_name}' not found in model map")
        return False

    column = getattr(model, column_name, None)
    if not column:
        st.error(f"Column '{column_name}' not found in table '{table_name}'")
        return False

    query = select(exists().where(column == value))

    with session_scope() as session:
        try:
            result = session.execute(query)
            return bool(result.scalar())
        except SQLAlchemyError as e:
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
        st.error(f"Table '{table_name}' not found in model map")
        return None

    # Verify that the columns exist in the table
    table_columns = inspect(model).c
    if filter_column not in table_columns or target_column not in table_columns:
        st.error(f"Column not found in table {model.__tablename__}")
        return None

    if operation not in ["first", "max"]:
        st.error("Invalid operation. Use 'first' or 'max'.")
        return None

    with session_scope() as session:
        try:
            if operation == "first":
                query = (
                    select(getattr(model, target_column)).where(getattr(model, filter_column) == filter_value).limit(1)
                )
                result = session.execute(query)
                value = result.scalar()
                return value

            elif operation == "max":
                query = select(func.max(getattr(model, target_column))).where(
                    getattr(model, filter_column) == filter_value
                )
                result = session.execute(query)
                value = result.scalar()
                return value
        except SQLAlchemyError as e:
            st.error(f"An error occurred while retrieving column value: {str(e)}")
            return None


# ---------------------------- Utility Functions ---------------------------- #


def test_postgres_connection():
    """
    Test the connection to the PostgreSQL database.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            st.write("Connection successful")
    except SQLAlchemyError as e:
        st.error(f"Connection failed: {e}")


def close_all_connections_sync():
    """Close all connections synchronously."""
    close_all_connections()
