import pandas as pd

from sqlalchemy import (
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
    URL,
)
from sqlalchemy.orm import sessionmaker, scoped_session, aliased, Query, joinedload
from sqlalchemy.dialects.postgresql import insert, ARRAY
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.expression import BinaryExpression
from sqlalchemy.sql import Select

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, aliased, joinedload
from sqlalchemy.future import select

import sqlglot
import operator
from contextlib import contextmanager
from db.models import Base, Users, UserWatchHistory, TraktMedia, IMDBMedia, TMDBMedia

import streamlit as st

# from streamlit import session_state as ss
from datetime import datetime, timedelta
from functools import wraps
import time
import psycopg2


from typing import List, Any, Optional, Union, Tuple, Dict

model_map = {
    "user": User,
    "user_watch_history": UserWatchHistory,
    "trakt_media": TraktMedia,
    "imdb_media": IMDBMedia,
    "tmdb_media": TMDBMedia,
}

DB_USER = "postgres"
DB_PASS = "sohyunmina89"
DB_PORT = "5432"
DB_NAME = "ws_media_db"

# Local Postgres
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@localhost:{DB_PORT}/{DB_NAME}"


# Xata connection details
# DATABASE_NAME = "media"
# BRANCH = "main"
# WORKSPACE_ID = "bpmg26"
# API_KEY = st.secrets.db.xata_api_key
# REGION = "eu-west-1"

# Construct the connection string

# Xata connection string
# DATABASE_URL = (
#     f"postgresql://{WORKSPACE_ID}:{API_KEY}@{REGION}.sql.xata.sh:5432/{DATABASE_NAME}:{BRANCH}?sslmode=require"
# )


# postgresql://bpmg26:xau_VzvO6GQkucnGGrQoGyk4TIiF50V8WuCG1@eu-west-1.sql.xata.sh:5432/media:main?sslmode=require

# # Neon connection string
# DATABASE_URL = "postgresql://media_owner:dUljTRZ0i9ON@ep-cold-night-a2p2srjx-pooler.eu-central-1.aws.neon.tech/media?sslmode=require"

# # Cockroachdb
# DATABASE_URL = "cockroachdb://rohit:M9p_JOuyouSpZlvHFZbvXQ@brave-hyena-10712.7tc.aws-eu-central-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full"

# # Cockroachdb GCP
# DATABASE_URL = "cockroachdb://rohit:yI-WIvlllEtjULIXZFC3wA@second-efreet-1378.jxf.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full"


# # Create engine
# engine = create_engine(
#     DATABASE_URL,
#     poolclass=QueuePool,
#     pool_size=5,  # Reduced from 4 to leave room for other potential connections
#     max_overflow=0,  # No overflow connections
#     pool_recycle=1800,
#     pool_timeout=30,
#     pool_pre_ping=True,  # Enable pre-ping to detect stale connections
#     # connect_args={"connect_timeout": 10, "options": "-c statement_timeout=30000"},
# )


# # Create session
# Session = sessionmaker(bind=engine)

# Create synchronous engine
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=0,
    pool_recycle=1800,
    pool_timeout=30,
    pool_pre_ping=True,
    echo=False,  # Set to True for SQL query logging
)

# Create synchronous session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

from contextlib import contextmanager


@contextmanager
def session_scope():
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


def test_postgres_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Connection successful")
    except SQLAlchemyError as e:
        print(f"Connection failed: {e}")


# ---- Helper Functions ----


def optimize_query(query: Union[str, Select, Query], params: dict = None):
    if isinstance(query, Query):
        stmt = query.statement
        compiled = stmt.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})
    elif isinstance(query, Select):
        compiled = query.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})
    else:
        with engine.connect() as connection:
            result = connection.execute(text(query), params)
        return result

    sql = str(compiled)

    # Log the original SQL for debugging
    print(f"Original SQL: {sql}")

    try:
        # Parse the SQL using sqlglot
        parsed_sql = sqlglot.parse_one(sql)

        # Optimize the parsed SQL
        optimized_sql = parsed_sql.transform()  # Use transform or any other optimization function
        final_sql = optimized_sql.sql()
    except Exception as e:
        print(f"Optimization error: {e}")
        # Fallback to the original SQL if optimization fails
        final_sql = sql

    # Log the optimized SQL for debugging
    print(f"Optimized SQL: {final_sql}")

    # Execute the optimized query synchronously
    with engine.connect() as connection:
        result = connection.execute(text(final_sql))

    return result


# def timeit(func):
#     @wraps(func)
#     def timeit_wrapper(*args, **kwargs):
#         start_time = time.perf_counter()
#         result = func(*args, **kwargs)
#         end_time = time.perf_counter()
#         total_time = end_time - start_time
#         st.write(f"Function {func.__name__} took {total_time:.2f} seconds")
#         return result

#     return timeit_wrapper

import asyncio
from functools import wraps


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        st.write(f"Function {func.__name__} took {total_time:.2f} seconds")
        return result

    return timeit_wrapper


def get_model_columns(model):
    return [column.key for column in model.__table__.columns]


def get_primary_keys(model):
    inspector = inspect(model)
    return [column.name for column in inspector.primary_key]


def _get_column(column_name: str, joined_tables: Dict[str, Any]) -> Column:
    """Helper function to get the column object from the appropriate table."""
    if "." in column_name:
        table_name, col_name = column_name.split(".")
        table = joined_tables.get(table_name)
        if not table:
            raise ValueError(
                f"Table '{table_name}' not found in joined tables. Available tables are: {', '.join(joined_tables.keys())}"
            )
    else:
        table = next((t for t in joined_tables.values() if hasattr(t, column_name)), None)
        if not table:
            raise ValueError(f"Column '{column_name}' not found in any joined table")
        col_name = column_name

    column = getattr(table, col_name, None)
    if not column:
        available_columns = [c.key for c in table.__table__.columns]
        raise AttributeError(
            f"'{table.__name__}' object has no attribute '{col_name}'. Available columns are: {', '.join(available_columns)}"
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


def close_all_connections():
    engine.dispose()


# ----


# def create_tables():
#     Base.metadata.create_all(engine)
#     print("Tables created successfully")


# def add_user(id, name):
#     with Session.begin() as session:
#         user = User(id=id, name=name)
#         session.add(user)
#         session.commit()
#     print(f"User {name} added successfully")


# def read_users():
#     with Session.begin() as session:
#         users = session.query(User).all()
#         for user in users:
#             print(user.trakt_user_id)


# def get_server_version():
#     with engine.connect() as conn:
#         version = conn.scalar(text("SELECT version()"))
#         print(f"Postgres version is {version}")


# def test_operations():
#     with Session.begin() as session:
#         dummy_user = User(
#             trakt_user_id="dummy_user_1223",
#             auth_token="dummy_auth_token",
#             refresh_token="dummy_refresh_token",
#             expires_at=datetime.utcnow() + timedelta(days=30),
#             last_db_update=datetime.utcnow(),
#         )
#         session.add(dummy_user)

#         dummy_media = MediaData(
#             trakt_url="https://trakt.tv/shows/dummy-show/seasons/2/episodes/1",
#             title="Dummy Show",
#             ep_title="Pilot",
#             media_type="episode",
#             season_num=1,
#             ep_num=1,
#             runtime=45,  # in minutes
#         )
#         session.add(dummy_media)

#         dummy_watch_history = UserWatchHistory(
#             trakt_user_id=dummy_user.trakt_user_id,
#             trakt_url=dummy_media.trakt_url,
#             watched_at=datetime.utcnow() - timedelta(days=1),
#         )
#         session.add(dummy_watch_history)

#         try:
#             session.commit()
#             print("Test operations completed successfully.")
#         except SQLAlchemyError as e:
#             session.rollback()
#             print(f"An error occurred during test operations: {str(e)}")


# def list_tables():
#     with engine.connect() as conn:
#         metadata = MetaData()
#         metadata.reflect(bind=engine)
#         tables = list(metadata.tables.keys())
#         print("Existing tables in the database:")
#         for table in tables:
#             print(f"- {table}")


# ---- Prod Functions ----


def create_schema(db_url: str = DATABASE_URL, recreate: bool = False) -> None:
    engine = create_engine(
        db_url,
    )
    inspector = inspect(engine)

    try:
        if recreate:
            # Drop all existing tables
            Base.metadata.drop_all(bind=engine)
            print("Dropped all existing tables.")

        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("Successfully created the schema.")

        # List all tables and their columns
        print("Current tables:")
        for table_name in inspector.get_table_names():
            print(f"- {table_name}")
            columns = inspector.get_columns(table_name)
            for column in columns:
                print(f"  - {column['name']} ({column['type']})")

    except SQLAlchemyError as e:
        print(f"An error occurred while creating the schema: {str(e)}")
    finally:
        engine.dispose()


@timeit
def add_data(df, uuid, model_name, operation="insert"):
    model = model_map[model_name]

    primary_keys = get_primary_keys(model)

    if model_name in ["user", "user_watch_history"]:
        df["trakt_uuid"] = uuid

    df = df.replace({pd.NA: None})
    df = df.drop_duplicates(subset=primary_keys)

    insert_cols = [col for col in df.columns if col in get_model_columns(model)]

    df = df[insert_cols]

    # Convert Int64 columns to object
    int64_mask = df.dtypes == "Int64"
    int64_columns = df.columns[int64_mask]
    if len(int64_columns) > 0:
        df[int64_columns] = df[int64_columns].astype(object)

    records = df.to_dict("records")

    with session_scope() as session:
        try:
            if operation == "insert":
                stmt = insert(model).values(records).on_conflict_do_nothing()
                result = session.execute(stmt)
                session.commit()
                st.write(f"Insert done for {result.rowcount} out of {len(records)} items into {model.__tablename__}.")

            elif operation == "upsert":
                stmt = insert(model).values(records)
                update_dict = {col: stmt.excluded[col] for col in insert_cols if col not in primary_keys}
                stmt = stmt.on_conflict_do_update(index_elements=primary_keys, set_=update_dict)
                result = session.execute(stmt)
                session.commit()
                st.write(f"Upsert done for {result.rowcount} out of {len(records)} items into {model.__tablename__}.")

            elif operation == "sync":
                # For UserWatchHistory, use trakt_uuid to sync
                assert model_name == "user_watch_history"

                # Delete records not in the new data
                event_ids = [record["event_id"] for record in records]
                delete_stmt = delete(model).where(model.trakt_uuid == uuid, ~model.event_id.in_(event_ids))
                delete_result = session.execute(delete_stmt)

                # Upsert new data
                stmt = insert(model).values(records)
                update_dict = {col: stmt.excluded[col] for col in insert_cols if col not in primary_keys}
                stmt = stmt.on_conflict_do_update(index_elements=primary_keys, set_=update_dict)
                upsert_result = session.execute(stmt)

                session.commit()
                st.write(
                    f"Sync done. {delete_result.rowcount} items deleted and {upsert_result.rowcount} items upserted in {model.__tablename__}."
                )

            else:
                st.error(f"Invalid operation: {operation}")

        except SQLAlchemyError as e:
            session.rollback()
            st.error(f"An error occurred while processing watch history data: {str(e)}")
        except Exception as e:
            session.rollback()
            st.error(f"An unexpected error occurred: {str(e)}")


def filter_new_data(df, df_col, model_name, model_col):
    model = model_map[model_name]

    # Convert Int64 columns to object
    int64_mask = df.dtypes == "Int64"
    int64_columns = df.columns[int64_mask]
    if len(int64_columns) > 0:
        df[int64_columns] = df[int64_columns].astype(object)

    df = df.replace({pd.NA: None})

    with session_scope() as session:
        result = session.execute(
            select(getattr(model, model_col)).where(getattr(model, model_col).in_(df[df_col].tolist()))
        )
        return result.scalars().all()


def join_tables(t1, t1_col, t2, t2_col):
    model1 = model_map[t1]
    model2 = model_map[t2]

    # Create a select statement
    stmt = select(model1, model2).join(model2, getattr(model1, t1_col) == getattr(model2, t2_col))

    # Compile the statement into a SQL string
    sql = stmt.compile(compile_kwargs={"literal_binds": True})

    with session_scope() as session:
        # Execute the compiled SQL synchronously
        result = session.execute(text(sql))
        records = result.fetchall()

    # Convert to DataFrame
    df = pd.DataFrame(records, columns=result.keys())

    # Remove duplicate columns (those ending with _1)
    df = df.loc[:, ~df.columns.str.endswith("_1")]

    return df


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
    main_table = model_map.get(table_name)
    if not main_table:
        raise ValueError(f"Table '{table_name}' not found in model map")

    query = select(main_table)
    joined_tables = {table_name: main_table}

    # Apply joins
    if joins:
        for left_table, right_table, left_col, right_col in joins:
            left_model = joined_tables.get(left_table)
            right_model = model_map.get(right_table)
            if not left_model:
                raise ValueError(f"Left table '{left_table}' not found in joined tables")
            if not right_model:
                raise ValueError(f"Right table '{right_table}' not found in model map")
            join_condition = getattr(left_model, left_col) == getattr(right_model, right_col)
            query = query.outerjoin(right_model, join_condition)
            joined_tables[right_table] = right_model

    # Select all columns except excluded ones and duplicates
    all_columns = []
    seen_columns = set()
    ex_columns = ex_columns or []  # Use an empty list if ex_columns is None
    for table in joined_tables.values():
        for c in table.__table__.columns:
            if c.name not in ex_columns and c.name not in seen_columns:
                all_columns.append(c)
                seen_columns.add(c.name)
    query = query.with_only_columns(*all_columns)

    # Apply filters
    if filters:
        for key, value in filters.items():
            column = _get_column(key, joined_tables)
            query = _apply_filter(query, column, value)

    # Apply ordering
    if order_by:
        order_columns = [order_by] if isinstance(order_by, str) else order_by
        for col in order_columns:
            column = _get_column(col, joined_tables)
            query = query.order_by(desc(column) if order_desc else asc(column))

    # Apply limit
    if limit is not None:
        query = query.limit(limit)

    # Execute the query synchronously
    with session_scope() as session:
        result = optimize_query(query)
        records = result.fetchall()

    # Convert the result to a DataFrame
    df = pd.DataFrame(records, columns=result.keys())

    return df


def check_value_exists(table_name: str, column_name: str, value: Any) -> bool:
    """
    Efficiently check if a given value exists in a specified column of a table.

    :param table_name: Name of the table to check
    :param column_name: Name of the column to check
    :param value: Value to search for
    :return: True if the value exists, False otherwise
    """
    if table_name not in model_map:
        raise ValueError(f"Table '{table_name}' not found in model map")

    table = model_map[table_name]
    column = getattr(table, column_name, None)

    if not column:
        raise ValueError(f"Column '{column_name}' not found in table '{table_name}'")

    query = select(exists().where(column == value))

    with session_scope() as session:
        result = optimize_query(query)
        return bool(result.scalar())

    # return bool(result)


def get_column_value(
    table_name: str,
    filter_column: str,
    filter_value: Any,
    target_column: str,
    operation: str = "first",
) -> Optional[Any]:
    if table_name not in model_map:
        raise ValueError(f"Table {table_name} not found in model map")

    table = model_map[table_name]

    # Verify that the columns exist in the table
    table_columns = inspect(table).c
    if filter_column not in table_columns or target_column not in table_columns:
        raise ValueError(f"Column not found in table {table.__tablename__}")

    if operation not in ["first", "max"]:
        raise ValueError("Invalid operation. Use 'first' or 'max'.")

    with session_scope() as session:
        if operation == "first":
            query = select(getattr(table, target_column)).where(getattr(table, filter_column) == filter_value).limit(1)
            result = session.execute(query)
            value = result.scalar()
            return value

        elif operation == "max":
            query = select(func.max(getattr(table, target_column))).where(getattr(table, filter_column) == filter_value)
            result = session.execute(query)
            value = result.scalar()
            return value


# def upsert_user_watch_history(df, uuid):
#     """
#     Upsert watch history data from a pandas DataFrame to the UserWatchHistory table.

#     :param df: pandas DataFrame containing watch history data
#     :param engine: SQLAlchemy engine instance
#     """

#     df["trakt_uuid"] = uuid
#     records = df[get_model_columns(UserWatchHistory)].to_dict("records")

#     records = records[:10]

#     with Session.begin() as session:
#         try:
#             stmt = insert(UserWatchHistory).values(records)

#             stmt = stmt.on_conflict_do_update(
#                 index_elements=["event_id"],
#                 set_={
#                     "trakt_uuid": stmt.excluded.trakt_uuid,
#                     "trakt_url": stmt.excluded.trakt_url,
#                     "watched_at": stmt.excluded.watched_at,
#                 },
#             )

#             session.execute(stmt)

#             st.write(f"Upsert for {len(records)} watch history records complete.")

#         except SQLAlchemyError as e:
#             session.rollback()
#             st.error(f"An error occurred while upserting watch history data: {str(e)}")


def read_user():
    with Session.begin() as session:
        return session.query(User).all()


# def main():
#     print("Starting database operations.")
#     # get_server_version()
#     # create_tables()
#     # create_schema()
#     # list_tables()
#     # test_operations()
#     # add_user(111, "John")
#     # read_users()


# if __name__ == "__main__":
#     main()
