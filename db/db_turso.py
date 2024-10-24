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
)
from sqlalchemy import insert, select, delete
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.expression import BinaryExpression
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import Insert
import json
from functools import partial
import io
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from sqlalchemy.pool import QueuePool


import operator
from contextlib import contextmanager
from db.models_turso import Base, User, UserWatchHistory, TraktMedia, IMDBMedia, TMDBMedia
import streamlit as st
from datetime import datetime, timedelta
from functools import wraps
import time
from typing import List, Any, Optional, Union, Tuple, Dict

model_map = {
    "user": User,
    "user_watch_history": UserWatchHistory,
    "trakt_media": TraktMedia,
    "imdb_media": IMDBMedia,
    "tmdb_media": TMDBMedia,
}
col_dict = {
    # db_columns
    "trakt_uuid": "string",
    "trakt_user_id": "string",
    "trakt_auth_token": "string",
    "last_db_update": "datetime64[ns]",
    #
    "title": "string",
    "ep_title": "string",
    "trakt_url": "string",
    "media_type": "string",
    "season_num": "Int64",
    "ep_num": "Int64",
    "ep_num_abs": "Int64",
    "total_episodes": "Int64",
    "status": "string",
    # "tmdb_status": "string",
    # "ep_plays": "Int64",
    "runtime": "Int64",
    # "plays": "Int64",
    # "watchtime": "Int64",
    "watched_at": "datetime64[ns]",
    # "tmdb_release_date": "datetime64[ns]",
    "released": "datetime64[ns]",
    "tmdb_last_air_date": "datetime64[ns]",
    "genres": "object",
    "imdb_genres": "object",  # Choose one
    "tmdb_genres": "object",
    "country": "string",
    # "imdb_country": "object",
    "tmdb_language": "string",
    "tmdb_certification": "string",
    "tmdb_networks": "object",
    "tmdb_collection": "string",
    "tmdb_keywords": "object",
    # "last_updated_at": "datetime64[ns]",
    "overview": "string",
    "ep_overview": "string",
    "show_trakt_id": "Int64",
    "show_imdb_id": "string",
    "show_tmdb_id": "Int64",
    "event_id": "Int64",
}

# Turso connection details
dbUrl = f"sqlite+{st.secrets.db.url}/?authToken={st.secrets.db.auth_token}&secure=true"

# Create engine
engine = create_engine(
    dbUrl,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=2,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"check_same_thread": False},
)


# Create session
Session = sessionmaker(bind=engine)


# @contextmanager
# def session_scope():
#     """Provide a transactional scope around a series of operations."""
#     session = Session()
#     try:
#         yield session
#         session.commit()
#     except:
#         session.rollback()
#         raise
#     finally:
#         session.close()


# ---- Helper Functions ----


def stdf_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return st.text(buffer.getvalue())


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
    elif isinstance(value, list):
        return query.filter(column.in_(value))
    else:
        return query.filter(func.lower(column) == value.lower() if isinstance(value, str) else column == value)


def close_all_connections():
    engine.dispose()


# @compiles(Insert, "sqlite")
# def sqlite_upsert(insert_stmt, compiler, **kw):
#     pk = insert_stmt.table.primary_key
#     pk_names = set(c.name for c in pk)
#     insert = compiler.visit_insert(insert_stmt, **kw)
#     ondup = f"ON CONFLICT ({','.join(pk_names)}) DO UPDATE SET "
#     ondup += ", ".join(f"{c.name}=excluded.{c.name}" for c in insert_stmt.table.columns if c.name not in pk_names)
#     return f"{insert} {ondup}"


# ----


def create_tables():
    Base.metadata.create_all(engine)
    st.write("Tables created successfully")


def add_user(id, name):
    with Session.begin() as session:
        user = User(id=id, name=name)
        session.add(user)
        session.commit()
    st.write(f"User {name} added successfully")


def read_users():
    with Session.begin() as session:
        users = session.query(User).all()
        for user in users:
            st.write(user.trakt_user_id)


def get_server_version():
    with engine.connect() as conn:
        version = conn.scalar(text("SELECT version()"))
        st.write(f"Postgres version is {version}")


def test_operations():
    with Session.begin() as session:
        dummy_user = User(
            trakt_user_id="dummy_user_1223",
            auth_token="dummy_auth_token",
            refresh_token="dummy_refresh_token",
            expires_at=datetime.utcnow() + timedelta(days=30),
            last_db_update=datetime.utcnow(),
        )
        session.add(dummy_user)

        dummy_media = MediaData(
            trakt_url="https://trakt.tv/shows/dummy-show/seasons/2/episodes/1",
            title="Dummy Show",
            ep_title="Pilot",
            media_type="episode",
            season_num=1,
            ep_num=1,
            runtime=45,  # in minutes
        )
        session.add(dummy_media)

        dummy_watch_history = UserWatchHistory(
            trakt_user_id=dummy_user.trakt_user_id,
            trakt_url=dummy_media.trakt_url,
            watched_at=datetime.utcnow() - timedelta(days=1),
        )
        session.add(dummy_watch_history)

        try:
            session.commit()
            st.write("Test operations completed successfully.")
        except SQLAlchemyError as e:
            session.rollback()
            st.write(f"An error occurred during test operations: {str(e)}")


# def list_tables():
#     with engine.connect() as conn:
#         metadata = MetaData()
#         metadata.reflect(bind=engine)
#         tables = list(metadata.tables.keys())
#         st.write("Existing tables in the database:")
#         for table in tables:
#             st.write(f"- {table}")


# ---- Prod Functions ----


@timeit
def create_schema(db_url=dbUrl, recreate=False):
    """
    Create the database schema based on the models defined in Base.

    :param db_url: SQLAlchemy database URL
    :param recreate: If True, drop all existing tables before creating new ones
    """
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    inspector = inspect(engine)

    try:
        if recreate:
            # Drop all existing tables
            Base.metadata.drop_all(bind=engine)
            st.write("Dropped all existing tables.")

        # Create all tables
        Base.metadata.create_all(engine)
        st.write("Successfully created the schema.")

        # List all tables and their columns
        st.write("Current tables:")
        for table_name in inspector.get_table_names():
            st.write(f"- {table_name}")
            for column in inspector.get_columns(table_name):
                st.write(f"  - {column['name']} ({column['type']})")

    except SQLAlchemyError as e:
        st.write(f"An error occurred while creating the schema: {str(e)}")
    finally:
        engine.dispose()


@timeit
def prepare_for_sqlite(df, model, primary_keys, dtype_dict=col_dict):

    def convert_to_json(val):
        if isinstance(val, (list, dict)):
            return json.dumps(val)
        elif pd.isna(val):
            return None
        return val

    def convert_datetime(val):
        if pd.isna(val):
            return None
        elif isinstance(val, str):
            return datetime.fromisoformat(val.rstrip("Z"))
        elif isinstance(val, (datetime, pd.Timestamp)):
            return val.to_pydatetime()
        return val

    # def convert_datetime(x):
    #     return x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else None

    # def convert_list(x):
    #     return json.dumps(x) if isinstance(x, list) else x

    stdf_info(df)

    # for col in df.columns:
    #     if df[col].dtype != "string":
    #         df[col] = df[col].replace({pd.NA: None})

    insert_cols = [col for col in df.columns if col in get_model_columns(model)]

    # insert_cols = set(df.columns) & set(model_cols)

    df = df[insert_cols]
    df = df.drop_duplicates(subset=primary_keys)

    stdf_info(df)

    for col in insert_cols:
        dtype = dtype_dict[col]

        # convert from pandas to python datetime
        if dtype == "datetime64[ns]":
            # st.write(col, "datetime")
            # # df[col] = df[col].values.astype("datetime64[ns]").astype(object)
            # df[col] = pd.to_datetime(df[col], errors="coerce")
            # df[col] = df[col].dt.to_pydatetime()
            # pass
            df[col] = df[col].apply(convert_datetime)

        # convert lists to json
        elif dtype == "object" and df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(convert_to_json)
        elif dtype == "Int64":
            df[col] = df[col].astype("Int64").replace({pd.NA: None})
        elif dtype == "string":
            df[col] = df[col].astype("string").replace({pd.NA: None})
        else:
            df[col] = df[col].astype(dtype)

        # elif dtype == 'Int64':
        #     df[col] = df[col].astype('Int64').where(pd.notnull, None)

    return df


@timeit
def add_data(df, uuid, model_name, operation="insert"):
    model = model_map[model_name]
    primary_keys = get_primary_keys(model)

    df = prepare_for_sqlite(df, model, primary_keys)

    if model_name == "tmdb_media":
        df = df.drop(columns=["tmdb_last_air_date"], errors="ignore")

    records = df.to_dict("records")

    # def json_deserialize(value):
    #     if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
    #         try:
    #             return json.loads(value)
    #         except json.JSONDecodeError:
    #             return value
    #     return value

    # # Deserialize JSON strings
    # for record in records:
    #     for key, value in record.items():
    #         record[key] = json_deserialize(value)

    chunk_size = 1000  # Adjust this value based on your specific use case

    st.write(records[0])

    with Session() as session:
        try:
            for i in range(0, len(records), chunk_size):
                chunk = records[i : i + chunk_size]

                if operation == "insert":
                    session.execute(model.__table__.insert(), chunk)
                elif operation == "upsert":
                    stmt = sqlite_upsert(model).values(chunk)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=primary_keys, set_={c.key: c for c in stmt.excluded if c.key not in primary_keys}
                    )
                    session.execute(stmt)
                elif operation == "sync":
                    if model_name != "user_watch_history":
                        raise ValueError("Sync operation is only supported for user_watch_history")

                    existing_ids = set(session.scalars(select(model.event_id).where(model.trakt_uuid == uuid)))
                    new_ids = set(df["event_id"])
                    ids_to_delete = existing_ids - new_ids

                    if ids_to_delete:
                        session.execute(delete(model).where(model.event_id.in_(ids_to_delete)))

                    stmt = sqlite_upsert(model).values(chunk)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=primary_keys, set_={c.key: c for c in stmt.excluded if c.key not in primary_keys}
                    )
                    session.execute(stmt)
                else:
                    raise ValueError(f"Invalid operation: {operation}")

                session.commit()

            st.write(f"{operation.capitalize()} done for {len(records)} items into {model.__tablename__}.")

        except SQLAlchemyError as e:
            st.write(f"An error occurred while processing data: {str(e)}")
            session.rollback()
            raise
        except Exception as e:
            st.write(f"An unexpected error occurred: {str(e)}")
            session.rollback()
            raise


@timeit
def add_data_old(df, uuid, model_name, operation="insert"):
    model = model_map[model_name]
    primary_keys = get_primary_keys(model)

    # if model_name in ["user", "user_watch_history"]:
    #     df["trakt_uuid"] = uuid

    # df = df.replace({pd.NA: None})

    # stdf_info(df)
    # df = df.drop_duplicates(subset=primary_keys)
    # stdf_info(df)

    # insert_cols = [col for col in df.columns if col in get_model_columns(model)]
    # df = df[insert_cols]

    # stdf_info(df)

    # # Convert Int64 columns to object
    # int64_columns = df.select_dtypes(include=["Int64"]).columns
    # df[int64_columns] = df[int64_columns].astype(object)

    # # Convert list columns to JSON strings
    # for col in df.columns:
    #     if pd.api.types.is_datetime64_any_dtype(df[col]):
    #         df[col] = df[col].dt.to_pydatetime()
    #     elif df[col].dtype == "object":
    #         if df[col].apply(lambda x: isinstance(x, list)).any():
    #             df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)

    df = prepare_for_sqlite(df, model, primary_keys)

    if model_name == "tmdb_media":
        df = df.drop(columns=["tmdb_last_air_date"])

    # stdf_info(df)
    st.write(pd.isna(df))

    records = df.to_dict("records")[:10]

    st.write(records[0])

    with Session.begin() as session:
        try:
            if operation == "insert":
                session.execute(insert(model), records)
                st.write(f"Insert done for {len(records)} items into {model.__tablename__}.")

            elif operation == "upsert":
                stmt = insert(model).values(records)
                session.execute(stmt)
                st.write(f"Upsert done for {len(records)} items into {model.__tablename__}.")

            elif operation == "sync":
                if model_name != "user_watch_history":
                    raise ValueError("Sync operation is only supported for user_watch_history")

                existing_ids = set(session.scalars(select(model.event_id).where(model.trakt_uuid == uuid)))
                new_ids = set(df["event_id"])
                ids_to_delete = existing_ids - new_ids

                if ids_to_delete:
                    session.execute(delete(model).where(model.event_id.in_(ids_to_delete)))

                stmt = insert(model).values(records)
                session.execute(stmt)

                st.write(
                    f"Sync done. {len(ids_to_delete)} items deleted and {len(records)} items upserted in {model.__tablename__}."
                )
            else:
                raise ValueError(f"Invalid operation: {operation}")

        except SQLAlchemyError as e:
            st.write(f"An error occurred while processing data: {str(e)}")
            raise
        except Exception as e:
            st.write(f"An unexpected error occurred: {str(e)}")
            raise


def convert_date_columns(df):
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except ValueError:
                pass  # Column couldn't be converted to datetime
    return df


@timeit
def filter_new_data(df, df_col, model_name, model_col):

    # Convert Int64 columns to object
    int64_mask = df.dtypes == "Int64"
    int64_columns = df.columns[int64_mask]
    if len(int64_columns) > 0:
        df[int64_columns] = df[int64_columns].astype(object)

    df = df.replace({pd.NA: None})

    model = model_map[model_name]

    with Session.begin() as session:
        return (
            session.execute(select(getattr(model, model_col)).where(getattr(model, model_col).in_(df[df_col])))
            .scalars()
            .all()
        )


@timeit
def join_tables(t1, t1_col, t2, t2_col):
    model1 = model_map[t1]
    model2 = model_map[t2]

    # Create a select statement
    stmt = select(model1, model2).join(model2, getattr(model1, t1_col) == getattr(model2, t2_col))

    with Session.begin() as session:
        # Compile the statement into a SQL string
        sql = stmt.compile(compile_kwargs={"literal_binds": True})

        # Use pandas to read the SQL directly
        df = pd.read_sql(sql, session.connection())

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
    with Session.begin() as session:
        main_table = model_map.get(table_name)
        if not main_table:
            raise ValueError(f"Table '{table_name}' not found in model map")

        query = select(main_table)
        joined_tables = {table_name: main_table}

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

        all_columns = []
        seen_columns = set()
        ex_columns = ex_columns or []
        for table in joined_tables.values():
            for c in table.__table__.columns:
                if c.name not in ex_columns and c.name not in seen_columns:
                    all_columns.append(c)
                    seen_columns.add(c.name)
        query = query.with_only_columns(*all_columns)

        if filters:
            for key, value in filters.items():
                column = _get_column(key, joined_tables)
                query = _apply_filter(query, column, value)

        if order_by:
            order_columns = [order_by] if isinstance(order_by, str) else order_by
            for col in order_columns:
                column = _get_column(col, joined_tables)
                query = query.order_by(desc(column) if order_desc else asc(column))

        if limit is not None:
            query = query.limit(limit)

        df = pd.read_sql(query, session.connection())

    # Convert JSON strings back to lists
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = df[col].apply(json.loads)
            except:
                pass

    return df


def check_value_exists(table_name: str, column_name: str, value: Any) -> bool:
    with Session.begin() as session:
        if table_name not in model_map:
            raise ValueError(f"Table '{table_name}' not found in model map")

        table = model_map[table_name]
        column = getattr(table, column_name, None)

        if not column:
            raise ValueError(f"Column '{column_name}' not found in table '{table_name}'")

        query = select(exists().where(column == value))

        result = session.execute(query).scalar()

    return bool(result)


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

    table_columns = inspect(table).c
    if filter_column not in table_columns or target_column not in table_columns:
        raise ValueError(f"Column not found in table {table.__tablename__}")

    if operation not in ["first", "max"]:
        raise ValueError("Invalid operation. Use 'first' or 'max'.")

    with Session.begin() as session:
        query = session.query(getattr(table, target_column)).filter(getattr(table, filter_column) == filter_value)

        if operation == "first":
            result = query.first()
        elif operation == "max":
            result = query.with_entities(func.max(getattr(table, target_column))).scalar()

        result = session.execute(query).scalar()

        return result[0] if result and isinstance(result, tuple) else result


# Custom SQLite-compatible UPSERT compilation
# @compiles(Insert, "sqlite")
# def sqlite_upsert(insert_stmt, compiler, **kw):
#     pk = insert_stmt.table.primary_key
#     insert = compiler.visit_insert(insert_stmt, **kw)
#     ondup = f"ON CONFLICT ({','.join(c.name for c in pk)}) DO UPDATE SET "
#     ondup += ", ".join(f"{c.name}=excluded.{c.name}" for c in insert_stmt.table.columns if c not in pk)
#     return f"{insert} {ondup}"


# @timeit
# def get_column_value(
#     table_name: str,
#     filter_column: str,
#     filter_value: Any,
#     target_column: str,
#     operation: str = "first",
# ) -> Optional[Any]:
#     """
#     Retrieve a value from a specific column based on the specified operation.

#     :param table_name: The name of the table to query
#     :param filter_column: The column name to filter on
#     :param filter_value: The value to filter by
#     :param target_column: The column name to retrieve the value from
#     :param operation: The type of operation to perform ('first' or 'max')
#     :param model_map: A dictionary mapping table names to SQLAlchemy model classes
#     :return: The requested value from the target column, or None if not found
#     :raises ValueError: If the filter column or target column doesn't exist, or if an invalid operation is specified
#     """
#     if table_name not in model_map:
#         raise ValueError(f"Table {table_name} not found in model map")

#     table = model_map[table_name]

#     # Verify that the columns exist in the table
#     table_columns = inspect(table).c
#     if filter_column not in table_columns or target_column not in table_columns:
#         raise ValueError(f"Column not found in table {table.__tablename__}")

#     if operation not in ["first", "max"]:
#         raise ValueError("Invalid operation. Use 'first' or 'max'.")

#     with Session.begin() as session:
#         query = session.query(getattr(table, target_column)).filter(getattr(table, filter_column) == filter_value)

#         if operation == "first":
#             result = query.first()
#         elif operation == "max":
#             result = query.with_entities(func.max(getattr(table, target_column))).scalar()

#         result = session.execute(query).scalar()

#         return result[0] if result and isinstance(result, tuple) else result


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
#     st.write("Starting database operations.")
#     # get_server_version()
#     # create_tables()
#     # create_schema()
#     # list_tables()
#     # test_operations()
#     # add_user(111, "John")
#     # read_users()


# if __name__ == "__main__":
#     main()
