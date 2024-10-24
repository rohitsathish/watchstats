from sqlalchemy import create_engine
import streamlit as st
from sqlalchemy.orm import Session
from sqlalchemy import select
from models import Base, Foo
import logging

dbUrl = f"sqlite+{st.secrets.db.url}/?authToken={st.secrets.db.auth_token}&secure=true"

engine = create_engine(
    dbUrl,
    connect_args={"check_same_thread": False},
)

Base.metadata.bind = engine
session = Session(engine)


def create_from_schema():
    Base.metadata.create_all(engine)


def add_foo(id, bar):
    # Add a new record
    new_foo = Foo(id=id, bar=bar)
    session.add(new_foo)
    session.commit()


def get_foo(id):
    # Query records
    foo = session.query(Foo).filter(Foo.id == id).first()
    if foo:
        print(foo)
    else:
        print("No record found")


add_foo("4", "Example Data")
get_foo("1")
