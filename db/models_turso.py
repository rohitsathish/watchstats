from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "user"

    trakt_uuid = Column(String, primary_key=True)
    trakt_user_id = Column(String, unique=True)
    trakt_auth_token = Column(String)
    trakt_refresh_token = Column(String)
    trakt_token_expires_at = Column(DateTime)
    last_db_update = Column(DateTime, default=datetime.utcnow)

    watch_history = relationship("UserWatchHistory", back_populates="user")


class UserWatchHistory(Base):
    __tablename__ = "user_watch_history"

    event_id = Column(BigInteger, primary_key=True)
    trakt_uuid = Column(String, ForeignKey("user.trakt_uuid"))
    trakt_url = Column(String, ForeignKey("trakt_media.trakt_url"))
    watched_at = Column(DateTime)
    runtime = Column(Integer)

    user = relationship("User", back_populates="watch_history")
    trakt = relationship("TraktMedia", back_populates="watch_events")


class TraktMedia(Base):
    __tablename__ = "trakt_media"

    trakt_url = Column(String, primary_key=True)
    title = Column(String)
    ep_title = Column(String)
    media_type = Column(String)
    season_num = Column(Integer)
    ep_num = Column(Integer)
    ep_num_abs = Column(Integer)
    total_episodes = Column(Integer)
    status = Column(String)
    runtime = Column(Integer)
    released = Column(DateTime)
    genres = Column(String)  # Store as comma-separated string
    country = Column(String)
    overview = Column(Text)
    ep_overview = Column(Text)
    show_trakt_id = Column(Integer)
    show_imdb_id = Column(String, ForeignKey("imdb_media.show_imdb_id"))
    show_tmdb_id = Column(Integer, ForeignKey("tmdb_media.show_tmdb_id"))

    watch_events = relationship("UserWatchHistory", back_populates="trakt")
    imdb = relationship("IMDBMedia", back_populates="trakt")
    tmdb = relationship("TMDBMedia", back_populates="trakt")


class IMDBMedia(Base):
    __tablename__ = "imdb_media"

    show_imdb_id = Column(String, primary_key=True)
    imdb_genres = Column(String)  # Store as comma-separated string

    trakt = relationship("TraktMedia", back_populates="imdb")


class TMDBMedia(Base):
    __tablename__ = "tmdb_media"

    show_tmdb_id = Column(Integer, primary_key=True)
    tmdb_last_air_date = Column(DateTime)
    tmdb_genres = Column(String)  # Store as comma-separated string
    tmdb_language = Column(String)
    tmdb_certification = Column(String)
    tmdb_networks = Column(String)  # Store as comma-separated string
    tmdb_collection = Column(String)
    tmdb_keywords = Column(String)  # Store as comma-separated string

    trakt = relationship("TraktMedia", back_populates="tmdb")
