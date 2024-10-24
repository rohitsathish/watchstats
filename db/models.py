from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, BigInteger
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from datetime import datetime

Base = declarative_base()


class Users(Base):
    __tablename__ = "users"

    trakt_uuid = Column(String, primary_key=True)
    trakt_user_id = Column(String, unique=True, nullable=False)
    auth_token = Column(String, nullable=True)
    refresh_token = Column(String, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    last_db_update = Column(DateTime, default=datetime.utcnow)

    # One-to-Many relationship with UserWatchHistory
    watch_history = relationship(
        "UserWatchHistory",
        backref=backref("user", lazy="joined"),
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Users(trakt_uuid='{self.trakt_uuid}', trakt_user_id='{self.trakt_user_id}')>"


class UserWatchHistory(Base):
    __tablename__ = "user_watch_history"

    event_id = Column(BigInteger, primary_key=True)
    trakt_uuid = Column(
        String, ForeignKey("users.trakt_uuid"), index=True, nullable=False
    )
    trakt_url = Column(
        String, ForeignKey("trakt_media.trakt_url"), index=True, nullable=False
    )
    watched_at = Column(DateTime, index=True, default=datetime.utcnow)
    runtime = Column(Integer, nullable=False)  # Runtime in minutes

    # One-to-Many relationship with TraktMedia is handled via backref in TraktMedia

    def __repr__(self):
        return (
            f"<UserWatchHistory(event_id={self.event_id}, trakt_uuid='{self.trakt_uuid}', "
            f"trakt_url='{self.trakt_url}', watched_at='{self.watched_at}', runtime={self.runtime})>"
        )


class TraktMedia(Base):
    __tablename__ = "trakt_media"

    trakt_url = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    ep_title = Column(String, nullable=True)
    media_type = Column(String, nullable=False)  # e.g., "movie", "episode"
    season_num = Column(Integer, nullable=True)
    ep_num = Column(Integer, nullable=True)
    ep_num_abs = Column(Integer, nullable=True)
    total_episodes = Column(Integer, nullable=True)
    status = Column(String, nullable=True)
    runtime = Column(Integer, nullable=True)  # Runtime in minutes
    released = Column(DateTime, nullable=True)
    show_released = Column(DateTime, nullable=True)
    genres = Column(ARRAY(String), nullable=True)
    country = Column(String, nullable=True)
    overview = Column(Text, nullable=True)
    ep_overview = Column(Text, nullable=True)
    show_trakt_id = Column(Integer, nullable=True)
    show_imdb_id = Column(
        String, ForeignKey("imdb_media.show_imdb_id"), index=True, nullable=True
    )
    show_tmdb_id = Column(
        Integer, ForeignKey("tmdb_media.show_tmdb_id"), index=True, nullable=True
    )

    # One-to-Many relationships with UserWatchHistory
    watch_events = relationship(
        "UserWatchHistory",
        backref=backref("trakt_media", lazy="joined"),
        cascade="all, delete-orphan",
    )

    # Many-to-One relationships with IMDBMedia and TMDBMedia
    imdb_media = relationship(
        "IMDBMedia", backref=backref("trakt_medias", lazy="dynamic"), lazy="joined"
    )
    tmdb_media = relationship(
        "TMDBMedia", backref=backref("trakt_medias", lazy="dynamic"), lazy="joined"
    )

    def __repr__(self):
        return f"<TraktMedia(trakt_url='{self.trakt_url}', title='{self.title}')>"


class IMDBMedia(Base):
    __tablename__ = "imdb_media"

    show_imdb_id = Column(String, primary_key=True)
    imdb_genres = Column(ARRAY(String), nullable=True)

    # One-to-Many relationship with TraktMedia is handled via backref in TraktMedia

    def __repr__(self):
        return f"<IMDBMedia(show_imdb_id='{self.show_imdb_id}')>"


class TMDBMedia(Base):
    __tablename__ = "tmdb_media"

    show_tmdb_id = Column(Integer, primary_key=True)
    tmdb_last_air_date = Column(DateTime, nullable=True)
    tmdb_genres = Column(ARRAY(String), nullable=True)
    tmdb_language = Column(String, nullable=True)
    tmdb_certification = Column(String, nullable=True)
    tmdb_networks = Column(ARRAY(String), nullable=True)
    tmdb_collection = Column(String, nullable=True)
    tmdb_keywords = Column(ARRAY(String), nullable=True)
    tmdb_poster_url = Column(String, nullable=True)

    # One-to-Many relationship with TraktMedia is handled via backref in TraktMedia

    def __repr__(self):
        return f"<TMDBMedia(show_tmdb_id={self.show_tmdb_id})>"
