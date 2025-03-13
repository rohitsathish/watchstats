-- SQLite3 schema for media visualization platform

-- User table for authentication and profile data
CREATE TABLE IF NOT EXISTS users (
    uuid TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    access_token TEXT NOT NULL,
    access_token_expires_at TEXT, -- Stored as ISO format string
    refresh_token TEXT NOT NULL,
    username TEXT,
    name TEXT,
    slug TEXT,
    avatar_url TEXT,
    timezone TEXT,
    last_db_update TEXT -- Stored as ISO format string
);

-- TMDB Media table for media data from TMDB
CREATE TABLE IF NOT EXISTS tmdb_media (
    show_tmdb_id INTEGER PRIMARY KEY,
    language TEXT,
    genres TEXT, -- JSON array stored as text
    keywords TEXT, -- JSON array stored as text
    certification TEXT,
    networks TEXT, -- JSON array stored as text
    collection TEXT,
    poster_url TEXT,
    last_air_date TEXT -- Stored as ISO format string
);

-- IMDB Media table for media data from IMDB
CREATE TABLE IF NOT EXISTS imdb_media (
    show_imdb_id TEXT PRIMARY KEY,
    genres TEXT -- JSON array stored as text
);

-- Trakt Media table for media data from Trakt
CREATE TABLE IF NOT EXISTS trakt_media (
    trakt_url TEXT PRIMARY KEY,
    media_type TEXT,
    title TEXT,
    ep_title TEXT,
    season_num INTEGER,
    ep_num INTEGER,
    ep_num_abs INTEGER,
    total_episodes INTEGER,
    status TEXT,
    ep_overview TEXT,
    overview TEXT,
    runtime INTEGER,
    released TEXT, -- Stored as ISO format string
    show_released TEXT, -- Stored as ISO format string
    genres TEXT, -- JSON array stored as text
    country TEXT,
    show_trakt_id INTEGER,
    show_imdb_id TEXT,
    show_tmdb_id INTEGER,
    FOREIGN KEY (show_imdb_id) REFERENCES imdb_media(show_imdb_id),
    FOREIGN KEY (show_tmdb_id) REFERENCES tmdb_media(show_tmdb_id)
);

-- User Watch History table
CREATE TABLE IF NOT EXISTS user_watch_history (
    event_id INTEGER PRIMARY KEY,
    uuid TEXT,
    trakt_url TEXT,
    watched_at TEXT NOT NULL, -- Stored as ISO format string
    runtime INTEGER,
    FOREIGN KEY (uuid) REFERENCES users(uuid),
    FOREIGN KEY (trakt_url) REFERENCES trakt_media(trakt_url)
);

-- User Ratings table
CREATE TABLE IF NOT EXISTS user_ratings (
    uuid TEXT,
    rated_at TEXT NOT NULL, -- Stored as ISO format string
    rating INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    title TEXT NOT NULL,
    trakt_url TEXT,
    season_num INTEGER,
    ep_num INTEGER,
    PRIMARY KEY (uuid, trakt_url),
    FOREIGN KEY (uuid) REFERENCES users(uuid),
    FOREIGN KEY (trakt_url) REFERENCES trakt_media(trakt_url)
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_watch_history_uuid ON user_watch_history(uuid);
CREATE INDEX IF NOT EXISTS idx_watch_history_trakt_url ON user_watch_history(trakt_url);

CREATE INDEX IF NOT EXISTS idx_trakt_media_show_imdb_id ON trakt_media(show_imdb_id);
CREATE INDEX IF NOT EXISTS idx_trakt_media_show_tmdb_id ON trakt_media(show_tmdb_id);