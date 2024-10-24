import heapq
import time
from collections import Counter
import logging
import requests as req
import streamlit as st
import pandas as pd
import numpy as np
from streamlit import session_state as ss
from db import db_o3
from db.db_o3 import add_data, read_table_df, check_value_exists, get_column_value, filter_new_data

# from db.db_o1 import
# import aiohttp
import asyncio
import httpx
import pickle

# from random import randint
from datetime import datetime as dt
import tqdm.asyncio as tqdma

# from stqdm import stqdm
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import os
import io
import diskcache

tmdb_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI2NTJhODdhZGM2NjQyNDcxMTliYWQ4NjlhZjA3MjhiMyIsInN1YiI6IjYyNjc2OTE3MTJhYWJjMDA1MTY1ZGU3YSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.nv3_UiTTaOl79hx5EADEoANpefDdJgflqzesbO7JTTs",
    # "Authorization": "Bearer 652a87adc664247119bad869af0728b3"
}

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# ---- Global Constants ----

TRAKT_API_BASE_URL = "https://api.trakt.tv"

col_dict = {
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
    "show_released": "datetime64[ns]",
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
    "tmdb_poster_url": "string",
    # "last_updated_at": "datetime64[ns]",
    "overview": "string",
    "ep_overview": "string",
    "show_trakt_id": "Int64",
    "show_imdb_id": "string",
    "show_tmdb_id": "Int64",
    "event_id": "Int64",
}

# ---- Cache Setup ----

cache = diskcache.Cache("assets/cache")


def stdf_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return st.text(buffer.getvalue())


# ---- Additional Data ----


def load_country_codes():
    with open("assets/country_codes_custom.pkl", "rb") as file:
        return pickle.load(file)


def load_lang_codes():
    with open("assets/lang_codes.pkl", "rb") as file:
        return pickle.load(file)


def load_tmdb_lang_codes(simplified=True):
    with open("assets/tmdb_lang_codes.pkl", "rb") as file:
        lang_code = pickle.load(file)
        if simplified:
            return {key: value["english_name"] for key, value in lang_code.items()}
        else:
            return lang_code


def load_trakt_imdb_genre_map():
    with open("assets/trakt_to_imdb_genres.pkl", "rb") as file:
        return pickle.load(file)


# ---- Request Handlers ----


async def api_call_wrapper(
    client,
    url,
    use_cache=True,
    headers=None,
    max_retries=5,
    current_retry=0,
    cache_store_days=14,
    rate_limit_hits=0,
):
    """
    Makes an asynchronous API call using the provided client and handles rate-limiting by retrying the call
    after the specified wait duration, up to a maximum number of retries.
    """
    cache_key = f"api_call_{url}"
    try:
        if use_cache:
            cached_response = cache.get(cache_key)
            if cached_response:
                return cached_response  # Deserialize cached JSON to Python object

        response = await client.get(url, headers=headers)

        if response.status_code == 200:
            cache.set(cache_key, response, expire=60 * 60 * 24 * cache_store_days)
            return response  # Return as Python object

        if response.status_code == 429:
            if current_retry < max_retries:
                wait_duration = int(response.headers.get("Retry-After", 10))
                await asyncio.sleep(wait_duration)
                logging.warning(f"Rate Limit Hit {rate_limit_hits + 1}")
                return await api_call_wrapper(
                    client,
                    url,
                    current_retry=current_retry + 1,
                    rate_limit_hits=rate_limit_hits + 1,
                )  # Retry the request
            else:
                response.raise_for_status()  # Optionally raise an exception or return an error

        # 520 and 522 are connect errors, 502 is a bad gateway error
        if response.status_code in [520, 522, 524, 401, 502]:
            logging.error(
                f"Network or auth error encountered: {response.status_code}. URL: {url}. Attempt {current_retry + 1} of {max_retries}"
            )
            if current_retry < max_retries:
                if "https://api.themoviedb.org" in url:
                    logging.warning(f"TMDB hit, {response.status_code}")
                    wait_duration = 120
                else:
                    wait_duration = 10
                await asyncio.sleep(wait_duration)
                return await api_call_wrapper(client, url, current_retry=current_retry + 1)
            else:
                response.raise_for_status()

        if response.status_code == 404:
            logging.error(f"404 Not Found error. URL: {url}")
            return None

        # Handle other HTTP errors
        response.raise_for_status()

    except httpx.ConnectError as e:
        logging.error(f"Network error encountered: {e}. URL: {url}. Attempt {current_retry + 1} of {max_retries}")
        if current_retry < max_retries:
            if "https://api.themoviedb.org" in url:
                logging.warning("TMDB hit, Connecterror")
                wait_duration = 120
            else:
                wait_duration = 10
            await asyncio.sleep(wait_duration)
            return await api_call_wrapper(client, url, current_retry=current_retry + 1)  # Retry the request
        else:
            raise  # Reraise the ConnectError after max retries
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP status error: {e}. URL: {url}")
        raise  # Reraise the HTTPStatusError to indicate failure
    except Exception as e:
        logging.error(f"Unexpected error: {e}. URL: {url}")
        raise  # Reraise for unexpected errors


def get_media_data(slug):
    return req.get(f"https://api.trakt.tv/{slug}?extended=full", headers=ss.user_headers).json()


def get_tmdb_media_data(id_, client=None, media_type="movie", ext=None):
    assert media_type in ["movie", "tv"]
    if not client:
        return req.get(get_tmdb_url(id_, media_type, ext), headers=tmdb_headers).json()
    else:
        return api_call_wrapper(client, get_tmdb_url(id_, media_type, ext), headers=tmdb_headers)


def get_tmdb_url(id_, media_type="movie", ext=None):
    assert media_type in ["movie", "tv", "episode"]
    if media_type == "episode":
        media_type = "tv"
    if not ext:
        return f"http://api.themoviedb.org/3/{media_type}/{id_}?language=en-US"
    else:
        return f"http://api.themoviedb.org/3/{media_type}/{id_}/{ext}"


# ---- High Level API Call Functions ----


def async_wrapper_watch_history(watch_history=True, test=True):
    def wrapper(watch_history):
        if watch_history:
            return build_watch_history(ss.trakt_user_id, test=test)
        else:
            return build_ratings(ss.trakt_user_id)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(wrapper(watch_history))
    loop.close()
    return result


def load_media_data(from_file=True, test=True):
    if from_file:
        return pd.read_pickle("assets/trakt_main.pkl")
    else:
        df = async_wrapper_watch_history(watch_history=True, test=test)
        # if not test:
        #     df.to_pickle("assets/trakt_main.pkl")
        # return df


def load_ratings_data(from_file=True):
    if from_file:
        return pd.read_pickle("assets/trakt_ratings.pkl")
    else:
        ratings_df = async_wrapper_watch_history(watch_history=False)
        ratings_df.to_pickle("assets/trakt_ratings.pkl")
        return ratings_df


# ---- Data Wrangling Post API Calls ----


def consolidate_trakt_genres(genres, mapping):
    """
    The approach taken here for the consolidated genres column is to use the imdb genres where available and a modified version of trakt genres when they are not available. The trakt genres dictionary has been modified to change or remove certain genres that do not align with IMDB genres. This function uses that dictionary to then create a consolidated genres column.
    """
    transformed_dict = {value["name"]: {**value, "slug": key} for key, value in mapping.items()}

    for value in transformed_dict.values():
        del value["name"]

    new_genres_list = []

    for g in genres:
        if transformed_dict[g]["type"] == "remove":
            pass
        elif transformed_dict[g]["type"] == "map":
            new_genres_list.append(transformed_dict[g]["value"])
        else:
            new_genres_list.append(g)

    return new_genres_list[:3]


def add_ratings_col(df, ratings_df, merge_col):
    assert merge_col in ["show", "season_num", "ep_num"]

    if merge_col == "season_num":
        ratings_df = ratings_df.drop("ep_num", axis=1)
    elif merge_col == "ep_num":
        ratings_df = ratings_df.drop("season_num", axis=1)
    else:
        ratings_df = ratings_df.drop(["season_num", "ep_num"], axis=1)

    if merge_col != "show":
        with_ratings_df = pd.merge(df, ratings_df, on=["show_trakt_id", "media_type", merge_col], how="left")
    else:
        with_ratings_df = pd.merge(df, ratings_df, on=["show_trakt_id", "media_type"], how="left")

    cols = list(df.columns)
    cols.insert(cols.index("watched_at"), "user_rating")
    with_ratings_df = with_ratings_df[cols].astype({"user_rating": "Int64"})
    return with_ratings_df


def adjust_wgm_runtime(df):
    # Create a mask for the conditions to avoid repetition
    mask_season4 = (df.title == "We Got Married") & (df.season_num == 4)
    mask_specific_eps = (df.ep_num >= 263 - 132) & (df.ep_num <= 286 - 132)

    # Adjust runtime for specific episodes in season 4
    df.loc[mask_season4 & mask_specific_eps, "runtime"] = (
        df.loc[mask_season4 & mask_specific_eps, "runtime"] * 2 / 3
    ).apply(np.floor)

    # Adjust runtime for other episodes in season 4
    df.loc[mask_season4 & ~mask_specific_eps, "runtime"] = (
        df.loc[mask_season4 & ~mask_specific_eps, "runtime"] / 3
    ).apply(np.floor)

    # Adjust runtime for all other seasons
    df.loc[(df.title == "We Got Married") & (df.season_num != 4), "runtime"] = (
        df.loc[(df.title == "We Got Married") & (df.season_num != 4), "runtime"] / 3
    ).apply(np.floor)

    return df


# ---- Data Fetching Functions ----


# --- Trakt ---


async def fetch_trakt_ratings(trakt_id):
    async with httpx.AsyncClient(
        timeout=200.0,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=40),
        http2=True,
    ) as client:
        url = f"https://api.trakt.tv/users/{trakt_id}/ratings/all"
        response = await api_call_wrapper(client, url, headers=ss.user_headers, use_cache=False)
    ratings = response.json()
    return ratings


# async def fetch_watch_history_page(client, trakt_id, start_date, end_date, page):
#     """
#     Fetch a single page of watch history data for the given user and date range.
#     """
#     url = f"{TRAKT_API_BASE_URL}/users/{trakt_id}/history?start_at={start_date}&end_at={end_date}&page={page}&limit=100"
#     response = await api_call_wrapper(client, url)
#     # st.write(pd.DataFrame(response.json()))
#     return response.json()


# async def fetch_trakt_history(
#     client,
#     trakt_id,
#     start_date,
#     end_date,
#     page_count,
# ):

#     progress_text = st.empty()

#     tasks = [
#         fetch_watch_history_page(client, trakt_id, start_date, end_date, page) for page in range(1, page_count + 1)
#     ]

#     async def progress_wrapper(tasks, index, total):
#         history_pages = await asyncio.gather(tasks)
#         progress_text.write(f"Trakt History Pages: Completed {index + 1} of {total} tasks.")
#         return history_pages

#     wrapped_tasks = [progress_wrapper(task, i, len(tasks)) for i, task in enumerate(tasks)]

#     results = await asyncio.gather(*wrapped_tasks)

#     all_history = [item for list_ in results for sublist in list_ for item in sublist]

#     return all_history


# async def fetch_trakt_data(client, media_type, slug, season_num=None, ep_num=None):
#     """
#     Fetches media details for a given media (movie or show) from the API.
#     """
#     if media_type == "movie":
#         url = f"{TRAKT_API_BASE_URL}/movies/{slug}?extended=full"
#         response = await api_call_wrapper(client, url)
#         data = response.json()
#         return {
#             "runtime": data.get("runtime"),
#             # "released": data.get("released"),
#             "country": data.get("country"),
#             # "language": data.get("language"),
#             "genres": data.get("genres"),
#             # "certification": data.get("certification"),
#             # "status": data.get("status"),
#             # "first_aired": data.get("first_aired"),
#             # "network": data.get("network"),
#         }
#     else:
#         url = f"{TRAKT_API_BASE_URL}/shows/{slug}/seasons/{season_num}/episodes/{ep_num}?extended=full"
#         response_ep = await api_call_wrapper(client, url)
#         data_ep = response_ep.json()

#         url = f"{TRAKT_API_BASE_URL}/shows/{slug}?extended=full"
#         response_show = await api_call_wrapper(client, url)
#         data_show = response_show.json()

#         # Here data_ep is used for runtime and final totals are correct. Check for released. It is better to get runtime here as we are doing episode level pulls, for tmdb, imdb we were doing show level
#         return {
#             "runtime": data_ep.get("runtime"),
#             # "released": data_show.get("released"),
#             "country": data_show.get("country"),
#             # "language": data_show.get("language"),
#             "genres": data_show.get("genres"),
#             # "certification": data_show.get("certification"),
#             # "status": data_show.get("status"),
#             # "first_aired": data_show.get("first_aired"),
#             # "network": data_show.get("network"),
#         }


# async def add_trakt_data(df):
#     """
#     Fetches media details for multiple media entries in parallel using asynchronous API calls
#     and updates the Streamlit UI to track progress accurately.
#     """
#     async with httpx.AsyncClient(
#         timeout=200.0,
#         limits=httpx.Limits(max_connections=100, max_keepalive_connections=40),
#         http2=True,
#     ) as client:
#         progress_text = st.empty()

#         tasks = [
#             fetch_trakt_data(client, row["media_type"], row["trakt_slug"], row.get("season_num"), row.get("ep_num"))
#             for row in df.to_dict("records")
#         ]

#         # Initialize a counter and a lock to safely increment the counter across tasks
#         completed_tasks = 0
#         lock = asyncio.Lock()

#         async def progress_wrapper(task):
#             nonlocal completed_tasks  # Refer to the non-local counter
#             result = await task
#             # Use a lock to safely increment and read the counter
#             async with lock:
#                 completed_tasks += 1
#                 progress_text.write(f"Trakt Data: Completed {completed_tasks} of {len(tasks)} tasks.")
#             return result

#         # Wrap each task with the progress wrapper
#         wrapped_tasks = [progress_wrapper(task) for task in tasks]
#         results = await asyncio.gather(*wrapped_tasks)

#         return results


# --- TMDB ---


# async def fetch_tmdb_image_data(client, media_type, id_, image_base_url="https://image.tmdb.org/t/p/original"):

#     url = get_tmdb_url(id_, media_type=media_type, ext="images")
#     response = await api_call_wrapper(client, url, headers=tmdb_headers)

#     try:
#         data = response.json()["posters"]
#     except AttributeError:
#         return {"tmdb_poster_url": None}

#     filtered_data = [img for img in data if img["aspect_ratio"] < 1]
#     img = heapq.nlargest(1, filtered_data, key=lambda x: x["vote_count"])

#     if img:
#         rel_url = img[0]["file_path"]
#         abs_url = image_base_url + rel_url
#         return {"tmdb_poster_url": abs_url}
#     else:
#         return {"tmdb_poster_url": None}


async def fetch_tmdb_certification_data(client, media_type, id_):
    """
    Fetches certification details for a given media (movie or show) from the API.
    """
    if pd.isna(id_) or id_ in [None, "nan"]:
        return {}
    else:
        if media_type == "movie":
            url = get_tmdb_url(id_, media_type=media_type, ext="release_dates")
            response = await api_call_wrapper(client, url, headers=tmdb_headers)
            try:
                data = response.json()
            except AttributeError:
                return {"tmdb_certification": None}

            us_data = next(
                (item["release_dates"] for item in data["results"] if item["iso_3166_1"] == "US"),
                [],
            )
            # Filter out empty or None certifications and collect all
            certifications = [
                release["certification"] for release in us_data if release["certification"] not in [None, "nan", ""]
            ]
            # Find the most common certification if there are multiple.
            # Could be replaced with just taking the first value to speed things up.
            if certifications:
                most_common_certification = Counter(certifications).most_common(1)[0][0]
                return {"tmdb_certification": most_common_certification}
            return {"tmdb_certification": None}
        else:
            url = get_tmdb_url(id_, media_type=media_type, ext="content_ratings")
            response = await api_call_wrapper(client, url, headers=tmdb_headers)
            try:
                data_show = response.json()
            except AttributeError:
                return {"tmdb_certification": None}

            certification = next(
                (item["rating"] for item in data_show["results"] if item["iso_3166_1"] == "US"),
                None,
            )

            return {"tmdb_certification": certification}


async def fetch_tmdb_keywords_data(client, media_type, id_):
    """
    Fetches keyword details for a given media (movie or show) from the API.
    """
    # movie - original language, production countries (list), production companies, status, belongs to collection, imdb_id
    # tv - in production, networks, origin_country, original language, production companies, production countries, status
    if pd.isna(id_) or id_ in [None, "nan"]:
        return {}
    else:
        url = get_tmdb_url(id_, media_type=media_type, ext="keywords")
        response = await api_call_wrapper(client, url, headers=tmdb_headers)
        data = response.json()

        # Try to fetch keywords from either 'results' or 'keywords', whichever is present. It seems like the keywords endpoint of TMDB is different for movies and shows. For shows its results and for movies its keywords
        keywords_list = data.get("results") or data.get("keywords") or []

        keywords = {"tmdb_keywords": [item.get("name") for item in keywords_list] if keywords_list else None}

        # st.write(keywords)
        return keywords


async def fetch_tmdb_data(client, media_type, id_):
    """
    Fetches media details for a given media (movie or show) from the API.
    """
    # movie - original language, production countries (list), production companies, status, belongs to collection, imdb_id
    # tv - in production, networks, origin_country, original language, production companies, production countries, status

    image_base_url = "https://image.tmdb.org/t/p/original"

    if pd.isna(id_) or id_ in [None, "nan"]:
        return {}
    else:
        if media_type == "movie":
            url = get_tmdb_url(id_, media_type=media_type)
            response = await api_call_wrapper(client, url, headers=tmdb_headers)
            data = response.json()
            # prod_countries = [country["name"] for country in data.get("production_countries", [])]
            # prod_companies = [company["name"] for company in data.get("production_companies", [])]
            genres = [g["name"] for g in data.get("genres", [])]

            return {
                # "tmdb_release_date": data.get("release_date"),
                "tmdb_language": data.get("original_language"),
                "tmdb_genres": genres,
                "tmdb_poster_url": image_base_url + data.get("poster_path"),
                # "tmdb_prod_countries": prod_countries,
                # "tmdb_prod_companies": prod_companies,
                # "tmdb_status": data.get("status"),
                "tmdb_collection": (
                    data.get("belongs_to_collection", {}).get("name") if data.get("belongs_to_collection") else None
                ),
                "tmdb_imdb_id": data.get("imdb_id"),
                "tmdb_last_air_date": None,
                "tmdb_networks": None,
            }
        else:
            # Fetch episode data
            url = get_tmdb_url(id_, media_type="tv")
            response_show = await api_call_wrapper(client, url, headers=tmdb_headers)
            data_show = response_show.json()
            # prod_countries = [
            #     country["name"] for country in data_show.get("production_countries", [])
            # ]
            # prod_companies = [
            #     company["name"] for company in data_show.get("production_companies", [])
            # ]
            genres = [g["name"] for g in data_show.get("genres", [])]
            networks = [network["name"] for network in data_show.get("networks", [])]

            return {
                # "tmdb_release_date": data_show.get("first_air_date"),
                "tmdb_language": data_show.get("original_language"),
                "tmdb_genres": genres,
                "tmdb_poster_url": image_base_url + data_show.get("poster_path"),
                # "tmdb_status": data_show.get("status"),
                "tmdb_collection": None,
                "tmdb_imdb_id": None,
                "tmdb_last_air_date": data_show.get("last_air_date"),
                # "tmdb_in_production": data_show.get("in_production"),
                "tmdb_networks": networks,
                # "tmdb_origin_country": data_show.get("origin_country"),
                # "tmdb_prod_countries": prod_countries,
                # "tmdb_prod_companies": prod_companies,
            }


async def add_tmdb_data(df):
    async with httpx.AsyncClient(
        timeout=200.0,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=40),
        http2=True,
    ) as client:
        progress_text = st.empty()
        tasks = []
        for row in df.to_dict("records"):
            tmdb_data = fetch_tmdb_data(client, row["media_type"], row["show_tmdb_id"])
            tmdb_keywords_data = fetch_tmdb_keywords_data(client, row["media_type"], row["show_tmdb_id"])
            tmdb_cert_data = fetch_tmdb_certification_data(client, row["media_type"], row["show_tmdb_id"])
            tasks.append((tmdb_data, tmdb_keywords_data, tmdb_cert_data))

        lock = asyncio.Lock()
        completed_tasks = 0

        async def progress_wrapper(task_trio, index, total):
            nonlocal completed_tasks
            tmdb_result, keywords_result, cert_result = await asyncio.gather(*task_trio)
            merged_result = {**tmdb_result, **keywords_result, **cert_result}
            async with lock:
                completed_tasks += 1
                progress_text.write(f"TMDB Data: Completed {completed_tasks} of {len(tasks)} tasks.")
            return merged_result

        wrapped_tasks = [progress_wrapper(task_trio, i, len(tasks)) for i, task_trio in enumerate(tasks)]
        results = await asyncio.gather(*wrapped_tasks)

    return results


# --- IMDB ---


async def fetch_imdb_data(client, imdb_id):
    """Fetch movie details from the OMDB API."""
    if pd.isnull(imdb_id):
        return {
            "imdb_genres": None,
            # "imdb_certification": None,
            # "imdb_language": None,
            # "imdb_country": None,
        }
    else:
        api_key = "fadeb875"
        # st.write(imdb_id)
        url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
        response = await api_call_wrapper(client, url)
        imdb_data = response.json()

        # Process IMDB Data
        imdb_genres = imdb_data.get("Genre")

        if imdb_genres in [None, "N/A"]:
            imdb_genres = []
        else:
            try:
                imdb_genres = imdb_genres.split(", ")
            except:
                imdb_genres = list(imdb_genres)

        # imdb_country = imdb_data.get("Country")

        # if imdb_country in [None, "N/A"]:
        #     imdb_country = []
        # else:
        #     try:
        #         imdb_country = imdb_country.split(", ")
        #     except:
        #         imdb_country = list(imdb_country)

        # imdb_certification = imdb_data.get("Rated")

        # if imdb_certification == "N/A":
        #     imdb_certification = None

        # imdb_language = imdb_data.get("Language")

        # if imdb_language in [None, "N/A"]:
        #     imdb_language = []
        # else:
        #     try:
        #         imdb_language = imdb_language.split(", ")
        #     except:
        #         imdb_language = list(imdb_language)

        return {
            "imdb_genres": imdb_genres,
            # "imdb_certification": imdb_certification,
            # "imdb_language": imdb_language,
            # "imdb_country": imdb_country,
        }


async def add_imdb_data(df):
    async with httpx.AsyncClient(
        timeout=200.0,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=40),
        http2=True,
    ) as client:
        progress_text = st.empty()

        tasks = [fetch_imdb_data(client, imdb_id) for imdb_id in df["show_imdb_id"]]

        lock = asyncio.Lock()
        completed_tasks = 0

        async def progress_wrapper(task, index, total):
            nonlocal completed_tasks
            result = await task
            async with lock:
                completed_tasks += 1
                progress_text.write(f"IMDB Data: Completed {completed_tasks} of {total} tasks.")
            return result

        wrapped_tasks = [progress_wrapper(task, i, len(tasks)) for i, task in enumerate(tasks)]
        results = await asyncio.gather(*wrapped_tasks)

        return results


# ---- Master Function ----


async def build_watch_history_old(
    trakt_id,
    test=True,
    start_date="2019-1-1",
    end_date=(dt.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
):
    """
    Builds a watch history for a user within a specified date range by fetching data from the API.
    The function processes the fetched data to create a DataFrame with relevant details.
    """

    async with httpx.AsyncClient(
        timeout=200.0,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=40),
        http2=True,
    ) as client:

        tt = time.time()

        watch_history = await api_call_wrapper(
            client,
            f"https://api.trakt.tv/users/{trakt_id}/history?extended=full&limit={10**8}",
            headers=ss.user_headers,
            use_cache=False,
        )

        watch_history = watch_history.json()

        st.write(time.time() - tt)

    df = pd.json_normalize(watch_history)
    st.write("Raw Data")
    st.write(df)

    df["movie.released"] = pd.to_datetime(df["movie.released"])
    df["movie.released"] = df["movie.released"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    consolidate_cols = [
        ("movie.title", "show.title", "title"),
        ("movie.country", "show.country", "country"),
        # ("movie.language", "show.language", "language"),
        # ("movie.certification", "show.certification", "certification"),
        ("movie.genres", "show.genres", "genres"),
        # ("movie.updated_at", "show.updated_at", "last_updated_at"),
        ("movie.status", "show.status", "status"),
        ("movie.runtime", "episode.runtime", "runtime"),
        ("movie.released", "show.first_aired", "released"),
        ("movie.overview", "show.overview", "overview"),
        ("movie.ids.trakt", "show.ids.trakt", "show_trakt_id"),
        ("movie.ids.tmdb", "show.ids.tmdb", "show_tmdb_id"),
        ("movie.ids.imdb", "show.ids.imdb", "show_imdb_id"),
        ("movie.ids.slug", "show.ids.slug", "trakt_slug"),
    ]

    for movie_col, show_col, new_col in consolidate_cols:
        df[new_col] = df[movie_col].combine_first(df[show_col])
        df = df.drop(columns=[movie_col, show_col])

    df = df.rename(
        columns={
            "id": "event_id",
            "type": "media_type",
            "episode.season": "season_num",
            "episode.number": "ep_num",
            "episode.title": "ep_title",
            "episode.number_abs": "ep_num_abs",
            "episode.overview": "ep_overview",
            "show.aired_episodes": "total_episodes",
        }
    )

    st.write("Trakt History Fetch")
    st.write(df)

    if test:
        df = df.sample(50, axis=0).reset_index()

    # Initial data type setting to handle functions correctly

    cur_col_dict = {k: v for k, v in col_dict.items() if k in df.columns}
    df = df.astype(cur_col_dict)

    # WGM runtime data needs to be adjusted

    df = adjust_wgm_runtime(df)

    # Get media details from imdb and tmdb

    stt = time.time()
    tmdb_data, imdb_data = await asyncio.gather(add_tmdb_data(df), add_imdb_data(df))
    st.write(("Data Loading time", time.time() - stt))

    tmdb_df = pd.DataFrame(tmdb_data)
    df = pd.concat([df, tmdb_df], axis=1)

    imdb_df = pd.DataFrame(imdb_data)
    df = pd.concat([df, imdb_df], axis=1)

    # Data modification

    # Create the URL from the slug
    df["trakt_url"] = np.where(
        df["media_type"] == "movie",
        "https://trakt.tv/movies/" + df["trakt_slug"],
        "https://trakt.tv/shows/"
        + df["trakt_slug"]
        + "/seasons/"
        + df["season_num"].astype(str)
        + "/episodes/"
        + df["ep_num"].astype(str),
    )

    df["country"] = df["country"].replace(load_country_codes())

    df["tmdb_language"] = df["tmdb_language"].replace(load_tmdb_lang_codes())

    # Filter out the necessary columns
    df = df[col_dict.keys()]
    df = df.astype(col_dict)

    st.write(df["runtime"].dtype)

    return df


async def build_watch_history_old1(
    test=True,
    start_date="2019-1-1",
    end_date=(dt.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
):
    """
    Builds a watch history for a user within a specified date range by fetching data from the API.
    The function processes the fetched data to create a DataFrame with relevant details.
    """

    async with httpx.AsyncClient(
        timeout=200.0,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=40),
        http2=True,
    ) as client:

        tt = time.time()

        watch_history = await api_call_wrapper(
            client,
            f"https://api.trakt.tv/users/{ss.trakt_user_id}/history?extended=full&limit={10**10}",
            headers=ss.user_headers,
            use_cache=False,
        )

        watch_history = watch_history.json()

        st.write(time.time() - tt)

    df = pd.json_normalize(watch_history)
    st.write("Raw Data")
    st.write(df)

    df["movie.released"] = pd.to_datetime(df["movie.released"])
    df["movie.released"] = df["movie.released"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    consolidate_cols = [
        ("movie.title", "show.title", "title"),
        ("movie.country", "show.country", "country"),
        # ("movie.language", "show.language", "language"),
        # ("movie.certification", "show.certification", "certification"),
        ("movie.genres", "show.genres", "genres"),
        # ("movie.updated_at", "show.updated_at", "last_updated_at"),
        ("movie.status", "show.status", "status"),
        ("movie.runtime", "episode.runtime", "runtime"),
        ("movie.released", "show.first_aired", "released"),
        ("movie.overview", "show.overview", "overview"),
        ("movie.ids.trakt", "show.ids.trakt", "show_trakt_id"),
        ("movie.ids.tmdb", "show.ids.tmdb", "show_tmdb_id"),
        ("movie.ids.imdb", "show.ids.imdb", "show_imdb_id"),
        ("movie.ids.slug", "show.ids.slug", "trakt_slug"),
    ]

    for movie_col, show_col, new_col in consolidate_cols:
        df[new_col] = df[movie_col].combine_first(df[show_col])
        df = df.drop(columns=[movie_col, show_col])

    df = df.rename(
        columns={
            "id": "event_id",
            "type": "media_type",
            "episode.season": "season_num",
            "episode.number": "ep_num",
            "episode.title": "ep_title",
            "episode.number_abs": "ep_num_abs",
            "episode.overview": "ep_overview",
            "show.aired_episodes": "total_episodes",
        }
    )

    st.write("Trakt History Fetch")
    st.write(df)

    if test:
        df = df.sample(50, axis=0).reset_index()

    # Initial data type setting to handle functions correctly

    cur_col_dict = {k: v for k, v in col_dict.items() if k in df.columns}
    df = df.astype(cur_col_dict)

    # WGM runtime data needs to be adjusted

    df = adjust_wgm_runtime(df)

    # Get media details from imdb and tmdb

    # stt = time.time()
    # tmdb_data, imdb_data = await asyncio.gather(add_tmdb_data(df), add_imdb_data(df))
    # st.write(("Data Loading time", time.time() - stt))

    # tmdb_df = pd.DataFrame(tmdb_data)
    # df = pd.concat([df, tmdb_df], axis=1)

    # imdb_df = pd.DataFrame(imdb_data)
    # df = pd.concat([df, imdb_df], axis=1)

    # Data modification

    # Create the URL from the slug
    df["trakt_url"] = np.where(
        df["media_type"] == "movie",
        "https://trakt.tv/movies/" + df["trakt_slug"],
        "https://trakt.tv/shows/"
        + df["trakt_slug"]
        + "/seasons/"
        + df["season_num"].astype(str)
        + "/episodes/"
        + df["ep_num"].astype(str),
    )

    df["country"] = df["country"].replace(load_country_codes())

    # df["tmdb_language"] = df["tmdb_language"].replace(load_tmdb_lang_codes())

    # Filter out the necessary columns

    col_dict_1 = {k: v for k, v in col_dict.items() if k in df.columns}
    # st.write(trakt_cols)
    df = df[col_dict_1.keys()]
    df = df.astype(col_dict_1)

    return df


# import pandas as pd
# import numpy as np
# import asyncio
# import httpx
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy import select
# from db.models import MediaData


async def trakt_history_and_data(shallow=True, start_dt=None, end_dt=dt.utcnow()):
    if shallow:
        assert all([start_dt is not None, end_dt is not None])
        base_url = f"https://api.trakt.tv/users/{ss.trakt_user_id}/history?extended=full&limit=100&start_at={start_dt}&end_at={end_dt}"
    else:
        base_url = f"https://api.trakt.tv/users/{ss.trakt_user_id}/history?extended=full&limit=100"

    async with httpx.AsyncClient(
        timeout=200.0, limits=httpx.Limits(max_connections=100, max_keepalive_connections=40), http2=True
    ) as client:

        tt = time.time()
        # Fetch first page to get pagination info
        first_page_response = await api_call_wrapper(
            client, f"{base_url}&page=1", headers=ss.user_headers, use_cache=False
        )

        first_page = first_page_response.json()
        total_pages = int(first_page_response.headers.get("X-Pagination-Page-Count", 1))

        # Set up progress tracking
        completed_tasks = 0
        lock = asyncio.Lock()
        progress_text = st.empty()

        async def progress_wrapper(url, page, total):
            nonlocal completed_tasks
            response = await api_call_wrapper(client, url, headers=ss.user_headers, use_cache=False)
            result = response.json()
            async with lock:
                completed_tasks += 1
                progress_text.write(f"Trakt History: Retrieved {completed_tasks + 1} of {total} pages.")
            return result

        # Fetch all pages concurrently with progress tracking
        tasks = [progress_wrapper(f"{base_url}&page={page}", page, total_pages) for page in range(2, total_pages + 1)]
        results = await asyncio.gather(*tasks)
        all_data = first_page + [item for page_result in results for item in page_result]

    st.write(time.time() - tt, "Load Trakt History")

    df = pd.json_normalize(all_data)

    df["movie.released"] = pd.to_datetime(df["movie.released"])
    df["movie.released"] = df["movie.released"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    consolidate_cols = [
        ("movie.title", "show.title", "title"),
        ("movie.country", "show.country", "country"),
        ("movie.genres", "show.genres", "genres"),
        ("movie.status", "show.status", "status"),
        ("movie.runtime", "episode.runtime", "runtime"),
        ("movie.released", "episode.first_aired", "released"),
        ("movie.overview", "show.overview", "overview"),
        ("movie.ids.trakt", "show.ids.trakt", "show_trakt_id"),
        ("movie.ids.tmdb", "show.ids.tmdb", "show_tmdb_id"),
        ("movie.ids.imdb", "show.ids.imdb", "show_imdb_id"),
        ("movie.ids.slug", "show.ids.slug", "trakt_slug"),
    ]

    for movie_col, show_col, new_col in consolidate_cols:
        df[new_col] = df[movie_col].combine_first(df[show_col])
        df = df.drop(columns=[movie_col, show_col])

    df = df.rename(
        columns={
            "id": "event_id",
            "type": "media_type",
            "episode.season": "season_num",
            "episode.number": "ep_num",
            "episode.title": "ep_title",
            "episode.number_abs": "ep_num_abs",
            "episode.overview": "ep_overview",
            "show.aired_episodes": "total_episodes",
            "show.first_aired": "show_released",
        }
    )

    # if test:
    #     df = df.sample(50, axis=0).reset_index(drop=True)

    cur_col_dict = {k: v for k, v in col_dict.items() if k in df.columns}
    df = df.astype(cur_col_dict)

    df = adjust_wgm_runtime(df)

    df["trakt_url"] = np.where(
        df["media_type"] == "movie",
        "https://trakt.tv/movies/" + df["trakt_slug"],
        "https://trakt.tv/shows/"
        + df["trakt_slug"]
        + "/seasons/"
        + df["season_num"].astype(str)
        + "/episodes/"
        + df["ep_num"].astype(str),
    )

    df["country"] = df["country"].replace(load_country_codes())

    col_dict_1 = {k: v for k, v in col_dict.items() if k in df.columns}
    # df = df[col_dict_1.keys()]
    df = df.astype(col_dict_1)

    return df


async def get_new_imdb_data(df):

    df = df.dropna(subset=["show_imdb_id"]).drop_duplicates(subset=["show_imdb_id"])

    existing_imdb_ids = filter_new_data(df, "show_imdb_id", "imdb_media", "show_imdb_id")

    df_to_fetch = df[~df["show_imdb_id"].isin(existing_imdb_ids)]

    if not df_to_fetch.empty:

        imdb_data = await add_imdb_data(df_to_fetch)

        imdb_df = pd.DataFrame(imdb_data)
        imdb_df["show_imdb_id"] = df_to_fetch["show_imdb_id"].values

        col_dict_1 = {k: v for k, v in col_dict.items() if k in imdb_df.columns}
        # df = df[col_dict_1.keys()]
        imdb_df = imdb_df.astype(col_dict_1)

        return imdb_df

    else:
        st.info("No new values in IMDB")
        return None


async def get_new_tmdb_data(df):

    df = df.dropna(subset=["show_tmdb_id"]).drop_duplicates(subset=["show_tmdb_id"])

    existing_tmdb_ids = filter_new_data(df, "show_tmdb_id", "tmdb_media", "show_tmdb_id")

    df_to_fetch = df[~df["show_tmdb_id"].isin(existing_tmdb_ids)]

    if not df_to_fetch.empty:

        tmdb_data = await add_tmdb_data(df_to_fetch)

        tmdb_df = pd.DataFrame(tmdb_data)
        tmdb_df["show_tmdb_id"] = df_to_fetch["show_tmdb_id"].values
        tmdb_df["tmdb_language"] = tmdb_df["tmdb_language"].replace(load_tmdb_lang_codes())

        col_dict_1 = {k: v for k, v in col_dict.items() if k in tmdb_df.columns}
        # st.write(col_dict_1)
        # df = df[col_dict_1.keys()]
        tmdb_df = tmdb_df.astype(col_dict_1)
        # st.write(tmdb_df)

        return tmdb_df

    else:
        st.info("No new values in TMDB")
        return None


async def build_watch_history(trakt_id, test=True):

    # check_value_exists("users", "trakt_uuid", ss.trakt_uuid)

    # st.write(
    #     db_o3.read_table_df(
    #         "tmdb_media",
    #     )
    # )

    user_df = pd.DataFrame(
        {
            "trakt_user_id": pd.Series([ss.trakt_user_id], dtype="string"),
            "trakt_uuid": pd.Series([ss.trakt_uuid], dtype="string"),
            "trakt_auth_token": pd.Series([ss.token["access_token"]], dtype="string"),
            "last_db_update": pd.Series([dt.utcnow() - relativedelta(days=10)], dtype="datetime64[ns]"),
        }
    )

    # st.write(user_df.to_dict("records"))
    add_data(user_df, ss.trakt_uuid, "users", operation="upsert")

    # st.write(get_column_value("users", "trakt_user_id", "maverick0213", "last_db_update"))

    # last_db_update = get_column_value("users", "trakt_user_id", "maverick0213", "last_db_update")

    trakt_df = await trakt_history_and_data(shallow=False)
    st.write(trakt_df)
    new_imdb_df = await get_new_imdb_data(trakt_df)
    new_tmdb_df = await get_new_tmdb_data(trakt_df)

    # st.write(new_imdb_df.head(1).to_dict("records"))
    # st.write(new_tmdb_df.head(1).to_dict("records"))

    if not new_imdb_df is None:
        add_data(new_imdb_df, ss.trakt_uuid, "imdb_media")
    if not new_tmdb_df is None:
        add_data(new_tmdb_df, ss.trakt_uuid, "tmdb_media")
    add_data(trakt_df, ss.trakt_uuid, "trakt_media", operation="sync")
    add_data(trakt_df, ss.trakt_uuid, "user_watch_history", operation="sync")


async def build_ratings(trakt_id):
    # No cases verified where show or episode have been rated

    ratings = await fetch_trakt_ratings(trakt_id)
    all_ratings = []
    for item in ratings:
        type_ = item.get("type")
        if type_ == "movie":
            get_show = "movie"
        else:
            get_show = "show"
        all_ratings.append(
            {
                "show_trakt_id": item.get(get_show).get("ids").get("trakt"),
                "media_type": type_,
                "ep_num": item.get("episode", {}).get("number"),
                "season_num": item.get("season", {}).get("number"),
                "user_rating": item.get("rating"),
            }
        )
    ratings_df = pd.DataFrame(all_ratings)
    cur_col_dict = {k: v for k, v in col_dict.items() if k in ratings_df.columns}
    ratings_df = ratings_df.astype(cur_col_dict)

    return ratings_df
    # all_ratings = [item for item in ratings]
    # return pd.DataFrame(all_ratings)
