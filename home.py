from math import e
from re import S
from tkinter.tix import IMAGE
from streamlit_plotly_events import plotly_events
from helpers.api import (
    api_call_wrapper,
    load_media_data,
    load_ratings_data,
    add_ratings_col,
    async_wrapper_watch_history,
    stdf_info,
    tmdb_headers,
    col_dict,
)
import helpers.auth as auth
import helpers.formatting as fm
import helpers.data_wrangling as dw
import helpers.plots as plots

from datetime import datetime as dt

from db.db_o3 import (
    close_all_connections,
    create_schema,
    test_postgres_connection,
    read_table_df,
)
import pandas as pd
import numpy as np

import locale
import streamlit as st
from streamlit import session_state as ss
from streamlit_profiler import Profiler
import requests as req
import diskcache
import os

from dateutil.relativedelta import relativedelta
from datetime import datetime as dt
from datetime import timedelta

from streamlit_echarts import st_echarts
import json
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from PIL import Image


from urllib.parse import urljoin


# profiler = Profiler(async_mode="disabled")


# ----- Global variables -----

# ----- Performance Tracker (to remove) ----

# profiler.start()

# ----- UI -----

locale.setlocale(locale.LC_ALL, "en_US")

st.set_page_config(
    page_title="CharmingGraph", layout="wide", initial_sidebar_state="collapsed"
)

pg = st.navigation(
    [
        st.Page("home.py", title="Home", icon="⭕"),
        st.Page("pages/1_About.py", title="About", icon="📚"),
    ]
)

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

c1a, c1b, c1c = st.columns(3)
c2a, c2b, c2c = st.columns(3)
c3a, c3b, c3c = st.columns(3)

# ---- Convenience Functions ----

with st.sidebar.expander("Convenience Functions"):

    if st.button("Clear Cache"):
        diskcache.Cache("assets/cache").clear()

    if st.button("Recreate Schema"):
        create_schema(create=True)
        # Remember to adapt tables on Xata
    if st.button("Close All DB Connections"):
        close_all_connections()

    if st.button("Test Con"):
        test_postgres_connection()

    if st.button("Clear Token"):
        auth.clear_token()

# ---- Init Session State ----

# st.write(api.get_media_d ata('movies/knives-out-2019'))
# st.write(api.get_tmdb_media_data("318846", ext="release_dates"))
# st.write(api.get_tmdb_media_data("87083", "tv", ext="content_ratings"))

# ---- Oauth ----

# DO_AUTH = True
# if DO_AUTH:
with st.sidebar:
    pass

from streamlit_js import st_js
import json
import logging


class LS:
    BASE = "_LS_"
    counter = 1

    @classmethod
    def set(cls, key, value):
        a = cls.load_all()
        logging.info(f"JSON.stringify('{value}')")

        if type(value) == dict:
            import json

            str_value = json.dumps(value)
            st_js(
                code=f"""
                console.log('{key}')
                
                localStorage.setItem('{key}','{str_value}')
                
                """,
                key="_set_" + str(cls.counter),
            )
        else:
            st_js(
                code=f"""
                console.log('{key}')
                console.log(JSON.stringify('{value}'))
                localStorage.setItem('{key}', JSON.stringify('{value}'))
                
                """,
                key="_set_" + str(cls.counter),
            )
        a[key] = value
        cls.counter += 1

    @classmethod
    def get(cls, key, default=None):
        a = cls.load_all()
        return a.get(key, default)

    @classmethod
    def load_all(cls):
        if cls.BASE + "all" not in st.session_state:
            cls._load_all()
        return st.session_state.get(cls.BASE + "all", {})

    @classmethod
    def keys(cls):
        a = cls.load_all()
        return list(a.keys())

    @classmethod
    def delete(cls, key):
        a = cls.load_all()
        # del st.session_state[cls.BASE + 'all'][key]
        st.session_state[cls.BASE + "all"] = {k: v for k, v in a.items() if k != key}
        st_js(
            code=f"""localStorage.removeItem('{key}')""", key="_del_" + str(cls.counter)
        )
        cls.counter += 1

    @classmethod
    def _load_all(cls):
        code = """
        // Create an empty object to store all key-value pairs
        let localStorageItems = {};

        // Iterate over all keys in localStorage
        for (let i = 0; i < localStorage.length; i++) {
            let key = localStorage.key(i);
            let value = JSON.parse(localStorage.getItem(key));
            localStorageItems[key] = value;
        }

        // The `localStorageItems` object now contains all key-value pairs
        return localStorageItems;
        """
        logging.info("Loading")
        items = st_js(code=code, key="_load_all")
        try:
            st.session_state[cls.BASE + "all"] = items[0]
            return items[0]
        except:
            return {}


# Streamlit UI to test functionality

st.title("Test LocalStorage Functionality")

# Input to set a key-value pair
key = st.text_input("Key")
value = st.text_input("Value")

if st.button("Set Value"):
    LS.set(key, value)
    x = LS.get(key)
    st.toast(x)

# Retrieve value by key
if st.button("Get Value"):
    stored_value = LS.get(key, "No value found")
    st.info(f"Value for {key}: {stored_value}")

# Delete key-value from localStorage
if st.button("Delete Key"):
    LS.delete(key)
    st.success(f"Deleted {key} from localStorage")

LS.load_all()


# st.write(req.get(f"https://api.trakt.tv/users/{ss['user_id']}?extended=full", headers=ss.user_headers).json())


# else:
#     ss.trakt_user_id = "maverick0213"
#     ss.trakt_uuid = "a743c1 b7 d8d0beddad63187f471d20370da2b362"
#     ss.token = {"access_token": "adc23ab5dcb5d30e4eaf3b560dc8061deba59c0936d5076356e53c765c089c60"}
#     ss.user_headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
#         "Content-Type": "application/json",
#         "Authorization": "Bearer 0358ac08961f7378cec27ab80cfe747164aedf06ffc252bebe6563b5bc208d94",
#         "trakt-api-key": "b8f321f93f6bc1d18d08e6d90fd65c2f43ff39801caee2bc76561827d51dfe19",
#         "trakt-api-version": "2",
#     }

# st.components.v1.iframe(
#     src="http://localhost:8501/component/streamlit_oauth.authorize_button/index.html",
#     # width=300,
#     # height=150,
#     scrolling=False,
# )

# st.markdown("""
#     <script>
#         document.querySelectorAll('p').forEach(function(p) {
#             if (p.textContent.includes('Demo Account')) {
#                 p.style.color = 'red';  // Change this to the desired style
#                 p.style.fontWeight = 'bold';
#                 p.style.fontSize = '20px';
#             }
#         });
#     </script>
# """, unsafe_allow_html=True)


st.write(ss)

# ---- Load streaming data ----

if not ss.get("trakt_uuid"):

    def login_dialog():
        c1, _, c2 = st.columns([1, 0.1, 1])
        with c1:
            st.write("Login to Trakt")
            # _, c, _ = st.columns([1, 1, 1])
            # with st.container():
            auth.authenticate()
        # st.divider()
        with c2:
            st.write("Try out a demo account")
            st.button(
                "Demo Account",
                on_click=lambda: ss.update(
                    {"trakt_uuid": "a743c1b7d8d0beddad63187f471d20370da2b362"},
                    key="demo_button",
                ),
            )

    # @st.dialog("Login")
    # def login_dialog():
    #     st.write("Login to Trakt")
    #     #_, c, _ = st.columns([1, 1, 1])
    #     auth.authenticate()
    #     st.divider()
    #     st.write("Try out a demo account")
    #     #st.write(" ")
    #     # _, c2, _ = st.columns([1,1,1])
    #     # with c2:
    #     st.button(
    #         "Demo Account",
    #         on_click=lambda: ss.update({"trakt_uuid": "a743c1b7d8d0beddad63187f471d20370da2b362"}, key="demo_button"),
    #     )

    _, c, _ = st.columns([1, 2, 1])
    with c:
        login_dialog()
else:
    user_details = req.get(
        f"https://api.trakt.tv/users/{ss['trakt_user_id']}?extended=full",
        headers=ss.user_headers,
    ).json()

    # st.write(user_details)

    name = user_details.get("name")
    avatar_url = user_details.get("images").get("avatar").get("full")

    st.sidebar.image(avatar_url, use_column_width=True)
    with st.sidebar:
        fm.center_text(name)

    from_file = c3a.selectbox("From File?", [True, False], index=0)
    test = c3b.selectbox("Test data?", [True, False], index=0)

    # ---- Filer Data Options ----

    if "most_recent_widget" not in ss:
        ss["most_recent_widget"] = None

    def dur_recent():
        ss["most_recent_widget"] = "duration"
        ss.year = None

    def year_recent():
        ss["most_recent_widget"] = "year"
        ss.duration = None

    dur_dict = {
        "Last 1 Month": relativedelta(months=1),
        "Last 3 Months": relativedelta(months=3),
        "Last 6 Months": relativedelta(months=6),
        "Last 1 Year": relativedelta(years=1),
        "All Time": None,
    }

    if not from_file:
        async_wrapper_watch_history(True, test)

    ratings_df = async_wrapper_watch_history(False, test)

    # df = load_media_data(from_file=from_file, test=test)
    # ratings_df = load_ratings_data(from_file=from_file)

    # add_data(df, ss.trakt_uuid, MediaData)
    # add_data(df, ss.trakt_uuid, UserWatchHistory)

    # upsert_user_watch_history(df, ss.trakt_uuid)
    # df[['event_id', 'trakt_url', '']]

    # st.write("Raw Data 1")
    # dw.format_df(df, {"small": ["trakt_url"]})

    # dw.format_df(
    #     join_tables('tmdb_media', 'show_tmdb_id', 'trakt_media', 'show_tmdb_id'), agg=True
    # )
    # import pickle

    # with open("assets/country_codes.pkl", "rb") as f:
    #     country_codes = pickle.load(f)
    #     # st.write(country_codes)

    # country_codes_custom = {
    #     "af": "Afghanistan",
    #     "al": "Albania",
    #     "dz": "Algeria",
    #     "as": "American Samoa",
    #     "ad": "Andorra",
    #     "ao": "Angola",
    #     "ai": "Anguilla",
    #     "aq": "Antarctica",
    #     "ag": "Antigua and Barbuda",
    #     "ar": "Argentina",
    #     "am": "Armenia",
    #     "aw": "Aruba",
    #     "au": "Australia",
    #     "at": "Austria",
    #     "az": "Azerbaijan",
    #     "bs": "The Bahamas",
    #     "bh": "Bahrain",
    #     "bd": "Bangladesh",
    #     "bb": "Barbados",
    #     "by": "Belarus",
    #     "be": "Belgium",
    #     "bz": "Belize",
    #     "bj": "Benin",
    #     "bm": "Bermuda",
    #     "bt": "Bhutan",
    #     "bo": "Bolivia",
    #     "ba": "Bosnia and Herzegovina",
    #     "bw": "Botswana",
    #     "bv": "Bouvet Island",
    #     "br": "Brazil",
    #     "io": "British Indian Ocean Territory",
    #     "bn": "Brunei",
    #     "bg": "Bulgaria",
    #     "bf": "Burkina Faso",
    #     "bi": "Burundi",
    #     "cv": "Cape Verde",
    #     "kh": "Cambodia",
    #     "cm": "Cameroon",
    #     "ca": "Canada",
    #     "ky": "Cayman Islands",
    #     "cf": "Central African Republic",
    #     "td": "Chad",
    #     "cl": "Chile",
    #     "cn": "China",
    #     "cx": "Christmas Island",
    #     "co": "Colombia",
    #     "km": "Comoros",
    #     "cg": "Congo",
    #     "cd": "Democratic Republic of the Congo",
    #     "ck": "Cook Islands",
    #     "cr": "Costa Rica",
    #     "hr": "Croatia",
    #     "cu": "Cuba",
    #     "cy": "Cyprus",
    #     "cz": "Czech Republic",
    #     "ci": "Ivory Coast",
    #     "dk": "Denmark",
    #     "dj": "Djibouti",
    #     "dm": "Dominica",
    #     "do": "Dominican Republic",
    #     "ec": "Ecuador",
    #     "eg": "Egypt",
    #     "sv": "El Salvador",
    #     "gq": "Equatorial Guinea",
    #     "er": "Eritrea",
    #     "ee": "Estonia",
    #     "sz": "Eswatini",
    #     "et": "Ethiopia",
    #     "fk": "Falkland Islands",
    #     "fo": "Faroe Islands",
    #     "fj": "Fiji",
    #     "fi": "Finland",
    #     "fr": "France",
    #     "gf": "French Guiana",
    #     "pf": "French Polynesia",
    #     "tf": "French Southern Territories",
    #     "ga": "Gabon",
    #     "gm": "The Gambia",
    #     "ge": "Georgia",
    #     "de": "Germany",
    #     "gh": "Ghana",
    #     "gi": "Gibraltar",
    #     "gr": "Greece",
    #     "gl": "Greenland",
    #     "gd": "Grenada",
    #     "gp": "Guadeloupe",
    #     "gu": "Guam",
    #     "gt": "Guatemala",
    #     "gn": "Guinea",
    #     "gw": "Guinea-Bissau",
    #     "gy": "Guyana",
    #     "ht": "Haiti",
    #     "va": "Vatican City",
    #     "hn": "Honduras",
    #     "hk": "Hong Kong",
    #     "hu": "Hungary",
    #     "is": "Iceland",
    #     "in": "India",
    #     "id": "Indonesia",
    #     "ir": "Iran",
    #     "iq": "Iraq",
    #     "ie": "Ireland",
    #     "il": "Israel",
    #     "it": "Italy",
    #     "jm": "Jamaica",
    #     "jp": "Japan",
    #     "jo": "Jordan",
    #     "kz": "Kazakhstan",
    #     "ke": "Kenya",
    #     "ki": "Kiribati",
    #     "kp": "North Korea",
    #     "kr": "South Korea",
    #     "kw": "Kuwait",
    #     "kg": "Kyrgyzstan",
    #     "la": "Laos",
    #     "lv": "Latvia",
    #     "lb": "Lebanon",
    #     "ls": "Lesotho",
    #     "lr": "Liberia",
    #     "ly": "Libya",
    #     "li": "Liechtenstein",
    #     "lt": "Lithuania",
    #     "lu": "Luxembourg",
    #     "mo": "Macau",
    #     "mg": "Madagascar",
    #     "mw": "Malawi",
    #     "my": "Malaysia",
    #     "mv": "Maldives",
    #     "ml": "Mali",
    #     "mt": "Malta",
    #     "mh": "Marshall Islands",
    #     "mq": "Martinique",
    #     "mr": "Mauritania",
    #     "mu": "Mauritius",
    #     "yt": "Mayotte",
    #     "mx": "Mexico",
    #     "fm": "Micronesia",
    #     "md": "Moldova",
    #     "mc": "Monaco",
    #     "mn": "Mongolia",
    #     "me": "Montenegro",
    #     "ms": "Montserrat",
    #     "ma": "Morocco",
    #     "mz": "Mozambique",
    #     "mm": "Myanmar",
    #     "na": "Namibia",
    #     "nr": "Nauru",
    #     "np": "Nepal",
    #     "nl": "Netherlands",
    #     "nc": "New Caledonia",
    #     "nz": "New Zealand",
    #     "ni": "Nicaragua",
    #     "ne": "Niger",
    #     "ng": "Nigeria",
    #     "nu": "Niue",
    #     "nf": "Norfolk Island",
    #     "mk": "North Macedonia",
    #     "mp": "Northern Mariana Islands",
    #     "no": "Norway",
    #     "om": "Oman",
    #     "pk": "Pakistan",
    #     "pw": "Palau",
    #     "ps": "Palestine",
    #     "pa": "Panama",
    #     "pg": "Papua New Guinea",
    #     "py": "Paraguay",
    #     "pe": "Peru",
    #     "ph": "Philippines",
    #     "pn": "Pitcairn Islands",
    #     "pl": "Poland",
    #     "pt": "Portugal",
    #     "pr": "Puerto Rico",
    #     "qa": "Qatar",
    #     "ro": "Romania",
    #     "ru": "Russia",
    #     "rw": "Rwanda",
    #     "re": "Réunion",
    #     "sh": "Saint Helena",
    #     "kn": "Saint Kitts and Nevis",
    #     "lc": "Saint Lucia",
    #     "vc": "Saint Vincent and the Grenadines",
    #     "ws": "Samoa",
    #     "sm": "San Marino",
    #     "st": "Sao Tome and Principe",
    #     "sa": "Saudi Arabia",
    #     "sn": "Senegal",
    #     "rs": "Serbia",
    #     "sc": "Seychelles",
    #     "sl": "Sierra Leone",
    #     "sg": "Singapore",
    #     "sk": "Slovakia",
    #     "si": "Slovenia",
    #     "sb": "Solomon Islands",
    #     "so": "Somalia",
    #     "za": "South Africa",
    #     "ss": "South Sudan",
    #     "es": "Spain",
    #     "lk": "Sri Lanka",
    #     "sd": "Sudan",
    #     "sr": "Suriname",
    #     "sj": "Svalbard and Jan Mayen",
    #     "se": "Sweden",
    #     "ch": "Switzerland",
    #     "sy": "Syria",
    #     "tw": "Taiwan",
    #     "tj": "Tajikistan",
    #     "tz": "Tanzania",
    #     "th": "Thailand",
    #     "tl": "Timor-Leste",
    #     "tg": "Togo",
    #     "tk": "Tokelau",
    #     "to": "Tonga",
    #     "tt": "Trinidad and Tobago",
    #     "tn": "Tunisia",
    #     "tr": "Turkey",
    #     "tm": "Turkmenistan",
    #     "tc": "Turks and Caicos Islands",
    #     "tv": "Tuvalu",
    #     "ug": "Uganda",
    #     "ua": "Ukraine",
    #     "ae": "United Arab Emirates",
    #     "gb": "United Kingdom",
    #     "us": "United States",
    #     "um": "United States Minor Outlying Islands",
    #     "uy": "Uruguay",
    #     "uz": "Uzbekistan",
    #     "vu": "Vanuatu",
    #     "ve": "Venezuela",
    #     "vn": "Vietnam",
    #     "vg": "British Virgin Islands",
    #     "vi": "U.S. Virgin Islands",
    #     "wf": "Wallis and Futuna",
    #     "ye": "Yemen",
    #     "zm": "Zambia",
    #     "zw": "Zimbabwe",
    # }
    # ###

    # # st.write(country_codes_custom.keys() == country_codes.keys())

    # if st.button("save"):
    #     with open("assets/country_codes_custom.pkl", "wb") as f:
    #         pickle.dump(country_codes_custom, f)

    df = read_table_df(
        "users",
        filters={"trakt_user_id": "maverick0213"},
        order_by="user_watch_history.watched_at",
        order_desc=True,
        joins=[
            ("users", "user_watch_history", "trakt_uuid", "trakt_uuid"),
            ("user_watch_history", "trakt_media", "trakt_url", "trakt_url"),
            ("trakt_media", "tmdb_media", "show_tmdb_id", "show_tmdb_id"),
            ("trakt_media", "imdb_media", "show_imdb_id", "show_imdb_id"),
        ],
    )

    df = df.astype(col_dict)

    def convert_to_int(x):
        if x is not None and not pd.isnull(x):
            return int(x)

    # df = read_table_df("trakt_media")

    # st.write("Read Table", df)

    # Get the columns and their types
    # columns = df.dtypes.reset_index()
    # st.write(columns)

    # df_test = read_table_df("user_watch_history")

    # st.write("Test Read - 1 Table", df_test)

    # st.write(
    #     read_table_df(
    #         'trakt_media',
    #         joins = [
    #             ('tmdb_media', 'show_tmdb_id', 'show_tmdb_id')
    #         ],
    #         filters = {
    #             'tmdb_media.tmdb_keywords': ['climate change'],
    #             "media_type": "movie",
    #         }
    #     )
    # )

    duration_select = c3a.selectbox(
        "Duration",
        list(dur_dict.keys()),
        index=None,
        on_change=dur_recent,
        key="duration",
    )

    year_select = c3b.selectbox(
        "Year",
        sorted(df["watched_at"].dt.year.unique().tolist()),
        index=None,
        on_change=year_recent,
        key="year",
    )

    if duration_select == None and year_select == None:
        pass
    elif ss.most_recent_widget == "duration":
        if duration_select != "All Time":
            df = df[df.watched_at >= dt.now() - dur_dict[duration_select]]
    else:
        df_original = df.copy(deep=True)
        df = df[df.watched_at.dt.year == year_select]
        c1, c2, _ = st.columns([1, 1, 6])
        c1.write(
            f"{df.runtime.sum() // 1440} days {df.runtime.sum() % 1440 // 60} hours"
        )
        c2.write(
            f"{df_original.runtime.sum() // 1440} days {df_original.runtime.sum() % 1440 // 60} hours",
        )
        c1.write(f"{df.runtime.sum() // 60} hours")
        c2.write(f"{df_original.runtime.sum() // 60} hours")
        st.write(f"{round(df.runtime.sum()*100/df_original.runtime.sum(), 2)}%")

    # with time_container:
    #     st.write(f"{df.runtime.sum() // 60} hours")
    #     st.write(round(df.runtime.sum() * 100 / total_watchtime, 2))

    with st.expander("Raw Data"):
        dw.format_df(df, agg=True)
        stdf_info(df)

    # st.write("Watch Count")
    df_watch_count = dw.get_watch_count(df)
    with st.expander("Watch Count"):
        dw.format_df(df_watch_count, agg=True)

    # st.write("Seasons DF")
    df_season = dw.group_by_season(df_watch_count)
    df_season = add_ratings_col(df_season, ratings_df, merge_col="season_num")
    with st.expander("Seasons DF"):
        dw.format_df(df_season, agg=True)

    # movie_tmdb_id = df_season[df_season.media_type == "movie"].iloc[0]["show_tmdb_id"]

    # import heapq

    #     return image_urls

    # st.write(get_poster_images(movie_tmdb_id, "movie"))
    # for img in get_poster_images(movie_tmdb_id, "movie"):
    #     st.image(img)

    # st.write(movie_tmdb_id)

    # endpoint = req.get(
    #     f"http://api.themoviedb.org/3/movie/{movie_tmdb_id}/images?include_image_language=en,null",
    #     headers=tmdb_headers,
    # ).json()["posters"][0]["file_path"]

    # st.write(endpoint)

    # st.image(f"http://image.tmdb.org/t/p/w500/{endpoint}")

    # if st.button("Export Data"):
    #     exp_df = df_season.rename(
    #         columns={
    #             "show_tmdb_id": "tmdbID",
    #             "show_imdb_id": "imdbID",
    #             "title": "Title",
    #             "watched_at": "WatchedDate",
    #             "user_rating": "Rating10",
    #         }
    #     )
    #     # exp_df["tmdbID"] = exp_df["tmdbID"].apply(lambda x: convert_to_int(x))
    #     exp_df[["tmdbID", "imdbID", "Title", "WatchedDate", "Rating10"]].loc[exp_df.media_type == "movie"].to_csv(
    #         "letterboxd.csv", index=False
    #     )

    # Export imdb_id, watched_at and user_rating with media type as movies as a csv from the dataframe

    # st.write(df_season[["show_imdb_id, watched_at, user_rating"]])

    # @st.fragment
    # def export_to_movielens(df, filename="movielens.csv"):
    #     if st.button("Export to Movielens"):

    #         # if df.user_rating.isna().sum() > 0:
    #         #     st.write(df[df.media_type == "movie"][df.user_rating.isna()])

    #         # df2['imdb'] = df['movie'].apply(lambda x: get_imbd(x))
    #         # df2 = df2[~df2.imdb.isna()]

    #         df2 = df[df.media_type == "movie"]

    #         df2["watched_at"] = dt.today().strftime("%d-%m-%Y")
    #         df2["user_rating"] = df2["user_rating"].astype(int)
    #         df2 = df2[["show_imdb_id", "user_rating", "watched_at"]]
    #         df2.columns = ["Const", "Your Rating", "Date Rated"]

    #         x = df2.columns.tolist()
    #         extra_columns = [
    #             "Title",
    #             "URL",
    #             "Title Type",
    #             "IMDb Rating",
    #             "Runtime (mins)",
    #             "Year",
    #             "Genres",
    #             "Num Votes",
    #             "Release Date",
    #             "Directors",
    #         ]
    #         x.extend(extra_columns)

    #         df2 = df2.reindex(columns=x, fill_value="")
    #         df2.to_csv(filename, index=False)

    # export_to_movielens(df_season)

    # @st.fragment
    # def list_titles():

    #     df = df_season[~df_season.user_rating.isna()]
    #     df = df[df.media_type == "movie"]

    #     title_list = "".join([f"{title} - {rating}, " for title, rating in zip(df.title, df.user_rating)])
    #     st.write(title_list)

    # list_titles()

    df_show = dw.group_by_show(df_watch_count)
    with st.expander("Shows DF"):
        dw.format_df(df_show, agg=True)
        st.write(df_show.head(10))

    @st.fragment
    def limited_df():
        columns_list = st.multiselect(
            "Selected Columns",
            df_show.columns.tolist(),
            ["title", "media_type", "watchtime"],
        )
        dw.format_df(df_show[columns_list], agg=True, max_width=300)

    with st.expander("Limited DF"):
        limited_df()

    # ---- Visualization ----

    # @st.fragment
    # def generate_grap h(df_, col, sum_100, type_, media_type):
    #     try:
    #         return plots.generate_chart(df_, col, type_, sum_100, media_type)
    #     except Exception as e:
    #         st.write(f"Error generating graph for {col}: {e}")
    #         return None  # Return None in case of error

    # @st.fragment
    # def generate_graph(df_, col, sum_100, type_, media_type):
    #     try:
    #         return plots.plot_with_echarts(df_, col)
    #     except Exception as e:
    #         st.write(f"Error generating graph for {col}: {e}")
    #         return None  # Return None in case of error

    # import streamlit as st
    # from streamlit_echarts import st_echarts

    # Usage example:
    # fig = generate_heatmap(df)
    # st.plotly_chart(fig, use_container_width=True)

    # Display the heatmap in Streamlit

    # st.write(event)

    # @st.fragment
    # def example_plotly_events():
    #     # fig = px.line(x=[1, 2, 3], y=[1, 3, 2])
    #     fig = plots.generate_heatmap(df)
    #     selected_points = plotly_events(fig, override_height=800)

    #     # st.write(selected_points)

    # example_plotly_events()``

    import plotly.graph_objects as go
    from datetime import datetime

    st.write(df)

    # st.plotly_chart(go.Figure())

    # st.plotly_chart(plots.ex1())

    # st.plotly_chart(plots.ex2())

    # st.plotly_chart(plots.generate_month_heatmap(df, 2024, 10))

    # st.plotly_chart(plots.generate_month_heatmap_alt3(df, 2024, 10))

    # st.plotly_chart(plots.generate_month_heatmap_subplots(df, 2024, 10))

    st.plotly_chart(plots.generate_month_heatmap_with_features_no_invalid(df, 2024, 10))

    # st.plotly_chart(plots.generate_month_heatmap_with_subplots(df, 2024, 10))

    @st.fragment
    def plot_data():

        selected_year = plots.pick_year_heatmap(df)

        with st.expander("Heatmap Original"):
            st.plotly_chart(
                plots.generate_heatmap(df, selected_year),
                use_container_width=True,
            )

        fig = plots.generate_heatmap(df, selected_year)
        # ontainer = st.empty()

        # with container:
        #     selected_points = plotly_events(fig, override_height=700, key="1")

        # x = selected_points

        # Show a Poster if Click Registered
        # if selected_points:
        #     with container:
        cols = st.columns([5, 1])
        with cols[0]:
            selected_points = plotly_events(
                fig, override_height=700, hover_event=True, key="2"
            )
        with cols[1]:
            # if selected_points == []:
            #     selected_points = x

            try:
                chosen_date = dt.strptime(
                    f"{selected_points[0]['x']} {selected_points[0]['y']} {selected_year}",
                    "%d %b %Y",
                )

                next_day = chosen_date + timedelta(days=1)

                if chosen_date <= dt.today():

                    # st.write(df.groupby("title").agg({"runtime": "sum"})[["runtime",]])

                    df_display = (
                        df[(df.watched_at >= chosen_date) & (df.watched_at < next_day)]
                        .groupby("title")
                        .agg({"runtime": "sum", "tmdb_poster_url": "first"})[
                            ["runtime", "tmdb_poster_url"]
                        ]
                        .sort_values("runtime", ascending=False)
                        .iloc[:2]
                        .reset_index()
                    )

                    if not df_display.empty:

                        center_text(
                            f"On {chosen_date.strftime('%d %B %Y')}, you watched:"
                        )

                        st.text(" ")

                        def poster_card(row):
                            st.image(
                                row.tmdb_poster_url,
                                use_column_width="always",
                                caption=f"{row.title}",
                            )
                            # center_text(f"{row.title} <br> {row.runtime // 60}h {row.runtime % 60}m", font_size=14)

                        for row in df_display.itertuples():

                            @st.cache_data(show_spinner=False)
                            def load_image_bytes(url):
                                response = req.get(url)
                                if response.status_code == 200:
                                    return BytesIO(response.content)
                                return None

                            @st.cache_data(show_spinner=False)
                            def resize_image(image_bytes, height):

                                image = Image.open(image_bytes)

                                if image is not None:
                                    return image.resize(
                                        (
                                            (image.size[0] * height) // image.size[1],
                                            height,
                                        ),
                                        Image.LANCZOS,
                                    )
                                return None

                            image = load_image_bytes(row.tmdb_poster_url)
                            resized_image = resize_image(image, 250)
                            st.image(resized_image, caption=row.title)

            except ValueError:
                pass

            except IndexError:
                chosen_date = dt.today()

                # st.write(df_grouped)

    plot_data()

    columns = [
        "title",
        "media_type",
        "country",
        "tmdb_language",
        # "genres",
        "imdb_genres",
        # "tmdb_genres",
        "tmdb_networks",
        "decade",
    ]

    c4a, c4b = st.columns(2)

    grid = []
    for i in range(len(columns)):
        if i % 2 == 0:
            grid.append(c4a.container())
        else:
            grid.append(c4b.container())

    # # Function to handle click event
    # def on_plot_click(trace, points, state):
    #     if len(points.point_inds) > 0:
    #         st.session_state.click_registered = 1

    # # Placeholder to register click
    # fig.update_traces(marker=dict(size=15))

    # # Display the Plot in Streamlit
    # cols = st.columns([3, 4])
    # with cols[0]:
    #     plotly_chart = st.plotly_chart(fig, use_container_width=True)

    # # Show a Poster if Click Registered
    # if st.session_state.get("click_registered", 0) != 0:
    #     with cols[1]:
    #         st.image("https://via.placeholder.com/300", caption="Poster Displayed Here")

    # Writes a component similar to st.write()
    # fig = px.line(x=[1], y=[1])
    # selected_points = plotly_events(fig)

    # st.write(selected_points)

    # Can write inside of things using with!
    # with st.expander("Plot"):
    #     fig = px.line(x=[1], y=[1])
    #     selected_points = plotly_events(fig)

    # # Select other Plotly events by specifying kwargs
    # fig = px.line(x=[1], y=[1])
    # selected_points = plotly_events(fig, click_event=False, hover_event=True)

    @st.fragment
    def plot_with_options(df_show, catg, plot, sum_100, radio_index, gen_chart=True):
        media_radio = st.radio(
            catg.upper(),
            ["movie", "show", "all"],
            radio_index,
            horizontal=True,
            key=catg,
        )

        if gen_chart:
            return st.plotly_chart(
                plots.generate_chart(df_show, catg, plot, sum_100, media_radio)
            )
        else:
            return st.plotly_chart(
                plots.plot_runtime_by_decade(df_show, media_radio),
                use_container_width=True,
            )

    # @st.fragment
    def graphs():
        for con in grid:
            if con:
                con.empty()
        con_dict = {}
        for i, catg in enumerate(columns):
            radio_index = 2
            if catg in ["tmdb_genres", "tmdb_networks"]:
                radio_index = 1
            if catg in ["imdb_genres"]:
                plot = "bar"
            else:
                plot = "pie"
            sum_100 = False
            # if catg == "imdb_genres":
            #     sum_100 = True
            # else:
            #     sum_100 = False
            con_dict[i] = grid[i].container()
            with con_dict[i]:

                sum_100 = False
                if catg == "imdb_genres":
                    sum_100 = st.radio(
                        "Sum to 100%",
                        [True, False],
                        0,
                        horizontal=True,
                        key=f"{catg}_sum100",
                    )
                if catg != "decade":
                    plot_with_options(df_show, catg, plot, sum_100, radio_index)
                else:
                    plot_with_options(
                        df, catg, plot, sum_100, radio_index, gen_chart=False
                    )

    graphs()

    # profiler.stop()
