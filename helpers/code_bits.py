@st.cache_data
def movies_watchtime(trakt_id):
    return (
        req.get(f"{base_url}users/{trakt_id}/stats", headers=headers).json()["movies"][
            "minutes"
        ]
        // 60
    )


@st.cache_data
def shows_watchtime(trakt_id):
    return (
        req.get(f"{base_url}users/{trakt_id}/stats", headers=headers).json()[
            "episodes"
        ]["minutes"]
        // 60
    )


@st.cache_data
def total_watchtime(trakt_id):
    return (
        req.get(f"{base_url}users/{trakt_id}/stats", headers=headers).json()["movies"][
            "minutes"
        ]
        + req.get(f"{base_url}users/{trakt_id}/stats", headers=headers).json()[
            "episodes"
        ]["minutes"]
    ) // 60


@st.cache_data
def get_avatar(trakt_id):
    return req.get(
        f"{base_url}users/{trakt_id}/?extended=full", headers=headers
    ).json()["images"]["avatar"]["full"]


@st.cache_data
def movies_count(trakt_id):
    return req.get(f"{base_url}users/{trakt_id}/stats", headers=headers).json()[
        "movies"
    ]["watched"]


@st.cache_data
def shows_count(trakt_id):
    return req.get(f"{base_url}users/{trakt_id}/stats", headers=headers).json()[
        "shows"
    ]["watched"]


@st.cache_data
def episodes_count(trakt_id):
    return req.get(f"{base_url}users/{trakt_id}/stats", headers=headers).json()[
        "episodes"
    ]["watched"]


@st.cache_data
def ratings_count(trakt_id):
    return req.get(f"{base_url}users/{trakt_id}/stats", headers=headers).json()[
        "ratings"
    ]["total"]


@st.cache_data
def average_rating(trakt_id):
    ratings_data = req.get(f"{base_url}users/{trakt_id}/stats", headers=headers).json()[
        "ratings"
    ]["distribution"]

    total_ratings = ratings_count(trakt_id)

    if total_ratings > 0:
        return (
            sum(int(rating) * count for rating, count in ratings_data.items())
            / total_ratings
        )
    else:
        return 0


# df_media_sum = df.groupby(["media_type"]).agg({"runtime": "sum"})
# df_media_sum

# show_only = df_show.loc[df_show.media_type == "show"]
# show_time = df_show.loc[df_show.media_type == "show"]["runtime"].sum()
# show_only["percentage_run"] = show_only["runtime"] * 100 / show_time
# show_only

# st.write(req.get(f"{func.base_url}movies/one-fine-spring-day-2001?extended=full", headers=func.headers).json())
