import pandas as pd
import streamlit as st
import requests as req
from st_aggrid import AgGrid, GridOptionsBuilder
import io

from helpers.api import col_dict

headers = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer dd9e05b92aecbe64a44f9fe2794045ccc88826e2bdf7f45729fd61048d11fb3e",
    "trakt-api-key": "7071ecc32e14fcdf7fdba0711d7fe29ed1290035bb78524abbc6913b45defc0b",
    "trakt-api-version": "2",
}


@st.fragment
def format_df(df, col_width_dict={"small": ["trakt_url"]}, agg=False, max_width=150):
    if agg:
        gb = GridOptionsBuilder.from_dataframe(df)

        # Example: Set all columns to be resizable
        gb.configure_default_column(resizable=True, filterable=True, maxWidth=max_width)

        # Configure other grid options (e.g., enabling sorting, filtering)

        # Placeholder: If you could hook into onGridReady from Python

        column_defs = []
        for col in df.columns:
            # col_def = {}
            col_def = {
                "headerName": col,
                "field": col,
                "filter": "true",
                # "resizable": "true",
                # "maxWidth": 150,
            }

            if col == "title":
                col_def["pinned"] = "left"

            # Add the column definition to the list
            column_defs.append(col_def)

        # column_widths_js = (
        #     "["
        #     + ", ".join(f"{{key: '{col_name}', newWidth: {150}}}" for col_name in df.columns)
        #     + "]"
        # )

        gridOptions = gb.build()
        # gridOptions['onGridReady']  = """
        #     function(params) {
        #         params.api.setColumnWidths([
        #             {key: 'title', newWidth: 20}
        #         ]);
        #     }
        #     """
        gridOptions["columnDefs"] = column_defs
        # gridOptions["defaultColDef"] = {
        #     "filter": "true",
        # }
        # gridOptions = {
        #     "autoSizeStrategy": {"type": "fitCellContents"},
        # }

        # Display the DataFrame with AgGrid
        AgGrid(df, gridOptions=gridOptions, enable_enterprise_modules=True)

        # small_col_width = 100
        # default_col_width = 250

        # # Define gridOptions with the dynamically created columnDefs
        # gridOptions = {
        #     # "autoSizeStrategy": {"type": "fitCellContents"},
        #     "columnDefs": column_defs,
        #     "defaultColDef": {"autoSizeStrategy": ({"type": "fitCellContents"})},
        # }

        # return AgGrid(df, gridOptions=gridOptions, allow_unsafe_jscode=True)
    else:
        col_config = {col: col for col in df.columns}
        for k in col_width_dict.keys():
            for col in col_width_dict[k]:
                col_config[col] = st.column_config.Column(width=k)

        return st.dataframe(df, column_config=col_config, use_container_width=True)


# @st.cache_data
def get_watch_count(df):

    # Determine the groupby columns based on media type
    df["grouping_col"] = df["show_trakt_id"].astype(str)

    # For episodes, adjust the grouping column
    df.loc[df["media_type"] == "episode", "grouping_col"] = (
        df["show_trakt_id"].astype(str)
        + "_"
        + df["season_num"].astype(str)
        + "_"
        + df["ep_num"].astype(str)
    )

    aggregations = {col: "first" for col in df.columns}
    aggregations.update({"event_id": "count", "watched_at": "max"})

    # Group by the generated column and aggregate
    result_df = (
        df.groupby("grouping_col")
        .agg(aggregations)
        .rename(columns={"event_id": "plays"})
        .reset_index(drop=True)
    )

    result_df["watchtime"] = result_df["plays"] * result_df["runtime"]

    new_cols = list(df.columns)
    new_cols = [col for col in new_cols if col not in ["event_id", "grouping_col"]]
    runtime_index = new_cols.index("runtime")
    new_cols[runtime_index + 1 : runtime_index + 1] = ["plays", "watchtime"]
    result_df = result_df[new_cols]

    cur_col_dict = {k: v for k, v in col_dict.items() if k in result_df.columns}
    result_df = result_df.astype(cur_col_dict)

    result_df = result_df.sort_values(by="watched_at", ascending=False).reset_index(
        drop=True
    )

    return result_df


# @st.cache_data
def group_by_season(df):
    none_df = df[df["ep_num"].isnull()]
    grouped_df = df.dropna(subset=["show_trakt_id", "ep_num"])
    grouped_df["media_type"] = "season"  # Set media type before any operations

    # Define aggregations
    aggregations = {column: "first" for column in df.columns}
    aggregations.update(
        {
            "runtime": "sum",
            "watchtime": "sum",
            "plays": "sum",
            "watched_at": "max",
            "released": "min",
        }
    )

    season_df = grouped_df.groupby(["show_trakt_id", "season_num"]).agg(aggregations)
    season_df = pd.concat([season_df, none_df], ignore_index=True)
    season_df["trakt_url"] = season_df["trakt_url"].str.split("/episodes").str[0]

    season_df = season_df.drop(
        columns=["ep_num", "ep_title", "ep_num_abs", "ep_overview"], axis=1
    )

    # season ep plays not part of this column list
    cur_col_dict = {k: v for k, v in col_dict.items() if k in season_df.columns}
    season_df = season_df.astype(cur_col_dict)

    season_df = season_df.sort_values(by="watched_at", ascending=False).reset_index(
        drop=True
    )

    return season_df


def group_by_show(df):
    none_df = df[df["ep_num"].isnull()]
    grouped_df = df.dropna(subset=["show_trakt_id", "ep_num"])
    grouped_df["media_type"] = "show"  # Set media type before any operations

    # Define aggregations
    aggregations = {column: "first" for column in df.columns}
    aggregations.update(
        {
            "runtime": "sum",
            "watchtime": "sum",
            "plays": "sum",
            "watched_at": "max",
            "released": "min",
        }
    )

    shows_df = grouped_df.groupby(["show_trakt_id"]).agg(aggregations)
    shows_df = pd.concat([shows_df, none_df], ignore_index=True)
    shows_df["trakt_url"] = shows_df["trakt_url"].str.split("/seasons").str[0]

    shows_df = shows_df.drop(
        columns=["ep_num", "season_num", "ep_title", "ep_num_abs", "ep_overview"],
        axis=1,
    )

    # season ep plays not part of this column list
    cur_col_dict = {k: v for k, v in col_dict.items() if k in shows_df.columns}
    shows_df = shows_df.astype(cur_col_dict)

    shows_df = shows_df.sort_values(by="watched_at", ascending=False).reset_index(
        drop=True
    )

    return shows_df


@st.cache_data
def wrangle_data_for_plots(
    df, column_name, media_type="all", sum_100=False, n=10, others_threshold=1
):
    """
    Wrangles data by filtering based on media type, handling missing values, and grouping smaller categories.

    Args:
        df (DataFrame): The dataframe to be processed.
        column_name (str): Column to group by.
        media_type (str): Filter based on media type ('all', 'show', 'movie').
        sum_100 (bool): Sum watchtime for exploded lists.
        n (int): Number of top categories to retain.
        others_threshold (float): Threshold for grouping smaller categories into 'Others'.

    Returns:
        DataFrame: Wrangled dataframe with watchtime and percentage calculations.
    """
    assert media_type in ["all", "show", "movie"]
    df_copy = df.copy()

    # Filter by media_type if required
    if media_type != "all":
        df_copy = df_copy[df_copy["media_type"] == media_type]

    total_time = df_copy["watchtime"].sum()

    # Handle NaN values and empty lists as 'Unknown'
    has_unknowns = (
        df_copy[column_name].isna().any()
        or df_copy[column_name].apply(lambda x: x == [] or x == "" or x == " ").any()
    )
    if has_unknowns:
        df_copy[column_name] = df_copy[column_name].apply(
            lambda x: (
                "Unknown"
                if (not isinstance(x, list) and pd.isnull(x))
                or (isinstance(x, list) and not x)
                else x
            )
        )

    # Handle list columns
    if df_copy[column_name].apply(isinstance, args=(list,)).any():
        df_copy = df_copy.explode(column_name)
        if sum_100:
            total_time = df_copy["watchtime"].sum()

    # Group by the specified column and aggregate on watchtime
    df_grouped = (
        df_copy.groupby(column_name)
        .agg({"watchtime": "sum"})
        .sort_values("watchtime", ascending=False)
        .reset_index()
    )

    # Percentage calculations and formatting
    df_grouped["%_tw"] = (df_grouped["watchtime"] / total_time) * 100

    # Group smaller categories into 'Others'
    others_mask = (
        (df_grouped.index >= n) | (df_grouped["%_tw"] < others_threshold)
    ) & (df_grouped[column_name] != "Unknown")
    others_sum = df_grouped.loc[others_mask, "watchtime"].sum()
    if others_sum > 0:
        others_percentage = others_sum / total_time * 100
        top_categories = df_grouped.loc[~others_mask]
        top_categories = pd.concat(
            [
                top_categories,
                pd.DataFrame(
                    [
                        {
                            column_name: "Others",
                            "watchtime": others_sum,
                            "%_tw": others_percentage,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        df_grouped = top_categories

    return df_grouped


@st.cache_data
def process_for_echarts(df, column_name):
    """
    Preprocess the dataframe for use with echarts.
    Groups data by the given column and formats it for the echarts input.
    Sorts the data in decreasing order of watchtime, with 'Others' and 'Unknown' at the end.

    Args:
        df (DataFrame): Wrangled dataframe.
        column_name (str): Column to be processed for echarts.

    Returns:
        list: List of dictionaries in the format [{'value': watchtime, 'name': column_value}, ...].
    """
    data = df.groupby(column_name)[["watchtime", "%_tw"]].sum().reset_index()
    data_list = data.apply(
        lambda row: {
            "value": row["watchtime"],
            "%_tw": row["%_tw"],
            "name": row[column_name],
        },
        axis=1,
    ).tolist()

    # Sort data in decreasing order of watchtime, with 'Others' and 'Unknown' at the end
    data_list = sorted(
        data_list, key=lambda x: (x["name"] not in ["Others", "Unknown"], -x["value"])
    )
    data_list.append(
        data_list.pop(
            data_list.index(next(filter(lambda x: x["name"] == "Others", data_list)))
        )
    )
    data_list.append(
        data_list.pop(
            data_list.index(next(filter(lambda x: x["name"] == "Unknown", data_list)))
        )
    )

    # st.write(data_list)

    return data_list


# def group_by_season(df):
#     none_df = df[df["ep_num"].isnull()]
#     grouped_df = df.dropna(subset=["show_trakt_id", "ep_num"])
#     grouped_df['media_type'] = 'season'  # Set media type before any operations

#     # Define aggregations
#     aggregations = {column: "first" for column in df.columns if column not in ["runtime", "watchtime", "ep_plays"]}
#     aggregations.update({
#         "runtime": "sum",
#         "watchtime": "sum",
#         "ep_plays": "sum",
#         "last_watched_at": "last",
#         "last_updated_at": "last"
#     })

#     season_df = grouped_df.groupby(["show_trakt_id", "season_num"]).agg(aggregations)
#     season_df = pd.concat([season_df, none_df], ignore_index=True)
#     season_df['trakt_url'] = season_df['trakt_url'].str.split("/episodes").str[0]

#     return season_df.sort_values(by="last_watched_at", ascending=False).reset_index(drop=True)

# def group_by_show(df):
#     none_df = df[df["ep_num"].isnull()]
#     grouped_df = df.dropna(subset=["show_trakt_id", "ep_num"])
#     grouped_df['media_type'] = 'show'  # Set media type before any operations

#     # Define aggregations
#     aggregations = {column: "first" for column in df.columns if column not in ["runtime", "watchtime"]}
#     aggregations.update({
#         "runtime": "sum",
#         "watchtime": "sum",
#         "last_watched_at": "last",
#         "last_updated_at": "last"
#     })

#     shows_df = grouped_df.groupby(["show_trakt_id"]).agg(aggregations)
#     shows_df = pd.concat([shows_df, none_df], ignore_index=True)
#     shows_df['trakt_url'] = shows_df['trakt_url'].str.split("/seasons").str[0]

#     return shows_df.sort_values(by="last_watched_at", ascending=False).reset_index(drop=True)


# # Define the aggregations for each column
# aggregations = {column: "first" for column in df.columns}
# aggregations["runtime"] = "sum"
# aggregations["watchtime"] = "sum"

# shows_df = df.groupby(["show_trakt_id"]).agg(aggregations)

# shows_df.loc[shows_df["media_type"] == "ep_num", "media_type"] = "show"
# shows_df['trakt_url'] = shows_df['trakt_url'].str.split("/seasons").str[0]

# new_cols = list(df.columns)
# new_cols = [col for col in new_cols if col not in ["ep_num", "season_num", "watch_count"]]
# shows_df = shows_df[new_cols]

# shows_df = shows_df.sort_values(by="last_watched_at", ascending=False).reset_index(drop=True)

# return shows_df
