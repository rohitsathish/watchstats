# Comments
# Add code to combine all categories after the top n into an others category for a cleaner view.

from enum import unique
from tkinter import font
from matplotlib import legend
from matplotlib.pyplot import margins
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
from datetime import datetime
import calendar
import random
from plotly.subplots import make_subplots


import helpers.data_wrangling as dw

plotly_font = "Source Sans Pro"

font_attrs = {
    "title_font": dict(family=plotly_font),
    "font": dict(family=plotly_font),
    "xaxis": dict(title_font=dict(family=plotly_font)),
    "yaxis": dict(title_font=dict(family=plotly_font)),
    "hoverlabel": dict(font=dict(family=plotly_font)),
}

# 0E1117, #262730

# ---- General Functions ----


def round_percentages(values):
    total = sum(values)
    whole_numbers = [int(value) for value in values]
    remainders = sorted(
        [(i, values[i] - whole_numbers[i]) for i in range(len(values))],
        key=lambda x: x[1],
        reverse=True,
    )

    total_rounded = sum(whole_numbers)
    while total_rounded < total:
        i, _ = remainders.pop(0)
        whole_numbers[i] += 1
        total_rounded += 1

    return whole_numbers


# ---- Plotting Functions ----
# def gen_st_cols(divs=2):
#     col_dict = {}
#     for div in range(divs):
#         col_dict[div] = st.columns(2)


def pick_year_heatmap(df):
    """
    Pick a year for the heatmap.
    """
    df["year"] = df["watched_at"].dt.year
    year_options = [str(year) for year in sorted(df["year"].unique())]
    selected_year = st.selectbox(
        "Select Year", options=year_options, index=len(year_options) - 1
    )
    return selected_year


def generate_heatmap(df, selected_year):
    """
    Could convert the colorbar into hour values. Currently in minutes.
    """
    # Extract year, month, and day from 'watched_at'
    df["year"] = df["watched_at"].dt.year
    df["month"] = df["watched_at"].dt.month
    df["day"] = df["watched_at"].dt.day

    # Filter data based on selection
    filtered_df = df[df["year"] == int(selected_year)]

    # Aggregate watchtime per day
    heatmap_data = filtered_df.groupby(["month", "day"])["runtime"].sum().reset_index()

    # Create a pivot table with months as rows and days as columns
    heatmap_pivot = heatmap_data.pivot(
        index="month", columns="day", values="runtime"
    ).fillna(0)

    # Ensure all months and days are present
    all_months = list(range(1, 13))
    all_days = list(range(1, 32))
    heatmap_pivot = heatmap_pivot.reindex(
        index=all_months, columns=all_days, fill_value=0
    )

    # Calculate the 95th percentile for watchtime to set color scale
    percentile_98 = np.percentile(heatmap_pivot.values.flatten(), 98)

    # Define month names in order from January to December
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Define a custom colorscale from black to bright red
    custom_colorscale = [
        [0.0, "#262730"],  # Start of valid watchtime
        [0.1, "#872b2b"],  # Start of valid watchtime
        [1.0, "#FF4B4B"],  # End of valid watchtime
    ]

    # -- INVALID DATES --#

    # Determine invalid dates based on selected_year
    invalid_dates = {
        2: [30, 31],
        4: [31],
        6: [31],
        9: [31],
        11: [31],
    }

    # Function to check if a year is a leap year
    def is_leap_year(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    # If a specific year is selected, handle Feb 29
    year = int(selected_year)
    if not is_leap_year(year):
        invalid_dates.setdefault(2, []).append(29)

    # Create a mask for invalid dates
    invalid_mask = np.zeros(heatmap_pivot.shape, dtype=bool)
    for month, days in invalid_dates.items():
        for day in days:
            if day in all_days:
                day_index = all_days.index(day)
                month_index = all_months.index(month)
                invalid_mask[month_index, day_index] = True

    # Create a z_mask array where invalid dates are 1 and others are np.nan
    z_mask = np.where(invalid_mask, 1, np.nan)

    # Define a custom colorscale for invalid dates (transparent for valid, gray for invalid)
    invalid_colorscale = [
        [0, "rgba(0,0,0,0)"],
        [1, "black"],
    ]  # Transparent for valid dates  # Gray for invalid dates

    # Create the secondary heatmap for invalid dates
    invalid_heatmap = go.Heatmap(
        z=z_mask,
        x=heatmap_pivot.columns,
        y=month_names,
        xgap=2,
        ygap=2,
        colorscale=invalid_colorscale,
        showscale=False,
        hoverinfo="skip",  # Disable hover for invalid heatmap
        opacity=1,  # Semi-transparent to allow primary heatmap hover
        textfont={"color": "rgba(0, 0, 0, 0)"},
    )

    # **New Section: Highlight Future Dates**

    # Get today's date
    today = datetime.today()

    # Initialize future_dates dictionary
    future_dates = {}

    if year > today.year:
        # All dates are in the past
        pass  # No future dates to mark
    elif year < today.year:
        # All dates are in the past
        pass  # No future dates to mark
    else:
        # year == today.year
        for month in all_months:
            if month < today.month:
                continue  # No future dates in past months
            elif month == today.month:
                # Days after today.day are future dates
                future_days = [day for day in all_days if day > today.day]
                if future_days:
                    future_dates[month] = future_days
            else:
                # All days in future months are future dates
                future_dates[month] = all_days.copy()

    # Create a mask for future dates
    future_mask = np.zeros(heatmap_pivot.shape, dtype=bool)
    for month, days in future_dates.items():
        for day in days:
            if day in all_days and not invalid_mask[month - 1, day - 1]:
                day_index = all_days.index(day)
                month_index = all_months.index(month)
                future_mask[month_index, day_index] = True

    # Create a z_future_mask array where future dates are 1 and others are np.nan
    z_future_mask = np.where(future_mask, 1, np.nan)

    # Define a custom colorscale for future dates (transparent for valid, blue for future)
    future_colorscale = [
        [0, "black"],  # Transparent for valid dates
        [1, "lightgrey"],  # Blue for future dates
    ]

    # Create the tertiary heatmap for future dates
    future_heatmap = go.Heatmap(
        z=z_future_mask,
        x=heatmap_pivot.columns,
        y=month_names,
        xgap=2,
        ygap=2,
        colorscale=future_colorscale,
        showscale=False,
        hoverinfo="skip",  # Disable hover for future heatmap
        opacity=1,  # Semi-transparent to allow primary heatmap hover and text visibility
        textfont={"color": "rgba(0, 0, 0, 0)"},
        # textfont={"size": 15, "color": "white"},  # Set the font size and color here
    )

    # -- WATCHTIME SETUP --#

    # Create a 2D list for customdata with date strings
    customdata = []
    hovertext = []
    for month in all_months:
        customdata_row = []
        hovertext_row = []
        for day in all_days:
            try:
                date = pd.Timestamp(year=int(selected_year), month=month, day=day)
                date_str = date.strftime("%d %B %Y")
                watchtime = heatmap_pivot.loc[month, day]
                hours = watchtime // 60
                minutes = watchtime % 60
                if hours >= 1:
                    watchtime_formatted = f"{int(hours)}h {int(minutes)}m"
                else:
                    watchtime_formatted = f"{int(minutes)}m"
                if invalid_mask[month - 1, day - 1]:
                    customdata_row.append(None)
                    hovertext_row.append(None)
                else:
                    customdata_row.append(date_str)
                    hovertext_row.append(
                        f"{date_str}<br>Watchtime: {watchtime_formatted}"
                    )
            except ValueError:
                customdata_row.append("")
                hovertext_row.append("")
        customdata.append(customdata_row)
        hovertext.append(hovertext_row)

    total_watchtime_per_month = filtered_df.groupby("month")["runtime"].sum()

    # Add annotations for total watchtime per month
    annotations = []
    for i, month in enumerate(month_names):
        total_minutes = total_watchtime_per_month.get(i + 1, 0)  # Months are 1-indexed
        total_watchtime_formatted = f"{round(total_minutes / 60)}h"
        annotations.append(
            dict(
                x=31,  # Position it slightly outside the last day (31)
                y=month,
                text=total_watchtime_formatted,
                showarrow=False,
                xanchor="left",
                font=dict(size=15, color="white"),  # **Changed Font Color to Blue**
                xshift=30,  # Set the x-offset to 10 pixels
                hovertext=f"Total Watchtime in {month}: {total_watchtime_formatted}",
                hoverlabel=dict(font=dict(size=14)),
            )
        )

    annotations = []
    for i, month in enumerate(month_names):
        for j, day in enumerate(all_days):
            value = heatmap_pivot.iloc[i, j]
            if value > 30:
                annotations.append(
                    dict(
                        x=day,
                        y=month,
                        text=f"{round(value / 60)}h",
                        showarrow=False,
                        font=dict(color="white"),
                    )
                )

    # -- PRIMARY HEATMAP --#

    # Create the primary heatmap with customdata and hovertemplate
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            xgap=2,
            ygap=2,
            y=month_names,
            colorscale=custom_colorscale,  # Use the custom colorscale here
            zmin=0,
            zmax=percentile_98,  # Use the 95th percentile for color scaling
            # colorbar=dict(title="Total Watchtime"),
            # text=heatmap_pivot.applymap(lambda val: f"{round(val / 60)}h" if val > 30 else "").values,
            # texttemplate="%{text}",
            customdata=customdata,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hovertext,
            textfont={"color": "white", "family": "sans-serif"},
            hoverlabel=dict(
                font=dict(
                    size=14, family="sans-serif"
                ),  # Set the font size and family for hovertext
                bordercolor="black",
            ),
            # font=dict(family="sans-serif"),  # Set the font size, color, and family here
            # Set the font size here
            showscale=False,
            colorbar=dict(
                orientation="h",
                tickfont=dict(color="white"),
            ),
            legendgrouptitle=dict(font=dict(color="white")),
        )
    )

    fig.add_trace(future_heatmap)

    # Add the invalid heatmap to the figure
    fig.add_trace(invalid_heatmap)

    # Add the future heatmap to the figure

    # Create the final layout with annotations
    fig.update_layout(
        # title=dict(
        #     text="Watchtime Heatmap",
        #     font=dict(
        #         color="white",  # Font color for the title
        #     ),
        #     x=0.5,
        #     pad=dict(
        #         t=10,  # Padding at the top of the title (space between title and top border)
        #         b=0,  # Padding at the bottom of the title (space between title and plot area)
        #     ),  # Set the x position of the title to center align
        # ),
        xaxis=dict(
            # title="Day of Month",
            # automargin="bottom",
            tickmode="linear",
            mirror=True,
            linewidth=0,
            zeroline=False,
            tick0=1,
            showgrid=False,
            range=[0.5, 31.5],
            color="white",
            # showline=False,  # Extended range to accommodate annotations
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=month_names,  # Ensures months are ordered from Jan to Dec
            mirror=True,
            autorange="reversed",  # Flips the y-axis to have Dec at top and Jan at bottom
            linewidth=0,
            showgrid=False,
            color="white",
            ticksuffix="  ",
            # showline=False,
        ),
        # coloraxis_colorbar=dict(
        #     tickfont=dict(color="white"),
        #     orientation="h",
        # ),
        font=dict(family="sans-serif"),
        hoverlabel=dict(
            font=dict(color="white"),  # Font color for hovertext
            bgcolor="black",  # Background color of hovertext
            bordercolor="black",  # Border color of hovertext
        ),
        legend=dict(
            font=dict(color="white"),  # Font color for hovertext
        ),
        # height=500,
        # autosize=True,
        annotations=annotations,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",  # Set the background color to a dark theme
        margin=dict(l=50, r=0, t=0),
    )

    # fig.update_traces(
    #     text=heatmap_pivot.applymap(lambda val: f"{round(val / 60)}h" if val > 30 else "").values,
    #     texttemplate="%{text}",
    # )

    # fig.show(config={"staticPlot": True})

    return fig


def generate_month_heatmap_with_features_no_invalid(df, selected_year, selected_month):

    # Ensure 'watched_at' is in datetime format
    if not np.issubdtype(df["watched_at"].dtype, np.datetime64):
        df["watched_at"] = pd.to_datetime(df["watched_at"])

    # Extract year, month, and day from 'watched_at'
    df["year"] = df["watched_at"].dt.year
    df["month"] = df["watched_at"].dt.month
    df["day"] = df["watched_at"].dt.day

    # Filter data based on selected_year and selected_month
    filtered_df = df[(df["year"] == selected_year) & (df["month"] == selected_month)]

    # Aggregate watchtime per day
    heatmap_data = (
        filtered_df.groupby("day")["runtime"]
        .sum()
        .reindex(range(1, 32), fill_value=0)
        .reset_index()
    )

    # Determine number of days in the selected month
    _, num_days = calendar.monthrange(selected_year, selected_month)

    # Get the first weekday of the month (0=Monday, 6=Sunday)
    first_weekday = datetime(selected_year, selected_month, 1).weekday()

    # Get today's date
    today = datetime.now()

    # Create a list of weeks, each week is a list of 7 days
    weeks = []
    day_counter = 1

    # Fill the first week
    current_week = [None] * 7  # Initialize with None for empty cells
    for i in range(first_weekday, 7):
        if day_counter > num_days:
            break
        runtime = heatmap_data.loc[
            heatmap_data["day"] == day_counter, "runtime"
        ].values[0]
        current_week[i] = runtime
        day_counter += 1
    weeks.append(current_week)

    # Fill the remaining weeks
    while day_counter <= num_days:
        current_week = [None] * 7
        for i in range(7):
            if day_counter > num_days:
                break
            runtime = heatmap_data.loc[
                heatmap_data["day"] == day_counter, "runtime"
            ].values[0]
            current_week[i] = runtime
            day_counter += 1
        weeks.append(current_week)

    # Build an alternating matrix: day numbers row, watchtime row
    alternating_matrix = []
    future_mask = []
    day_num_counter = 1
    for week_idx, week in enumerate(weeks):
        day_numbers_row = []
        day_numbers_future_row = []
        watchtime_row = []
        watchtime_future_row = []
        for i in range(7):
            if week_idx == 0 and i < first_weekday:
                # Empty cells before the first day of the month
                day_numbers_row.append("")
                day_numbers_future_row.append(False)
                watchtime_row.append(None)
                watchtime_future_row.append(False)
            elif day_num_counter > num_days:
                # Empty cells after the last day of the month
                day_numbers_row.append("")
                day_numbers_future_row.append(False)
                watchtime_row.append(None)
                watchtime_future_row.append(False)
            else:
                runtime = week[i]
                date = datetime(selected_year, selected_month, day_num_counter)
                is_future = date.date() > today.date()
                day_numbers_row.append(str(day_num_counter))
                day_numbers_future_row.append(is_future)
                watchtime_row.append(runtime)
                watchtime_future_row.append(is_future)
                day_num_counter += 1
        alternating_matrix.append(day_numbers_row)
        future_mask.append(day_numbers_future_row)
        alternating_matrix.append(watchtime_row)
        future_mask.append(watchtime_future_row)

    # Convert to NumPy arrays
    heatmap_array = np.array(alternating_matrix, dtype=object)
    future_mask_array = np.array(future_mask)

    # Calculate the 98th percentile for watchtime to set color scaling
    watchtime_values = heatmap_array[1::2]  # Extract watchtime rows
    watchtime_values_flat = np.array(
        [item for sublist in watchtime_values for item in sublist if item is not None]
    )
    percentile_98 = (
        np.percentile(watchtime_values_flat, 98)
        if watchtime_values_flat.size > 0
        else 0
    )

    # Define days of the week starting from Monday
    days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Define colorscales
    day_colorscale = [[0, "#0E1117"], [1, "#0E1117"]]
    watch_colorscale = [
        [0.0, "#262730"],
        [0.1, "#872b2b"],
        [1.0, "#FF4B4B"],
    ]
    future_colorscale = [
        [0, "rgba(0,0,0,0)"],  # Transparent
        [1, "#262730"],  # Light grey for future dates
    ]

    # Create list of traces
    traces = []
    y_values = []
    for row_idx in range(heatmap_array.shape[0]):
        z = [heatmap_array[row_idx]]
        y_value = -row_idx
        if row_idx % 2 == 0:
            # Day numbers row
            z_numeric = [[1 if val else 0 for val in heatmap_array[row_idx]]]
            trace = go.Heatmap(
                z=z_numeric,
                x=days_of_week,
                y=[y_value],
                xgap=12,
                ygap=12,
                colorscale=day_colorscale,
                showscale=False,
                text=z,  # Display the day numbers inside the heatmap cells
                texttemplate="%{text}",
                textfont={"color": "white", "size": 12},
                hoverinfo="skip",
            )
            # scatter_trace = go.Scatter(
            #     x=days_of_week,
            #     y=[y_value] * len(days_of_week),  # Ensure the y-values match the heatmap row
            #     mode="markers",  # Use markers to create circles
            #     marker=dict(
            #         symbol="circle",  # Circle marker
            #         size=20,  # Adjust the size of the marker
            #         color="rgba(255, 255, 255, 0.1)",  # Semi-transparent white color
            #         line=dict(color="white", width=2),  # Optional: white border for the circle
            #     ),
            #     text=z,  # Display the text value inside the circle
            #     textposition="middle center",  # Position text inside the marker
            #     textfont=dict(size=12, color="black"),  # Adjust text color as needed
            #     hoverinfo="skip",
            # )
            traces.append(trace)
            # traces.append(scatter_trace)
        else:
            # Watchtime row
            formatted_watchtime = [
                (
                    f"{round(runtime / 60)}h"
                    if runtime is not None and runtime > 30
                    else ""
                )
                for runtime in heatmap_array[row_idx]
            ]
            z_numeric = [
                [
                    runtime if runtime is not None else np.nan
                    for runtime in heatmap_array[row_idx]
                ]
            ]
            trace = go.Heatmap(
                z=z_numeric,
                x=days_of_week,
                y=[y_value],
                xgap=12,
                ygap=12,
                colorscale=watch_colorscale,
                showscale=False,
                zmin=0,
                zmax=percentile_98,
                text=[formatted_watchtime],
                texttemplate="%{text}",
                textfont={"color": "white", "size": 12},
                hoverinfo="text",
                hovertext=[
                    [f"Watchtime: {wt}" if wt else "" for wt in formatted_watchtime]
                ],
            )
            traces.append(trace)

        y_values.append(y_value)

    # st.write(round(48 / 60))

    # Create future heatmap
    z_future_mask = np.where(future_mask_array, 1, np.nan)
    future_heatmap = go.Heatmap(
        z=z_future_mask,
        x=days_of_week,
        y=y_values,
        colorscale=future_colorscale,
        showscale=False,
        hoverinfo="skip",
        opacity=1,
        textfont={"color": "grey"},
    )

    # Create figure and add all traces
    fig = go.Figure(data=traces)
    fig.add_trace(future_heatmap)

    # Update layout
    fig.update_layout(
        title=f"📅 Watchtime Heatmap for {calendar.month_name[selected_month]} {selected_year}",
        title_font=dict(size=20, color="white"),
        xaxis=dict(
            # title="Days of the Week",
            tickmode="array",
            tickvals=days_of_week,
            ticktext=days_of_week,
            tickfont=dict(color="white"),
            showgrid=False,
            zeroline=False,
            showline=False,
            side="top",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_values,
            ticktext=[""] * len(y_values),
            tickfont=dict(color="white"),
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        plot_bgcolor="#0E1117",  # Dark background
        paper_bgcolor="#0E1117",  # Dark background
        height=600,
    )

    # Update hover label styling
    fig.update_layout(
        hoverlabel=dict(
            font=dict(color="white"),
            bgcolor="black",
            bordercolor="black",
        ),
    )

    return fig


def process_decade_chart_data(df_, media_type):
    # Split 'watched_at' column into decade
    assert media_type in ["all", "show", "movie"]

    if media_type == "show":
        media_type = "episode"

    if media_type != "all":
        df = df_.copy()[df_["media_type"] == media_type]
    else:
        df = df_.copy()

    df["year"] = df["released"].dt.year
    df["decade"] = ((df["year"] // 10) * 10).astype(str) + "s"

    # st.write(
    #     df.groupby("title")
    #     .agg({"decade": lambda x: set(x), "runtime": "sum", "media_type": "first"})
    #     .sort_values("runtime", ascending=False)
    # )

    return df.groupby("decade", as_index=False)["runtime"].sum()


def plot_runtime_by_decade(df, media_type):
    """
    Plots a horizontal bar graph in Plotly with each decade's bar labeled with the percentage of total runtime.

    Parameters:
    df_decade (pd.DataFrame): Dataframe with 'decade' and summed 'runtime' columns.

    Returns:
    plotly.graph_objs._figure.Figure: The Plotly figure object representing the bar graph.
    """

    df_decade = process_decade_chart_data(df, media_type)

    # Calculate the percentage of runtime for each decade
    total_runtime = df_decade["runtime"].sum()
    df_decade["percent"] = (df_decade["runtime"] / total_runtime) * 100

    # Ensure that the decade values chosen for y are complete
    min_decade = df_decade["decade"].min()
    max_decade = df_decade["decade"].max()
    all_decades = [
        str(decade) + "s"
        for decade in range(int(min_decade[:-1]), int(max_decade[:-1]) + 1, 10)
    ]

    df_decade.set_index("decade", inplace=True)

    # Fill missing decades with 0 runtime
    df_decade = df_decade.reindex(all_decades, fill_value=0)

    df_decade.reset_index(inplace=True)

    # Create the horizontal bar graph with percentage labels
    fig = px.bar(
        df_decade,
        y="decade",
        x="runtime",
        text=df_decade["percent"].apply(lambda x: f"{x:.1f}%"),
        labels={"runtime": "Total Runtime", "decade": "Decade"},
        orientation="h",
    )

    # Update text position and layout
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title="",
        xaxis=dict(range=[0, max(df_decade["runtime"]) * 1.1]),
        # title_font=dict(size=1, color="white"),
        yaxis_title="Decade",
        xaxis_title="Total Runtime",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        yaxis=dict(autorange="reversed"),
        margin=dict(
            r=0,
            t=0,
        ),
    )

    return fig


@st.cache_data
def process_chart_data(
    df_, column_name, sum_100=False, media_type="all", n=10, others_threshold=1
):
    """
    Process data for visualization.
    Handles NaN and empty lists, marks them as 'Unknown', and groups smaller categories into 'Others'.
    """
    # Filter data based on media type
    assert media_type in ["all", "show", "movie"]
    if media_type != "all":
        df = df_.copy()[df_["media_type"] == media_type]
    else:
        df = df_.copy()

    total_time = df["watchtime"].sum()

    # Identify entries needing to be set as 'Unknown'
    has_unknowns = (
        df[column_name].isna().any()
        or df[column_name].apply(lambda x: x == [] or x == "" or x == " ").any()
    )

    # Handle NaN values and empty lists
    if has_unknowns:
        df[column_name] = df[column_name].apply(
            lambda x: (
                "Unknown"
                if (not isinstance(x, list) and pd.isnull(x))
                or (isinstance(x, list) and not x)
                else x
            )
        )

    # Handle list columns
    if df[column_name].apply(isinstance, args=(list,)).any():
        df = df.explode(column_name)
        if sum_100:
            total_time = df["watchtime"].sum()

    # Group by the specified column and aggregate watchtime
    df_grouped = (
        df.groupby(column_name)
        .agg({"watchtime": "sum"})
        .sort_values("watchtime", ascending=False)
        .reset_index()
    )

    # Percentage calculations and formatting
    df_grouped["%_tw"] = (df_grouped["watchtime"] / total_time) * 100
    df_grouped["formatted_time"] = df_grouped["watchtime"].apply(
        lambda x: f"{x // 1440} days {x % 1440 // 60} hours"
    )

    with st.expander("Data"):
        st.write(df_grouped)

    # Group smaller categories into 'Others'
    others_mask = (
        (df_grouped.index >= n) | (df_grouped["%_tw"] < others_threshold)
    ) & (df_grouped[column_name] != "Unknown")
    others_sum = df_grouped.loc[others_mask, "watchtime"].sum()
    if others_sum > 0:
        others_percentage = others_sum / total_time * 100
        others_formatted_time = (
            f"{others_sum // 1440} days {others_sum % 1440 // 60} hours"
        )
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
                            "formatted_time": others_formatted_time,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        df_grouped = top_categories

    # Ensure 'Unknown' and 'Others' are at the end of the order
    df_grouped = pd.concat(
        [
            df_grouped[~df_grouped[column_name].isin(["Unknown", "Others"])],
            df_grouped[df_grouped[column_name] == "Others"],
            df_grouped[df_grouped[column_name] == "Unknown"],
        ]
    )

    return df_grouped


def generate_chart(
    df_,
    column_name,
    chart_type="pie",
    sum_100=False,
    media_type="all",
    n=10,
    others_threshold=1,
):
    """
    Generate a chart based on the processed data.
    Allows selection between 'bar' or 'pie' chart.
    """
    # Process data for visualization
    df_grouped = process_chart_data(
        df_, column_name, sum_100, media_type, n, others_threshold
    )

    # with st.expander("Data"):
    #     st.write(df_grouped)

    # Generate the requested chart type
    if chart_type == "bar":
        fig = px.bar(
            df_grouped,
            x=column_name,
            y="watchtime",
            # title=f"Total Watch Time by {column_name}",
            text=df_grouped["%_tw"].apply(lambda x: f"{x:.0f}%"),
            hover_name=column_name,
            hover_data=["formatted_time"],
        )
        fig.update_traces(textposition="outside")

    elif chart_type == "pie":
        fig = px.pie(
            df_grouped,
            names=column_name,
            values="watchtime",
            # title=f"Total Watch Time by {column_name}",
            hover_name=column_name,
            hover_data=["formatted_time"],
            color_discrete_map={"Unknown": "grey", "Others": "lightgrey"},
        )

        # Custom text for slices based on threshold
        threshold = 5
        labels = [
            f"{name}<br>{percent:.0f}%" if percent >= threshold else ""
            for name, percent in zip(df_grouped[column_name], df_grouped["%_tw"])
        ]

        # Define colors
        color_map = {"Unknown": "grey", "Others": "lightgrey"}

        # Apply colors and labels manually using update_traces
        fig.update_traces(
            text=labels,
            textinfo="text",
            insidetextorientation="horizontal",
            sort=False,
            marker=dict(
                colors=[
                    color_map.get(name, "default_color")
                    for name in df_grouped[column_name]
                ]
            ),
        )

    else:
        raise ValueError("chart_type must be either 'bar' or 'pie'.")

    # Apply font attributes, assumed to be defined elsewhere in your script
    fig.update_layout(**font_attrs, title="")

    return fig


# @st.cache_data
# def generate_chart(df_, column_name, sum_100=False, media_type="all", type="bar", n=10, others_threshold=1):
#     """
#     Generate a chart based on a specified column of the dataframe.
#     Handles NaN and empty lists by marking them as 'Unknown'.
#     Groups smaller categories into 'Others' if their count exceeds a threshold n or their percentage is less than the others_threshold.
#     """
#     assert media_type in ["all", "show", "movie"]
#     if media_type != "all":
#         df = df_.copy()[df_["media_type"] == media_type]
#     else:
#         df = df_.copy()

#     total_time = df["watchtime"].sum()

#     # Identifying if there are any entries that need to be set to "Unknown"
#     has_unknowns = df[column_name].isna().any() or df[column_name].apply(lambda x: x == [] or x == "" or x == " ").any()

#     # Handling NaN values, empty lists as 'Unknown' only if there are any
#     if has_unknowns:
#         df[column_name] = df[column_name].apply(
#             lambda x: "Unknown" if (not isinstance(x, list) and pd.isnull(x)) or (isinstance(x, list) and not x) else x
#         )

#     # Handle list columns
#     if df[column_name].apply(isinstance, args=(list,)).any():
#         df = df.explode(column_name)
#         if sum_100:
#             total_time = df["watchtime"].sum()

#     # Group by the specified column and aggregate on watchtime
#     df_grouped = (
#         df.groupby(column_name).agg({"watchtime": "sum"}).sort_values("watchtime", ascending=False).reset_index()
#     )

#     # Percentage calculations and formatting
#     df_grouped["%_tw"] = (df_grouped["watchtime"] / total_time) * 100
#     df_grouped["formatted_time"] = df_grouped["watchtime"].apply(
#         lambda x: f"{x // 1440} days {x % 1440 // 60} hours"
#         # lambda x: f"{x // 60} hours"
#     )

#     with st.expander("Data"):
#         st.write(df_grouped)

#     # Group smaller categories into 'Others'
#     others_mask = ((df_grouped.index >= n) | (df_grouped["%_tw"] < others_threshold)) & (
#         df_grouped[column_name] != "Unknown"
#     )
#     others_sum = df_grouped.loc[others_mask, "watchtime"].sum()
#     if others_sum > 0:
#         others_percentage = others_sum / total_time * 100
#         others_formatted_time = f"{others_sum // 1440} days {others_sum % 1440 // 60} hours"
#         top_categories = df_grouped.loc[~others_mask]
#         top_categories = pd.concat(
#             [
#                 top_categories,
#                 pd.DataFrame(
#                     [
#                         {
#                             column_name: "Others",
#                             "watchtime": others_sum,
#                             "%_tw": others_percentage,
#                             "formatted_time": others_formatted_time,
#                         }
#                     ]
#                 ),
#             ],
#             ignore_index=True,
#         )
#         df_grouped = top_categories

#     # Create the chart
#     if type == "bar":
#         fig = px.bar(
#             df_grouped,
#             x=column_name,
#             y="watchtime",
#             title=f"Total Watch Time by {column_name}",
#             text=df_grouped["%_tw"].apply(lambda x: f"{x:.0f}%"),
#             hover_name=column_name,
#             hover_data=["formatted_time"],
#         )
#         fig.update_traces(textposition="outside")
#     else:
#         # Order the categories explicitly
#         # Ensure 'Unknown' and 'Others' are at the end of the pie chart
#         df_grouped = pd.concat(
#             [
#                 df_grouped[~df_grouped[column_name].isin(["Unknown", "Others"])],
#                 df_grouped[df_grouped[column_name] == "Others"],
#                 df_grouped[df_grouped[column_name] == "Unknown"],
#             ]
#         )

#         fig = px.pie(
#             df_grouped,
#             names=column_name,
#             values="watchtime",
#             title=f"Total Watch Time by {column_name}",
#             hover_name=column_name,
#             hover_data=["formatted_time"],
#             color_discrete_map={"Unknown": "grey", "Others": "lightgrey"},
#         )

#         # Custom text for slices based on threshold
#         threshold = 5
#         labels = [
#             f"{name}<br>{percent:.0f}%" if percent >= threshold else ""
#             for name, percent in zip(df_grouped[column_name], df_grouped["%_tw"])
#         ]

#         # Define your colors here
#         color_map = {"Unknown": "grey", "Others": "lightgrey"}

#         # Apply colors manually using update_traces

#         fig.update_traces(
#             text=labels,
#             textinfo="text",
#             insidetextorientation="horizontal",
#             sort=False,
#             marker=dict(colors=[color_map.get(name, "default_color") for name in df_grouped[column_name]]),
#         )

#     # Apply font attributes, assumed to be defined elsewhere in your script
#     fig.update_layout(**font_attrs)

#     return fig


def plot_with_echarts(df, column_name):
    """
    Plot data using echarts in Streamlit.

    Args:
        data (list): Processed data for echarts.
        column_name (str): Title of the chart.

    Returns:
        None: Displays the chart in Streamlit.
    """

    df_grouped = dw.wrangle_data_for_plots(df, column_name)
    data = dw.process_for_echarts(df_grouped, column_name)

    # st.write(data)

    # Sort legend items such that 'Others' and 'Unknown' are at the bottom
    legend_data = [
        item["name"] for item in data if item["name"] not in ["Others", "Unknown"]
    ] + ["Others", "Unknown"]

    # Dynamically reduce the number of labels for better readability
    # label_threshold = 10
    label_threshold = 1
    for i, item in enumerate(data):
        percentage = item["%_tw"]
        rounded_percentage = round(percentage)  # Round to 0 decimal places
        if percentage < label_threshold or i >= 6:
            item["label"] = {"show": False}
        else:
            item["label"] = {
                "show": True,
                "position": "inside" if percentage > 10 else "outside",
                "formatter": f"{item['name']}\n{rounded_percentage}%",  # Assuming 'name' is the label name
                # "rich": {
                #     "d": {
                #         "fontSize": 14,
                #         "fontFamily": "sans-serif",
                #         "color": "white",
                #         "align": "center",  # Horizontal alignment
                #         "verticalAlign": "middle",  # Vertical alignment
                #         "lineHeight": 20,  # Adjust as needed for better vertical centering
                #     }
                # },
                "labelLine": {
                    "show": False,
                },
                # "offset": [-15, -15],
                # You can add more complex rules or callbacks here
                "textStyle": {
                    "fontFamily": "sans-serif",
                    "color": "white",
                    "fontSize": 13,
                    "align": "center",  # Horizontal alignment
                    "verticalAlign": "middle",  # Vertical alignment
                },
                # Optionally, you can also add padding or other styling properties if needed
            }

    # st.write(data)

    option = {
        "color": [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ],
        "title": {
            "text": f"Watch Time by {column_name}",
            "left": "center",
            "textStyle": {
                "fontFamily": "sans-serif",
                "fontSize": 18,
                "color": "white",
                "fontWeight": "500",
            },
        },
        "tooltip": {
            "trigger": "item",
            "textStyle": {"fontFamily": "sans-serif", "fontSize": 14},
        },
        "legend": {
            "orient": "vertical",
            "left": "right",
            "data": legend_data,
            "textStyle": {
                "fontFamily": "sans-serif",
                "color": "white",
            },
        },
        "series": [
            {
                "name": "Watchtime",
                "type": "pie",
                "radius": "65%",
                "data": data,
                "label": {
                    "textStyle": {
                        "fontFamily": "sans-serif",
                        "color": "white",
                    }
                },
                "backgroundColor": "transparent",
            }
        ],
        "backgroundColor": "transparent",
    }

    return st_echarts(options=option, height="400px")
