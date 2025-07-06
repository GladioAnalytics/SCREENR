import streamlit as st
import pandas as pd
from ETF_FILTER import (
    plot_density, scatter_with_regression, plot_top_positions,
    plot_top_n_counts, create_treemap_2, get_benchmark_returns,
    get_multiple_returns, combine_returns, trim_combined_df,
    plot_cumulative_and_underwater, plot_correlation_heatmap,
    generate_etf_grid_charts_with_allocations,
    generate_duration_maturity_charts,
    generate_radar_chart
)

# Set up Streamlit page
st.set_page_config(page_title="Dynamic ETF Analysis Dashboard", layout="wide")

# Initialize session state for plots
if "plots" not in st.session_state:
    st.session_state.plots = {}

# Load Data
@st.cache_data
def load_etf_data():
    df = pd.read_csv("etf_data.csv", index_col=None)  # <- DON'T set index_col
    df.columns = df.columns.str.strip()
    return df

etf_df = load_etf_data()
etf_df.columns = etf_df.columns.str.strip()

# Identify float and string columns
float_columns = etf_df.select_dtypes(include=["float", "int"]).columns.tolist()
string_columns = etf_df.select_dtypes(include=["object", "string"]).columns.tolist()

# Sidebar: Dataset Info
st.sidebar.header("Dataset Info")
st.sidebar.write(f"Total Rows: {etf_df.shape[0]}")
st.sidebar.write(f"Total Columns: {etf_df.shape[1]}")
st.sidebar.write(f"Float Columns: {len(float_columns)}")
st.sidebar.write(f"String Columns: {len(string_columns)}")

# Sidebar: Select Variables to Filter
st.sidebar.header("Select Variables to Filter")
float_filter_vars = st.sidebar.multiselect("Select Numeric Variables", float_columns)
string_filter_vars = st.sidebar.multiselect("Select String Variables", string_columns)

# Filter Section
st.sidebar.header("Apply Filters")
filtered_df = etf_df.copy()

# Numeric Filters
if float_filter_vars:
    st.sidebar.subheader("Numeric Filters")
    for col in float_filter_vars:
        min_val, max_val = float(etf_df[col].min()), float(etf_df[col].max())
        selected_range = st.sidebar.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
        filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]

# String Filters (Side by Side)
if string_filter_vars:
    st.sidebar.subheader("String Filters")
    filter_columns = st.sidebar.columns(len(string_filter_vars))

    for col_idx, col in enumerate(string_filter_vars):
        with filter_columns[col_idx]:
            inclusion_choice = st.radio(f"{col} - Include/Exclude", ["Include", "Exclude"], index=0)

            unique_values = etf_df[col].dropna().unique()
            select_all = st.checkbox(f"Select All {col}", value=(inclusion_choice == "Include"))

            selected_items = []
            for val in unique_values:
                is_checked = st.checkbox(str(val), value=select_all, key=f"{col}_{val}")
                if is_checked:
                    selected_items.append(val)

            if inclusion_choice == "Include":
                filtered_df = filtered_df[filtered_df[col].isin(selected_items)]
            else:
                filtered_df = filtered_df[~filtered_df[col].isin(selected_items)]

# Display Filtered Data
st.header("Filtered Dataset")
st.write(f"Filtered Rows: {filtered_df.shape[0]} / {etf_df.shape[0]}")
st.dataframe(filtered_df)

# Main Dashboard Sections

# Section 1: Density Plot
st.header("Density Plot")
density_column = st.selectbox("Select Column for Density Plot", options=[""] + filtered_df.select_dtypes(include=['float']).columns.tolist())
density_outlier_method = st.selectbox("Outlier Removal Method", ["percentile", "IQR", "zscore"], key="density_outliers")
if st.button("Generate Density Plot"):
    if density_column:
        st.session_state.plots["density"] = plot_density(filtered_df, density_column, remove_outliers_method=density_outlier_method)
    else:
        st.warning("Select a column for the Density Plot.")

if "density" in st.session_state.plots:
    st.plotly_chart(st.session_state.plots["density"])

# Section 2: Scatter Plot
st.header("Scatter Plot")
scatter_x = st.selectbox("X Column", options=[""] + filtered_df.select_dtypes(include=['float']).columns.tolist(), key="scatter_x")
scatter_y = st.selectbox("Y Column", options=[""] + filtered_df.select_dtypes(include=['float']).columns.tolist(), key="scatter_y")
scatter_standardize = st.checkbox("Standardize Data", value=False)
if st.button("Generate Scatter Plot"):
    if scatter_x and scatter_y:
        st.session_state.plots["scatter"] = scatter_with_regression(filtered_df, scatter_x, scatter_y, standardize=scatter_standardize)
    else:
        st.warning("Select both X and Y columns for the Scatter Plot.")

if "scatter" in st.session_state.plots:
    st.plotly_chart(st.session_state.plots["scatter"])

# Section 3: Treemap
st.header("Treemap")
treemap_path_levels = st.multiselect("Select Hierarchy Levels", options=filtered_df.columns.tolist(), default=[])
treemap_color_column = st.selectbox("Select Color Column for Treemap", options=[""] + filtered_df.select_dtypes(include=['float', 'int']).columns.tolist())
treemap_remove_na_or_zero = st.checkbox("Remove NA or Zero", value=False)
if st.button("Generate Treemap"):
    if treemap_path_levels and treemap_color_column:
        st.session_state.plots["treemap"] = create_treemap_2(
            data=filtered_df,
            path_levels=treemap_path_levels,
            color_column=treemap_color_column,
            remove_na_or_zero=treemap_remove_na_or_zero
        )
    else:
        st.warning("Select hierarchy levels and a color column for the Treemap.")

if "treemap" in st.session_state.plots:
    st.plotly_chart(st.session_state.plots["treemap"])

# Section 4: Top N Counts
st.header("Top N Counts")
top_n_column = st.selectbox("Select Column for Top N Counts", options=[""] + filtered_df.select_dtypes(include=['object']).columns.tolist())
top_n_value = st.slider("Top N", min_value=5, max_value=50, value=10)
if st.button("Generate Top N Counts"):
    if top_n_column:
        st.session_state.plots["top_n"] = plot_top_n_counts(filtered_df, column=top_n_column, top_n=top_n_value)
    else:
        st.warning("Select a column for the Top N Counts.")

if "top_n" in st.session_state.plots:
    st.plotly_chart(st.session_state.plots["top_n"])

# Section 5: Benchmark and Ticker Analysis
st.header("Ticker Analysis")
benchmark_ticker = st.text_input("Enter Benchmark Ticker (default: SPY)", value="SPY")

tickers_input = st.text_input("Enter Tickers (comma-separated)")

start_date = st.date_input("Start Date", value=pd.to_datetime("2014-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))
if st.button("Generate Ticker Analysis"):
    tickers = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]
    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        benchmark_returns = get_benchmark_returns(benchmark_ticker)
        returns_df = get_multiple_returns(tickers)
        combined_returns = combine_returns(returns_df, benchmark_returns)
        trimmed_df = trim_combined_df(combined_returns, start_date=start_date, end_date=end_date)

        st.session_state.plots["cumulative"] = plot_cumulative_and_underwater(trimmed_df)
        st.session_state.plots["heatmap"] = plot_correlation_heatmap(trimmed_df)

if "cumulative" in st.session_state.plots:
    st.subheader("Cumulative and Underwater Returns")
    st.plotly_chart(st.session_state.plots["cumulative"])

if "heatmap" in st.session_state.plots:
    st.subheader("Correlation Heatmap")
    st.plotly_chart(st.session_state.plots["heatmap"])

# Additional Sections: ETF Grid Charts, Duration & Maturity Charts, and Radar Chart
if tickers_input:
    tickers = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]
    if not tickers:
        st.warning("Please enter valid tickers.")
    else:
        # ETF Grid Charts with Allocations
        st.header("ETF Grid Charts with Allocations")
        etf_grid_fig = generate_etf_grid_charts_with_allocations(tickers, filtered_df)
        if etf_grid_fig:
            st.plotly_chart(etf_grid_fig)

        # Duration and Maturity Charts
        st.header("Duration and Maturity Charts")
        duration_maturity_fig = generate_duration_maturity_charts(tickers, filtered_df)
        if duration_maturity_fig:
            st.plotly_chart(duration_maturity_fig)

        # Radar Chart for Financial Metrics
        st.header("Radar Chart for Financial Metrics")
        radar_fig = generate_radar_chart(tickers, filtered_df)
        if radar_fig:
            st.plotly_chart(radar_fig)