# IMPORT YAHOO FINANCE MODULES 
import yfinance as yf
import yahoo_fin

# IMPORT STANDARD DATA SCIENCE STACK
import pandas as pd
import numpy as np

# IMPIRT PLOTTING LIBRARIES
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# PROGRESS BAR AND WARNINGS
from tqdm import tqdm
import warnings
import time

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore



def get_ticker_data(ticker):
    try:
        # Initialize Ticker
        fund = yf.Ticker(ticker)
        data = fund.funds_data

        # Fund Overview
        try:
            df = pd.DataFrame([data.fund_overview])
            df.columns = df.columns.str.replace(r'(?<!^)([A-Z])', r' \1', regex=True).str.title()
            result = df
        except Exception as e:
            result = pd.DataFrame()

        # Fund Operations
        try:
            df = pd.DataFrame(data.fund_operations[ticker]).T
            df.reset_index(drop=True, inplace=True)
            result = pd.concat([result, df], axis=1)
        except Exception as e:
            pass

        # Asset Classes
        try:
            df = pd.DataFrame([data.asset_classes])
            df.columns = df.columns.str.replace(r'(?<!^)([A-Z])', r' \1', regex=True).str.title()
            result = pd.concat([result, df], axis=1)
        except Exception as e:
            pass

        # Equity Holdings
        try:
            df = pd.DataFrame(data.equity_holdings[ticker])
            df = df.T
            df.reset_index(drop=True, inplace=True)
            result = pd.concat([result, df], axis=1)
        except Exception as e:
            pass

        # Bond Holdings
        try:
            df = pd.DataFrame(data.bond_holdings[ticker]).T
            df.reset_index(drop=True, inplace=True)
            result = pd.concat([result, df], axis=1)
        except Exception as e:
            pass

        # Bond Ratings
        try:
            df = pd.DataFrame([data.bond_ratings])
            df.columns = df.columns.astype(str).str.replace('_', ' ').str.title()
            result = pd.concat([result, df], axis=1)
        except Exception as e:
            pass

        # Sector Weightings
        try:
            df = pd.DataFrame([data.sector_weightings])
            df.columns = df.columns.str.replace(r'(?<!^)([A-Z])', r' \1', regex=True).str.title()
            df.columns = df.columns.astype(str).str.replace('_', ' ')
            result = pd.concat([result, df], axis=1)
        except Exception as e:
            pass

        # Top Holdings
        try:
            df = pd.DataFrame(data.top_holdings['Holding Percent'])
            top_holding_tickers = df.index
            nr_holdings = len(df.index)+1
            top_holding_tickers = pd.DataFrame(top_holding_tickers).T
            top_holding_tickers.columns = [f"Top Position {i}" for i in range(1, nr_holdings)]
            top_holding_tickers.reset_index(drop=True, inplace=True)

            top_holding_tickers_weight = df['Holding Percent']
            top_holding_tickers_weight = pd.DataFrame(top_holding_tickers_weight.values).T
            top_holding_tickers_weight.columns = [f"Weight of Top Position {i}" for i in range(1, nr_holdings)]
            top_holding_tickers_weight = (top_holding_tickers_weight * 100).round(2)

            temp_df = pd.concat([top_holding_tickers, top_holding_tickers_weight], axis=1)
            result = pd.concat([result, temp_df], axis=1)
        except Exception as e:
            pass

        return result
    except Exception as e:
        raise RuntimeError(f"Error fetching data for ticker {ticker}: {e}")
    

def remove_outliers(data, method="IQR", **kwargs):
    if method == "IQR":
        lower_quantile = kwargs.get("lower_quantile", 0.25)
        upper_quantile = kwargs.get("upper_quantile", 0.75)
        factor = kwargs.get("factor", 1.5)
        q1 = data.quantile(lower_quantile)
        q3 = data.quantile(upper_quantile)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]
    elif method == "zscore":
        threshold = kwargs.get("threshold", 3)
        z_scores = zscore(data)
        return data[abs(z_scores) <= threshold]
    elif method == "percentile":
        lower_percentile = kwargs.get("lower_percentile", 0.01)
        upper_percentile = kwargs.get("upper_percentile", 0.99)
        lower_bound = data.quantile(lower_percentile)
        upper_bound = data.quantile(upper_percentile)
        return data[(data >= lower_bound) & (data <= upper_bound)]
    else:
        raise ValueError("Invalid method. Choose from 'IQR', 'zscore', or 'percentile'.")

def plot_density(etf_df, column_name, remove_outliers_method=None, **kwargs):
    filtered_data = etf_df[column_name].dropna()
    if remove_outliers_method:
        filtered_data = remove_outliers(filtered_data, method=remove_outliers_method, **kwargs)
    fig = ff.create_distplot([filtered_data],group_labels=[column_name],show_hist=False,show_rug=False)
    fig.update_layout(title=f"Density Plot of {column_name}",xaxis_title=column_name,yaxis_title="Density",template="plotly_white")
    return fig

def scatter_with_regression(etf_df, x_col, y_col, standardize=False, remove_outliers_method=None, **kwargs):
    data = etf_df[[x_col, y_col]].dropna()
    if remove_outliers_method:
        data[x_col] = remove_outliers(data[x_col], method=remove_outliers_method, **kwargs)
        data[y_col] = remove_outliers(data[y_col], method=remove_outliers_method, **kwargs)
        data = data.dropna() 
    if standardize:
        scaler = StandardScaler()
        data[x_col] = scaler.fit_transform(data[[x_col]])
        data[y_col] = scaler.fit_transform(data[[y_col]])
    x = data[x_col].values
    y = data[y_col].values
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    correlation = np.corrcoef(x, y)[0, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Observations', marker=dict(color='black')))
    fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', name='Regression Line', line=dict(color='blue')))
    fig.update_layout(title=f"Scatter Plot of {x_col} vs {y_col} (Corr: {correlation:.2f})",xaxis_title=x_col,yaxis_title=y_col,template="plotly_white",showlegend=False)
    return fig

def plot_top_positions(etf_df, position=1, top_count=30):
    column_name = f"Top Position {position}"
    filtered_data = etf_df[column_name].dropna()
    value_counts = filtered_data.value_counts()
    top_values = value_counts.head(top_count)
    fig = px.bar(top_values,x=top_values.index,y=top_values.values,labels={"x": "Value", "y": "Count"},title=f"Top {top_count} Unique Non-NA Values in {column_name}",)
    fig.update_layout(xaxis_title="Ticker",yaxis_title="Count",xaxis_tickangle=-45,template="plotly_white")
    return fig






def get_benchmark_returns(benchmark_ticker):
    benchmark = yf.download(benchmark_ticker,auto_adjust=False,period='max')
    if isinstance(benchmark.columns, pd.MultiIndex):
        benchmark.columns = benchmark.columns.droplevel(level=1)
    benchmark = benchmark['Adj Close']
    benchmark = benchmark.pct_change().dropna()
    benchmark = pd.DataFrame(benchmark)
    benchmark.columns = ['Benchmark']
    return benchmark

def get_returns(ticker):
    hist = yf.download(ticker,auto_adjust=False,period='max')
    hist.columns = hist.columns.droplevel(level=1)
    prices = hist['Adj Close']
    returns = prices.pct_change().dropna()
    returns = pd.DataFrame(returns)
    returns.columns = [ticker]
    return returns


def get_multiple_returns(ticker_list):
    return_list = []
    for ticker in ticker_list:
        try:
            returns = get_returns(ticker)
            return_list.append(returns)
        except Exception as e:
            print(f"Skipping {ticker} due to error: {e}")
    return_df = pd.concat(return_list, axis=1) if return_list else pd.DataFrame()
    return return_df

def combine_returns(returns_df, benchmark_df):
    combined_returns = pd.concat([returns_df, benchmark_df], axis=1, join='outer')
    return combined_returns

def trim_combined_df(df, start_date=None, end_date=None):
    if start_date is None:
        start_date = df.index.min()
    if end_date is None:
        end_date = df.index.max()
    trimmed_df = df.loc[start_date:end_date]
    return trimmed_df


def plot_cumulative_and_underwater(df):
    cumulative_returns = (1 + df).cumprod()
    drawdowns = cumulative_returns / cumulative_returns.cummax() - 1

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,  
        vertical_spacing=0.1, 
        subplot_titles=("Cumulative Returns", "Underwater Plot (Drawdowns)")
    )

    default_colors = px.colors.qualitative.Plotly
    color_map = {}

    for i, column in enumerate(df.columns):
        color = "black" if column == "Benchmark" else default_colors[i % len(default_colors)]
        color_map[column] = color
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=cumulative_returns[column],
                mode="lines",
                name=column,  
                line=dict(color=color, width=2)
            ),
            row=1, col=1
        )

    for column in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=drawdowns[column],
                mode="lines",
                name=column,
                line=dict(color=color_map[column], width=2, dash="dot"),
                showlegend=False  
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=700,  
        title="Cumulative Returns and Underwater Plot",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        xaxis2_title="Date",
        yaxis2_title="Drawdowns",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig


def plot_top_n_counts(data, column, top_n=30):
    value_counts = data[column].value_counts()
    top_values = value_counts.head(top_n)
    title = f"Top {top_n} Counts of Variable {column}"
    fig = px.bar(top_values,x=top_values.index,y=top_values.values,labels={"x": column, "y": "Count"},title=title)
    fig.update_layout(xaxis_title=column,yaxis_title="Count",xaxis_tickangle=-45,template="plotly_white")
    return fig

def plot_correlation_heatmap(trimmed_df):
    correlation_matrix = trimmed_df.corr(method='pearson', min_periods=1).round(4)
    mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool))
    correlation_matrix = correlation_matrix.where(~mask)
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,  # Annotate each cell with its value
        color_continuous_scale="RdBu_r",  # Diverging color scale
        title="Lower Triangular Correlation Heatmap of Returns",
        labels=dict(x="Assets", y="Assets", color="Correlation"),)
    fig.update_layout(
        xaxis=dict(title="Assets", tickfont=dict(family="Arial", size=12, color="black", weight="bold")),
        yaxis=dict(title="Assets", tickfont=dict(family="Arial", size=12, color="black", weight="bold")),
        coloraxis_colorbar=dict(title="Correlation"),
        margin=dict(l=40, r=40, t=40, b=40),)
    return fig




def create_treemap(data, path_levels, color_column, outlier_method="IQR", remove_na_or_zero=True, **kwargs):
    """
    Creates a treemap visualization.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        path_levels (list of str): Columns representing the hierarchy levels.
        color_column (str): The column used for coloring.
        outlier_method (str): Method to remove outliers from the color scale. Default is "IQR".
        remove_na_or_zero (bool): Remove rows with NA or zero in the color column.
        **kwargs: Additional arguments for outlier removal.

    Returns:
        plotly.graph_objs._figure.Figure: The treemap figure.
    """
    if not path_levels or len(path_levels) < 1:
        raise ValueError("At least one path level must be specified in `path_levels`.")

    # Remove rows with missing values in the path_levels
    data = data.dropna(subset=path_levels)

    if remove_na_or_zero and color_column != "Annual Report Expense Ratio":
        data = data[(data[color_column].notna()) & (data[color_column] != 0)]

    hierarchy = " → ".join(path_levels)
    title = f"Treemap of {hierarchy} Colored by {color_column}"

    # Handle hover data
    def format_hover_data(row):
        positions = [
            f"{row[f'Top Position {i}']}: {row[f'Weight of Top Position {i}']}%"
            for i in range(1, 11)
            if pd.notna(row[f'Top Position {i}']) and pd.notna(row[f'Weight of Top Position {i}'])
        ]

        columns = ["Expense Ratio", "Annual Holdings Turnover", "Total Net Assets",
            "Cash Position", "Stock Position", "Bond Position", "Preferred Position",
            "Convertible Position", "Other Position", "Price/Earnings", "Price/Book", "Price/Sales",
            "Price/Cashflow", "Duration", "Maturity", "Bb", "Aa", "Aaa", "A", "Other", "B",
            "Bbb", "Below B", "Us Government", "Realestate", "Consumer Cyclical",
            "Basic Materials", "Consumer Defensive", "Technology", "Communication Services",
            "Financial Services", "Utilities", "Industrials", "Energy", "Healthcare"]

        additional_info = []
        for col in columns:
            if pd.notna(row[col]):
                if col in ["Total Net Assets", "Annual Holdings Turnover"]:
                    additional_info.append(f"{col}: {row[col]:,.2f}")
                else:
                    additional_info.append(f"{col}: {row[col] * 100:.2f}%")

        combined_info = "<br>".join(positions + additional_info)
        return combined_info if combined_info else "No data available"

    data["Hover Info"] = data.apply(format_hover_data, axis=1)

    # Handle outliers for color column
    def remove_outliers(column, method="IQR", **kwargs):
        if method == "IQR":
            q1 = column.quantile(0.25)
            q3 = column.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return column[(column >= lower_bound) & (column <= upper_bound)]
        else:
            return column  # Default: no outlier handling

    clean_color_column = remove_outliers(data[color_column], method=outlier_method, **kwargs)
    color_min = clean_color_column.min()
    color_max = clean_color_column.max()

    # Create the treemap
    fig = px.treemap(
        data,
        path=path_levels + ["Ticker"],  
        color=color_column,
        title=title,
        color_continuous_scale="Viridis",
        range_color=(color_min, color_max)
    )

    # Add hover customization
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>"  
            "<br>%{customdata[1]}"  
        ),
        customdata=data[["Ticker", "Hover Info"]].values,
    )

    # Layout customization
    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig







def generate_radar_chart(etf_tickers, df):
    # Define variables
    radar_variables_financial = ['Price/Earnings', 'Price/Book', 'Price/Sales', 'Price/Cashflow']

    # Filter valid tickers with data for financial metrics
    radar_data_financial = []
    for ticker in etf_tickers:
        etf_data = df[df['Ticker'] == ticker]

        # Check if all radar variables are NaN or zero
        if (etf_data[radar_variables_financial].fillna(0) == 0).all(axis=None):
            print(f"Skipping {ticker}: All data is NaN or zero for financial metrics.")
            continue

        # Append the radar data for the ticker
        radar_data_financial.append((ticker, etf_data[radar_variables_financial].iloc[0].dropna()))

    if not radar_data_financial:
        print("No valid tickers with data to display.")
        return

    # Create financial radar chart
    fig_radar = go.Figure()
    for ticker, metrics in radar_data_financial:
        fig_radar.add_trace(
            go.Scatterpolar(
                r=metrics.values,
                theta=metrics.index,
                fill='toself',
                name=ticker
            )
        )

    fig_radar.update_layout(
        title=dict(
            text="ETF Metrics: Financial Radar Chart",
            x=0.5,
            xanchor="center",
            font=dict(size=20)
        ),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max([max(metrics.values) for _, metrics in radar_data_financial])] if radar_data_financial else [0, 1])
        ),
        legend=dict(
            title="Tickers",
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Show the radar chart
    return fig_radar




def generate_duration_maturity_charts(etf_tickers, df):

    horizontal_bars = ['Duration', 'Maturity']
    credit_variables = ['Bb', 'Aa', 'Aaa', 'A', 'Other', 'B', 'Bbb', 'Below B', 'Us Government']
    allocation_variables = ['Cash Position', 'Stock Position', 'Bond Position', 
                            'Preferred Position', 'Convertible Position', 'Other Position']

    # Identify valid tickers with data
    valid_tickers = []
    row_specs = []

    for ticker in etf_tickers:
        etf_data = df[df['Ticker'] == ticker]

        # Check if both Duration & Maturity and Credit Info are NaN or zero
        if (
            (etf_data[horizontal_bars].fillna(0) == 0).all(axis=None) or
            (etf_data[credit_variables].fillna(0) == 0).all(axis=None)
        ):
            print(f"Skipping {ticker}: No valid data in Duration/Maturity or Credit Info.")
            continue

        valid_tickers.append(ticker)
        row_specs.append([{"type": "bar"}, {"type": "pie"}, {"type": "bar"}])

    if not valid_tickers:
        print("No valid tickers with data to display.")
        return

    # Create subplots: 3 columns (Duration/Maturity, Credit Info, Allocations) for each row (valid ticker)
    fig = make_subplots(
        rows=len(valid_tickers), cols=3,
        subplot_titles=[
            f"{ticker} - Duration & Maturity" if col == 1 else
            f"{ticker} - Credit Info" if col == 2 else
            f"{ticker} - Allocations"
            for ticker in valid_tickers for col in range(1, 4)
        ],
        specs=row_specs,
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )

    for i, ticker in enumerate(valid_tickers, start=1):
        etf_data = df[df['Ticker'] == ticker]

        # Horizontal Bar Chart: Duration and Maturity
        if not (etf_data[horizontal_bars].fillna(0) == 0).all(axis=None):
            fig.add_trace(
                go.Bar(
                    x=etf_data[horizontal_bars].iloc[0],
                    y=horizontal_bars,
                    orientation='h',
                    name=f"{ticker} - Duration & Maturity",
                    showlegend=False
                ),
                row=i, col=1
            )

        # Pie Chart: Credit Info
        if not (etf_data[credit_variables].fillna(0) == 0).all(axis=None):
            credit_data = etf_data.melt(
                value_vars=credit_variables,
                var_name='Credit Variable',
                value_name='Value'
            )
            fig.add_trace(
                go.Pie(
                    labels=credit_data['Credit Variable'],
                    values=credit_data['Value'],
                    name=f"{ticker} - Credit Info",
                    legendgroup="Credit Info",
                    showlegend=(i == 1)  # Show legend only for the first chart
                ),
                row=i, col=2
            )

        # Horizontal Bar Chart: Allocations
        if not (etf_data[allocation_variables].fillna(0) == 0).all(axis=None):
            fig.add_trace(
                go.Bar(
                    x=etf_data[allocation_variables].iloc[0],
                    y=allocation_variables,
                    orientation='h',
                    name=f"{ticker} - Allocations",
                    showlegend=False
                ),
                row=i, col=3
            )

    # Update layout for better spacing and unified legend
    fig.update_layout(
        height=400 * len(valid_tickers),  # Dynamically adjust height
        title_text="ETF Overview: Duration, Maturity, Credit Info, and Allocations",
        legend=dict(
            title="Credit Info Legend",
            orientation="h",  # Horizontal legend
            yanchor="top",
            y=-0.2,  # Position below the grid
            xanchor="center",
            x=0.5  # Center horizontally
        )
    )

    # Show the grid of charts
    return fig




def generate_etf_grid_charts_with_allocations(etf_tickers, df):
    top_positions = [
        'Top Position 1', 'Top Position 2', 'Top Position 3', 'Top Position 4', 'Top Position 5',
        'Top Position 6', 'Top Position 7', 'Top Position 8', 'Top Position 9', 'Top Position 10'
    ]
    top_weights = [
        'Weight of Top Position 1', 'Weight of Top Position 2', 'Weight of Top Position 3',
        'Weight of Top Position 4', 'Weight of Top Position 5', 'Weight of Top Position 6',
        'Weight of Top Position 7', 'Weight of Top Position 8', 'Weight of Top Position 9',
        'Weight of Top Position 10'
    ]
    sectors = [
        'Realestate', 'Consumer Cyclical', 'Basic Materials', 'Consumer Defensive',
        'Technology', 'Communication Services', 'Financial Services', 'Utilities',
        'Industrials', 'Energy'
    ]
    allocations = [
        'Cash Position', 'Stock Position', 'Bond Position', 'Preferred Position',
        'Convertible Position', 'Other Position'
    ]

    valid_tickers = []
    row_specs = []

    for ticker in etf_tickers:
        etf_data = df[df['Ticker'] == ticker]

        # Skip ticker if Sectors or Top Positions contain only NaN or zero
        if (
            (etf_data[top_weights].fillna(0) == 0).all(axis=None) or
            (etf_data[sectors].fillna(0) == 0).all(axis=None)
        ):
            print(f"Skipping {ticker}: Sectors or Top Positions contain only NaN or zero.")
            continue

        valid_tickers.append(ticker)
        row_specs.append([{"type": "bar"}, {"type": "pie"}, {"type": "bar"}])

    if not valid_tickers:
        print("No valid tickers with data to display.")
        return

    fig = make_subplots(
        rows=len(valid_tickers), cols=3,
        subplot_titles=[
            f"{ticker} - Top Positions" if col == 1 else
            f"{ticker} - Sectors" if col == 2 else
            f"{ticker} - Allocations"
            for ticker in valid_tickers for col in range(1, 4)
        ],
        specs=row_specs,
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )

    for i, ticker in enumerate(valid_tickers, start=1):
        etf_data = df[df['Ticker'] == ticker]

        # Vertical Bar Chart: Top Positions and Weights
        if not (etf_data[top_weights].fillna(0) == 0).all(axis=None):
            positions_data = etf_data.melt(
                value_vars=top_weights,
                var_name='Weight Column',
                value_name='Weight'
            )
            positions_data['Asset'] = positions_data['Weight Column'].map(
                dict(zip(top_weights, etf_data[top_positions].iloc[0]))
            )
            fig.add_trace(
                go.Bar(
                    x=positions_data['Asset'],
                    y=positions_data['Weight'],
                    name=f"{ticker} - Top Positions",
                    showlegend=False
                ),
                row=i, col=1
            )

        # Pie Chart: Sector Contributions
        if not (etf_data[sectors].fillna(0) == 0).all(axis=None):
            sector_data = etf_data.melt(
                value_vars=sectors,
                var_name='Sector',
                value_name='Contribution'
            )
            fig.add_trace(
                go.Pie(
                    labels=sector_data['Sector'],
                    values=sector_data['Contribution'],
                    name="Sectors",
                    legendgroup="Sectors",  # Unified legend for sectors
                    showlegend=(i == 1)  # Show legend only for the first chart
                ),
                row=i, col=2
            )

        # Horizontal Bar Chart: Allocations
        if not (etf_data[allocations].fillna(0) == 0).all(axis=None):
            fig.add_trace(
                go.Bar(
                    x=etf_data[allocations].iloc[0],
                    y=allocations,
                    orientation='h',
                    name=f"{ticker} - Allocations",
                    showlegend=False
                ),
                row=i, col=3
            )

    # Update layout for better spacing and unified legend
    fig.update_layout(
        height=400 * len(valid_tickers),  # Adjust height dynamically based on the number of rows
        title_text="ETF Overview: Top Positions, Sector Contributions, and Allocations",
        legend=dict(
            title="Sectors Legend",
            orientation="h",  # Horizontal legend
            yanchor="top",  # Anchor the top of the legend box
            y=-0.2,  # Position below the plot
            xanchor="center",  # Center horizontally
            x=0.5,  # Position at the center horizontally
            tracegroupgap=30  # Spacing between groups
        ),
    )

    # Show the grid of charts
    return fig



def create_treemap_2(data, path_levels, color_column, title="Treemap Visualization",
                     outlier_method="IQR", remove_na_or_zero=True, **kwargs):
    
    # Normalize columns
    data.columns = data.columns.str.strip()

    # Check required columns
    required_cols = path_levels + ["Ticker"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        st.warning(f"Missing required columns: {missing_cols}")
        return None

    # Filter NA or zero values for coloring
    if remove_na_or_zero and color_column != "Annual Report Expense Ratio":
        data = data[(data[color_column].notna()) & (data[color_column] != 0)]

    # Drop rows missing hierarchy columns
    data = data.dropna(subset=required_cols)
    if data.empty:
        st.warning("No data available after filtering path levels and Ticker.")
        return None

    # Generate hover text
    def format_hover_data(row):
        positions = [
            f"{row[f'Top Position {i}']}: {row[f'Weight of Top Position {i}']}%"
            for i in range(1, 11)
            if pd.notna(row.get(f'Top Position {i}')) and pd.notna(row.get(f'Weight of Top Position {i}'))
        ]
        info_columns = [
            'Annual Report Expense Ratio', 'Annual Holdings Turnover', 'Total Net Assets',
            'Cash Position', 'Stock Position', 'Bond Position', 'Preferred Position',
            'Convertible Position', 'Other Position', 'Bb', 'Aa', 'Aaa', 'A', 'Other', 'B', 
            'Bbb', 'Below B', 'Us Government', 'Realestate', 'Consumer Cyclical', 
            'Basic Materials', 'Consumer Defensive', 'Technology', 
            'Communication Services', 'Financial Services', 'Utilities', 'Industrials', 
            'Energy'
        ]
        additional_info = []
        for col in info_columns:
            val = row.get(col)
            if pd.notna(val):
                if col in ["Total Net Assets", "Annual Holdings Turnover"]:
                    additional_info.append(f"{col}: {val:,.2f}")
                else:
                    additional_info.append(f"{col}: {val * 100:.2f}%")
        return "<br>".join(positions + additional_info) or "No data available"

    data["Hover Info"] = data.apply(format_hover_data, axis=1)

    data["Unique Identifier"] = data[required_cols].astype(str).agg("_".join, axis=1)

    clean_color_column = remove_outliers(data[color_column], method=outlier_method, **kwargs)

    if clean_color_column.isna().all():
        st.warning("All values in the color column are NaN after outlier removal.")
        return None

    color_min = clean_color_column.min()
    color_max = clean_color_column.max()

    try:
        fig = px.treemap(
            data,
            path=path_levels + ["Ticker"],
            color=color_column,
            title=title,
            color_continuous_scale="Viridis",
            range_color=(color_min, color_max)
        )

        fig.update_traces(
            hovertemplate="<b>%{label}</b><br><br>%{customdata[1]}",
            customdata=data[["Ticker", "Hover Info"]].values,
        )
        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        return fig
    except Exception as e:
        st.error(f"Failed to create treemap: {e}")
        return None
