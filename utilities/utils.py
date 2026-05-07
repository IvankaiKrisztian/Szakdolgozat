"""
Data preparation and visualisation utilities.
"""

import io
import logging
from datetime import datetime, timedelta
import calendar
import holidays
import numpy as np
import pandas

import pandas as pd
from matplotlib import pyplot as plt, pyplot
from models.fuzzy import get_all_fuzzy_membership_values


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def parse_date(date: str):
    """Parse an ISO-format date string ('YYYY-MM-DD') into a datetime.date."""
    return datetime.strptime(date, '%Y-%m-%d').date()


def show_full_pd_df():
    """Remove pandas column-count and width limits — useful for notebook inspection."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)


def get_split_date(df, split_percentage):
    """
    Return the cut-off date for the train/test temporal split.

    Selects the date at position floor(N * split_percentage) in chronological
    order.  All observations with date < split_date form the training set;
    the remainder form the test set.

    Parameters
    ----------
    df : pd.DataFrame  Must contain a 'date' column.
    split_percentage : float  Fraction of observations for training (e.g. 0.8).

    Returns
    -------
    datetime  Last date of the training window.
    """
    number_of_observations = df['date'].count().max()
    first_observations = df.head(int(number_of_observations * split_percentage)).tail(1).reset_index(drop=True)
    return first_observations['date'][0]


def get_date_features(sales_and_stock):
    """
    Append calendar features to the sales/stock DataFrame.

    Derived columns: day_of_week (0=Mon), month (1–12), quarter (1–4),
    year (str), year_quarter (e.g. '2024Q1').  Used for exploratory analysis;
    not used as model inputs.

    Parameters
    ----------
    sales_and_stock : pd.DataFrame  Must contain a 'date' column in YYYY.MM.DD format.

    Returns
    -------
    pd.DataFrame  Original DataFrame with calendar feature columns added.
    """
    sales_and_stock['date'] = pd.to_datetime(sales_and_stock['date'], format='%Y.%m.%d')
    sales_and_stock['day_of_week'] = sales_and_stock['date'].dt.dayofweek
    sales_and_stock['month'] = sales_and_stock['date'].dt.month
    sales_and_stock['quarter'] = sales_and_stock['date'].dt.quarter
    sales_and_stock['year'] = sales_and_stock['date'].dt.year
    sales_and_stock['year_quarter'] = (
        sales_and_stock['year'].astype('str') + 'Q' + sales_and_stock['quarter'].astype('str')
    )
    sales_and_stock['year_quarter'] = sales_and_stock['year_quarter'].astype("str")
    sales_and_stock['year'] = sales_and_stock['year'].astype("str")
    return sales_and_stock


# ---------------------------------------------------------------------------
# Demand construction
# ---------------------------------------------------------------------------

def get_prepared_demand_df(sales_and_stock):
    """
    Construct a clean daily demand series from raw sales and stock data.

    The key modelling decision here is how to handle stockout days: on a day
    where stock = 0, recorded sales understate true demand (censored observation).
    The approach taken is to treat stockout observations as missing and impute
    via linear interpolation between the last in-stock and first post-stockout
    observations.  This is a common pragmatic choice when actual demand during
    stockouts cannot be recovered.

    Steps
    -----
    1. Expand the date range to a contiguous daily sequence (no gaps).
    2. Clip negative stock/sales to zero.
    3. Flag stockout days (stock <= 0) → out_of_stock column.
    4. Set demand = sales on non-stockout days; demand = NaN on stockout days.
    5. Linearly interpolate the NaN demand values.

    Parameters
    ----------
    sales_and_stock : pd.DataFrame  Columns: date, sales, stock.

    Returns
    -------
    pd.DataFrame  Columns: date, sales, stock, out_of_stock, demand.
    """
    dates = pd.DataFrame(
        {'date': pd.date_range(start=sales_and_stock['date'].min(),
                               end=sales_and_stock['date'].max(),
                               freq='D')}
    )
    demand = dates.merge(sales_and_stock, on='date', how='left')
    demand['stock'] = demand['stock'].clip(lower=0).fillna(0)
    demand['sales'] = demand['sales'].clip(lower=0)
    demand['out_of_stock'] = demand['stock'] <= 0

    # Stockout days get NaN demand; non-stockout days use observed sales
    demand['demand'] = np.where(demand['stock'] > 0, demand['sales'].fillna(0), None)
    demand['demand'] = pd.to_numeric(demand['demand'])
    demand['demand'] = demand['demand'].interpolate(method='linear')
    demand['date'] = pd.to_datetime(demand['date'], format='%Y.%m.%d')
    return demand


# ---------------------------------------------------------------------------
# Demand characterisation
# ---------------------------------------------------------------------------

def calculate_adi(sales_and_stock, column_to_calculate_on='sales'):
    """
    Average Demand Interval (ADI) = N / N_nonzero.

    ADI = 1 implies demand occurs every period (smooth/continuous pattern).
    ADI > 1.32 (Syntetos-Boylan threshold) implies intermittent demand.

    Parameters
    ----------
    sales_and_stock : pd.DataFrame
    column_to_calculate_on : str  Column to evaluate (default 'sales').

    Returns
    -------
    float
    """
    observations = sales_and_stock['date'].count()
    non_zero_demand_observations = (
        sales_and_stock[sales_and_stock[column_to_calculate_on] != 0]['date'].count()
    )
    return observations / non_zero_demand_observations


def calculate_cv(sales_and_stock, column_to_calculate_on='sales'):
    """
    Coefficient of Variation (CV) = sigma / mu.

    CV^2 > 0.49 (Syntetos-Boylan threshold) implies erratic demand.
    Combined with ADI, places the series in the Syntetos-Boylan demand
    classification matrix.

    Parameters
    ----------
    sales_and_stock : pd.DataFrame
    column_to_calculate_on : str  Column to evaluate (default 'sales').

    Returns
    -------
    float
    """
    s_deviation = sales_and_stock[column_to_calculate_on].std()
    mean_demand = sales_and_stock[column_to_calculate_on].mean()
    return s_deviation / mean_demand


# ---------------------------------------------------------------------------
# Holidays
# ---------------------------------------------------------------------------

def get_holidays_df():
    """
    Return Hungarian statutory public holidays for 2024–2026.

    "Pihenőnap" entries (compensatory rest days) are excluded because they are
    not statutory holidays — they simply shift the working day pattern without
    representing abnormal demand events.

    Returns
    -------
    pd.DataFrame  Columns: date (datetime), holiday (str).
    """
    hu_holidays = holidays.country_holidays('HU', years=[2024, 2025, 2026])
    holidays_pd = pd.DataFrame([hu_holidays]).melt(var_name="date", value_name="holiday")
    holidays_pd = holidays_pd[~holidays_pd['holiday'].str.contains('Pihenőnap')]
    holidays_pd['date'] = pd.to_datetime(holidays_pd['date'])
    return holidays_pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_not_fired_rules(show=True):
    """
    Configure a StringIO log handler to capture "Rule not fired!" debug messages.

    After the forecasting loop, the buffer can be scanned to count the number
    of timesteps where no rule achieved a positive firing strength.

    Parameters
    ----------
    show : bool  If True, DEBUG-level messages also appear in the console.

    Returns
    -------
    io.StringIO  Buffer containing all captured log output.
    """
    logging.basicConfig(level=logging.DEBUG, force=show)
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)
    return log_stream


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------

def assert_pd_equal_ignore_order(actual, expected):
    """
    Assert DataFrame equality independent of row order.

    Both frames are sorted by all columns before comparison.  Used in unit
    tests where output row ordering is not guaranteed (e.g. after groupby).
    """
    return pd.testing.assert_frame_equal(
        actual.sort_values(by=list(actual.columns)).reset_index(drop=True),
        expected.sort_values(by=list(expected.columns)).reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_average_by_group(sales, group_by, plot_rotation=0):
    """
    Line plot of mean sales grouped by a calendar feature.

    Parameters
    ----------
    sales : pd.DataFrame  Must contain 'sales' and the group_by column.
    group_by : str        One of: "month", "day_of_week", "year_quarter", "year".
    plot_rotation : int   X-tick label rotation in degrees.
    """
    plot_df = sales.groupby(group_by)['sales'].mean()

    fig, ax = plt.subplots()
    ax.plot(plot_df.index, plot_df.values)

    ax.set_xticks(plot_df.index)
    if group_by == 'month':
        ax.set_xticklabels([calendar.month_abbr[m] for m in plot_df.index], rotation=plot_rotation)
    elif group_by == 'day_of_week':
        ax.set_xticklabels([calendar.day_name[d] for d in plot_df.index], rotation=plot_rotation)

    ax.set_xlabel(group_by)
    ax.set_ylabel('Average Sales')
    plt.tight_layout()
    plt.show()


def plot_fuzzy_sets_memberships(fuzzy_sets, min_value, max_value):
    """
    Plot all membership functions mu_i(x) over the demand domain.

    Evaluates each integer x in [min_value-50, max_value+50] against every
    fuzzy set and overlays the curves in a single figure.

    Parameters
    ----------
    fuzzy_sets : list[dict]
    min_value : int  Left bound of the plot domain.
    max_value : int  Right bound of the plot domain.
    """
    fuzzy_list = []
    for i in range(min_value - 50, max_value + 50):
        value = get_all_fuzzy_membership_values(fuzzy_sets, i)
        value["demand"] = int(i)
        fuzzy_list.append(value)

    show_sets = pd.DataFrame(fuzzy_list).melt(
        id_vars="demand", var_name="fuzzy_set", value_name="membership"
    )
    order = ["VeryLowDemand", "LowDemand", "MediumDemand", "HighDemand", "VeryHighDemand"]
    for group in order:
        data = show_sets[show_sets["fuzzy_set"] == group]
        plt.plot(data["demand"], data["membership"], label=group)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_imputed_values_with_missing_stock(demand):
    """
    Plot the demand series with stockout-imputed points highlighted in red.

    Provides a visual audit of the linear interpolation applied on stockout days.

    Parameters
    ----------
    demand : pd.DataFrame  Columns: date, demand, out_of_stock.
    """
    fig, ax = plt.subplots()
    ax.plot(demand['date'], demand['demand'], label='Demand')

    oos = demand[demand['out_of_stock']]
    oos['date'] = pd.to_datetime(oos['date'], format='%Y.%m.%d')
    ax.scatter(oos['date'], oos['demand'], color='red', s=10, label='Out of stock', zorder=5)

    ax.legend()
    fig.autofmt_xdate()
    plt.show()


def plot_forecasts(test_data, forecast):
    """
    Overlay actual demand and one model's predictions on a single axis.

    Parameters
    ----------
    test_data : pd.DataFrame  Columns: date, demand.
    forecast : pd.DataFrame   Columns: date, prediction.
    """
    plt.figure(figsize=(20, 10))
    plt.plot(test_data["date"], test_data["demand"],
             label="Actual", color="steelblue", linewidth=2)
    plt.plot(forecast["date"], forecast["prediction"],
             label="Prediction", color="tomato", linewidth=2, linestyle="--")

    plt.title("Actual vs Predicted Demand")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_actual_demand_and_2_prediction_model(test_data, fuzzy_model_pred, average_pred):
    """
    Three-series comparison plot: actual demand, fuzzy predictions, moving average predictions.

    Parameters
    ----------
    test_data : pd.DataFrame       Columns: date, demand.
    fuzzy_model_pred : pd.DataFrame  Columns: date, prediction.
    average_pred : pd.DataFrame    Columns: date, prediction.
    """
    fig, ax = plt.subplots(figsize=(50, 25))

    ax.plot(test_data["date"], test_data["demand"],
            label="Actual", color="steelblue", linewidth=2)
    ax.plot(fuzzy_model_pred["date"], fuzzy_model_pred["prediction"],
            label="Fuzzy prediction", color="tomato", linewidth=2, linestyle="--")
    ax.plot(average_pred["date"], average_pred["prediction"],
            label="Moving average prediction", color="green", linewidth=2, linestyle=":")

    ax.set_title("Actual demand vs Fuzzy prediction vs Moving average prediction", fontsize=30)
    ax.set_xlabel("Date", fontsize=30)
    ax.set_ylabel("Demand", fontsize=30)
    ax.tick_params(axis="both", labelsize=30)
    plt.xticks(rotation=45)
    ax.legend(fontsize=30)
    ax.grid(True, alpha=0.3)

    ax.margins(x=0)
    ax.set_ylim(0, 125)
    ax.set_xlim(test_data["date"].min(), test_data["date"].max())

    plt.tight_layout()
    plt.show()
