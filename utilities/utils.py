from datetime import datetime, timedelta

import pandas as pd
from matplotlib import pyplot as plt


def parse_date(date: str):
    return datetime.strptime(date, '%Y-%m-%d').date()


def show_full_pd_df():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)


def assert_pd_equal_ignore_order(actual, expected):
    return pd.testing.assert_frame_equal(
        actual.sort_values(by=list(actual.columns)).reset_index(drop=True),
        expected.sort_values(by=list(expected.columns)).reset_index(drop=True)
    )


def plot_average_by_group(sales, group_by,plot_rotation=0):
    plot_df = sales.groupby(group_by)['sales'].mean()
    plot_df.plot(kind='line')
    plt.xlabel(group_by)
    plt.ylabel('Average Sales')
    plt.xticks(rotation=plot_rotation)
    plt.show()


def get_split_date(date_array,split_percentage):
    start_date = date_array.min()
    end_date = date_array.max()
    date_range = end_date - start_date
    days_to_add_to_split = int(date_range.days * split_percentage)
    return start_date + timedelta(days=days_to_add_to_split)


def get_date_features(sales_and_stock):
    sales_and_stock['date'] = pd.to_datetime(sales_and_stock['date'], format='%Y.%m.%d')
    sales_and_stock['day_of_week'] = sales_and_stock['date'].dt.dayofweek + 1
    sales_and_stock['month'] = sales_and_stock['date'].dt.month
    sales_and_stock['quarter'] = sales_and_stock['date'].dt.quarter
    sales_and_stock['year'] = sales_and_stock['date'].dt.year
    sales_and_stock['year_quarter'] = sales_and_stock['year'] * 100 + sales_and_stock['quarter']
    sales_and_stock['year_quarter'] = sales_and_stock['year_quarter'].astype("str")
    sales_and_stock['year'] = sales_and_stock['year'].astype("str")
    return sales_and_stock
