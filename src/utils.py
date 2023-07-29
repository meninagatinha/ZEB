#!/usr/bin/env python3
import typing
import re
from bisect import bisect
from datetime import timedelta
from collections import defaultdict, namedtuple
from pandas import DataFrame, Series, isnull


TIME_INTERVALS = [timedelta(minutes=x) for x in range(0, 24 * 60, 15)]


def get_time_interval(time: timedelta, **time_interval_args: object) -> namedtuple:
    """
    
    """
    interval_size = timedelta(**time_interval_args)
    n_goes_into = time // interval_size
    Interval = namedtuple('Interval', ['index', 'interval'])
    return Interval(n_goes_into, n_goes_into * interval_size)


def add_interval_counts(row: Series,
                        interval_count_map: typing.DefaultDict[timedelta, int],
                        **time_interval_args) -> Series:
    """
    Takes as input a row from a DataFrame, assuming departure_time and next_arrival_time exist
    Finds the first interval during which the bus is waiting at the terminal
    Finds the last interval during which the bus is waiting at the terminal
    Adds +1 to the counts for the range(lo, hi) in the interval_count_map
    """
    lower_bound = upper_bound = -2
    interval_size = timedelta(**time_interval_args)
    if not isnull(row.next_arrival_time):
        lower_bound: namedtuple = get_time_interval(row.final_departure_time, **time_interval_args)
        upper_bound: namedtuple = get_time_interval(row.next_arrival_time, **time_interval_args)
        for i in range(upper_bound.index - lower_bound.index + 1):
            current_interval = lower_bound.interval + i * interval_size
            interval_count_map[current_interval] += 1
    row['left_index'], row['right_index'] = lower_bound.index, upper_bound.index
    return row


def add_interval_counts_wrapper(dataframe: DataFrame) -> DataFrame:
    """
    A wrapper function to apply add_interval_counts to a grouped Series or DataFrame
    """
    d = defaultdict(int)
    dataframe.apply(lambda x: add_interval_counts(x, d), axis=1)
    return DataFrame(dict(interval=[str(TIME_INTERVALS[key]) for key in d.keys()],
                          num_buses=d.values()),
                     index=list(d.keys()))


def parse_dates(date_str: str) -> timedelta:
    """
    Departure Time and Arrival Time show up in the format "HH:MM:SS"

    Want to convert this to a timedelta type so it is easier to work with and sort.
    """
    pattern = r'([0-2]?\d|2[0-3]):([0-5]?\d):([0-5]?\d)'
    hms = [int(x) for x in re.search(pattern, date_str).groups()]
    return timedelta(hours=hms[0], minutes=hms[1], seconds=hms[2])
