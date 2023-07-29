#!/usr/bin/env python3
import src.utils
import pytest
import typing
from datetime import timedelta
from pandas import Series
from collections import defaultdict


@pytest.mark.parametrize(
    'date_str,expected',
    [
        pytest.param(
            '24:10:10', timedelta(days=1, seconds=610), id='24hours'
        ),
        pytest.param(
            '00:00:00', timedelta(seconds=0), id="midnight"
        ),
        pytest.param(
            '24:00:00', timedelta(days=1), id='1day'
        ),
        pytest.param(
            '13:14:59', timedelta(hours=13, minutes=14, seconds=59), id='normal'
        )
    ],
)
def test_parse_dates(date_str: str, expected: str):
    assert src.utils.parse_dates(date_str) == expected


@pytest.mark.parametrize(
    'time,expected,kwargs',
    [
        pytest.param(
            timedelta(hours=1), timedelta(hours=1), dict(minutes=15), id='1hour,15minutes'
        ),
        pytest.param(
            timedelta(hours=1, minutes=1), timedelta(hours=1), dict(minutes=15), id="1hour1min,15minutes"
        ),
        pytest.param(
            timedelta(seconds=0), timedelta(seconds=0), dict(minutes=30),  id='0hours,30minutes'
        ),
        pytest.param(
            timedelta(days=1, hours=12), timedelta(days=1), dict(days=1),  id='1.5days,1day'
        )
    ],
)
def test_get_time_interval(time: timedelta, expected: timedelta, kwargs):
    assert src.utils.get_time_interval(time=time, **kwargs) == expected


@pytest.mark.parametrize(
    'row,interval_count_map,kwargs,expected',
    [
        pytest.param(
            Series([timedelta(hours=7, minutes=3), timedelta(hours=7, minutes=30)],
                   index=['final_departure_time', 'next_arrival_time']),
            defaultdict(int),
            dict(minutes=15),
            Series([timedelta(hours=7, minutes=3),
                    timedelta(hours=7, minutes=30),
                    28,
                    30], index=['final_departure_time', 'next_arrival_time', 'left_index', 'right_index']),
            id='1hour,15minutes'
        ),
    ],
)
def test_add_interval_counts(row: Series, interval_count_map: typing.DefaultDict[timedelta, int], kwargs, expected):
    assert src.utils.add_interval_counts(row=row, interval_count_map=interval_count_map, **kwargs).equals(expected)
