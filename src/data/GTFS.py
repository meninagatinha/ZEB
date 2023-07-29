#!/usr/bin/env python3
import pandas as pd
import requests
import zipfile
import io
from src.utils import parse_dates
from datetime import timedelta, date


LINKED_TRIP_GROUPS = ['block_id', 'service_id']
PRIMARY_KEY_SET = LINKED_TRIP_GROUPS + ['trip_id']
TIME_COLS = ['departure_time', 'arrival_time']


class GTFS:
    """
    This is a container class for the GTFS data. 
    
    Reads in data from the latest URL, which can be found by visiting:
     https://github.com/MobilityData/mobility-database-catalogs
    and downloading the sources spreadsheet, found at:
     https://bit.ly/catalogs-csv
    """

    GTFS_PROVIDER = 'New Jersey Transit (NJ Transit)'
    GTFS_CATALOGS_CSV = 'https://bit.ly/catalogs-csv'
    FILE_EXTENSION = '.txt'
    URL_COLUMN = 'urls.latest'
    SOURCES_NAME_VALUE = 'Bus'

    def __init__(self, gtfs_output_path):
        self.source_url = self.get_current_source_url()
        self.download_gtfs_data(gtfs_output_path)
        self.calendar_dates = pd.read_csv(gtfs_output_path + 'calendar_dates.txt', parse_dates=['date'])
        self.routes = pd.read_csv(gtfs_output_path + 'routes.txt')
        self.shapes = pd.read_csv(gtfs_output_path + 'shapes.txt')
        self.stop_times = pd.read_csv(gtfs_output_path + 'stop_times.txt', parse_dates=TIME_COLS,
                                      date_parser=parse_dates)
        self.stops = pd.read_csv(gtfs_output_path + 'stops.txt')
        self.trips = pd.read_csv(gtfs_output_path + 'trips.txt')
        self._add_descending_stop_sequence_rank()
        self._add_trip_departure_arrival_times()

    def get_current_source_url(self):
        """
        Automatically pulls in the sources csv from the link provided on the GitHub page
        and gets the NJ Transit most recent source URL
        """
        sources_csv = pd.read_csv(self.GTFS_CATALOGS_CSV)
        source_url = sources_csv[(sources_csv.provider == self.GTFS_PROVIDER) &
                                 (sources_csv.name == self.SOURCES_NAME_VALUE)][self.URL_COLUMN]
        return source_url.iloc[0]

    def download_gtfs_data(self, gtfs_output_path):
        """
        Downloads a zip file containing the following relevant datasets:
        Calendar Dates
        Routes
        Shapes
        Stop Times
        Trips
        See documentation for descriptions. Writes the zip to a folder in the current directory.
        """
        r = requests.get(self.source_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        today = date.today().strftime('%Y-%m-%d')
        z.extractall(gtfs_output_path + today)

    def _add_descending_stop_sequence_rank(self):
        """
        Looks at the stop_times dataset and adds a column that ranks the stop_sequence in descending order.
        Allows for filtering on the last stop in the sequence
        """
        self.stop_times['stop_sequence_rank'] = self.stop_times.groupby('trip_id')['stop_sequence'].rank(
            ascending=False)

    def _add_trip_departure_arrival_times(self):
        """
        The stop_times data has information on the arrival & departure time for a given trip & stop.
        This function will take the first arrival time for the trip and the last arrival time for the trip
        from the stop_times dataset. This is equivalent to taking arrival time when stop_sequence = 1
        and taking departure time when stop_sequence = max(stop_sequence) when grouped by trip. 
        These values are added to the trips dataset and presupposes _add_descending_stop_sequence_rank has been called.
        """
        max_stop_sequences = self.stop_times[self.stop_times.stop_sequence_rank == 1][['trip_id',
                                                                                       'departure_time',
                                                                                       'stop_id',
                                                                                       'shape_dist_traveled']]
        min_stop_sequences = self.stop_times[self.stop_times.stop_sequence == 1][['trip_id', 'arrival_time']]
        self.trips = self.trips.merge(max_stop_sequences, on='trip_id').merge(min_stop_sequences, on="trip_id")
        self.trips = self.trips.rename(
            {'departure_time': 'final_departure_time', 'arrival_time': 'first_arrival_time',
             'stop_id': 'final_stop_id'}, axis=1)
        self.trips['duration'] = self.trips.final_departure_time - self.trips.first_arrival_time

    @staticmethod
    def add_trip_next_arrival_time(trips_df):
        """
        I would make this a normal method; however, it is unnecessary to get the next arrival time for all trips
        since we just care about Wayne for this analysis. 
        
        Rather than arbitrarily filter the data within this class, I'd rather leave the data as general as possible,
        and any manipulations that result in a reduction of data (filtering, dropping columns & bad observations)
        should be left out of this class and performed as a separate analysis

        This method takes a trips_df: DataFrame, which is a subset of the self.trips defined above.
        It then adds a column called "next_arrival_time" which is the forward lag of the arrival_time column.
        time_waiting is also calculated.
        """
        trips_df = trips_df.sort_values(by=LINKED_TRIP_GROUPS + ['final_departure_time', 'first_arrival_time'])
        service_blocks = trips_df.groupby(LINKED_TRIP_GROUPS)
        trips_df['next_arrival_time'] = service_blocks.first_arrival_time.shift(-1)
        trips_df['time_waiting'] = trips_df.next_arrival_time - trips_df.final_departure_time
        return trips_df


if __name__ == '__main__':
    gtfs = GTFS('datasets/data/')
    assert gtfs.get_current_source_url() == 'https://storage.googleapis.com/storage/v1/b/mdb-latest/o/us-new-jersey-new-jersey-transit-nj-transit-gtfs-508.zip?alt=media'
    assert parse_dates('24:10:10') == timedelta(days=1, seconds=610)
