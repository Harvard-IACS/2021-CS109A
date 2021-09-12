from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd


def haversine(pt, lat2=42.355589, lon2=-71.060175):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1 = pt[0]
    lat1 = pt[1]

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3956  # Radius of earth in miles
    return c * r


def get_distance():
    # Read the data from the file "hubway_stations.csv"
    stations = pd.read_csv("hubway_stations.csv")

    # Read the data from the file "hubway_trips.csv"
    trips = pd.read_csv("hubway_trips.csv")

    station_counts = np.unique(trips['strt_statn'].dropna(), return_counts=True)
    counts_df = pd.DataFrame({'id': station_counts[0], 'checkouts': station_counts[1]})
    counts_df = counts_df.join(stations.set_index('id'), on='id')
    # add to the pandas dataframe the distance using the function we defined above and using map 
    counts_df.loc[:, 'dist_to_center'] = list(map(haversine, counts_df[['lng', 'lat']].values))
    return counts_df
