import os

import pandas as pd
from shapely import Point, wkt
import numpy as np
import re
from rtree import index
from tqdm import tqdm


def get_region():
    _dir = f"../data/{city_name}/Boundary/"
    file_name = _dir + "union.csv"
    df = pd.read_csv(file_name)
    return df['the_geom'].apply(wkt.loads).tolist()

def crime_count():
    raw_dir = f"../data/{city_name}/Crime/"
    res = [0] * region_cnt
    if city == 0:
        file_name = raw_dir + "Crimes_01_2014.csv"
    else:
        file_name = raw_dir + "NYPD_Complaint_Data_Current__Year_To_Date.csv"
    df = pd.read_csv(file_name)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if pd.isna(row['Longitude']) or pd.isna(row['Latitude']):
            continue
        geometry = Point(float(row['Longitude']), float(row['Latitude']))
        for j, geom in enumerate(geoms):
            if geom.contains(geometry):
                res[j] += 1
    path = f"../data/{city_name}/crime.csv"
    columns = [f"region_{i}" for i in range(region_cnt)]
    ans = pd.DataFrame(columns=columns)
    ans.loc[len(ans)] = res
    ans.to_csv(path, index=False)
    print(ans.shape)
    print(ans.head())


if __name__ == '__main__':
    for city in range(2):
        city_names = ["Chicago", "Manhattan"]
        city_name = city_names[city]
        geoms = get_region()
        region_cnt = len(geoms)
        crime_count()
