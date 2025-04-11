import os

import pandas as pd
from shapely import Point, wkt
import numpy as np
import re
from datetime import datetime

from tqdm import tqdm


def get_region():
    _dir = f"../data/{city_name}/Boundary/"
    file_name = _dir + "union.csv"
    df = pd.read_csv(file_name)
    return df['the_geom'].apply(wkt.loads).tolist()


def to_hour(time_str):
    time_obj = datetime.strptime(time_str, "%m/%d/%Y %I:%M:%S %p")

    # 设定起始时间 (1月1日 12:00:00 AM)
    start_time = datetime.strptime("01/01/2014 12:00:00 AM", "%m/%d/%Y %I:%M:%S %p")

    # 计算时间差
    time_diff = time_obj - start_time

    # 返回时间差的小时数
    return int(time_diff.total_seconds() // 3600)


def taxi_flow():
    raw_taxi_data_dir = f"../data/{city_name}/Taxi/"
    ans = np.zeros([31 * 24, region_cnt])  # just Jau
    for name in os.listdir(raw_taxi_data_dir):
        print("Begin Constructing {}.".format(name))
        df = pd.read_csv(raw_taxi_data_dir + name)
        if city == 0 and "_01" not in name:
            continue
        for i, row in tqdm(df.iterrows(), total=len(df)):
            stime = row["Trip Start Timestamp"] if city == 0 else row["pickup_datetime"]
            if city == 0 and pd.isna(row["Pickup Centroid Location"]):
                continue
            point = wkt.loads(str(row["Pickup Centroid Location"])) if city == 0 else Point(
                float(row['Pickup_longitude']),
                float(row['Pickup_latitude'])) if "Green" in name else Point(float(row['pickup_longitude']),
                                                                             float(row['pickup_latitude']))
            if point.x == 0 or point.y == 0:
                continue
            sid = -1
            for j, geom in enumerate(geoms):
                if geom.contains(point):
                    sid = j
                    break
            if 0 <= sid <= region_cnt:
                ans[to_hour(stime)][int(sid)] += 1
        print("Constructing {} finished.".format(name))
    columns = [f"region_{i}" for i in range(region_cnt)]
    df = pd.DataFrame(ans, columns=columns)
    path = f"../data/{city_name}/taxi_hour.csv"
    df.to_csv(path, index=False)


if __name__ == '__main__':
    for city in range(2):
        if city == 0:
            continue
        city_names = ["Chicago", "Manhattan"]
        city_name = city_names[city]
        geoms = get_region()
        region_cnt = len(geoms)
        taxi_flow()
        exit(0)
