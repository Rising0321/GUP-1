import pandas as pd
from shapely import wkt
from shapely import Point, Polygon
import numpy as np
import json
from tqdm import tqdm

def get_region():
    _dir = f"../data/{city_name}/Boundary/"
    file_name = _dir + "union.csv"
    df = pd.read_csv(file_name)
    return df['the_geom'].apply(wkt.loads).tolist()


def get_category():
    facility_type = {
        1: "Residential",
        2: "Education Facility",
        3: "Cultural Facility",
        4: "Recreational Facility",
        5: "Social Services",
        6: "Transportation Facility",
        7: "Commercial",
        8: "Government Facility (non public safety)",
        9: "Religious Institution",
        10: "Health Services",
        11: "Public Safety",
        12: "Water",
        13: "Miscellaneous"
    }
    with open(f"../data/{city_name}/POI/category.json", 'r') as file:
        return facility_type, json.load(file)


def poi():
    _dir = f"../data/{city_name}/POI/"
    columns = [f"region_{i}" for i in range(region_cnt)]
    ans = pd.DataFrame(columns=columns)
    poi_type = pd.DataFrame(columns=["type"])
    if city == 0:
        file_name = _dir + "illinois-latest.osm.csv"
        df = pd.read_csv(file_name, sep='|')
        res_tables = {}
        for i, row in  tqdm(df.iterrows(), total=len(df)):
            geometry = wkt.loads(row['WKT'])
            # if row['amenity'] is not None:
            if geometry.is_valid:
                for j, geom in enumerate(geoms):
                    if geom.contains(geometry):
                        now_pair = row['SUBCATEGORY']
                        if now_pair not in res_tables:
                            res_tables[now_pair] = [0] * region_cnt
                        res_tables[now_pair][j] += 1
        for key in res_tables:
            if sum(res_tables[key]) >= 10:
                ans.loc[len(ans)] = res_tables[key]
                poi_type.loc[len(poi_type)] = key
    else:
        file_name = _dir + "new-york-latest.osm.csv"
        df = pd.read_csv(file_name, sep='|')
        res_tables = {}
        # facility_type, facility = get_category()
        for i, row in  tqdm(df.iterrows(), total=len(df)):
            point = wkt.loads(row['WKT'])
            for j, geom in enumerate(geoms):
                if geom.contains(point):
                    # T = facility_type[row['FACILITY_T']]
                    # try:
                    #     DOM = facility[T][str(row['FACI_DOM'])]
                    # except KeyError:
                    #     DOM = "Other"
                    now_pair = row['SUBCATEGORY']
                    if now_pair not in res_tables:
                        res_tables[now_pair] = [0] * region_cnt
                    res_tables[now_pair][j] += 1
        for key in res_tables:
            if sum(res_tables[key]) >= 10:
                ans.loc[len(ans)] = res_tables[key]
                poi_type.loc[len(poi_type)] = key
    path = f"../data/{city_name}/poi_v4.csv"
    ans.to_csv(path, index=False)
    assert (len(ans) == len(poi_type))
    poi_type.to_csv(f"../data/{city_name}/poi_type_v4.csv", index=False)


if __name__ == '__main__':
    for city in range(2):
        city_names = ["Chicago", "Manhattan"]
        city_name = city_names[city]
        geoms = get_region()
        region_cnt = len(geoms)
        poi()
        exit(0)
