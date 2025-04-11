import pandas as pd
from shapely import wkt
from shapely import Point, Polygon
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform
import os


def get_region():
    _dir = f"../data/{city_name}/Boundary/"
    file_name = _dir + "union.csv"
    df = pd.read_csv(file_name)
    return df['the_geom'].apply(wkt.loads).tolist()


def pixel_2_84(coord, src_crs, dst_crs):
    res = transform(src_crs, dst_crs, [coord[0]], [coord[1]])
    return [res[0][0], res[1][0]]


def eight4_2_origin(coord, src_crs, dst_crs, transform_matrix):
    temp = transform(dst_crs, src_crs, [coord[0]], [coord[1]])
    return rasterio.transform.rowcol(transform_matrix, temp[0], temp[1])


def get_coords(src, row, col, src_crs, dst_crs):
    top_left_latlon = pixel_2_84(src.xy(row, col, offset='ul'), src_crs, dst_crs)
    top_right_latlon = pixel_2_84(src.xy(row, col, offset='ur'), src_crs, dst_crs)
    bottom_left_latlon = pixel_2_84(src.xy(row, col, offset='ll'), src_crs, dst_crs)
    bottom_right_latlon = pixel_2_84(src.xy(row, col, offset='lr'), src_crs, dst_crs)
    return [top_left_latlon, top_right_latlon, bottom_left_latlon, bottom_right_latlon]


def get_mean_coords(src, row, col, src_crs, dst_crs):
    top_left_latlon = pixel_2_84(src.xy(row, col, offset='ul'), src_crs, dst_crs)
    bottom_left_latlon = pixel_2_84(src.xy(row, col, offset='ll'), src_crs, dst_crs)
    return [(top_left_latlon[0] + bottom_left_latlon[0]) / 2, (top_left_latlon[1] + bottom_left_latlon[1]) / 2]


def get_bbx(coords):
    # print(coords, coords[0][0])
    x = [i[0] for i in coords]
    y = [i[1] for i in coords]
    return [(min(x), min(y)), (max(x), max(y))]


def merge_bbx(bbx1, bbx2):
    return [(min(bbx1[0][0], bbx2[0][0]), min(bbx1[0][1], bbx2[0][1])),
            (max(bbx1[1][0], bbx2[1][0]), max(bbx1[1][1], bbx2[1][1]))]


def select():
    columns = [f"region_{i}" for i in range(region_cnt)]
    ans = pd.DataFrame(columns=columns)
    res = [0] * region_cnt

    from tqdm import tqdm

    sum_value = 0
    cnt_value = 0
    src = rasterio.open(file_name)
    data = src.read(1)  # 读取第一个波段的数据

    print(src.meta)
    # exit(0)
    src_crs = src.crs
    dst_crs = 'EPSG:4326'
    transform_matrix = src.transform
    for i in tqdm(range(region_cnt)):
        geom = geoms[i]
        # print(exact_boundary)
        bbx_boundary = geom.bounds

        row_min, col_min = eight4_2_origin(bbx_boundary[0:2], src_crs, dst_crs, transform_matrix)
        row_max, col_max = eight4_2_origin(bbx_boundary[2:4], src_crs, dst_crs, transform_matrix)

        row_min, col_min = row_min[0], col_min[0]
        row_max, col_max = row_max[0], col_max[0]
        if row_min > row_max:
            row_min, row_max = row_max, row_min
        if col_min > col_max:
            col_min, col_max = col_max, col_min
        # print(row_min, row_max, col_min, col_max)

        value_max = -123456789
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                # coords = get_mean_coords(src, row, col, src_crs, dst_crs)
                # now_point = Point(coords[0], coords[1])
                value_max = max(value_max, data[row][col])
        # print(row_max - row_min, row_min, value_max)
        if value_max != -123456789:
            sum_value += value_max
            cnt_value += 1
        res[i] = value_max

    for i in range(region_cnt):
        if res[i] == -123456789:
            res[i] = sum_value / cnt_value
    ans.loc[len(ans)] = res
    ans.to_csv(ans_name, index=False)


def clip():
    columns = [f"region_{i}" for i in range(region_cnt)]
    ans = pd.DataFrame(columns=columns)

    _dir = f"../data/{city_name}/Boundary/"
    file_name = _dir + "union.csv"
    df = pd.read_csv(file_name)

    if city == 0:
        res = df['TOTAL POPULATION'].to_list()
    else:
        res = df["POPN_TOTAL"].to_list()

    assert (len(res) == region_cnt)
    ans.loc[len(ans)] = res
    ans.to_csv(ans_name, index=False)


if __name__ == '__main__':
    for city in range(2):
        for target in range(1):
            city_names = ["Chicago", "Manhattan"]
            city_name = city_names[city]
            target_names = ["nightlight", "carbon"]
            target_name = target_names[target]
            ans_name = f"../data/{city_name}/{target_name}.csv"
            geoms = get_region()
            region_cnt = len(geoms)
            file_name = f"../data/{target_name}.tif"
            # if target == 0:
            #     clip()
            # else:
            select()
