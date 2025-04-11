import pandas as pd
import os

from shapely import unary_union, wkt
from shapely.geometry import base
import geopandas as gpd
import shapefile


# def filter_top_ten():
#     df = pd.read_csv('../data/Chicago/Boundary/CensusBlockTIGER2010.csv')
#
#     pop = pd.read_csv("../data/Chicago/Boundary/Population_by_2010_Census_Block.csv")
#
#     merged_df = pd.merge(df, pop[['CENSUS BLOCK FULL', 'TOTAL POPULATION']], left_on='GEOID10',
#                          right_on='CENSUS BLOCK FULL')
#
#     merged_df = merged_df[(merged_df['TOTAL POPULATION'] != 0) & (merged_df['TOTAL POPULATION'] <= 2000)]
#
#     merged_df = merged_df.sort_values(by='TOTAL POPULATION', ascending=False)
#
#     top_10_percent = int(len(merged_df) * 0.1)
#
#     top_10_population = merged_df.head(top_10_percent)
#
#     top_10_population = top_10_population.sample(frac=1, random_state=42).reset_index(drop=True)
#
#     top_10_population.to_csv('../data/Chicago/Boundary/top_10_regions.csv', index=False)


def merge_geometries(group):
    return unary_union(group['the_geom'])


def union():
    if city == 0:
        file_name = _dir + "CensusBlockTIGER2010.csv"
        df = pd.read_csv(file_name)
        pop = pd.read_csv("../data/Chicago/Boundary/Population_by_2010_Census_Block.csv")
        merged_df = pd.merge(df, pop[['CENSUS BLOCK FULL', 'TOTAL POPULATION']], left_on='GEOID10',
                             right_on='CENSUS BLOCK FULL')
        merged_df['the_geom'] = merged_df['the_geom'].apply(wkt.loads)
        merged_df['TOTAL POPULATION'] = merged_df.groupby('TRACTCE10')['TOTAL POPULATION'].transform('sum')
        df = merged_df.groupby('TRACTCE10').agg({
            'TOTAL POPULATION': 'first',
            'the_geom': lambda x: unary_union(x)
        }).reset_index()
    else:
        file_name = _dir + "36061_block.csv"
        df = pd.read_csv(file_name)
        pop = pd.read_csv('../data/Manhattan/Boundary/36061_pop.csv')
        pop = pop[pop['BORONAME'] == "Manhattan"]
        print(pop.shape)
        merged_df = pd.merge(df, pop[['BLKID', 'POPN_TOTAL']], left_on='GEOID10',
                             right_on='BLKID')
        merged_df['the_geom'] = merged_df['geometry'].apply(wkt.loads)
        merged_df['POPN_TOTAL'] = merged_df.groupby('TRACTCE10')['POPN_TOTAL'].transform('sum')
        df = merged_df.groupby('TRACTCE10').agg({
            'POPN_TOTAL': 'first',  # 保留总人口（相同值）
            'the_geom': lambda x: unary_union(x)  # 合并几何
        }).reset_index()
    print(df.shape)
    df.to_csv(_dir + "union.csv", index=False)


if __name__ == '__main__':
    for city in range(2):
        city_names = ["Chicago", "Manhattan"]
        city_name = city_names[city]

        _dir = f"../data/{city_name}/Boundary/"
        union()
        exit(0)
