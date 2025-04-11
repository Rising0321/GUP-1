# GUP-1

## Data Processing

###  Download Data

Download the Data from the site below.

Data Source:
- Region division data, Population data: [US Census Bureau](https://www.census.gov/en.html).
- Crime data: [Chicago](https://data.cityofchicago.org/Public-Safety/Crimes-2014/qnmj-8ku6/about_data)  & [Manhattan](https://opendata.cityofnewyork.us/)
- POI data: [POI](http://download.slipo.eu/results/osm-to-csv/)
  - `pd.read_csv(sep=’|’)`
- Human mobility data: [Chicago](https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew/about_data) & [Manhattan](https://data.cityofnewyork.us/Transportation/2014-Yellow-Taxi-Trip-Data/gkne-dk5s/about_data)
- Satellite: Google Map
- Nightlight data: [Earth Observation Group](https://eogdata.mines.edu/products/vnl/#eguide)
- Carbon emission data: [ODIAC](https://db.cger.nies.go.jp/dataset/ODIAC/)
Also, we provide the processed data and processing code for data preprocessing.

### Prepare

Place the downloaded data in the `data` folder. The directory structure should look like this:

```python
├─data
│  ├─Chicago
│  │  ├─Boundary
│  │  │  ├─CensusBlockTIGER2010.csv # file about Chicago's region division
│  │  │  └─Population_by_2010_Census_Block.csv # file about Chicago regions' population
│  │  ├─Crime
│  │  │  └─Crimes_01_2014.csv # file about Chicago's crime data
│  │  ├─POI
│  │  │  └─illinois-latest.osm.csv # file about Chicago's POI data
│  │  ├─Satellite
│  │  │  ├─0.tif # satellite image about Chicago region 1
│  │  │  ├─1.tif
│  │  │  ├─...
│  │  │  └─x.tif
│  │  └─Taxi
│  │  │  └─Taxi_Trips_2014.csv # file about Chicago's taxi flow data
│  ├─Manhattan
│  │  ├─Boundary
│  │  │  ├─CensusBlockTIGER2010.csv # file about Manhattan's region division
│  │  │  └─Population_by_2010_Census_Block.csv # file about Manhattan regions' population
│  │  ├─Crime
│  │  │  └─NYPD_Complaint_Data_Current__Year_To_Date.csv # file about Manhattan's crime data
│  │  ├─POI
│  │  │  └─new-york-latest.osm.csv # file about Manhattan's POI data
│  │  ├─Satellite
│  │  │  ├─0.tif # satellite image about Manhattan region 1
│  │  │  ├─1.tif
│  │  │  ├─...
│  │  │  └─x.tif
│  │  └─Taxi
│  │  │  ├─2014_Yellow_Taxi_Trip_Data_Jau.csv # file about Manhattan's taxi flow data.
│  │  │  └─2014_Green_Taxi_Trip_Data_Jau.csv # file about Manhattan's taxi flow data.
├─preprocess
│  ├─_0_filter_union.py # Aggregate census block regions into census tracts.
│  ├─_2_getBoundary_v2.py  # Download satellite imagery based on regional boundaries..
│  ├─_3_getPOI_v4.py # Proprecess POI data and POI type.
│  ├─_4_getTaxiFlow_v2.py # Proprecess taxi flow data.
│  ├─_5_processCarandNight_v2.py # Proprecess carbon and nightlight data.
│  └─_6_getCrime_v2.py # Proprecess crime data.
```

### Process

Process the data using the scripts in the `preprocess` folder. The scripts are as follows:


```
_0_filter_union.py # Aggregate census block regions into census tracts.
_2_getBoundary_v2.py  # Download satellite imagery based on regional boundaries.
_3_getPOI_v4.py # Proprecess POI data and POI type.
_4_getTaxiFlow_v2.py # Proprecess taxi flow data.
_5_processCarandNight_v2.py # Proprecess carbon and nightlight data.
_6_getCrime_v2.py # Proprecess crime data.

```

## Obtain the Baseline Embedding

For each of the baseline, the region embedding should be processed to $R^{N\times D}$, which N is the number of regions and D is the dimension of the embedding. The embedding should be saved in the `embeddings` folder. The directory structure should look like this:

For the GCN used to embedding the road topology, you should run basline/GCN/GCN_main.py to obtain the region embedding.
```
├─embeddings
│  ├─AutoST
│  │  ├─Chicago.npy 
│  │  │─Manhattan.npy
```


## Pretrain the model

```python
python  MAE_main.py
        --dataset Chicago # select city Chicago or Manhattan
        --use_embedding 2 # use_embedding==2 means use three modalities
        --seed 42 # choose a random seed
        --log_path ./logs/YourName.csv # choose a name
        --gpu cuda:2 # choose a gpu
        --batch_size 32 # choose a batch size
        --lr 1e-4 # choose a learning rate
        --name test # specefy a name
```

## test the model

```python
python  MAE_main.py
        --dataset Chicago # select city Chicago or Manhattan
        --use_embedding 2 # use_embedding==2 means use three modalities
        --seed 42 # choose a random seed
        --log_path ./logs/YourName.csv # choose a name
        --gpu cuda:2 # choose a gpu
        --batch_size 32 # choose a batch size
        --lr 1e-4 # choose a learning rate
        --name test # specefy a name
        --few_shot 10 # sepecify the number of examples
        --test 1 # test the model
        --indicator nightlight # specify the indicator
```


## FineTune the model
```python
python  MAE_FT_main.py
        --dataset Chicago # select city Chicago or Manhattan
        --use_embedding 2 # use_embedding==2 means use three modalities
        --seed 42 # choose a random seed
        --log_path ./logs/YourName.csv # choose a name
        --gpu cuda:2 # choose a gpu
        --batch_size 32 # choose a batch size
        --lr 1e-4 # choose a learning rate
        --name test # specefy a name
        --indicator nightlight # specify the indicator
        --few_shot 10 # sepecify the number of examples
```

