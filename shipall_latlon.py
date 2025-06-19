#!/usr/bin/env python3
import pandas as pd
import numpy as np
import geopandas as gpd
from pyproj import CRS, Transformer
import xarray as xr
import os
import pyreadr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import datetime
import pytz
import xesmf as xe
import rioxarray
from netCDF4 import Dataset
import netCDF4
import matplotlib.pyplot as plt

readRDS = robjects.r['readRDS']
Rdata_dir = "/mnt/scratch_lustre/ar_vems_scratch/shipping/Data/Shipping_original_20250306/"

def utc_to_aest(utc_dt):
     return utc_dt.replace(tzinfo=pytz.utc).astimezone(tz=pytz.timezone("Australia/NSW"))

#
# Insert columns easting and northing to the dataframe
# Lower left corner is 103,000 m easting, 5782000m northing
# Upper right corner 903000m Easting, 6912000 Northing
# 801 x 1131 cells
#
# Define the coordinate vectors
#x = np.linspace(1, 801, 801)
#y = np.linspace(1, 1131, 1131)
# Create the meshgrid
#i, j = np.meshgrid(x, y)

easting_min = 103    # in km
easting_max = 903
northing_min = 5782
northing_max = 6912
resolution_x = 1
resolution_y = 1
northings = np.arange(northing_min, northing_max + resolution_y, resolution_y)
eastings = np.arange(easting_min, easting_max + resolution_x, resolution_x)

lat_min = -38.02
lon_min = 148.48
lat_max = -27.86
lon_max = 157.09
#resol_lat = (lat_max - lat_min)/1130    # 0.009
#resol_lon = (lon_max - lon_min)/800   # 0.0107
lat_resolution = 0.01
lon_resolution = 0.01

lat_1d = np.arange(lat_min, lat_max + lat_resolution, lat_resolution)
lon_1d = np.arange(lon_min, lon_max + lon_resolution, lon_resolution)

# Create a new 1D target grid
ds_target = xr.Dataset({
    'lat': (['lat'], lat_1d),
    'lon': (['lon'], lon_1d),
})

#nested_grid = xe.util.grid_2d(easting_min, northing_max, resolution_x,  # longitude boundary range and resolution
#                        northing_min, northing_max, resolution_y)  # latitude boundary range and resolution
#nested_grid

ds_out = xr.Dataset(
    {
        "northing": (["northing"], np.arange(5782000, 6912000, 1000), {"units": "m"}),
        "easting": (["easting"], np.arange(103000, 903000, 1000), {"units": "m"}),
    }
)

#ds_out['co2_kg'] = (['northing', 'easting'], np.random.rand(1130,800))
zero_array = np.empty((1130, 800))
#zero_array.fill(0)
#ds_out['ch4_kg'] = (['northing', 'easting'], zero_array)
#zero_array.fill(np.nan)
#ds_out['no2_kg'] = (['northing', 'easting'], zero_array)
#ds_out = ds_out.rename({'northing': 'y', 'easting': 'x'})

# Loop through files
#collecting datasets when looping over shipping RDat files
list_da = []
list_date = []
list_outlatlon = []
month = 3
for day in range(1,1,1):
    #for hour in [20,21]:
    for hour in range(0,24,1):
        filename = f"2023-{month:02d}-{day:02d} {hour:02d}h.RDS"
        dt_start = f"2023-{month:02d}-{day:02d} {hour:02d}:00"
        print(filename)
        # Load RDS file using pyreadr (cant read this as it has issues with list in RDat)
        #result = pyreadr.read_r(filename)
        #result = readRDS("2023-02-09 20h.RDS")
        result = readRDS(Rdata_dir + filename)
        #shipei = result[None]  # RDS contains one object
        # Convert to DataFrame
        #dfship = pd.DataFrame(shipei)
        dfship = pd.DataFrame(result)
        #
        # Specify here the starting date 
        dt_first = datetime.datetime.strptime(dt_start, '%Y-%m-%d %H:%M') # native datetime
        list_date.append(dt_first)
        #datetimelistUTC= [dt_first + datetime.timedelta(hours=i) for i in range (0, 25)]
        #
        # Remove geometry column (index 13 in Python, equivalent to 14 in R)
        # dfshipn = dfship.drop(dfship.columns[13], axis=1)
        dfship=dfship.drop([13])  # drop the last colum ('geometry')
        result.colnames[0:13]  # last column (14) name is 'geometry' 
        #
        dfshipt = dfship.T   # transpose columns and rows
        dfshipt.columns=result.colnames[0:13]
        # Ensure 'i' and 'j' columns exist
        if 'i' not in dfshipt.columns or 'j' not in dfshipt.columns:
            raise ValueError("Missing 'i' or 'j' columns")
        #
        # Fill missing combinations of i and j
        df_complete = (
            dfshipt
            .set_index(['i', 'j'])
            .unstack(fill_value=np.nan)
            .stack()
            .reset_index()
        )
        #
        # Add easting and northing
        df_complete['easting'] = (df_complete['i'] - 1) * 1000 + 103000
        df_complete['northing'] = (df_complete['j'] - 1) * 1000 + 5782000
        #
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df_complete,
            geometry=gpd.points_from_xy(df_complete['easting'], df_complete['northing']),
            crs=CRS.from_proj4("+proj=utm +south +zone=56 +datum=WGS84")
        )
        #
        # Transform to lat/lon
        gdf = gdf.to_crs("EPSG:4326")
        #gdf = gdf.rename(columns={gdf.columns[13]: 'lon', gdf.columns[14]: 'lat'})
        transformer = Transformer.from_crs(
            "+proj=utm +zone=56 +south +datum=WGS84",  # source CRS
            "EPSG:4326",                               # target CRS (lon/lat)
            always_xy=True
        )
        #
        # Assuming df is your DataFrame with 'easting' and 'northing' columns
        gdf['lon'], gdf['lat'] = transformer.transform(gdf['easting'].values, gdf['northing'].values)
        #
        gdfn = gdf.drop(['i','j','easting','northing','geometry'], axis=1)
        #gdf_array = xr.Dataset.from_dataframe(gdfn.set_index(['lat', 'lon']))
        gdfni=gdfn.set_index(['lat', 'lon'])
        gdfni = gdfni.assign_coords(time = dt_first)
        gdfni = gdfni.expand_dims(dim="time")
        list_outlatlon.append(gdfni)
        # Now stack dataarrays in list
        dslatlon = xr.combine_by_coords(list_outlatlon)
        gdfni.to_xarray().to_netcdf("output2.nc")
        #
        gdf_array = xr.Dataset.from_dataframe(gdf.set_index(['northing', 'easting']))
        gdf_arrayshort=gdf_array.drop_vars(['geometry', 'lon', 'lat', 'i', 'j'])
        #
        dt_first = pd.to_datetime(dt_first)
        gdf_arrayshort = gdf_arrayshort.assign_coords(time = dt_first)
        gdf_arrayshort = gdf_arrayshort.expand_dims(dim="time")
        #gdf_arrayshort.to_netcdf('output2.nc')
        list_da.append(gdf_arrayshort)
        print(result.colnames[2:12])
        for item in result.colnames:
           #print(item)
           if item not in ['i', 'j', 'geometry']:
              print(item)
              ds_out[item] = (['northing', 'easting'], zero_array)
              zero_array.fill(np.nan)
        # Now stack dataarrays in list
        ds = xr.combine_by_coords(list_da)
        #ds_out = ds_out.assign_coords(time = dt_first)
        ds_out = ds_out.assign_coords(time = list_date)
        # ds_out = ds_out.expand_dims(dim="time")
        #ds_out = ds_out.rename({'northing': 'y', 'easting': 'x'})
        # Step 1: Copy and make writable
        ds_out = ds_out.copy()
        for var in ds_out.data_vars:
            ds_out[var].data = ds_out[var].data.copy()
        # Step 2: Use .update to patch values into ds_out
        #ds_out.update(gdf_arrayshort)
        ds_out.update(ds)
        #gdf_arrayshort.combine_first(ds_out)
        # Check the update to ds_out from gdf_arrayshort is correct
        #for var in ds_out.data_vars:
        #    ds_out[var] = ds_out[var].astype("float64")
        #
        for var in gdf_arrayshort.data_vars:
            gdf_arrayshort[var] = gdf_arrayshort[var].astype("float64")
        # Check the update to ds_out from gdf_arrayshort is correct
        for var in ds_out.data_vars:
            ds_out[var] = ds_out[var].astype("float64")
        for var in ds.data_vars:
            ds[var] = ds[var].astype("float64")
        #for var in ds.data_vars:
        #    updated_vals = ds_out[var].sel(
        #        easting=ds.easting,
        #        northing=ds.northing
        #    )
        #    source_vals = ds[var]
        #    if np.allclose(updated_vals, source_vals, equal_nan=True):
        #        print(f"{var}: All values match ✅")
        #    else:
        #        print(f"{var}: Values differ ❌")
        # Output to netcdf file
        fname = list_date[0].strftime('%d-%m-%Y')
        ds_out.to_netcdf(fname + ".nc")
        #ds.to_netcdf("ds.nc")
        #ds_out.to_netcdf("ds_out.nc")
        # 5. (Optional) drop old UTM coordinates if not needed
        #ds_out = ds_out.drop_vars(['easting', 'northing'])
       
 
