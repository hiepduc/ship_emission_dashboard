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

readRDS = robjects.r['readRDS']

# Loop through files
month = 2
for day in [9]:
    for hour in [20]:
        filename = f"2023-{month:02d}-{day:02d} {hour:02d}h.RDS"
        dt_start = f"2023-{month:02d}-{day:02d} {hour:02d}:00"
        print(filename)
        # Load RDS file using pyreadr (cant read this as it has issues with list in RDat)
        #result = pyreadr.read_r(filename)
        #result = readRDS("2023-02-09 20h.RDS")
        result = readRDS(filename)
        #shipei = result[None]  # RDS contains one object
        # Convert to DataFrame
        #dfship = pd.DataFrame(shipei)
        dfship = pd.DataFrame(result)

# Specify here the starting date and ending date of whe emission calculation
dt_start_str = '2021-04-20 00:00'
nday = 30
dt_first = datetime.datetime.strptime(dt_start_str, '%Y-%m-%d %H:%M') # native datetime
dt_first = datetime.datetime.strptime(dt_start, '%Y-%m-%d %H:%M') # native datetime
datetimelistUTC= [dt_first + datetime.timedelta(hours=i) for i in range (0, 25)]

def utc_to_aest(utc_dt):
     return utc_dt.replace(tzinfo=pytz.utc).astimezone(tz=pytz.timezone("Australia/NSW"))
#
# Insert columns easting and northing to the dataframe
# Lower left corner is 103,000 m easting, 5782000m northing
# Upper right corner 903000m Easting, 6912000 Northing
# 801 x 1131 cells
# 


# Remove geometry column (index 13 in Python, equivalent to 14 in R)
# dfshipn = dfship.drop(dfship.columns[13], axis=1)
dfship=dfship.drop([13])  # drop the last colum ('geometry')
result.colnames[0:13]  # last column (14) name is 'geometry' 

dfshipt = dfship.T   # transpose columns and rows
dfshipt.columns=result.colnames[0:13]

# Ensure 'i' and 'j' columns exist
if 'i' not in dfshipt.columns or 'j' not in dfshipt.columns:
    raise ValueError("Missing 'i' or 'j' columns")

# Fill missing combinations of i and j
df_complete = (
    dfshipt
    .set_index(['i', 'j'])
    .unstack(fill_value=np.nan)
    .stack()
    .reset_index()
)

# Add easting and northing
df_complete['easting'] = (df_complete['i'] - 1) * 1000 + 103000
df_complete['northing'] = (df_complete['j'] - 1) * 1000 + 5782000

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df_complete,
    geometry=gpd.points_from_xy(df_complete['easting'], df_complete['northing']),
    crs=CRS.from_proj4("+proj=utm +south +zone=56 +datum=WGS84")
)

# Transform to lat/lon
gdf = gdf.to_crs("EPSG:4326")
#gdf = gdf.rename(columns={gdf.columns[13]: 'lon', gdf.columns[14]: 'lat'})
transformer = Transformer.from_crs(
    "+proj=utm +zone=56 +south +datum=WGS84",  # source CRS
    "EPSG:4326",                               # target CRS (lon/lat)
    always_xy=True
)

# Assuming df is your DataFrame with 'easting' and 'northing' columns
gdf['lon'], gdf['lat'] = transformer.transform(gdf['easting'].values, gdf['northing'].values)

gdfn = gdf.drop(['i','j','easting','northing','geometry'], axis=1)
gdf_array = xr.Dataset.from_dataframe(gdfn.set_index(['lat', 'lon']))
gdfni=gdfn.set_index(['lat', 'lon'])
gdf_array = gdfni.to_xarray()
gdf_array.to_netcdf("output2.nc")

gdf_array = xr.Dataset.from_dataframe(gdf.set_index(['northing', 'easting']))
gdf_array.drop_vars(['geometry', 'lon', 'lat'])
gdf_arrayshort=gdf_array.drop_vars(['geometry', 'lon', 'lat'])

dt_first = pd.to_datetime(dt_first)
gdf_arrayshort = gdf_arrayshort.assign_coords(time = dt_first)
gdf_arrayshort = gdf_arrayshort.expand_dims(dim="time")

#gdf_arrayshort = gdf_arrayshort.expand_dims(time=dt_first)
gdf_arrayshort.to_netcdf('output2.nc')

# Convert from utm to lat lon
import rioxarray
gdf_arrayshort = gdf_arrayshort.rename({'northing': 'y', 'easting': 'x'})
gdf_arrayshort = gdf_arrayshort.rio.write_crs("EPSG:32756")

gdf_arrayshort_latlon = gdf_arrayshort.rio.reproject("EPSG:4326")

# Create a pivot table for NetCDF
df_wide = gdf.pivot_table(index='northing', columns='easting', values=['co2_kg','ch4_kg','n2o_kg','nox_kg', 'no2_kg', 'so2_kg', 'pm25_kg','pm10_kg', 'co_kg', 'nmvoc_kg', 'co2e_kg'])
df_widelatlon = gdf.pivot_table(index='lat', columns='lon', values=['co2_kg','ch4_kg','n2o_kg','nox_kg', 'no2_kg', 'so2_kg', 'pm25_kg','pm10_kg', 'co_kg', 'nmvoc_kg', 'co2e_kg'])

# Convert to xarray for NetCDF writing
ds = xr.DataArray(df_wide).to_dataset(name='value')
ds.to_netcdf("output.nc")

for varname, dr in gdf.data_vars.items():
  # Olny pick variables we need
  if varname in ['TEMP2']:
    temp = xr.DataArray(
       data=dr[:,0,:,:],
       dims=["Time","y", "x"],
       coords=dict(
           Time=datetimelistUTC,
           lat=(["y","x"], lats),
           lon=(["y","x"], lons),
       ),
    )
    dr_temp = regridder_bilinear(temp)
    bilinear_list.append(dr_temp)
#We can merge a list of DataArray to a single Dataset

bilinear_result = xr.merge(bilinear_list)  # merge a list of DataArray to a single Dataset
# NOTE: The next version of xESMF (v0.2) will be able to directly regrid a Dataset,
# so you will not need those additional code. But it is a nice coding exercise anyway.
bilinear_result

# Read a specific file (for checking or exploration)
result = pyreadr.read_r("2023-02-09 20h.RDS")
ship9feb2023 = result[None]
print(ship9feb2023['co2_kg'])

# Remove geometry and convert to DataFrame
ship9feb2023.pop('geometry', None)
dfship = pd.DataFrame(ship9feb2023)


