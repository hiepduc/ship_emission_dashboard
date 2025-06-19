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

ds_out 
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
month = 2
for day in [10]:
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

#for var in gdf_arrayshort.data_vars:
#    updated_vals = ds_out[var].sel(
#        easting=gdf_arrayshort.easting,
#        northing=gdf_arrayshort.northing
#    )
#    source_vals = gdf_arrayshort[var]
#    if np.allclose(updated_vals, source_vals, equal_nan=True):
#        print(f"{var}: All values match ✅")
#    else:
#        print(f"{var}: Values differ ❌")

# Check the update to ds_out from gdf_arrayshort is correct
for var in ds_out.data_vars:
    ds_out[var] = ds_out[var].astype("float64")

for var in ds.data_vars:
    ds[var] = ds[var].astype("float64")

for var in ds.data_vars:
    updated_vals = ds_out[var].sel(
        easting=ds.easting,
        northing=ds.northing
    )
    source_vals = ds[var]
    if np.allclose(updated_vals, source_vals, equal_nan=True):
        print(f"{var}: All values match ✅")
    else:
        print(f"{var}: Values differ ❌")

for var in ds.data_vars:
    ds[var].attrs["_FillValue"] = -9999.0
    ds[var].attrs["missing_value"] = -9999.0
    # Replace np.nan with -9999.0
    ds[var] = ds[var].fillna(-9999.0)

for var in ds_out.data_vars:
    ds_out[var].attrs["_FillValue"] = -9999.0
    ds_out[var].attrs["missing_value"] = -9999.0
    # Replace np.nan with -9999.0
    ds_out[var] = ds_out[var].fillna(-9999.0)

# Outpout to netcdf file
fname = list_date[0].strftime('%d-%m-%Y')

ds_out.to_netcdf(fname + ".nc")
ds.to_netcdf("ds.nc")
ds_out.to_netcdf("ds_out.nc")
gdf_arrayshort.to_netcdf("gdf_arrayshort.nc")
ds_out.to_netcdf("ds_out.nc", encoding={var: {"zlib": True} for var in ds_out.data_vars})
gdf_arrayshort.to_netcdf("gdf_arrayshort.nc", encoding={var: {"zlib": True} for var in gdf_arrayshort.data_vars})

# Choose a variable to plot (e.g., "co2_kg")
var_to_plot = "co2_kg"

fig, axs = plt.subplots(2, 2, figsize=(14, 6))

# Plot ds_out
ds_out[var_to_plot][1,:,:].plot(ax=axs[0,0], cmap="viridis")
axs[0,0].set_title(f"ds_out — {var_to_plot}")

# Plot gdf_arrayshort
gdf_arrayshort[var_to_plot].plot(ax=axs[0,1], cmap="viridis")
axs[0,1].set_title(f"gdf_arrayshort — {var_to_plot}")

# Plot ds
ds[var_to_plot][1,:,:].plot(ax=axs[1,0], cmap="viridis")
axs[1,0].set_title(f"ds — {var_to_plot}")

for ax in axs:
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')

plt.tight_layout()
plt.show()

# Another check
# Choose a variable to check
var_to_plot = "co2_kg"

# Pull the two fields aligned on the same coords
out_vals = ds_out[var_to_plot].sel(
    easting=gdf_arrayshort.easting,
    northing=gdf_arrayshort.northing
)

gdf_vals = gdf_arrayshort[var_to_plot]

# Compute the difference
diff = out_vals - gdf_vals

# Set up plots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Full ds_out
ds_out[var_to_plot].plot(ax=axs[0], cmap="viridis")
axs[0].set_title("ds_out (full)")

# gdf_arrayshort subset
gdf_vals.plot(ax=axs[1], cmap="viridis")
axs[1].set_title("gdf_arrayshort (subset)")

# Difference
diff.plot(ax=axs[2], cmap="coolwarm", center=0)
axs[2].set_title("Difference (ds_out - gdf)")

for ax in axs:
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

plt.tight_layout()
plt.show()

# Use 1D lat lon
# Create regridder
regridder = xe.Regridder(ds_out, ds_target, method='bilinear')  # or 'nearest_s2d'

# Apply regridding
ds_out_regridded = regridder(ds_out)

# Convert from utm to lat lon
import pyproj

# 1. Create 2D grid of all points
easting_2d, northing_2d = np.meshgrid(ds_out['easting'].values, ds_out['northing'].values)

# 2. Setup the transformer
transformer = pyproj.Transformer.from_crs(
    f"epsg:32756",   # 32756 = UTM Zone 56S
    "epsg:4326",     # WGS84
    always_xy=True
)

# 3. Do the transformation
lon_vals, lat_vals = transformer.transform(easting_2d, northing_2d)

# 4. Assign to ds_out
ds_out = ds_out.assign_coords(
    lon=(("y", "x"), lon_vals),
    lat=(("y", "x"), lat_vals)
)

ds_out.rename({"y": "lat", "x": "lon"})

# 5. (Optional) drop old UTM coordinates if not needed
ds_out = ds_out.drop_vars(['easting', 'northing'])

# Create Netcdf file
# First set the time
nhour=1
datetimelistAEST = [] # an emtpy list to hold date (AEST)
datetimelistUTC = [] # an emtpy list to hold regridding result
hour_first = dt_first.hour
dt_firstutc = pytz.utc.localize(dt_first)  # UTC time
#dt_last = dt_first + datetime.timedelta(hours=days*24)
dt_last = dt_first + datetime.timedelta(hours=nhour)
dt_lastutc = pytz.utc.localize(dt_last)  # UTC time
for i in range(0, nhour, 1):
    dt_temp = dt_firstutc + datetime.timedelta(hours=i)
    datetimelistUTC.append(dt_temp)

# AEST date time and weekday weekend lists
dt_firstaest = dt_firstutc.astimezone(pytz.timezone("Australia/NSW"))

#for i in range(0, days*24, 1):
for i in range(0, nhour, 1):
    dt_temp = dt_firstaest + datetime.timedelta(hours=i)
    datetimelistAEST.append(dt_temp)



# --------------------------------------------------------------------------
# Create a netCDF file containing all hourly shipping species
# --------------------------------------------------------------------------
ship_list = list(ds_out.data_vars)
lats = latemis[:,1]   # array size of emission
lons = latemis[1,:]

nc = Dataset('emission_ship.nc', 'w', format='NETCDF4_CLASSIC')
nc.Conventions = 'CF-1.7'
nc.title = 'Shipping emission'
nc.institution = 'Department of Climate Change, Energy, the Environment and Water, NSW'
nc.source = 'Robin Smith'
nc.history = str(datetime.datetime.utcnow()) + ' Python'
nc.references = ''
nc.comment = 'Start date and time: ' + dt_first.strftime('%d/%m/%Y %H:%M')

# 3km spacing in x and y
#x = np.arange(-150, 153, 3)
#y = np.arange(-100, 100, 3)
nc.createDimension('Time', None)
nc.createDimension('lon', lons.size)
nc.createDimension('lat', lats.size)

# Create coordinate variables lat and lon

clon = nc.createVariable('lon', np.float32, ('lon',))
clon[:] = lons[:]
clon.units = 'degrees'
clon.axis = 'long' # Optional
clon.standard_name = 'longitude'
clon.long_name = 'longitude lat-lon projection'

clat = nc.createVariable('lat', np.float32, ('lat',))
clat[:] = lats[:]
clat.units = 'degrees'
clat.axis = 'lat' # Optional
clat.standard_name = 'latitude'
clat.long_name = 'latitude lat-lon projection'

ntime=1
specsarr = np.zeros((ntime,lats.size,lons.size,)).astype(np.float32)
k = 0
for key in ship_list:
   print ('key=', key)
   temps_var = nc.createVariable(key, datatype=np.float32,
                              dimensions=('Time','lat', 'lon'),
                              zlib=True)
   for i in range(ntime):
      specsarr[i,:,:] =  eval("ds_out." + varname + "[0,:,:]")    #hdd_emis_xarray[i,:,:,k]  # at hour i UTC, species k
   temps_var[:,:,:] = specsarr
   temps_var
   temps_var.units = 'kg/hour'
   temps_var.standard_name = key
   temps_var.long_name = key
#  temps_var.missing_value = -9999
   temps_var


times = datetimelistUTC
calendar = 'standard'
units = 'days since 1970-01-01 00:00'

timevar = nc.createVariable(varname='Times', dimensions=('Time',),
                              datatype='float64')
timevar[:] = netCDF4.date2num(times, units=units, calendar=calendar)
timevar.units = units


nc.close()




#-------------------------------------------------------------------------------------------
# Another way (does not work)
proj_utm = pyproj.CRS.from_epsg(32756)  # UTM Zone 56S
proj_wgs84 = pyproj.CRS.from_epsg(4326)  # WGS84

transformer = pyproj.Transformer.from_crs(proj_utm, proj_wgs84, always_xy=True)

# Convert coordinates
lon_vals, lat_vals = transformer.transform(ds_out['easting'].values*1000, ds_out['northing'].values*1000)

# Add new coordinates
ds_out = ds_out.assign_coords(lon=("easting", lon_vals), lat=("northing", lat_vals))

# Rename dimensions
ds_out = ds_out.rename({'easting': 'x', 'northing': 'y'})

# Fill NaNs and set _FillValue
fill_value = -9999

for var in ds_out.data_vars:
    ds_out[var] = ds_out[var].fillna(fill_value)
    ds_out[var].attrs["_FillValue"] = fill_value

# Save corrected file
ds_out.to_netcdf(
    "ds_out_wgs84.nc",
    encoding={var: {"zlib": True} for var in ds_out.data_vars}
)

print("✅ Exported ds_out_wgs84.nc! You can open with ncview now")

# nother way utm to lat lon using rio
import rioxarray
gdf_arrayshort = gdf_arrayshort.rename({'northing': 'y', 'easting': 'x'})
gdf_arrayshort = gdf_arrayshort.rio.write_crs("EPSG:32756")

gdf_arrayshort_latlon = gdf_arrayshort.rio.reproject("EPSG:4326")


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



