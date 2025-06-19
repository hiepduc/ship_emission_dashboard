#!/usr/bin/env python3

''' This program calculate the hdd-scaled wood heater emission. The base July emission data is read
    and then is scaled based on temperature at each grid point as obtained from the meteorogical data file
This version for CCAM meteorological data
'''
import cartopy      # always use cartopy first
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import datetime as dt  # Python standard library datetime  module
import numpy as np
import glob
import cartopy.feature as cfeature
from datetime import datetime
import os
import xarray as xr
import xesmf as xe
from xgcm.autogenerate import generate_grid_ds
from xgcm import Grid
from scipy import stats
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords)
from cartopy.feature import NaturalEarthFeature
from matplotlib.cm import get_cmap
import datetime
import pytz

def utc_to_aest(utc_dt):
     return utc_dt.replace(tzinfo=pytz.utc).astimezone(tz=pytz.timezone("Australia/NSW"))
#    return utc_dt.replace(tzinfo=pytz.timezone("Australia/NSW"))

# ----------------------------------------------------------------------------------------
# Open the shipping emission files 
# ---------------------------------------------------------------------------------------
emission_ship = '/home/duch/shipping/monthdailyavg/daysumapr2023.nc'
emis_shipid = Dataset(emission_ship, 'r')
e_ds = xr.open_dataset(emission_ship)

pm25 = emis_shipid.variables['pm25_kg']
pm10 = emis_shipid.variables['pm10_kg']
co = emis_shipid.variables['co_kg']
nox = emis_shipid.variables['nox_kg']
co2e = emis_shipid.variables['co2e_kg']

e_ds.nox_kg[1,:,:].plot()
e_ds.co[1,:,:].plot()
e_ds.co2e[1,:,:].plot()
e_ds.pm10[1,:,:].plot()
plt.show()

emission_timesum = '/home/duch/shipping/monthtimesum/timesumapr2023.nc'
emis_timesumid = Dataset(emission_timesum, 'r')
e_dstimesum = xr.open_dataset(emission_timesum)

pm25 = emis_timesumid.variables['pm25_kg']
pm10 = emis_timesumid.variables['pm10_kg']
co = emis_timesumid.variables['co_kg']
nox = emis_timesumid.variables['nox_kg']
co2e = emis_timesumid.variables['co2e_kg']

e_dstimesum.nox_kg[0,:,:].plot()
plt.show()
e_dstimesum.co[0,:,:].plot()
plt.show()
e_dstimesum.co2e[0,:,:].plot()
plt.show()
e_dstimesum.pm10[0,:,:].plot()
plt.show()

emission_fieldsum = '/home/duch/shipping/monthfieldsum/fieldsumapr2023.nc'
emis_fieldsumid = Dataset(emission_fieldsum, 'r')
e_dsfieldsum = xr.open_dataset(emission_fieldsum)

pm25 = emis_fieldsumid.variables['pm25_kg']
pm10 = emis_fieldsumid.variables['pm10_kg']
co = emis_fieldsumid.variables['co_kg']
nox = emis_fieldsumid.variables['nox_kg']
co2e = emis_fieldsumid.variables['co2e_kg']

e_dsfieldsum.nox_kg[0,:,:].plot()
plt.show()
e_dsfieldsum.co[0,:,:].plot()
plt.show()
e_dsfieldsum.co2e[0,:,:].plot()
plt.show()
e_dsfieldsum.pm10[0,:,:].plot()
plt.show()

#lats_emis = emis_shipid.variables['lat']
#lons_emis = emis_shipid.variables['lon']
lats_emis = emis_shipid.variables['northing']
lons_emis = emis_shipid.variables['easting']

latsize = lats_emis.size
Lat_max = lats_emis[0]; Lat_min= lats_emis[latsize-1]
lonsize = lons_emis.size
Lon_min = lons_emis[0]; Lon_max = lons_emis[lonsize-1]
#resolution_y = lats_emis[1] - lats_emis[0]   # should be same as lats_emis[n] - lats-emis[n-1]
#resolution_x = lons_emis[1] - lons_emis[0]   # should be same as lons_emis[n] - lons-emis[n-1]
resolution_y = (Lat_max - Lat_min)/latsize   # should be same as lats_emis[n] - lats-emis[n-1]
resolution_x = (Lon_max - Lon_min)/lonsize   # should be same as lons_emis[n] - lons-emis[n-1]
resolution = 0.01
resolution_true = 0.01
resolution_sum = 0.1

#nested_grid = xe.util.grid_2d(Lon_min-resolution/2, Lon_max+resolution/2, resolution,  # longitude boundary range and resolution
#                        Lat_min-resolution/2, Lat_max+resolution/2, resolution)  # latitude boundary range and resolution
#nested_grid = xe.util.grid_2d(Lon_min, Lon_max, resolution,  # longitude boundary range and resolution
#                        Lat_min, Lat_max, resolution)  # latitude boundary range and resolution
#nested_grid

nested_grid = xe.util.grid_2d(Lon_min, Lon_max, resolution_x,  # longitude boundary range and resolution
                        Lat_min, Lat_max, resolution_y)  # latitude boundary range and resolution
nested_grid

# Create 2D lat/lon arrays for Basemap
lon2d, lat2d = np.meshgrid(lons_emis, lats_emis)

# using xarray ds
pm25ds = getattr(e_ds, 'pm25_kg')
pm10ds = e_ds.pm10_kg

# Plot Emission data using xarray ds (ds_pma25, ds_pm10) and netcdf variables (pm25, pm10)
proj = ccrs.PlateCarree()

fig, axes = plt.subplots(2,2,figsize=[18,10], subplot_kw={'projection': proj})

ds_pm25 = (e_ds['pm25_kg'])[1,:,:]  # at hour 1
ds_pm10 = (e_ds['pm10_kg'])[18,:,:]  # at hour 18
ds_co = (e_ds['co_kg'])[20,:,:]  # at hour 20

ds_pm25.plot(ax=axes.flatten()[0], cmap='coolwarm', transform=ccrs.PlateCarree(), x='easting', y='northing', vmin=0.05, vmax=0.4,
                        cbar_kwargs={'shrink': 0.5, 'label': 'kg/hour'},infer_intervals=True)

ds_pm10.plot(ax=axes.flatten()[1], cmap='coolwarm', transform=ccrs.PlateCarree(), x='easting', y='northing', vmin=0.05, vmax=0.5,
                        cbar_kwargs={'shrink': 0.5, 'label': 'kg/hour'},infer_intervals=True)

ds_co.plot(ax=axes.flatten()[2], cmap='coolwarm', transform=ccrs.PlateCarree(), x='easting', y='northing', vmin=0.05, vmax=0.5,
                        cbar_kwargs={'shrink': 0.5, 'label': 'kg/hour'},infer_intervals=True)

# using netcdf variables
#CS=plt.contourf(lon2d, lat2d, pm25[1,:,:], 10, transform=ccrs.PlateCarree(),  cmap=get_cmap("jet"))
#CS = axes[1,1].contourf(lon2d, lat2d, pm25[20,:,:], 10, transform=ccrs.PlateCarree(),  cmap=get_cmap("jet"))
#CS = axes[1,1].contourf(lon2d, lat2d, pm25[20,:,:], np.arange(0.05, .5, .05), extend='both', transform=ccrs.PlateCarree(),  cmap=get_cmap("jet"))
CS = axes[1,1].contourf(lons_emis, lats_emis, pm25[20,:,:], np.arange(0.05, .5, .05), extend='both', transform=ccrs.PlateCarree(),  cmap=get_cmap("jet"))
# Add a color bar
fig.colorbar(CS, ax=axes[1,1], shrink=0.9, label='kg/hour')

for ax in axes.flatten():
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top=False; gl.ylabels_right=False

axes.flatten()[0].set_title('PM2.5 daily emission on 2 April')
axes.flatten()[1].set_title('PM10 daily emission on 19 April')
axes.flatten()[2].set_title('CO daily emission on 21 April')
axes.flatten()[3].set_title('PM2.5 daily emission on 21 April')

plt.show()

