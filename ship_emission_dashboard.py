import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
import os
import datetime
import matplotlib.colors as mcolors

# Configuration
DATA_DIR = "/home/duch/shipping/monthdailysum"
FILE_TEMPLATE = "daysum{month}2023.nc"

pollutants = {
    "CO2": "co2_kg",
    "CH4": "ch4_kg",
    "N2O": "n2o_kg",
    "NOx": "nox_kg",
    "NO2": "no2_kg",
    "SO2": "so2_kg",
    "PM2.5": "pm25_kg",
    "CO": "co_kg",
    "NMVOC": "nmvoc_kg",
    "CO2e": "co2e_kg",
}

# Sidebar

selected_year = 2023
selected_month = st.sidebar.selectbox("Select month", list(range(1,13)), format_func=lambda x: datetime.date(1900, x, 1).strftime('%B'))

# Calculate days in month:
if selected_month == 12:
    days_in_month = 31
else:
    next_month = datetime.date(selected_year, selected_month + 1, 1)
    this_month = datetime.date(selected_year, selected_month, 1)
    days_in_month = (next_month - this_month).days

selected_day = st.sidebar.selectbox("Select day", list(range(1, days_in_month + 1)))

selected_date = datetime.date(selected_year, selected_month, selected_day)

#selected_date = st.sidebar.date_input("Select date", datetime.date(2023, 2, 1))
selected_pollutant = st.sidebar.selectbox("Select pollutant", list(pollutants.keys()))

# Resolve filename
month_str = selected_date.strftime("%b").lower()
nc_file = os.path.join(DATA_DIR, FILE_TEMPLATE.format(month=month_str))

# Load data
if not os.path.exists(nc_file):
    st.error(f"NetCDF file not found: {nc_file}")
    st.stop()

try:
    ds = xr.open_dataset(nc_file)
except Exception as e:
    st.error(f"Failed to open dataset: {e}")
    st.stop()

# Ensure time variable is present
if "time" not in ds.variables:
    st.error("No 'time' variable in dataset.")
    st.stop()

time_var = ds["time"]
time_units = time_var.attrs.get("units", None)

# Handle missing 'units'
if time_units is None:
    #st.warning("No 'units' attribute found in 'time'. Using fallback.")
    base_date = datetime.date(2023, 2, 1)
else:
    try:
        base_date = datetime.datetime.strptime(time_units.split("since")[1].strip(), "%Y-%m-%d %H:%M:%S").date()
    except Exception as e:
        st.error(f"Failed to parse time units: {e}")
        st.stop()

day_index = (selected_date - base_date).days

# Check bounds
if day_index < 0 or day_index >= ds.dims["time"]:
    st.warning(f"Selected date {selected_date} is out of range.")
    st.stop()

# Get selected variable
var_name = pollutants[selected_pollutant]
if var_name not in ds.variables:
    st.error(f"Variable {var_name} not found in dataset.")
    st.stop()

data = ds[var_name][day_index, :, :]

# Mask NaNs
masked_data = np.ma.masked_invalid(data.values)

# Normalize using 99th percentile
try:
    masked_data_np = np.array(masked_data.filled(np.nan), copy=True)
    vmax = np.nanpercentile(masked_data_np, 99)
    norm = mcolors.LogNorm(vmin=1, vmax=max(vmax, 10))
except Exception as e:
    st.error(f"Error during normalization: {e}")
    st.stop()

# Get UTM grid and convert to Lat/Lon
east = ds["easting"].values
north = ds["northing"].values
east_grid, north_grid = np.meshgrid(east, north)

transformer = Transformer.from_crs("epsg:32756", "epsg:4326", always_xy=True)  # UTM zone 56S -> WGS84
lon, lat = transformer.transform(east_grid, north_grid)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_title(f"{selected_pollutant} Emissions on {selected_date}", fontsize=14)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

pc = ax.pcolormesh(lon, lat, masked_data_np, cmap='YlOrRd', norm=norm, shading="auto", transform=ccrs.PlateCarree())
plt.colorbar(pc, ax=ax, label=f"{selected_pollutant} (kg)")

# Add gridlines with labels (tick marks and values)
gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, color='gray')
gl.top_labels = False      # Remove labels on top edge
gl.right_labels = False    # Remove labels on right edge
#gl.xformatter = ccrs.LongitudeFormatter()
#gl.yformatter = ccrs.LatitudeFormatter()
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 12, 'color': 'black'}
gl.ylabel_style = {'size': 12, 'color': 'black'}

st.pyplot(fig)

