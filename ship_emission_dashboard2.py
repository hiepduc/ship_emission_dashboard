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
import calendar

# Configuration
DATA_DIR = "/home/duch/shipping/monthdailysum"

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

# Available months based on actual files
available_months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "nov"]
month_name_map = {m: calendar.month_name[i+1] for i, m in enumerate(available_months)}

# Sidebar selections
st.sidebar.title("Shipping Emission Dashboard")
selected_month_abbr = st.sidebar.selectbox("Select month", available_months, format_func=lambda m: m.capitalize())
selected_year = 2023
selected_month_num = list(calendar.month_abbr).index(selected_month_abbr.capitalize())

# Determine number of days in month
days_in_month = (datetime.date(selected_year, selected_month_num % 12 + 1, 1) - datetime.timedelta(days=1)).day if selected_month_num != 12 else 31
selected_day = st.sidebar.selectbox("Select day", list(range(1, days_in_month + 1)))
selected_date = datetime.date(selected_year, selected_month_num, selected_day)

selected_pollutant = st.sidebar.selectbox("Select pollutant", list(pollutants.keys()))

# Construct file path
nc_file = os.path.join(DATA_DIR, f"daysum{selected_month_abbr}2023.nc")

# Load dataset
if not os.path.exists(nc_file):
    st.error(f"NetCDF file not found: {nc_file}")
    st.stop()

try:
    ds = xr.open_dataset(nc_file)
except Exception as e:
    st.error(f"Failed to open dataset: {e}")
    st.stop()

# Check time variable
if "time" not in ds.variables:
    st.error("No 'time' variable in dataset.")
    st.stop()

time_var = ds["time"]
time_units = time_var.attrs.get("units", None)

if time_units is None:
    #st.warning("No 'units' attribute found in 'time'. Using fallback.")
    base_date = datetime.date(selected_year, selected_month_num, 1)
else:
    try:
        base_date = datetime.datetime.strptime(time_units.split("since")[1].strip(), "%Y-%m-%d %H:%M:%S").date()
    except Exception as e:
        st.error(f"Failed to parse time units: {e}")
        st.stop()

# Index in time dimension
day_index = (selected_date - base_date).days

if day_index < 0 or day_index >= ds.dims["time"]:
    st.warning(f"Selected date {selected_date} is out of range.")
    st.stop()

# Variable name
var_name = pollutants[selected_pollutant]
if var_name not in ds.variables:
    st.error(f"Variable {var_name} not found in dataset.")
    st.stop()

data = ds[var_name][day_index, :, :]

# Mask NaNs
masked_data = np.ma.masked_invalid(data.values)

# Normalization
try:
    masked_data_np = np.array(masked_data.filled(np.nan), copy=True)
    vmax = np.nanpercentile(masked_data_np, 99)
    norm = mcolors.LogNorm(vmin=1, vmax=max(vmax, 10))
except Exception as e:
    st.error(f"Error during normalization: {e}")
    st.stop()

# Grid
east = ds["easting"].values
north = ds["northing"].values
east_grid, north_grid = np.meshgrid(east, north)

# UTM to lat/lon
transformer = Transformer.from_crs("epsg:32756", "epsg:4326", always_xy=True)
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

# Add lat/lon gridlines and labels
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# Coordinates for Sydney (example)
marker_lon, marker_lat = 151.2093, -33.8688

# Add marker
ax.plot(marker_lon, marker_lat, marker='*', color='red', markersize=10, transform=ccrs.PlateCarree())
ax.text(marker_lon + 0.3, marker_lat, "Sydney", transform=ccrs.PlateCarree(), fontsize=10, color='red')

st.pyplot(fig)

