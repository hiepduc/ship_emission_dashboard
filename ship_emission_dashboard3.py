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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from matplotlib.ticker import LongitudeFormatter, LatitudeFormatter

# === Configuration ===
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

# === Sidebar ===
st.sidebar.title("Shipping Emission Dashboard")
selected_month = st.sidebar.selectbox("Select month", [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "nov"
])

selected_pollutant = st.sidebar.selectbox("Select pollutant", list(pollutants.keys()))

# === Load corresponding NetCDF file ===
nc_file = os.path.join(DATA_DIR, FILE_TEMPLATE.format(month=selected_month))
if not os.path.exists(nc_file):
    st.error(f"NetCDF file not found: {nc_file}")
    st.stop()

try:
    ds = xr.open_dataset(nc_file)
except Exception as e:
    st.error(f"Failed to open dataset: {e}")
    st.stop()

# === Time Handling ===
if "time" not in ds.variables:
    st.error("No 'time' variable in dataset.")
    st.stop()

time_var = ds["time"]
time_units = time_var.attrs.get("units", None)

if time_units is None:
    #st.warning("No 'units' attribute found in 'time'. Using fallback.")
    base_date = datetime.date(2023, 2, 1)
else:
    try:
        base_date = datetime.datetime.strptime(time_units.split("since")[1].strip(), "%Y-%m-%d %H:%M:%S").date()
    except Exception as e:
        st.error(f"Failed to parse time units: {e}")
        st.stop()

num_days = ds.dims["time"]
selected_day = st.sidebar.slider("Select day of month", 1, num_days, 1)
day_index = selected_day - 1  # 0-based index

plot_date = base_date + datetime.timedelta(days=day_index)

# === Get pollutant variable ===
var_name = pollutants[selected_pollutant]
if var_name not in ds.variables:
    st.error(f"Variable {var_name} not found in dataset.")
    st.stop()

data = ds[var_name][day_index, :, :]

# === Mask and normalize ===
masked_data = np.ma.masked_invalid(data.values)
try:
    masked_data_np = np.array(masked_data.filled(np.nan), copy=True)
    vmax = np.nanpercentile(masked_data_np, 99)
    norm = mcolors.LogNorm(vmin=1, vmax=max(vmax, 10))
except Exception as e:
    st.error(f"Error during normalization: {e}")
    st.stop()

# === Convert UTM to Lat/Lon ===
east = ds["easting"].values
north = ds["northing"].values
east_grid, north_grid = np.meshgrid(east, north)

transformer = Transformer.from_crs("epsg:32756", "epsg:4326", always_xy=True)  # UTM zone 56S -> WGS84
lon, lat = transformer.transform(east_grid, north_grid)

# === Plotting ===
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_title(f"{selected_pollutant} Emissions on {plot_date}", fontsize=14)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# Draw data
pc = ax.pcolormesh(lon, lat, masked_data_np, cmap='YlOrRd', norm=norm, shading="auto", transform=ccrs.PlateCarree())
plt.colorbar(pc, ax=ax, label=f"{selected_pollutant} (kg)")

# === Add map marker ===
marker_lon, marker_lat = 151.2093, -33.8688  # Sydney
ax.plot(marker_lon, marker_lat, marker='*', color='red', markersize=10, transform=ccrs.PlateCarree())
ax.text(marker_lon + 0.3, marker_lat, "Sydney", transform=ccrs.PlateCarree(), fontsize=10, color='red')

# === Grid, ticks, labels ===
ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
ax.set_xticks(np.linspace(np.min(lon), np.max(lon), 5), crs=ccrs.PlateCarree())
ax.set_yticks(np.linspace(np.min(lat), np.max(lat), 5), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

st.pyplot(fig)

