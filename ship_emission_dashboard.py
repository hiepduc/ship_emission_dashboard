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
import pandas as pd
import h5py
h5py._errors.silence_errors()

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

# === Sidebar Inputs ===
st.sidebar.title("Shipping Emission Dashboard")

selected_month = st.sidebar.selectbox("Select month", [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
])
selected_pollutant = st.sidebar.selectbox("Select pollutant", list(pollutants.keys()))
show_monthly_plot = st.sidebar.checkbox("Show daily emission time series")

# === Load NetCDF Data ===
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

# Use selected_month to assign base_date
month_map = {
    "jan": "2023-01-01",
    "feb": "2023-02-01",
    "mar": "2023-03-01",
    "apr": "2023-04-01",
    "may": "2023-05-01",
    "jun": "2023-06-01",
    "jul": "2023-07-01",
    "aug": "2023-08-01",
    "sep": "2023-09-01",
    "oct": "2023-10-01",
    "nov": "2023-11-01",
    "dec": "2023-12-01"
}

#if time_units is None:
#    base_date = datetime.date(2023, 2, 1)
#else:
try:
    base_date = datetime.datetime.strptime(month_map[selected_month], "%Y-%m-%d").date()
except KeyError:
    st.error(f"Invalid month selection: {selected_month}")
    st.stop()

# Get number of days and selected index
num_days = ds.dims["time"]
selected_day = st.sidebar.slider("Select day of month", 1, num_days, 1)
day_index = selected_day - 1
plot_date = base_date + datetime.timedelta(days=day_index)

# === Get Pollutant Variable ===
var_name = pollutants[selected_pollutant]
if var_name not in ds.variables:
    st.error(f"Variable {var_name} not found in dataset.")
    st.stop()

data = ds[var_name][day_index, :, :]

# === Mask and Normalize ===
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
transformer = Transformer.from_crs("epsg:32756", "epsg:4326", always_xy=True)
lon, lat = transformer.transform(east_grid, north_grid)

# === Plotting ===
fig = plt.figure(figsize=(12, 9))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_title(f"{selected_pollutant} Emissions on {plot_date}", fontsize=14)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

pc = ax.pcolormesh(
    lon, lat, masked_data_np,
    cmap='YlOrRd', norm=norm, shading="auto", transform=ccrs.PlateCarree()
)
plt.colorbar(pc, ax=ax, label=f"{selected_pollutant} (kg)")

# Optional marker: Sydney
marker_lon, marker_lat = 151.2093, -33.8688
ax.plot(marker_lon, marker_lat, marker='*', color='red', markersize=10, transform=ccrs.PlateCarree())
ax.text(marker_lon + 0.3, marker_lat, "Sydney", transform=ccrs.PlateCarree(), fontsize=10, color='red')

# Clean gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 10}
gl.ylabel_style = {"size": 10}

st.pyplot(fig)

# === Emission Summary ===
valid_data = masked_data_np[~np.isnan(masked_data_np)]
max_val = np.max(valid_data) if valid_data.size > 0 else np.nan
avg_val = np.mean(valid_data) if valid_data.size > 0 else np.nan
total_val = np.nansum(valid_data) if valid_data.size > 0 else np.nan

st.subheader("Emission Summary for Selected Day")
st.write(f"**Max {selected_pollutant}:** {max_val:.2f} kg")
st.write(f"**Average {selected_pollutant}:** {avg_val:.2f} kg")
st.write(f"**Total {selected_pollutant}:** {total_val:.2f} kg")

# === Monthly Total Emission ===
try:
    monthly_data = ds[var_name][:, :, :]
    monthly_sum = np.nansum(monthly_data.values)
    st.write(f"**Monthly Total {selected_pollutant}:** {monthly_sum:.2f} kg")
except Exception as e:
    st.error(f"Error calculating monthly total: {e}")

# === Data Table ===
#st.subheader("Emission Data Table for Selected Day")
#
#flat_lat = lat.flatten()
#flat_lon = lon.flatten()
#flat_data = masked_data_np.flatten()
#df_table = pd.DataFrame({
#    "Latitude": flat_lat,
#    "Longitude": flat_lon,
#    f"{selected_pollutant} (kg)": flat_data
#}).dropna().reset_index(drop=True)

#st.dataframe(df_table)

# === Time Series Chart and Download ===
if show_monthly_plot:
    st.subheader(f"Daily Total {selected_pollutant} for {selected_month.capitalize()} 2023")

    try:
        daily_totals = ds[var_name].sum(dim=["northing", "easting"]).values
        days = [base_date + datetime.timedelta(days=i) for i in range(num_days)]

        df_daily = pd.DataFrame({
            "Date": days,
            f"{selected_pollutant} (kg)": daily_totals
        })

        st.line_chart(df_daily.set_index("Date"))
        st.bar_chart(df_daily.set_index("Date"))

        csv = df_daily.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Daily Emissions CSV",
            data=csv,
            file_name=f"{selected_pollutant}_{selected_month}_2023.csv",
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Error generating time series chart: {e}")

