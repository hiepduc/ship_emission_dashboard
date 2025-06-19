import streamlit as st
import xarray as xr
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

# File path pattern
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
}

# Streamlit UI
st.title("Daily Ship Emissions Dashboard")
selected_date = st.sidebar.date_input("Select date", datetime.date(2023, 2, 1))
selected_pollutant = st.sidebar.selectbox("Select pollutant", list(pollutants.keys()))

# Format file name based on month
month_str = selected_date.strftime("%b").lower()
nc_file = os.path.join(DATA_DIR, FILE_TEMPLATE.format(month=month_str))

# Load and validate dataset
if not os.path.exists(nc_file):
    st.error(f"NetCDF file not found for {month_str}")
else:
    # Load NetCDF file using netddf4  engine
    ds = xr.open_dataset(nc_file, engine="netcdf4")

    # Check for time variable and units
    time_var = ds.get("time")

    if time_var is None:
        st.error("No 'time' variable in dataset.")
    else:
        time_units = time_var.attrs.get("units")
    
        if time_var is None:
            st.error("No 'time' variable in dataset.")
        else:
            time_units = time_var.attrs.get("units", None)
            if time_units is None:
                st.warning("No 'units' attribute found in 'time'. Using fallback.")
                base_date = datetime.datetime(2023, 2, 1)
            else:
                base_date = datetime.datetime.strptime(time_units.split("since")[1].strip(), "%Y-%m-%d %H:%M:%S")

            day_index = (selected_date - base_date.date()).days

            if 0 <= day_index < len(time_var):
                easting = ds["easting"].values
                northing = ds["northing"].values
                data = ds[pollutants[selected_pollutant]][day_index, :, :]
                st.write("Selected day index:", day_index)
                st.write("Time length in dataset:", len(ds["time"]))

                st.write("Data shape:", data.shape)
                st.write("Data min:", np.nanmin(data.values))
                st.write("Data max:", np.nanmax(data.values))
                st.write("NaN count:", np.isnan(data.values).sum())
                st.write("Selected pollutant:", selected_pollutant)
                st.write("Variable in dataset:", pollutants[selected_pollutant] in ds)
                st.write("Easting:", easting.min(), "to", easting.max())
                st.write("Northing:", northing.min(), "to", northing.max())
                st.write("Sample data values (center):", data.values[data.shape[0]//2, data.shape[1]//2])

                fig, ax = plt.subplots(figsize=(10, 6))

                # Mask NaNs for plotting
                masked_data = np.ma.masked_invalid(data.values)

                # Optional: clip extreme values to improve contrast
                percentile_99 = np.nanpercentile(masked_data.data, 99)
                clipped = np.clip(masked_data, 0, percentile_99)

                # Define extent to match UTM coordinates for axes
                extent = [ds.easting.min(), ds.easting.max(), ds.northing.min(), ds.northing.max()]

                im = ax.imshow(clipped, origin="lower", cmap="YlOrRd", extent=extent)
                plt.colorbar(im, ax=ax, label=f"{selected_pollutant} (kg)")
                ax.set_xlabel("Easting (m)")
                ax.set_ylabel("Northing (m)")
                st.pyplot(fig)

            else:
                st.error(f"Date {selected_date} out of range.")
