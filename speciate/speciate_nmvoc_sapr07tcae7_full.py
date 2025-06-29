import xarray as xr
import numpy as np
import os

# === USER INPUTS ===
input_nc_file = "/home/duch/shipping/monthdailysum/daysumapr2023.nc"  # Change per month
gspro_file = "/home/duch/shipping/speciate/S2S-Tool/output/gspro.SAPRC07TC_AE7_CRITERIA_VOC.CMAQ.2025-06-29.txt"
profile_id = "121GROC"  # Ship emissions profile ID
output_nc_file = input_nc_file.replace(".nc", "_saprc07tcae7_full.nc")

# === STEP 1: LOAD NMVOC FROM NETCDF ===
ds = xr.open_dataset(input_nc_file)

if "nmvoc_kg" not in ds:
    raise ValueError("Variable 'nmvoc_kg' not found in input file.")

nmvoc = ds["nmvoc_kg"]

# === STEP 2: PARSE GSPRO FILE FOR FRACTIONS ===
speciation = {}
with open(gspro_file) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        prof, group, species, frac_str = parts[:4]
        if prof == profile_id:
            try:
                frac = float(frac_str)
                speciation[species] = frac
            except ValueError:
                continue

if not speciation:
    raise ValueError(f"No species found for profile ID {profile_id} in {gspro_file}")

# === STEP 3: COPY ORIGINAL VARIABLES EXCEPT nmvoc_kg ===
ds_out = xr.Dataset()

for var in ds.data_vars:
    #if var != "nmvoc_kg":
    #    ds_out[var] = ds[var]
    ds_out[var] = ds[var]

# === STEP 4: ADD SPECIATED NMVOC SPECIES ===
for species, frac in speciation.items():
    varname = species.lower() + "_kg"
    ds_out[varname] = nmvoc * frac
    ds_out[varname].attrs["units"] = "kg"
    ds_out[varname].attrs["long_name"] = f"{species} from NMVOC speciation"

# === STEP 5: COPY COORDINATES ===
for coord in ds.coords:
    ds_out[coord] = ds[coord]

# === STEP 6: SAVE TO NETCDF ===
ds_out.to_netcdf(output_nc_file)
print(f"✅ Full speciated + original emissions written to: {output_nc_file}")

