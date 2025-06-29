import pandas as pd

input_file = "gscnv.CB6r3_ae7_criteria.CMAQ.2022Feb02.txt"
output_file = "cb6r3_mapping.csv"

records = []

with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) != 4 or parts[0] != "VOC":
            continue
        raw_id = parts[2]
        mech_fraction = parts[3]

        try:
            fraction = float(mech_fraction)
        except ValueError:
            continue

        # Use raw_id as is; let user handle mapping later
        records.append((raw_id, fraction))

# Convert to DataFrame
df = pd.DataFrame(records, columns=["SPECIES_ID_OR_NAME", "MECH_FRACTION"])

# Extract MECH_SPECIES from string-based IDs (e.g., CARB3101), else leave blank
df["MECH_SPECIES"] = df["SPECIES_ID_OR_NAME"].apply(
    lambda x: x if not x.isnumeric() else "")

# Convert numeric IDs to int
df["SPECIES_ID"] = pd.to_numeric(
    df["SPECIES_ID_OR_NAME"], errors="coerce", downcast="integer")

# Reorder columns
df = df[["SPECIES_ID", "MECH_SPECIES", "MECH_FRACTION"]]

# Save to CSV
df.to_csv(output_file, index=False)
print(f"✅ Saved mapping to {output_file} with {len(df)} entries.")

