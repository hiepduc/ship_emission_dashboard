import pandas as pd

input_file = "gscnv.CB6r3_ae7_criteria.CMAQ.2022Feb02.txt"
output_file = "cb6r3_mapping.csv"

records = []

with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        # Skip comments or blank lines
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        # Expecting lines with: VOC TOG SPECIES_ID MECH_SPECIES FRACTION
        if len(parts) == 5 and parts[0] == "VOC":
            species_id = parts[2]
            mech_species = parts[3]
            try:
                fraction = float(parts[4])
            except ValueError:
                continue
            records.append((species_id, mech_species, fraction))

# Convert to DataFrame
df = pd.DataFrame(records, columns=["SPECIES_ID", "MECH_SPECIES", "MECH_FRACTION"])
df["SPECIES_ID"] = pd.to_numeric(df["SPECIES_ID"], errors='coerce', downcast="integer")
df = df.dropna(subset=["SPECIES_ID"])

# Save to CSV
df.to_csv(output_file, index=False)
print(f"✅ Saved CB6r3 mapping to {output_file} with {len(df)} entries.")

