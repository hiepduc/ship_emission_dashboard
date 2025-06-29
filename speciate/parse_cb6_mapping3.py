import pandas as pd
import re

input_path = "/home/duch/shipping/speciate/gscnv.CB6r3_ae7_criteria.CMAQ.2022Feb02.txt"
output_path = "/home/duch/shipping/speciate/cb6r3_clean_mapping.csv"

rows = []
with open(input_path, 'r') as f:
    for line in f:
        if line.strip().startswith("#") or not line.strip():
            continue  # Skip comments and blank lines

        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) != 4:
            continue  # Skip malformed lines

        inventory, surrogate, mech_species, factor = parts
        try:
            factor = float(factor)
        except ValueError:
            continue

        # Check if surrogate is a valid SPECIES_ID (numeric)
        if surrogate.isdigit():
            species_id = int(surrogate)
        else:
            continue  # Ignore CARB**** or custom surrogates

        rows.append({
            "SPECIES_ID": species_id,
            "MECH_SPECIES": mech_species.strip(),
            "MECH_FRACTION": factor
        })

df = pd.DataFrame(rows)
df.to_csv(output_path, index=False)
print(f"✅ Clean CB6r3 mapping saved to {output_path} with {len(df)} entries.")

