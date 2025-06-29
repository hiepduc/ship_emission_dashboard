import pandas as pd
import re

# Known CB6r3_AE7 species names
cb6_species_list = {
    'PAR', 'ETH', 'OLE', 'IOLE', 'FORM', 'ALD2', 'ETOH', 'ISOP', 'TOL',
    'XYL', 'BENZ', 'ACET', 'MEK', 'PRPE', 'NAPH', 'OPEN'
}

input_path = "gscnv.CB6r3_ae7_criteria.CMAQ.2022Feb02.txt"
output_path = "cb6r3_clean_mapping.csv"

rows = []
with open(input_path, 'r') as f:
    for line in f:
        if line.strip().startswith("#") or not line.strip():
            continue  # Skip comments and empty lines

        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) != 4:
            continue  # Skip malformed lines

        _, _, surrogate, factor = parts
        try:
            factor = float(factor)
        except ValueError:
            continue

        if surrogate in cb6_species_list:
            rows.append({
                "SPECIES_ID": None,
                "MECH_SPECIES": surrogate,
                "MECH_FRACTION": factor
            })
        elif surrogate.isdigit():
            rows.append({
                "SPECIES_ID": int(surrogate),
                "MECH_SPECIES": None,
                "MECH_FRACTION": factor
            })

df = pd.DataFrame(rows)
df.to_csv(output_path, index=False)
print(f"✅ Clean CB6r3 mapping saved to {output_path} with {len(df)} entries.")

