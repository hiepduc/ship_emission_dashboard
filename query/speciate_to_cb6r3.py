import pandas as pd
import matplotlib.pyplot as plt

# === File paths ===
speciate_file = "/home/duch/shipping/query/shipping_profile_121GROC.csv"
mapping_file = "/home/duch/shipping/speciate/cb6r3_mapping.csv"
output_excel = "/home/duch/shipping/query/shipping_profile_121GROC_speciated_cb6r3.xlsx"

# === Load data ===
speciate_df = pd.read_csv(speciate_file)
mapping_df = pd.read_csv(mapping_file)

# Drop rows with missing MECH_SPECIES
mapping_df = mapping_df.dropna(subset=['MECH_SPECIES'])

# Ensure SPECIES_ID is numeric for join
speciate_df['SPECIES_ID'] = pd.to_numeric(speciate_df['SPECIES_ID'], errors='coerce')
mapping_df['SPECIES_ID'] = pd.to_numeric(mapping_df['SPECIES_ID'], errors='coerce')

# === Merge by SPECIES_ID ===
merged = speciate_df.merge(mapping_df, on='SPECIES_ID', how='inner')

# Calculate CB6-weighted VOC
merged['CB6_MASS'] = merged['WEIGHT_PERCENT'] * merged['MECH_FRACTION']

# Sum up by CB6 species
cb6_profile = merged.groupby('MECH_SPECIES')['CB6_MASS'].sum().reset_index()
cb6_profile = cb6_profile.sort_values(by='CB6_MASS', ascending=False)

# === Total NMVOC weight from speciation ===
total_nmvoc = speciate_df['WEIGHT_PERCENT'].sum()

print(f"\nTotal NMVOC in profile: {total_nmvoc:.2f}")
print(f"Mapped to {len(cb6_profile)} CB6 species.")
print(cb6_profile.head(10))

# === Plot ===
plt.figure(figsize=(10, 6))
cb6_profile.head(20).plot.bar(x='MECH_SPECIES', y='CB6_MASS', legend=False)
plt.title("Top 20 CB6r3_AE7 VOC Species")
plt.ylabel("CB6 Mass Fraction (from NMVOC)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("/home/duch/shipping/query/cb6_profile_121GROC.png", dpi=300)
print("📊 Plot saved to cb6_profile_121GROC.png")

# === Save to Excel with formatting ===
with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
    cb6_profile.to_excel(writer, sheet_name='CB6_Profile', index=False)
    merged.to_excel(writer, sheet_name='Full_Mapping', index=False)
    workbook  = writer.book
    fmt_pct = workbook.add_format({'num_format': '0.00'})
    writer.sheets['CB6_Profile'].set_column('B:B', 15, fmt_pct)

print(f"✅ CB6r3 speciation profile saved to {output_excel}")

