import pandas as pd

# Load data files
speciated_df = pd.read_excel("/home/duch/shipping/speciate/excel/SPECIES.xlsx")
svoc_factors = pd.read_excel("/home/duch/shipping/speciate/excel/List of SVOC Splitting Factors.xlsx")
species_synonyms = pd.read_excel("/home/duch/shipping/speciate/excel/SPECIES_SYNONYMS.xlsx")
profiles = pd.read_excel("/home/duch/shipping/speciate/excel/PROFILES.xlsx")

# Find shipping-related profiles (example keywords)
keywords = ["ship", "marine", "Marine", "vessel", "Vessel", "boat", "barge", "port", "ferry", "Ferry"]
pattern = '|'.join(keywords)
shipping_profiles = profiles[
    profiles['PROFILE_NAME'].str.contains(pattern, case=False, na=False) &
    (profiles['PROFILE_TYPE'].str.upper() == 'GAS')
]
print(f"Found {len(shipping_profiles)} candidate shipping-related profiles:\n")

print("All shipping-related profiles:\n")
for i, (code, name) in enumerate(zip(shipping_profiles['PROFILE_CODE'], shipping_profiles['PROFILE_NAME'])):
    print(f"{i}: {code} - {name}")

choice = input("\nEnter the number of the PROFILE_CODE to select: ").strip()
try:
    idx = int(choice)
    profile_code = shipping_profiles.iloc[idx]['PROFILE_CODE']
except (ValueError, IndexError):
    print("Invalid selection.")
    exit()

print(f"\nUsing PROFILE_CODE {profile_code} for detailed species fractions...")

# Get species fractions from SPECIATED (SPECIES.xlsx) for selected profile_code
fractions = speciated_df[speciated_df['PROFILE_CODE'] == profile_code].copy()

if fractions.empty:
    print(f"No species found for PROFILE_CODE {profile_code}")
    exit()

# Convert SPECIES_ID to int for merging consistency
fractions['SPECIES_ID'] = fractions['SPECIES_ID'].astype(int)
svoc_factors['SPECIES_ID'] = svoc_factors['SPECIES_ID'].astype(int)
species_synonyms['SPECIES_ID'] = species_synonyms['SPECIES_ID'].astype(int)

# Debug print to check ID overlaps
species_ids = set(fractions['SPECIES_ID'])
svoc_ids = set(svoc_factors['SPECIES_ID'])
syn_ids = set(species_synonyms['SPECIES_ID'])

print(f"\nSpecies IDs in fractions: {len(species_ids)}")
print(f"Species IDs in SVOC factors: {len(svoc_ids)}")
print(f"Species IDs in synonyms: {len(syn_ids)}")
print(f"Intersection fractions & SVOC factors: {len(species_ids.intersection(svoc_ids))}")
print(f"Intersection fractions & synonyms: {len(species_ids.intersection(syn_ids))}")

species_props = pd.read_excel("/home/duch/shipping/speciate/excel/SPECIES_PROPERTIES.xlsx")
species_props['SPECIES_ID'] = species_props['SPECIES_ID'].astype(int)

merged = fractions.merge(
    species_props[['SPECIES_ID', 'SPECIES_NAME']],
    on='SPECIES_ID',
    how='left'
)


#merged['SPECIES_NAME'] = merged['SPECIES_NAME'].fillna(f"Unnamed_{merged['SPECIES_ID']}")
unnamed = merged[merged['SPECIES_NAME'].isna()]
if not unnamed.empty:
    print(f"\n⚠️ {len(unnamed)} species had no name in SVOC or synonyms. You may want to review them.")

# Select final columns
merged = merged[['SPECIES_ID', 'SPECIES_NAME', 'WEIGHT_PERCENT']].sort_values(by='WEIGHT_PERCENT', ascending=False)

print(f"\nSpecies emission fractions for PROFILE_CODE {profile_code}:")
print(merged.head(20))

# Save detailed profile to CSV
output_file = f"shipping_profile_{profile_code}.csv"
merged.to_csv(output_file, index=False)
print(f"\nSaved detailed profile to {output_file}")

