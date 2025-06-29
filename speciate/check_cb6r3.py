import os
import pandas as pd

# Folder with your Excel files
excel_folder = "/home/duch/shipping/speciate/excel"

# Known CB6r3 mechanism species keywords (expand as needed)
cb6_species_keywords = [
    'ETH', 'PAR', 'FORM', 'ACET', 'ALD2', 'OLE', 'TOL', 'XYL', 'ISOP', 'ALD2', 'BUT', 'BUTA', 'HEX', 'ALK', 'OLE', 'CH3O2', 'MO2',
    'ALD', 'ETHP', 'ALD', 'MEOH', 'MEK', 'MVK', 'C2H5OH', 'ALDX', 'BENZ', 'CRES', 'XYL', 'ISOP', 'ALK4', 'ALK3', 'ALK2', 'ALK1'
]

def search_cb6_in_excel(file_path):
    try:
        xls = pd.ExcelFile(file_path)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return

    found_rows = []
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
        except Exception as e:
            print(f"Failed to parse {sheet} in {file_path}: {e}")
            continue

        # Check each column of object type (likely string)
        for col in df.select_dtypes(include=['object']).columns:
            # Create mask for rows containing any CB6 species keyword (case-insensitive)
            mask = df[col].astype(str).str.upper().str.contains('|'.join(cb6_species_keywords), na=False)
            if mask.any():
                # Extract matched rows with info
                matched = df.loc[mask, [col]].copy()
                matched['Sheet'] = sheet
                matched['File'] = os.path.basename(file_path)
                matched['Column'] = col
                found_rows.append(matched)

    if found_rows:
        return pd.concat(found_rows, ignore_index=True)
    else:
        return None

def main():
    all_results = []
    for file in os.listdir(excel_folder):
        if file.endswith(".xlsx"):
            full_path = os.path.join(excel_folder, file)
            print(f"Scanning {file}...")
            result = search_cb6_in_excel(full_path)
            if result is not None:
                all_results.append(result)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"\nFound {len(combined)} matches containing CB6 species keywords.\n")
        # Save to CSV for review
        combined.to_csv("cb6_species_search_results.csv", index=False)
        print("Saved matches to cb6_species_search_results.csv")
    else:
        print("No matches found in any Excel file.")

if __name__ == "__main__":
    main()

