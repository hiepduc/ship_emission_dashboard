import pandas as pd
import os

folder = "/home/duch/shipping/speciate/excel"

for filename in os.listdir(folder):
    if filename.endswith(".xlsx"):
        path = os.path.join(folder, filename)
        try:
            df = pd.read_excel(path)
            print(f"{filename}: {df.columns.tolist()}")
        except Exception as e:
            print(f"Failed to read {filename}: {e}")

