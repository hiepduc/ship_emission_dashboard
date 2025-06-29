import subprocess
import pandas as pd
from io import StringIO

def read_access_table(mdb_path, table_name):
    """
    Reads a table from an Access .mdb/.accdb file using mdbtools and returns a pandas DataFrame.

    Parameters:
        mdb_path (str): Path to the .mdb or .accdb file
        table_name (str): Name of the table to export

    Returns:
        pd.DataFrame: Table data as a pandas DataFrame
    """
    try:
        result = subprocess.run(
            ['mdb-export', mdb_path, table_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        csv_data = result.stdout.decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))
        return df
    except subprocess.CalledProcessError as e:
        print(f"Error exporting table {table_name}: {e.stderr.decode()}")
        return pd.DataFrame()  # Return empty DataFrame on error

def list_access_tables(mdb_path):
    """
    Lists all tables in an Access .mdb/.accdb file using mdbtools.

    Parameters:
        mdb_path (str): Path to the .mdb or .accdb file

    Returns:
        list: List of table names
    """
    try:
        result = subprocess.run(
            ['mdb-tables', '-1', mdb_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        tables = result.stdout.decode('utf-8').splitlines()
        return tables
    except subprocess.CalledProcessError as e:
        print(f"Error listing tables: {e.stderr.decode()}")
        return []

mdb_file = "/home/duch/shipping/speciate/final-speciate-5.3-_-10-26-2023/FinalSPECIATE5.3_10-26-2023.accdb"

# List available tables
tables = list_access_tables(mdb_file)
print("Tables found:", tables)

# Read a specific table (e.g., "PROFILE")
df_profile = read_access_table(mdb_file, "PROFILE")
print(df_profile.head())

