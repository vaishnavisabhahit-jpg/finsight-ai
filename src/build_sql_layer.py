import os
import sqlite3
import pandas as pd

def setup_sql_layer():
    data_csv_path = os.path.join("data", "raw_financials.csv")
    db_path = os.path.join("data", "finsight.db")
    sql_file_path = os.path.join("sql", "analytical_views.sql")
    
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"Missing {data_csv_path}. Run generate_data.py first.")
        
    # Load CSV into Pandas
    df = pd.read_csv(data_csv_path)
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Ingest table into SQLite
    df.to_sql("raw_financials", conn, if_exists="replace", index=False)
    print(f"Successfully loaded {len(df)} records into SQLite table 'raw_financials'.")
    
    # Read and execute the SQL file
    with open(sql_file_path, "r") as f:
        sql_script = f.read()
        
    cursor.executescript(sql_script)
    conn.commit()
    print("Successfully created analytical SQL views: v_financial_ratios, v_financial_trends, v_budget_variance.")
    
    # Quick validation check: sample 3 rows from the trends view
    sample_df = pd.read_sql_query("SELECT Date, BusinessUnit, Revenue, Rolling3MonthAvgRevenue, RevenueGrowthMoMPct FROM v_financial_trends LIMIT 3;", conn)
    print("\n--- Sample Output from v_financial_trends ---")
    print(sample_df)
    
    conn.close()

if __name__ == "__main__":
    setup_sql_layer()