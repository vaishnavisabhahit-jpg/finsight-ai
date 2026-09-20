import os
import sqlite3
import numpy as np
import pandas as pd

def run_feature_engineering():
    db_path = os.path.join("data", "finsight.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError("Database not found. Run src/build_sql_layer.py first.")
        
    conn = sqlite3.connect(db_path)
    
    # Pull base financial records ordered by entity and time
    query = """
    SELECT 
        Date,
        BusinessUnit,
        Revenue,
        COGS,
        OperatingExpenses,
        TotalExpenses,
        OperatingProfit,
        AccountsReceivable,
        AccountsPayable,
        CashAndEquivalents,
        TotalDebt,
        NetCashFlow,
        BudgetRevenue,
        BudgetExpenses
    FROM raw_financials
    ORDER BY BusinessUnit, Date ASC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    featured_dfs = []
    
    # Process per business unit to prevent cross-entity data leakage
    for entity, group in df.groupby("BusinessUnit"):
        g = group.copy().sort_values("Date").reset_index(drop=True)
        
        # --- 1. Core Financial Ratios ---
        g["GrossMarginPct"] = ((g["Revenue"] - g["COGS"]) / g["Revenue"]) * 100
        g["OperatingMarginPct"] = (g["OperatingProfit"] / g["Revenue"]) * 100
        g["ReceivablesToRevenueRatio"] = g["AccountsReceivable"] / g["Revenue"]
        g["CashBufferRatio"] = g["CashAndEquivalents"] / g["TotalExpenses"]
        g["DebtToCashRatio"] = g["TotalDebt"] / g["CashAndEquivalents"]
        
        # --- 2. Historical Lags (Strictly past information: t-1, t-2) ---
        g["Revenue_Lag1"] = g["Revenue"].shift(1)
        g["Revenue_Lag2"] = g["Revenue"].shift(2)
        g["Expenses_Lag1"] = g["TotalExpenses"].shift(1)
        g["OperatingProfit_Lag1"] = g["OperatingProfit"].shift(1)
        g["OperatingMargin_Lag1"] = g["OperatingMarginPct"].shift(1)
        
        # --- 3. Momentum & Growth Indicators ---
        g["RevenueGrowthMoM"] = g["Revenue"].pct_change(1) * 100
        g["ExpenseGrowthMoM"] = g["TotalExpenses"].pct_change(1) * 100
        g["OperatingProfitGrowthMoM"] = g["OperatingProfit"].pct_change(1) * 100
        
        # --- 4. Rolling Baseline Indicators (Excluding future data) ---
        g["Rolling3M_AvgRev"] = g["Revenue"].shift(1).rolling(window=3).mean()
        g["Rolling3M_AvgExp"] = g["TotalExpenses"].shift(1).rolling(window=3).mean()
        
        # --- 5. Forward-Looking Risk Target Definition (T+1) ---
        # Forward operating margin and forward cash buffer
        future_op_margin = g["OperatingMarginPct"].shift(-1)
        future_cash_buffer = g["CashBufferRatio"].shift(-1)
        
        # Target rule: 1 = High Financial Risk next period, 0 = Healthy/Stable
        # Risk triggers if next month's operating margin drops below 18% OR cash buffer drops below 1.2x expenses
        g["Target_NextPeriodHighRisk"] = np.where(
            (future_op_margin < 18.0) | (future_cash_buffer < 1.25), 
            1, 
            0
        )
        
        # Drop the very last row per entity where future target is unknown (NaN)
        g = g.iloc[:-1].copy()
        
        # Drop the first 3 rows per entity where rolling/lagged features are NaN
        g = g.dropna().reset_index(drop=True)
        
        featured_dfs.append(g)
        
    final_df = pd.concat(featured_dfs, ignore_index=True)
    
    output_csv = os.path.join("data", "featured_financials.csv")
    final_df.to_csv(output_csv, index=False)
    
    print(f"Feature engineering completed successfully.")
    print(f"Output saved to: {output_csv}")
    print(f"Dataset shape: {final_df.shape}")
    print(f"High Risk class distribution:\n{final_df['Target_NextPeriodHighRisk'].value_counts(normalize=True)}")

if __name__ == "__main__":
    run_feature_engineering()