import os
import sqlite3
import numpy as np
import pandas as pd


def compute_emerging_z_score(df: pd.DataFrame) -> pd.Series:
  working_capital = (
      df["AccountsReceivable"]
      + df["CashAndEquivalents"]
      - df["AccountsPayable"]
  )
  operating_assets = (
      df["AccountsReceivable"]
      + df["CashAndEquivalents"]
      + (df["Revenue"] * 0.25)
  )
  operating_assets = operating_assets.replace(0, np.nan)

  x1 = working_capital / operating_assets
  x2 = df["NetCashFlow"] / df["TotalExpenses"].replace(0, np.nan)
  x3 = df["OperatingProfit"] / df["TotalExpenses"].replace(0, np.nan)
  x4 = df["CashAndEquivalents"] / df["TotalDebt"].replace(0, np.nan)
  x5 = df["Revenue"] / df["TotalExpenses"].replace(0, np.nan)

  z_proxy = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5
  return z_proxy


def run_feature_engineering():
  db_path = os.path.join("data", "finsight.db")
  if not os.path.exists(db_path):
    raise FileNotFoundError(
        "Database not found. Run src/build_sql_layer.py first."
    )

  conn = sqlite3.connect(db_path)
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

  df["Date"] = pd.to_datetime(df["Date"])
  featured_dfs = []

  for entity, group in df.groupby("BusinessUnit"):
    g = group.copy().sort_values("Date").reset_index(drop=True)

    z_proxy = compute_emerging_z_score(g)

    # Relative empirical threshold: lowest 25th percentile Z-score OR negative net cash flow with margin compression
    z_cutoff = z_proxy.quantile(0.30)
    margin_burn = (g["OperatingProfit"] < g["OperatingProfit"].median()) & (
        g["NetCashFlow"] < 0
    )
    distress_flag = ((z_proxy <= z_cutoff) | margin_burn).astype(int)

    # Next-period target
    g["Target_NextPeriodHighRisk"] = distress_flag.shift(-1)
    g["Target_NextPeriodOperatingProfit"] = g["OperatingProfit"].shift(-1)

    # Financial features
    g["GrossMarginPct"] = (
        (g["Revenue"] - g["COGS"]) / g["Revenue"].replace(0, np.nan)
    ) * 100
    g["OperatingMarginPct"] = (
        g["OperatingProfit"] / g["Revenue"].replace(0, np.nan)
    ) * 100
    g["ReceivablesToRevenueRatio"] = g["AccountsReceivable"] / g[
        "Revenue"
    ].replace(0, np.nan)
    g["CashBufferRatio"] = g["CashAndEquivalents"] / g["TotalExpenses"].replace(
        0, np.nan
    )
    g["DebtToCashRatio"] = g["TotalDebt"] / g["CashAndEquivalents"].replace(
        0, np.nan
    )
    g["BudgetVarianceRevenue"] = (
        (g["Revenue"] - g["BudgetRevenue"])
        / g["BudgetRevenue"].replace(0, np.nan)
    ) * 100
    g["BudgetVarianceExpenses"] = (
        (g["TotalExpenses"] - g["BudgetExpenses"])
        / g["BudgetExpenses"].replace(0, np.nan)
    ) * 100

    # Lags & Momentum
    g["Revenue_Lag1"] = g["Revenue"].shift(1)
    g["Revenue_Lag2"] = g["Revenue"].shift(2)
    g["Expenses_Lag1"] = g["TotalExpenses"].shift(1)
    g["OperatingProfit_Lag1"] = g["OperatingProfit"].shift(1)
    g["OperatingMargin_Lag1"] = g["OperatingMarginPct"].shift(1)

    g["RevenueGrowthMoM"] = g["Revenue"].pct_change(1) * 100
    g["ExpenseGrowthMoM"] = g["TotalExpenses"].pct_change(1) * 100
    g["OperatingProfitGrowthMoM"] = g["OperatingProfit"].pct_change(1) * 100

    g["Rolling3M_AvgRev"] = g["Revenue"].shift(1).rolling(window=3).mean()
    g["Rolling3M_AvgExp"] = g["TotalExpenses"].shift(1).rolling(window=3).mean()
    g["Rolling3M_AvgMargin"] = (
        g["OperatingMarginPct"].shift(1).rolling(window=3).mean()
    )

    # Clean NaNs and future drop
    g = g.iloc[:-1].copy()
    g = g.dropna().reset_index(drop=True)
    g["Target_NextPeriodHighRisk"] = g["Target_NextPeriodHighRisk"].astype(int)

    featured_dfs.append(g)

  final_df = pd.concat(featured_dfs, ignore_index=True)

  output_csv = os.path.join("data", "featured_financials.csv")
  final_df.to_csv(output_csv, index=False)

  print("=" * 60)
  print("Feature engineering completed successfully.")
  print(f"Dataset shape: {final_df.shape}")
  print("Target Class Distribution:")
  print(final_df["Target_NextPeriodHighRisk"].value_counts(dropna=False))
  print(final_df["Target_NextPeriodHighRisk"].value_counts(normalize=True))
  print("=" * 60)


if __name__ == "__main__":
  run_feature_engineering()