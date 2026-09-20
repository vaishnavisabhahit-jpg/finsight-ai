import os
import numpy as np
import pandas as pd

def generate_financial_dataset(start_date="2020-01-01", periods=60, seed=42):
    np.random.seed(seed)
    
    # 60 months = 5 years of historical financial records
    dates = pd.date_range(start=start_date, periods=periods, freq="MS")
    entities = ["Retail & E-commerce", "Enterprise Solutions", "Cloud Services"]
    
    records = []
    
    for entity in entities:
        base_rev = {
            "Retail & E-commerce": 500_000,
            "Enterprise Solutions": 800_000,
            "Cloud Services": 650_000
        }[entity]
        
        growth_rate = {
            "Retail & E-commerce": 0.006,
            "Enterprise Solutions": 0.009,
            "Cloud Services": 0.012
        }[entity]
        
        current_cash = 1_200_000
        current_debt = 900_000
        
        for i, dt in enumerate(dates):
            # Baseline trend + seasonal variation + realistic variance
            trend = base_rev * ((1 + growth_rate) ** i)
            seasonality = 1.0 + 0.12 * np.sin(2 * np.pi * (dt.month / 12.0))
            noise = np.random.normal(1.0, 0.03)
            
            revenue = trend * seasonality * noise
            
            # Direct costs (COGS)
            cogs_ratio = np.random.uniform(0.40, 0.48)
            cogs = revenue * cogs_ratio
            
            # Operating Expenses breakdown
            mkt_ratio = np.random.uniform(0.08, 0.13)
            mkt_expense = revenue * mkt_ratio
            
            payroll = (base_rev * 0.22) * ((1 + growth_rate * 0.6) ** i) + np.random.normal(0, 3000)
            other_opex = revenue * np.random.uniform(0.05, 0.08)
            operating_expenses = mkt_expense + payroll + other_opex
            
            total_expenses = cogs + operating_expenses
            operating_profit = revenue - total_expenses
            
            # Balance sheet & working capital fields
            accounts_receivable = revenue * np.random.uniform(0.70, 0.95)
            accounts_payable = cogs * np.random.uniform(0.55, 0.75)
            inventory = cogs * np.random.uniform(0.40, 0.60) if entity != "Cloud Services" else 0.0
            
            # Controlled risk anomalies for testing our risk and anomaly modules
            if entity == "Retail & E-commerce" and i in [40, 41, 42]:
                operating_expenses *= 1.35
                accounts_receivable *= 1.40
                
            if entity == "Enterprise Solutions" and i in [48, 49]:
                revenue *= 0.75
                
            # Cash flow & debt tracking
            cash_inflow = revenue * np.random.uniform(0.85, 1.05)
            cash_outflow = total_expenses * np.random.uniform(0.90, 1.05)
            net_cash_flow = cash_inflow - cash_outflow
            current_cash = max(100_000, current_cash + net_cash_flow)
            
            # Planned budget for variance analysis
            budget_revenue = trend * seasonality
            budget_expenses = budget_revenue * 0.75
            
            records.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "BusinessUnit": entity,
                "Revenue": round(revenue, 2),
                "COGS": round(cogs, 2),
                "MarketingExpense": round(mkt_expense, 2),
                "PayrollExpense": round(payroll, 2),
                "OtherOpex": round(other_opex, 2),
                "OperatingExpenses": round(operating_expenses, 2),
                "TotalExpenses": round(total_expenses, 2),
                "OperatingProfit": round(operating_profit, 2),
                "AccountsReceivable": round(accounts_receivable, 2),
                "AccountsPayable": round(accounts_payable, 2),
                "Inventory": round(inventory, 2),
                "CashAndEquivalents": round(current_cash, 2),
                "TotalDebt": round(current_debt, 2),
                "NetCashFlow": round(net_cash_flow, 2),
                "BudgetRevenue": round(budget_revenue, 2),
                "BudgetExpenses": round(budget_expenses, 2)
            })
            
    df = pd.DataFrame(records)
    output_path = os.path.join("data", "raw_financials.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at: {output_path}")
    print(f"Total rows: {len(df)} | Columns: {len(df.columns)}")

if __name__ == "__main__":
    generate_financial_dataset()