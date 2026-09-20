import pandas as pd
import numpy as np
from pathlib import Path

def generate_sec_features(input_path='data/real_sec_financials.csv', output_path='data/sec_features.csv'):
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Company', 'Date']).reset_index(drop=True)
    
    # Financial Ratios
    df['OperatingMarginPct'] = np.where(df['Revenue'] > 0, (df['OperatingProfit'] / df['Revenue']) * 100, 0.0)
    df['DebtToCashRatio'] = np.where(df['Cash'] > 0, df['Debt'] / df['Cash'], 0.0)
    
    # Chronological Lag Features per Company (No cross-company leakage)
    df['Lag_Revenue_1'] = df.groupby('Company')['Revenue'].shift(1)
    df['Lag_OperatingProfit_1'] = df.groupby('Company')['OperatingProfit'].shift(1)
    df['Lag_OperatingMargin_1'] = df.groupby('Company')['OperatingMarginPct'].shift(1)
    df['Lag_DebtToCash_1'] = df.groupby('Company')['DebtToCashRatio'].shift(1)
    
    # Growth Momentum
    df['Revenue_QoQ_Growth'] = np.where(
        df['Lag_Revenue_1'] > 0, 
        (df['Revenue'] - df['Lag_Revenue_1']) / df['Lag_Revenue_1'], 
        0.0
    )
    
    # Binary Distress Indicator for Ranking (Operating Loss / Severe Margin Contraction)
    df['IsDistressed'] = ((df['OperatingProfit'] < 0) | (df['OperatingMarginPct'] < 2.0)).astype(int)
    
    # Drop first quarter per company (which won't have lag-1 history)
    df = df.dropna().reset_index(drop=True)
    
    # Out-of-Time Chronological Split: Last 4 quarters per company reserved as unseen holdout test set
    df['QuarterRank'] = df.groupby('Company')['Date'].rank(ascending=False, method='first')
    df['Split'] = np.where(df['QuarterRank'] <= 4, 'Test', 'Train')
    df = df.drop(columns=['QuarterRank'])
    
    Path('data').mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Feature engineering complete: {len(df)} rows written to {output_path}")
    print(f"Train set: {len(df[df['Split'] == 'Train'])}, Test set: {len(df[df['Split'] == 'Test'])}")
    print(f"Distress prevalence: {df['IsDistressed'].mean():.2%}")

if __name__ == '__main__':
    generate_sec_features()