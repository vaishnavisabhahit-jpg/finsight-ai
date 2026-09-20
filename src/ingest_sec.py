import time
import requests
import pandas as pd
from pathlib import Path

HEADERS = {'User-Agent': 'FinSightAI Research contact@finsight.local'}

# Multi-sector cohort: Apple, Microsoft, Walmart, Target, Ford, Caterpillar, Amazon, Intel
COMPANIES = [
    ('0000320193', 'Apple'),
    ('0000789019', 'Microsoft'),
    ('0000104169', 'Walmart'),
    ('0000027419', 'Target'),
    ('0000037996', 'Ford'),
    ('0000018230', 'Caterpillar'),
    ('0001018724', 'Amazon'),
    ('0000050863', 'Intel')
]

def fetch_quarterly_facts(cik: str, company_name: str) -> pd.DataFrame:
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json'
    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        print(f'Failed to fetch CIK {cik} (Status: {res.status_code})')
        return pd.DataFrame()
    
    data = res.json().get('facts', {}).get('us-gaap', {})
    
    def extract_metric(tag: str, col_name: str):
        if tag not in data:
            return pd.DataFrame()
        units = data[tag].get('units', {})
        unit_key = 'USD' if 'USD' in units else (list(units.keys())[0] if units else None)
        if not unit_key:
            return pd.DataFrame()
        df = pd.DataFrame(units[unit_key])
        # Filter strictly for 10-Q (Quarterly) filings
        df = df[df['form'] == '10-Q'].copy()
        if 'end' in df.columns and 'val' in df.columns:
            return df[['end', 'val']].drop_duplicates(subset=['end']).rename(columns={'end': 'Date', 'val': col_name})
        return pd.DataFrame()

    # Revenue
    rev = extract_metric('Revenues', 'Revenue')
    if rev.empty:
        rev = extract_metric('RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenue')
    
    # Operating Income
    op_inc = extract_metric('OperatingIncomeLoss', 'OperatingProfit')
    
    # Cash & Short Term Liquidity
    cash = extract_metric('CashAndCashEquivalentsAtCarryingValue', 'Cash')
    
    # Long Term Debt
    debt = extract_metric('LongTermDebtNoncurrent', 'Debt')
    
    if rev.empty or op_inc.empty:
        print(f'Insufficient line items for {company_name}')
        return pd.DataFrame()
        
    merged = rev.merge(op_inc, on='Date', how='inner')
    if not cash.empty:
        merged = merged.merge(cash, on='Date', how='left')
    else:
        merged['Cash'] = 0.0
        
    if not debt.empty:
        merged = merged.merge(debt, on='Date', how='left')
    else:
        merged['Debt'] = 0.0
        
    merged['Company'] = company_name
    merged['CIK'] = cik
    return merged

def main():
    Path('data').mkdir(exist_ok=True)
    all_records = []
    print('Connecting to SEC EDGAR API...')
    
    for cik, name in COMPANIES:
        print(f'Pulling 10-Q filings for {name}...')
        df = fetch_quarterly_facts(cik, name)
        if not df.empty:
            all_records.append(df)
            print(f'  -> Loaded {len(df)} quarters for {name}')
        time.sleep(0.2)  # Respect SEC rate limits (<10 req/sec)
        
    if all_records:
        master_df = pd.concat(all_records, ignore_index=True)
        master_df.sort_values(by=['Company', 'Date'], inplace=True)
        master_df.to_csv('data/real_sec_financials.csv', index=False)
        print(f'Successfully compiled {len(master_df)} real quarterly records to data/real_sec_financials.csv')
    else:
        print('No records collected.')

if __name__ == '__main__':
    main()
