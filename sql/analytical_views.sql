-- 1. Base view with core financial ratios
DROP VIEW IF EXISTS v_financial_ratios;
CREATE VIEW v_financial_ratios AS
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
    ROUND((OperatingProfit / NULLIF(Revenue, 0)) * 100, 2) AS OperatingMarginPct,
    ROUND(((Revenue - COGS) / NULLIF(Revenue, 0)) * 100, 2) AS GrossMarginPct,
    ROUND((AccountsReceivable / NULLIF(Revenue, 0)) * 100, 2) AS ReceivableIntensityPct,
    ROUND((CashAndEquivalents / NULLIF(TotalExpenses, 0)), 2) AS CashBufferRatio
FROM raw_financials;

-- 2. Rolling financial trends and growth metrics per Business Unit
DROP VIEW IF EXISTS v_financial_trends;
CREATE VIEW v_financial_trends AS
SELECT 
    Date,
    BusinessUnit,
    Revenue,
    TotalExpenses,
    OperatingProfit,
    ROUND(AVG(Revenue) OVER (
        PARTITION BY BusinessUnit 
        ORDER BY Date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS Rolling3MonthAvgRevenue,
    ROUND(AVG(TotalExpenses) OVER (
        PARTITION BY BusinessUnit 
        ORDER BY Date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS Rolling3MonthAvgExpenses,
    ROUND(
        ((Revenue - LAG(Revenue, 1) OVER (PARTITION BY BusinessUnit ORDER BY Date)) 
        / NULLIF(LAG(Revenue, 1) OVER (PARTITION BY BusinessUnit ORDER BY Date), 0)) * 100, 
        2
    ) AS RevenueGrowthMoMPct,
    ROUND(
        ((TotalExpenses - LAG(TotalExpenses, 1) OVER (PARTITION BY BusinessUnit ORDER BY Date)) 
        / NULLIF(LAG(TotalExpenses, 1) OVER (PARTITION BY BusinessUnit ORDER BY Date), 0)) * 100, 
        2
    ) AS ExpenseGrowthMoMPct
FROM raw_financials;

-- 3. Budget vs Actual variance performance
DROP VIEW IF EXISTS v_budget_variance;
CREATE VIEW v_budget_variance AS
SELECT 
    Date,
    BusinessUnit,
    Revenue,
    BudgetRevenue,
    ROUND(Revenue - BudgetRevenue, 2) AS RevenueVariance,
    ROUND(((Revenue - BudgetRevenue) / NULLIF(BudgetRevenue, 0)) * 100, 2) AS RevenueVariancePct,
    TotalExpenses,
    BudgetExpenses,
    ROUND(TotalExpenses - BudgetExpenses, 2) AS ExpenseVariance,
    ROUND(((TotalExpenses - BudgetExpenses) / NULLIF(BudgetExpenses, 0)) * 100, 2) AS ExpenseVariancePct
FROM raw_financials;