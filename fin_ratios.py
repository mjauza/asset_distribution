from financetoolkit import Toolkit
import os
import pandas as pd


def get_fin_ratios(
    companies_symbols,
    pr_select = ['Gross Margin', 'Operating Margin', 'Net Profit Margin'],
    er_select = ['Days of Inventory Outstanding (DIO)','Days of Sales Outstanding (DSO)', 'Operating Cycle (CC)'],
    lr_select = ['Current Ratio', 'Quick Ratio', 'Cash Ratio'],
    sr_select = ['Debt-to-Assets Ratio', 'Debt-to-Equity Ratio'],
    vr_select = ['Earnings per Share (EPS)', 'Revenue per Share (RPS)','Price-to-Earnings (PE)']
):
    FMP_KEY = os.getenv("FMP")
    companies = Toolkit(companies_symbols, api_key=FMP_KEY)

    profitability_ratios = companies.ratios.collect_profitability_ratios()
    efficiency_ratios = companies.ratios.collect_efficiency_ratios()
    liquidity_ratios = companies.ratios.collect_liquidity_ratios()
    solvency_ratios = companies.ratios.collect_solvency_ratios()
    valuation_ratios = companies.ratios.collect_valuation_ratios()

    if len(companies_symbols) == 1:
        df = pd.concat([
            profitability_ratios.loc[pr_select, :],
            efficiency_ratios.loc[er_select, :],
            liquidity_ratios.loc[lr_select, :],
            solvency_ratios.loc[sr_select, :],
            valuation_ratios.loc[vr_select, :]
        ], axis=0)
        df.dropna(axis=1, inplace=True)
        return {companies_symbols[0] : df}

    res = {}
    for symbol in companies_symbols:
        df = pd.concat([
            profitability_ratios.loc[symbol].loc[pr_select, :],
            efficiency_ratios.loc[symbol].loc[er_select, :],
            liquidity_ratios.loc[symbol].loc[lr_select, :],
            solvency_ratios.loc[symbol].loc[sr_select, :],
            valuation_ratios.loc[symbol].loc[vr_select, :]
        ], axis=0)
        df.dropna(axis=1, inplace=True)
        res[symbol] = df

    return res


if __name__ == "__main__":
    res = get_fin_ratios(["AAPL"])
    print(res)