"""Append preprocessing function to sec_fundamentals.py"""

code = '''

# ==============================================================================
# SEC Fundamental Preprocessing for Coverage Enhancement
# ==============================================================================


SEC_FUNDAMENTAL_COLS = [
    "assets", "liabilities", "equity", "cash", "shares_out",
    "leverage", "cash_to_assets", "book_to_assets", "mktcap",
]


def preprocess_fundamentals_for_coverage(
    df: "pd.DataFrame",
    *,
    ffill: bool = True,
    fill_method: str = "median",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> "pd.DataFrame":
    """
    Preprocess SEC fundamental columns to improve coverage.
    
    PIT-safe preprocessing:
    1. Forward-fill within each ticker (only uses past values)
    2. Cross-sectional fill per date (median or zero)
    """
    if df.empty:
        return df
    
    sec_cols = [c for c in SEC_FUNDAMENTAL_COLS if c in df.columns]
    
    if not sec_cols:
        return df
    
    result = df.copy()
    
    # Step 1: Forward-fill within each ticker (PIT-safe)
    if ffill and ticker_col in result.columns:
        result = result.sort_values([ticker_col, date_col]).reset_index(drop=True)
        for col in sec_cols:
            result[col] = result.groupby(ticker_col)[col].ffill()
    
    # Step 2: Cross-sectional fill per date
    if fill_method == "median" and date_col in result.columns:
        for col in sec_cols:
            date_medians = result.groupby(date_col)[col].transform("median")
            result[col] = result[col].fillna(date_medians)
    elif fill_method == "zero":
        for col in sec_cols:
            result[col] = result[col].fillna(0.0)
    
    # Ensure float32 dtype
    for col in sec_cols:
        result[col] = result[col].astype("float32")
    
    return result
'''

with open('src/sec_fundamentals.py', 'a', encoding='utf-8') as f:
    f.write(code)

print('Successfully appended preprocess_fundamentals_for_coverage to sec_fundamentals.py')
