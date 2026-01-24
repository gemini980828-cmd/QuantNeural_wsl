"""
Audit Manifest vs Stooq Tickers and Build Stooq-Aligned Manifest.

Diagnoses ticker-key mismatch between Stooq universe and SEC manifest,
produces deterministic audit reports and a new manifest aligned to Stooq tickers.

All implementer prompts must be written in English (must include this statement verbatim).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_ticker(s: str) -> str:
    """
    Normalize ticker string for deterministic matching.
    
    Rules (in order):
    1. Strip whitespace
    2. Upper-case
    3. Replace '-' with '.'
    4. If contains ':', keep only last token (e.g., "NYSE:BRK.B" -> "BRK.B")
    5. Collapse multiple dots to single dot
    6. Remove trailing ".US" suffix
    
    Parameters
    ----------
    s : str
        Raw ticker string
        
    Returns
    -------
    str
        Normalized ticker
    """
    if not isinstance(s, str):
        s = str(s)
    
    # 1. Strip whitespace
    s = s.strip()
    
    # 2. Upper-case
    s = s.upper()
    
    # 3. Replace '-' with '.'
    s = s.replace('-', '.')
    
    # 4. If contains ':', keep only last token
    if ':' in s:
        s = s.split(':')[-1]
    
    # 5. Collapse multiple dots to single dot
    s = re.sub(r'\.+', '.', s)
    
    # 6. Remove trailing ".US" suffix
    if s.endswith('.US'):
        s = s[:-3]
    
    return s


def extract_stooq_tickers(stooq_data_dir: str | Path) -> pd.DataFrame:
    """
    Scan Stooq directory and extract tickers from filenames.
    
    Parameters
    ----------
    stooq_data_dir : str | Path
        Path to directory containing Stooq files
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: stooq_ticker_raw, stooq_ticker_norm
    """
    dir_path = Path(stooq_data_dir)
    
    if not dir_path.exists():
        logger.warning(f"Stooq directory does not exist: {stooq_data_dir}")
        return pd.DataFrame(columns=['stooq_ticker_raw', 'stooq_ticker_norm'])
    
    # Supported extensions
    extensions = {'.csv', '.txt', '.gz'}
    
    tickers = []
    for f in dir_path.iterdir():
        if not f.is_file():
            continue
        
        # Get stem, handling double extensions like .csv.gz
        name = f.name
        stem = name
        for ext in ['.csv.gz', '.txt.gz', '.csv', '.txt']:
            if name.lower().endswith(ext):
                stem = name[:-len(ext)]
                break
        
        if stem:
            tickers.append({
                'stooq_ticker_raw': stem,
                'stooq_ticker_norm': normalize_ticker(stem),
            })
    
    df = pd.DataFrame(tickers)
    
    # Deterministic sort
    if not df.empty:
        df = df.sort_values('stooq_ticker_raw').reset_index(drop=True)
    
    return df


def load_manifest(manifest_csv: str | Path) -> pd.DataFrame:
    """
    Load and parse SEC manifest CSV.
    
    Parameters
    ----------
    manifest_csv : str | Path
        Path to manifest CSV
        
    Returns
    -------
    pd.DataFrame
        Filtered manifest with columns: manifest_ticker_raw, manifest_ticker_norm,
        companyfacts_status, companyfacts_path
        
    Raises
    ------
    ValueError
        If required columns are missing
    """
    path = Path(manifest_csv)
    
    if not path.exists():
        logger.warning(f"Manifest file does not exist: {manifest_csv}")
        return pd.DataFrame(columns=[
            'manifest_ticker_raw', 'manifest_ticker_norm',
            'companyfacts_status', 'companyfacts_path'
        ])
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.warning(f"Failed to read manifest CSV: {e}")
        return pd.DataFrame(columns=[
            'manifest_ticker_raw', 'manifest_ticker_norm',
            'companyfacts_status', 'companyfacts_path'
        ])
    
    # Validate required columns
    required = {'ticker', 'companyfacts_status', 'companyfacts_path'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest CSV missing required columns: {missing}")
    
    # Filter to status == "ok" and non-empty path
    df = df[df['companyfacts_status'] == 'ok'].copy()
    df = df[df['companyfacts_path'].notna() & (df['companyfacts_path'].astype(str).str.strip() != '')].copy()
    
    # Create normalized columns
    df['manifest_ticker_raw'] = df['ticker'].astype(str)
    df['manifest_ticker_norm'] = df['manifest_ticker_raw'].apply(normalize_ticker)
    
    # Keep required columns
    df = df[['manifest_ticker_raw', 'manifest_ticker_norm', 
             'companyfacts_status', 'companyfacts_path']].copy()
    
    # Deterministic sort
    df = df.sort_values('manifest_ticker_raw').reset_index(drop=True)
    
    return df


def build_matches(
    stooq_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build matched, stooq-only, and manifest-only DataFrames.
    
    Parameters
    ----------
    stooq_df : pd.DataFrame
        Stooq tickers with stooq_ticker_raw, stooq_ticker_norm
    manifest_df : pd.DataFrame
        Manifest tickers with manifest_ticker_raw, manifest_ticker_norm,
        companyfacts_status, companyfacts_path
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (matches, stooq_only, manifest_only)
    """
    if stooq_df.empty or manifest_df.empty:
        matches = pd.DataFrame(columns=[
            'stooq_ticker_raw', 'stooq_ticker_norm',
            'manifest_ticker_raw', 'manifest_ticker_norm',
            'companyfacts_path', 'match_type'
        ])
        stooq_only = stooq_df[['stooq_ticker_raw', 'stooq_ticker_norm']].copy() if not stooq_df.empty else pd.DataFrame()
        manifest_only = manifest_df[['manifest_ticker_raw', 'manifest_ticker_norm']].copy() if not manifest_df.empty else pd.DataFrame()
        return matches, stooq_only, manifest_only
    
    # Deduplicate manifest by normalized ticker
    # Keep lexicographically smallest manifest_ticker_raw for each norm
    manifest_dedup = (
        manifest_df
        .sort_values('manifest_ticker_raw')
        .drop_duplicates(subset='manifest_ticker_norm', keep='first')
        .copy()
    )
    
    # Build lookup: manifest_ticker_norm -> row
    manifest_lookup = manifest_dedup.set_index('manifest_ticker_norm').to_dict('index')
    
    matches_list = []
    stooq_only_list = []
    matched_manifest_norms = set()
    
    for _, row in stooq_df.iterrows():
        stooq_raw = row['stooq_ticker_raw']
        stooq_norm = row['stooq_ticker_norm']
        
        if stooq_norm in manifest_lookup:
            m = manifest_lookup[stooq_norm]
            matches_list.append({
                'stooq_ticker_raw': stooq_raw,
                'stooq_ticker_norm': stooq_norm,
                'manifest_ticker_raw': m['manifest_ticker_raw'],
                'manifest_ticker_norm': stooq_norm,
                'companyfacts_path': m['companyfacts_path'],
                'match_type': 'norm_exact',
            })
            matched_manifest_norms.add(stooq_norm)
        else:
            stooq_only_list.append({
                'stooq_ticker_raw': stooq_raw,
                'stooq_ticker_norm': stooq_norm,
            })
    
    # Manifest-only: those not matched
    manifest_only_list = []
    for _, row in manifest_dedup.iterrows():
        if row['manifest_ticker_norm'] not in matched_manifest_norms:
            manifest_only_list.append({
                'manifest_ticker_raw': row['manifest_ticker_raw'],
                'manifest_ticker_norm': row['manifest_ticker_norm'],
            })
    
    matches = pd.DataFrame(matches_list)
    stooq_only = pd.DataFrame(stooq_only_list)
    manifest_only = pd.DataFrame(manifest_only_list)
    
    # Deterministic sort
    if not matches.empty:
        matches = matches.sort_values('stooq_ticker_raw').reset_index(drop=True)
    if not stooq_only.empty:
        stooq_only = stooq_only.sort_values('stooq_ticker_raw').reset_index(drop=True)
    if not manifest_only.empty:
        manifest_only = manifest_only.sort_values('manifest_ticker_raw').reset_index(drop=True)
    
    return matches, stooq_only, manifest_only


def build_stooq_aligned_manifest(
    stooq_df: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build manifest aligned to Stooq tickers.
    
    Parameters
    ----------
    stooq_df : pd.DataFrame
        All Stooq tickers
    matches : pd.DataFrame
        Matched tickers with companyfacts_path
        
    Returns
    -------
    pd.DataFrame
        Stooq-aligned manifest with columns: ticker, companyfacts_status, companyfacts_path
    """
    if stooq_df.empty:
        return pd.DataFrame(columns=['ticker', 'companyfacts_status', 'companyfacts_path'])
    
    # Build lookup from stooq_ticker_raw -> companyfacts_path
    match_lookup = {}
    if not matches.empty:
        match_lookup = matches.set_index('stooq_ticker_raw')['companyfacts_path'].to_dict()
    
    rows = []
    for _, row in stooq_df.iterrows():
        stooq_raw = row['stooq_ticker_raw']
        if stooq_raw in match_lookup:
            rows.append({
                'ticker': stooq_raw,
                'companyfacts_status': 'ok',
                'companyfacts_path': match_lookup[stooq_raw],
            })
        else:
            rows.append({
                'ticker': stooq_raw,
                'companyfacts_status': 'missing',
                'companyfacts_path': '',
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('ticker').reset_index(drop=True)
    
    return df


def run_audit(
    stooq_data_dir: str,
    manifest_csv: str,
    out_dir: str,
) -> dict:
    """
    Run full audit and write outputs.
    
    Parameters
    ----------
    stooq_data_dir : str
        Path to Stooq data directory
    manifest_csv : str
        Path to manifest CSV
    out_dir : str
        Output directory for artifacts
        
    Returns
    -------
    dict
        Audit summary
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Extract tickers
    stooq_df = extract_stooq_tickers(stooq_data_dir)
    manifest_df = load_manifest(manifest_csv)
    
    # Build matches
    matches, stooq_only, manifest_only = build_matches(stooq_df, manifest_df)
    
    # Build stooq-aligned manifest
    aligned_manifest = build_stooq_aligned_manifest(stooq_df, matches)
    
    # Compute summary
    stooq_count = len(stooq_df)
    manifest_ok_count = len(manifest_df.drop_duplicates('manifest_ticker_norm'))
    overlap_count = len(matches)
    stooq_only_count = len(stooq_only)
    manifest_only_count = len(manifest_only)
    overlap_ratio = overlap_count / stooq_count if stooq_count > 0 else 0.0
    
    summary = {
        'stooq_count': stooq_count,
        'manifest_ok_count': manifest_ok_count,
        'overlap_count': overlap_count,
        'stooq_only_count': stooq_only_count,
        'manifest_only_count': manifest_only_count,
        'overlap_ratio': round(overlap_ratio, 4),
        'notes': 'Non-overlap tickers cannot receive SEC facts. Use stooq_aligned_manifest.csv for alpha dataset.',
    }
    
    # Write outputs
    with open(out_path / 'audit_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    matches.to_csv(out_path / 'matches.csv', index=False)
    stooq_only.to_csv(out_path / 'stooq_only.csv', index=False)
    manifest_only.to_csv(out_path / 'manifest_only.csv', index=False)
    aligned_manifest.to_csv(out_path / 'stooq_aligned_manifest.csv', index=False)
    
    print(f"Audit complete. Outputs written to: {out_path}")
    print(f"  Stooq tickers: {stooq_count}")
    print(f"  Manifest OK:   {manifest_ok_count}")
    print(f"  Overlap:       {overlap_count} ({overlap_ratio:.1%})")
    print(f"  Stooq-only:    {stooq_only_count}")
    print(f"  Manifest-only: {manifest_only_count}")
    
    return summary


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Audit manifest vs Stooq tickers and build aligned manifest"
    )
    parser.add_argument(
        '--stooq-data-dir', required=True,
        help='Path to Stooq data directory containing ticker files'
    )
    parser.add_argument(
        '--manifest-csv', required=True,
        help='Path to SEC manifest CSV'
    )
    parser.add_argument(
        '--out-dir', required=True,
        help='Output directory for audit artifacts'
    )
    
    args = parser.parse_args()
    
    try:
        run_audit(
            stooq_data_dir=args.stooq_data_dir,
            manifest_csv=args.manifest_csv,
            out_dir=args.out_dir,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
