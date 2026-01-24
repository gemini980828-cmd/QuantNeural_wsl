"""
Real-Data End-to-End Runner for QUANT-NEURAL.

Integrates all real-data components into a single pipeline:
1. Build X features using provided feature_builder
2. Build Y labels from Stooq price data
3. Align X and Y on index intersection
4. Run health gates on X
5. Drop NaN-label rows from Y
6. Train/evaluate baseline MLP

Point-in-Time (PIT) Rules:
- All components respect as_of_date cutoff
- Train-only fit for RankGauss
- No shuffle for MLP training
- Deterministic given fixed inputs + seed
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from src.real_data_dataset import build_monthly_next_returns_from_stooq
from src.real_data_health_gates import run_real_data_health_gates, assert_real_data_health_gates
from src.real_data_train_eval import (
    run_baseline_real_data_mlp_experiment,
    run_shadow_scoring_mlp,
)
from src.shadow_risk_exposure import (
    run_shadow_risk_exposure_logit,
    run_shadow_risk_exposure_logit_with_metrics,
    run_shadow_risk_exposure_mlp_with_metrics,
    run_shadow_risk_overlay_spy_only,
)
from src.shadow_risk_decision_gate import evaluate_shadow_risk_promotion_decision


def drop_nan_label_rows(
    X: pd.DataFrame,
    Y: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows where Y has any NaN values.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature frame.
    Y : pd.DataFrame
        Label frame (must have same index as X).
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (X2, Y2) with NaN-label rows removed.
    
    Raises
    ------
    ValueError
        If result becomes empty after dropping NaN rows.
    """
    # Find rows where Y has no NaN (all values finite)
    valid_mask = ~Y.isna().any(axis=1)
    
    X2 = X.loc[valid_mask].copy()
    Y2 = Y.loc[valid_mask].copy()
    
    if len(X2) == 0:
        raise ValueError("No valid rows remaining after dropping NaN labels")
    
    return X2, Y2


def run_real_data_end_to_end_baseline_mlp(
    *,
    feature_builder: Callable[..., pd.DataFrame],
    feature_builder_kwargs: dict,
    stooq_csv_by_ticker_for_labels: dict[str, str],
    label_tickers_in_order: list[str],
    as_of_date: str,
    train_end: str,
    val_end: str,
    seed: int = 42,
    rankgauss: bool = True,
    epochs: int = 1,
    batch_size: int = 32,
    health_min_months: int = 18,
    health_missing_threshold: float = 0.30,
    health_ignore_first_n_rows: int = 12,
    health_min_sector_firms: int = 3,
    health_max_low_count_month_ratio: float = 1.0,
    # OPTIONAL: Shadow MLP export (no trading impact)
    shadow_mlp_output_csv_path: str | None = None,
    shadow_mlp_epochs: int = 10,
    shadow_mlp_sector_to_tickers: dict[str, list[str]] | None = None,
    # OPTIONAL: Shadow Risk Exposure export (no trading impact)
    shadow_risk_output_csv_path: str | None = None,
    shadow_risk_spy_ticker: str = "SPY",
    shadow_risk_horizon_days: int = 63,
    shadow_risk_metrics_output_json_path: str | None = None,
    # OPTIONAL: Shadow Risk Overlay export (no trading impact)
    shadow_risk_overlay_output_csv_path: str | None = None,
    shadow_risk_overlay_metrics_output_json_path: str | None = None,
    # OPTIONAL: Shadow Risk Decision Gate (no trading impact)
    shadow_risk_decision_gate_output_json_path: str | None = None,
    # OPTIONAL: Shadow Risk MLP export (no trading impact)
    shadow_risk_mlp_output_csv_path: str | None = None,
    shadow_risk_mlp_metrics_output_json_path: str | None = None,
) -> dict:
    """
    Run complete end-to-end baseline MLP experiment on real data.
    
    Parameters
    ----------
    feature_builder : Callable
        Function that builds feature frame (X). Must accept as_of_date kwarg.
    feature_builder_kwargs : dict
        Keyword arguments to pass to feature_builder (except as_of_date).
    stooq_csv_by_ticker_for_labels : dict[str, str]
        Mapping of ticker -> CSV path for label construction.
    label_tickers_in_order : list[str]
        Ordered list of 10 tickers for Y columns.
    as_of_date : str
        PIT cutoff date in "YYYY-MM-DD" format.
    train_end : str
        End date for training set.
    val_end : str
        End date for validation set.
    seed : int
        Random seed for reproducibility.
    rankgauss : bool
        Whether to apply RankGauss transformation.
    epochs : int
        Training epochs.
    batch_size : int
        Training batch size.
    health_min_months : int
        Minimum months required for health gates.
    health_missing_threshold : float
        Maximum missing ratio for health gates.
    health_ignore_first_n_rows : int
        Initial rows to ignore for missing ratio computation.
    shadow_mlp_output_csv_path : str | None, optional
        If provided, exports shadow MLP scores to this CSV path (no trading impact).
    shadow_mlp_epochs : int
        Epochs for shadow MLP training (only used if shadow_mlp_output_csv_path provided).
    shadow_mlp_sector_to_tickers : dict | None
        Optional sector-to-ticker mapping for broadcasting scores.
    
    Returns
    -------
    dict
        Results with keys:
        - "health_report": dict from health gates
        - "train_eval": dict from train/eval
        - "n_rows_xy_before_drop": int
        - "n_rows_xy_after_drop": int
    
    Raises
    ------
    ValueError
        If health gates fail or no valid rows remain.
    """
    # =========================================================================
    # 1. Build X features
    # =========================================================================
    X = feature_builder(**feature_builder_kwargs, as_of_date=as_of_date)
    
    # =========================================================================
    # 2. Build Y labels
    # =========================================================================
    Y = build_monthly_next_returns_from_stooq(
        stooq_csv_by_ticker_for_labels,
        tickers_in_order=label_tickers_in_order,
        as_of_date=as_of_date,
    )
    
    # =========================================================================
    # 3. Align X and Y on index intersection
    # =========================================================================
    idx = X.index.intersection(Y.index)
    X = X.loc[idx].copy()
    Y = Y.loc[idx].copy()
    
    n_rows_xy_before_drop = len(X)
    
    # =========================================================================
    # 4. Run health gates on X
    # =========================================================================
    health_report = run_real_data_health_gates(
        X,
        as_of_date=as_of_date,
        min_months=health_min_months,
        max_feature_missing_ratio=health_missing_threshold,
        ignore_first_n_rows_for_missing=health_ignore_first_n_rows,
        min_sector_firms=health_min_sector_firms,
        max_low_count_month_ratio=health_max_low_count_month_ratio,
    )
    
    # This will raise ValueError if gates fail
    assert_real_data_health_gates(health_report)
    
    # =========================================================================
    # 5. Drop NaN-label rows
    # =========================================================================
    X2, Y2 = drop_nan_label_rows(X, Y)
    
    n_rows_xy_after_drop = len(X2)
    
    # =========================================================================
    # 6. Train/evaluate baseline MLP
    # =========================================================================
    train_eval_result = run_baseline_real_data_mlp_experiment(
        X2, Y2,
        train_end=train_end,
        val_end=val_end,
        seed=seed,
        rankgauss=rankgauss,
        epochs=epochs,
        batch_size=batch_size,
    )
    
    # =========================================================================
    # 7. OPTIONAL: Shadow MLP scoring export (no trading impact)
    # =========================================================================
    shadow_mlp_csv_exported = None
    if shadow_mlp_output_csv_path is not None:
        # Train a separate MLP and export shadow scores to CSV
        run_shadow_scoring_mlp(
            X2, Y2,
            train_end=train_end,
            val_end=val_end,
            output_csv_path=shadow_mlp_output_csv_path,
            sector_to_tickers=shadow_mlp_sector_to_tickers,
            seed=seed,
            rankgauss=rankgauss,
            epochs=shadow_mlp_epochs,
            batch_size=batch_size,
        )
        shadow_mlp_csv_exported = shadow_mlp_output_csv_path
    
    # =========================================================================
    # 8. OPTIONAL: Shadow Risk Exposure export (no trading impact)
    # =========================================================================
    shadow_risk_csv_exported = None
    shadow_risk_metrics_json_exported = None
    if shadow_risk_output_csv_path is not None:
        # Build daily prices panel from label ticker CSVs
        from src.stooq_prices import load_stooq_daily_prices
        import pandas as pd
        
        prices_dict = {}
        for ticker, csv_path in stooq_csv_by_ticker_for_labels.items():
            try:
                df = load_stooq_daily_prices(csv_path, as_of_date=as_of_date, ticker=ticker)
                if df is not None and len(df) > 0:
                    close_series = df.set_index("date")["close"]
                    close_series = close_series[~close_series.index.duplicated(keep="last")].sort_index()
                    prices_dict[ticker.replace(".US", "")] = close_series
            except Exception:
                pass  # Will be handled by fail-safe in run_shadow_risk_exposure_logit
        
        if prices_dict:
            prices_panel = pd.DataFrame(prices_dict)
            prices_panel = prices_panel.dropna(how='all')
            prices_panel = prices_panel.sort_index()
            
            # Use with_metrics if JSON path provided, else use basic function
            if shadow_risk_metrics_output_json_path is not None:
                run_shadow_risk_exposure_logit_with_metrics(
                    prices=prices_panel,
                    as_of_date=as_of_date,
                    train_end=train_end,
                    val_end=val_end,
                    output_csv_path=shadow_risk_output_csv_path,
                    output_metrics_json_path=shadow_risk_metrics_output_json_path,
                    spy_ticker=shadow_risk_spy_ticker,
                    horizon_days=shadow_risk_horizon_days,
                    seed=seed,
                )
                shadow_risk_metrics_json_exported = shadow_risk_metrics_output_json_path
            else:
                run_shadow_risk_exposure_logit(
                    prices=prices_panel,
                    as_of_date=as_of_date,
                    train_end=train_end,
                    val_end=val_end,
                    output_csv_path=shadow_risk_output_csv_path,
                    spy_ticker=shadow_risk_spy_ticker,
                    horizon_days=shadow_risk_horizon_days,
                    seed=seed,
                )
            shadow_risk_csv_exported = shadow_risk_output_csv_path
    
    # =========================================================================
    # 9. OPTIONAL: Shadow Risk Overlay export (no trading impact)
    # =========================================================================
    shadow_risk_overlay_csv_exported = None
    shadow_risk_overlay_metrics_json_exported = None
    if (
        shadow_risk_overlay_output_csv_path is not None and
        shadow_risk_overlay_metrics_output_json_path is not None and
        shadow_risk_output_csv_path is not None
    ):
        # Use same prices_panel as shadow risk (must be built above)
        if prices_dict:
            run_shadow_risk_overlay_spy_only(
                prices=prices_panel,
                as_of_date=as_of_date,
                train_end=train_end,
                val_end=val_end,
                shadow_csv_path=shadow_risk_output_csv_path,
                output_overlay_csv_path=shadow_risk_overlay_output_csv_path,
                output_overlay_metrics_json_path=shadow_risk_overlay_metrics_output_json_path,
                spy_ticker=shadow_risk_spy_ticker,
            )
            shadow_risk_overlay_csv_exported = shadow_risk_overlay_output_csv_path
            shadow_risk_overlay_metrics_json_exported = shadow_risk_overlay_metrics_output_json_path
    
    # =========================================================================
    # 9.5: OPTIONAL: Shadow Risk MLP export (no trading impact)
    # =========================================================================
    shadow_risk_mlp_csv_exported = None
    shadow_risk_mlp_metrics_json_exported = None
    if (
        shadow_risk_mlp_output_csv_path is not None
        and shadow_risk_mlp_metrics_output_json_path is not None
    ):
        # Requires prices_panel from shadow risk (must be built above)
        if prices_dict:
            run_shadow_risk_exposure_mlp_with_metrics(
                prices=prices_panel,
                as_of_date=as_of_date,
                train_end=train_end,
                val_end=val_end,
                output_csv_path=shadow_risk_mlp_output_csv_path,
                output_metrics_json_path=shadow_risk_mlp_metrics_output_json_path,
                spy_ticker=shadow_risk_spy_ticker,
                horizon_days=shadow_risk_horizon_days,
                seed=seed,
            )
            shadow_risk_mlp_csv_exported = shadow_risk_mlp_output_csv_path
            shadow_risk_mlp_metrics_json_exported = shadow_risk_mlp_metrics_output_json_path
    
    # =========================================================================
    # 10. OPTIONAL: Shadow Risk Decision Gate (no trading impact)
    # =========================================================================
    shadow_risk_decision_gate_json_exported = None
    if shadow_risk_decision_gate_output_json_path is not None:
        # Requires all 9.5.x artifacts to exist
        risk_json = shadow_risk_metrics_output_json_path
        overlay_csv = shadow_risk_overlay_output_csv_path
        overlay_json = shadow_risk_overlay_metrics_output_json_path
        
        if risk_json and overlay_csv and overlay_json:
            evaluate_shadow_risk_promotion_decision(
                risk_metrics_json_path=str(risk_json),
                overlay_csv_path=str(overlay_csv),
                overlay_metrics_json_path=str(overlay_json),
                output_decision_json_path=shadow_risk_decision_gate_output_json_path,
            )
            shadow_risk_decision_gate_json_exported = shadow_risk_decision_gate_output_json_path
    
    return {
        "health_report": health_report,
        "train_eval": train_eval_result,
        "n_rows_xy_before_drop": n_rows_xy_before_drop,
        "n_rows_xy_after_drop": n_rows_xy_after_drop,
        # Metadata for meaning alignment
        "label_tickers_in_order": list(label_tickers_in_order),
        "label_semantics": "next_month_returns_of_label_tickers",
        "feature_columns": list(X.columns),
        "y_columns": list(Y.columns),
        # Health params used (resolved from config/defaults)
        "health_params_used": {
            "min_sector_firms": health_min_sector_firms,
            "max_low_count_month_ratio": health_max_low_count_month_ratio,
        },
        # Shadow MLP export path (None if not requested)
        "shadow_mlp_csv_exported": shadow_mlp_csv_exported,
        # Shadow Risk export paths (None if not requested)
        "shadow_risk_csv_exported": shadow_risk_csv_exported,
        "shadow_risk_metrics_json_exported": shadow_risk_metrics_json_exported,
        # Shadow Risk Overlay export paths (None if not requested)
        "shadow_risk_overlay_csv_exported": shadow_risk_overlay_csv_exported,
        "shadow_risk_overlay_metrics_json_exported": shadow_risk_overlay_metrics_json_exported,
        # Shadow Risk Decision Gate export path (None if not requested)
        "shadow_risk_decision_gate_json_exported": shadow_risk_decision_gate_json_exported,
        # Shadow Risk MLP export paths (None if not requested)
        "shadow_risk_mlp_csv_exported": shadow_risk_mlp_csv_exported,
        "shadow_risk_mlp_metrics_json_exported": shadow_risk_mlp_metrics_json_exported,
    }

