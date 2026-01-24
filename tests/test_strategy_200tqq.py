import pandas as pd

from src.strategy_200tqq import Strategy200TQQConfig, compute_200tqq_state, run_200tqq_backtest


def _make_ohlcv(
    dates: pd.DatetimeIndex,
    *,
    close: list[float],
    open_: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
    volume: list[float] | None = None,
    dividend: list[float] | None = None,
    split_ratio: list[float] | None = None,
) -> pd.DataFrame:
    if open_ is None:
        open_ = close
    if high is None:
        high = close
    if low is None:
        low = close
    if volume is None:
        volume = [0.0] * len(close)
    if dividend is None:
        dividend = [0.0] * len(close)
    if split_ratio is None:
        split_ratio = [1.0] * len(close)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "dividend": dividend,
            "split_ratio": split_ratio,
        },
        index=dates,
    )


def test_compute_200tqq_state_boundaries_are_focus():
    # window=2 keeps the test small; we only assert day-2 state.
    dates = pd.date_range("2024-01-01", periods=2, freq="D")

    # close == SMA => FOCUS
    s1 = pd.Series([100.0, 100.0], index=dates)
    out1 = compute_200tqq_state(s1, sma_window=2, overheat_mult=1.05)
    assert out1.loc[dates[1], "state"] == "FOCUS"

    # close == SMA*1.05 => FOCUS
    # If prev=95, curr=105 then SMA=(95+105)/2=100, upper=105.
    s2 = pd.Series([95.0, 105.0], index=dates)
    out2 = compute_200tqq_state(s2, sma_window=2, overheat_mult=1.05)
    assert out2.loc[dates[1], "state"] == "FOCUS"

    # close > upper => OVERHEAT
    s3 = pd.Series([95.0, 106.0], index=dates)
    out3 = compute_200tqq_state(s3, sma_window=2, overheat_mult=1.05)
    assert out3.loc[dates[1], "state"] == "OVERHEAT"

    # close < SMA => DOWN
    s4 = pd.Series([100.0, 90.0], index=dates)
    out4 = compute_200tqq_state(s4, sma_window=2, overheat_mult=1.05)
    assert out4.loc[dates[1], "state"] == "DOWN"


def test_down_to_focus_requires_one_day_confirmation_before_entry():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")

    # Construct closes so:
    # day1 => DOWN (c1 < c0)
    # day2 => FOCUS
    # day3 => FOCUS (confirmation)
    tqqq_close = [100.0, 90.0, 95.0, 95.0, 95.0]

    data = {
        "TQQQ": _make_ohlcv(dates, close=tqqq_close),
        "SPLG": _make_ohlcv(dates, close=[100.0] * 5),
        "BIL": _make_ohlcv(dates, close=[100.0] * 5),
    }

    cfg = Strategy200TQQConfig(
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        initial_equity=1.0,
        sma_window=2,
        overheat_mult=1.05,
        apply_entry_confirmation=True,
        stop_loss_pct=0.0,
        initial_mode="strategy",
        safe_ticker="SGOV",  # not provided; proxy is used
        safe_proxy_ticker="BIL",
    )

    result = run_200tqq_backtest(data, cfg=cfg)
    trades = result["trades"]

    buy_tqqq = trades[(trades["action"] == "BUY") & (trades["ticker"] == "TQQQ")]
    assert len(buy_tqqq) == 1
    # Entry happens on day4 open (after day2 signal + day3 confirmation).
    assert pd.to_datetime(buy_tqqq.iloc[0]["date"]).normalize() == dates[4]


def test_stop_loss_triggers_and_reenters_when_close_above_sma():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")

    # day1 close makes FOCUS with window=2 (100,100) => entry scheduled.
    # day2 low triggers stop (avg=100, stop=95, low=94), but close stays 100 (FOCUS) => re-enter next day.
    tqqq = _make_ohlcv(
        dates,
        close=[100.0, 100.0, 100.0, 100.0],
        low=[100.0, 100.0, 94.0, 100.0],
        open_=[100.0, 100.0, 100.0, 100.0],
    )

    data = {
        "TQQQ": tqqq,
        "SPLG": _make_ohlcv(dates, close=[100.0] * 4),
        "BIL": _make_ohlcv(dates, close=[100.0] * 4),
    }

    cfg = Strategy200TQQConfig(
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        initial_equity=1.0,
        sma_window=2,
        overheat_mult=1.05,
        apply_entry_confirmation=False,
        stop_loss_pct=0.05,
        initial_mode="safe",
        safe_ticker="SGOV",
        safe_proxy_ticker="BIL",
        cost_bps=0.0,
        slippage_bps=0.0,
    )

    result = run_200tqq_backtest(data, cfg=cfg)
    trades = result["trades"]

    sell_stop = trades[(trades["action"] == "SELL_STOP") & (trades["ticker"] == "TQQQ")]
    assert len(sell_stop) == 1
    assert pd.to_datetime(sell_stop.iloc[0]["date"]).normalize() == dates[2]

    buy_tqqq = trades[(trades["action"] == "BUY") & (trades["ticker"] == "TQQQ")]
    assert len(buy_tqqq) == 2
    # First entry: day2 open, re-entry: day3 open
    assert pd.to_datetime(buy_tqqq.iloc[0]["date"]).normalize() == dates[2]
    assert pd.to_datetime(buy_tqqq.iloc[1]["date"]).normalize() == dates[3]


def test_official_defaults_locked():
    """Verify Strategy200TQQConfig defaults match the official strategy rules."""
    cfg = Strategy200TQQConfig()

    # Official settings that must not change
    assert cfg.take_profit_mode == "official", "take_profit_mode must default to 'official'"
    assert cfg.take_profit_reinvest == "all_splg", "take_profit_reinvest must default to 'all_splg'"
    assert cfg.stop_loss_pct == 0.05, "stop_loss_pct must default to 0.05 (5%)"
    assert cfg.apply_entry_confirmation is True, "apply_entry_confirmation must default to True"
    assert cfg.sma_window == 200, "sma_window must default to 200"
    assert cfg.overheat_mult == 1.05, "overheat_mult must default to 1.05 (+5%)"
    assert cfg.initial_mode == "strategy", "initial_mode must default to 'strategy'"
    assert cfg.overheat_start_splg_weight == 1.0, "overheat_start_splg_weight must default to 1.0"


def test_take_profit_trade_generated_at_10_pct_roi():
    """Verify SELL_TAKE_PROFIT trade is generated when TQQQ ROI reaches +10%."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")

    # Construct prices so FOCUS entry on day2, price rises 15% by day4 (triggers +10% take profit)
    # SMA(window=2): day1=100, day2=(100+105)/2=102.5 => close=105 > 102.5 => FOCUS
    tqqq_close = [100.0, 105.0, 105.0, 117.0, 117.0, 117.0]  # 100 -> 117 = +17%
    tqqq_open = [100.0, 100.0, 105.0, 115.0, 117.0, 117.0]

    data = {
        "TQQQ": _make_ohlcv(dates, close=tqqq_close, open_=tqqq_open),
        "SPLG": _make_ohlcv(dates, close=[100.0] * 6, open_=[100.0] * 6),
        "BIL": _make_ohlcv(dates, close=[100.0] * 6, open_=[100.0] * 6),
    }

    cfg = Strategy200TQQConfig(
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        initial_equity=1.0,
        sma_window=2,
        overheat_mult=1.05,
        apply_entry_confirmation=False,  # Skip confirmation for simplicity
        stop_loss_pct=0.0,  # Disable stop-loss
        initial_mode="safe",
        safe_ticker="SGOV",
        safe_proxy_ticker="BIL",
        take_profit_mode="official",  # Must be official
        take_profit_reinvest="all_splg",
    )

    result = run_200tqq_backtest(data, cfg=cfg)
    trades = result["trades"]

    # Check that a SELL_TAKE_PROFIT trade was generated
    take_profit_trades = trades[trades["action"] == "SELL_TAKE_PROFIT"]
    assert len(take_profit_trades) >= 1, "Expected at least one SELL_TAKE_PROFIT trade when ROI > +10%"

    # Check that proceeds went to SPLG (rule 9-A: all_splg reinvest)
    splg_buys_after_tp = trades[
        (trades["action"] == "BUY") & 
        (trades["ticker"] == "SPLG") & 
        (trades["date"] >= take_profit_trades.iloc[0]["date"])
    ]
    assert len(splg_buys_after_tp) >= 1, "Take-profit proceeds should be reinvested into SPLG"
