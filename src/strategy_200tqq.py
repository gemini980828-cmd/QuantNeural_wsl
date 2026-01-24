"""
200티큐 전략 (TQQQ/SPLG/SGOV 스위칭) 백테스트 엔진.

핵심 요약
---------
- 신호는 TQQQ 종가와 SMA200, SMA200*1.05(+5% 상단선)로만 계산한다.
- 상태(STATE): DOWN / FOCUS / OVERHEAT
  - DOWN      : close < SMA200
  - FOCUS     : SMA200 <= close <= SMA200*1.05  (경계 포함: 보수적으로 중간 취급)
  - OVERHEAT  : close > SMA200*1.05
- 실행은 "다음 거래일 시가" 기준(일봉 백테스트 보수적 가정).
- DOWN→FOCUS 전환은 1일 추가 확인 후 진입(2일 연속 SMA200 위).
- DOWN이면 성장자산(TQQQ/SPLG) 전량 매도 후 안전자산(SGOV/대체)로.
- FOCUS면 보유 중 안전자산 전량을 TQQQ로 교체(단, SPLG는 유지).
- OVERHEAT면 기존 TQQQ는 유지(추가 교체 없음). 신규 자금은 SPLG로.

주의/제약
---------
- 장중 스탑로스(-5%)는 일봉 OHLC로 근사한다(저가가 트리거 이하이면 체결로 간주).
- 배당금/분배금은 입력 데이터의 dividend(현금/주) 컬럼이 있을 때만 반영된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


State = Literal["NO_SIGNAL", "DOWN", "FOCUS", "OVERHEAT"]
TakeProfitMode = Literal["none", "official", "high"]
TakeProfitReinvest = Literal["all_splg", "split_principal_profit"]
InitialMode = Literal["safe", "strategy"]


@dataclass(frozen=True)
class Strategy200TQQConfig:
    start_date: str = "2010-01-01"
    end_date: str = "2024-12-31"

    initial_equity: float = 1.0

    tqqq_ticker: str = "TQQQ"
    splg_ticker: str = "SPLG"
    safe_ticker: str = "SGOV"
    safe_proxy_ticker: str = "BIL"

    sma_window: int = 200
    overheat_mult: float = 1.05

    # DOWN->FOCUS 진입 "하루 더 확인"
    apply_entry_confirmation: bool = True

    # 신규자금(월급/배당 등) 월별 투입(0이면 비활성)
    monthly_contribution: float = 0.0

    # 거래비용(매수/매도 각각 동일하게 적용)
    cost_bps: float = 0.0
    slippage_bps: float = 0.0

    # 스탑로스(평단 대비 -X%, 0이면 비활성)
    stop_loss_pct: float = 0.05

    # 부분익절(선택)
    take_profit_mode: TakeProfitMode = "official"
    take_profit_reinvest: TakeProfitReinvest = "all_splg"

    # 시작(첫 진입) 규칙 적용 여부
    initial_mode: InitialMode = "strategy"
    # 시작 시 OVERHEAT이면 SPLG 비중(나머지는 안전자산)
    overheat_start_splg_weight: float = 1.0


def _ensure_columns(df: pd.DataFrame, *, required: list[str]) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    out = df.copy()
    # Optional columns default
    if "dividend" not in out.columns:
        out["dividend"] = 0.0
    if "split_ratio" not in out.columns:
        out["split_ratio"] = 1.0
    # Coerce numeric columns
    for col in required + ["dividend", "split_ratio"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Price DataFrame index must be a DatetimeIndex")
    return out


def compute_200tqq_state(
    tqqq_close: pd.Series,
    *,
    sma_window: int,
    overheat_mult: float,
) -> pd.DataFrame:
    """
    Compute SMA + envelope(+5%) and state series from TQQQ close.

    Boundaries are treated as FOCUS:
    - close == SMA => FOCUS
    - close == SMA*mult => FOCUS
    """
    if sma_window <= 1:
        raise ValueError(f"sma_window must be >= 2, got {sma_window}")
    if overheat_mult <= 1.0:
        raise ValueError(f"overheat_mult must be > 1.0, got {overheat_mult}")

    close = tqqq_close.astype(float).copy()
    sma = close.rolling(window=sma_window, min_periods=sma_window).mean()
    upper = sma * float(overheat_mult)

    state = pd.Series(index=close.index, dtype="object", name="state")
    state.loc[sma.isna()] = "NO_SIGNAL"
    mask = sma.notna()
    # Order matters: DOWN first, OVERHEAT second, else FOCUS.
    state.loc[mask & (close < sma)] = "DOWN"
    state.loc[mask & (close > upper)] = "OVERHEAT"
    state.loc[mask & ~(close < sma) & ~(close > upper)] = "FOCUS"

    return pd.DataFrame(
        {"close": close, "sma": sma, "upper": upper, "state": state.astype(str)},
        index=close.index,
    )


def _fee_rate(cfg: Strategy200TQQConfig) -> float:
    return float(cfg.cost_bps + cfg.slippage_bps) / 10000.0


def _sell(
    shares: float,
    *,
    price: float,
    fee_rate: float,
) -> tuple[float, float]:
    if shares <= 0:
        return 0.0, 0.0
    gross = float(shares) * float(price)
    fee = gross * fee_rate
    return gross - fee, fee


def _buy(
    cash: float,
    *,
    price: float,
    fee_rate: float,
) -> tuple[float, float]:
    if cash <= 0:
        return 0.0, 0.0
    fee = float(cash) * fee_rate
    net = float(cash) - fee
    shares = net / float(price)
    return shares, fee


def _first_trading_day_each_month(dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
    if len(dates) == 0:
        return set()
    s = pd.Series(1, index=dates)
    firsts = s.groupby([dates.year, dates.month]).head(1).index
    return set(pd.to_datetime(firsts))


def run_200tqq_backtest(
    data_by_ticker: dict[str, pd.DataFrame],
    *,
    cfg: Strategy200TQQConfig,
) -> dict:
    """
    Run the 200티큐 전략 backtest.

    Parameters
    ----------
    data_by_ticker:
        Dict[ticker -> DataFrame] where DataFrame index is trading dates and
        columns include: open, high, low, close, volume.
        Optional: dividend (cash/share), split_ratio (share multiplier).
    cfg:
        Strategy configuration.
    """
    # ---- Validate + normalize inputs
    tqqq = cfg.tqqq_ticker
    splg = cfg.splg_ticker
    safe = cfg.safe_ticker
    proxy = cfg.safe_proxy_ticker

    for key in [tqqq, splg, proxy]:
        if key not in data_by_ticker:
            raise ValueError(f"Missing required ticker in data_by_ticker: {key}")

    required_cols = ["open", "high", "low", "close", "volume"]
    df_tqqq = _ensure_columns(data_by_ticker[tqqq], required=required_cols)
    df_splg = _ensure_columns(data_by_ticker[splg], required=required_cols)
    df_proxy = _ensure_columns(data_by_ticker[proxy], required=required_cols)
    df_safe = None
    if safe in data_by_ticker:
        df_safe = _ensure_columns(data_by_ticker[safe], required=required_cols)

    # Trading calendar: require TQQQ/SPLG/proxy all present.
    start_dt = pd.Timestamp(cfg.start_date)
    end_dt = pd.Timestamp(cfg.end_date)
    dates = df_tqqq.index.intersection(df_splg.index).intersection(df_proxy.index)
    dates = dates[(dates >= start_dt) & (dates <= end_dt)]
    dates = dates.sort_values()
    if len(dates) == 0:
        raise ValueError("No overlapping trading dates in requested range.")

    df_tqqq = df_tqqq.reindex(dates)
    df_splg = df_splg.reindex(dates)
    df_proxy = df_proxy.reindex(dates)
    if df_safe is not None:
        df_safe = df_safe.reindex(dates)

    # Precompute state series from TQQQ close
    state_df = compute_200tqq_state(
        df_tqqq["close"], sma_window=cfg.sma_window, overheat_mult=cfg.overheat_mult
    )
    state = state_df["state"].astype(str)

    monthly_contrib_days = (
        _first_trading_day_each_month(dates) if cfg.monthly_contribution > 0 else set()
    )

    # ---- Portfolio state
    fee_rate = _fee_rate(cfg)
    cash = float(cfg.initial_equity)
    shares: dict[str, float] = {tqqq: 0.0, splg: 0.0, safe: 0.0, proxy: 0.0}

    tqqq_cost_basis = 0.0  # total cost basis in currency
    take_profit_done: set[str] = set()

    pending_entry_confirm = False
    start_state_consumed = cfg.initial_mode == "safe"

    trades: list[dict] = []
    daily_rows: list[dict] = []

    def safe_ticker_for(d: pd.Timestamp) -> str:
        if df_safe is not None:
            v = df_safe.at[d, "open"]
            if np.isfinite(v):
                return safe
        return proxy

    def price_at(d: pd.Timestamp, ticker: str, field: str) -> float:
        if ticker == safe and df_safe is None:
            raise KeyError("safe ticker requested but not provided")
        frame = df_safe if ticker == safe else (df_proxy if ticker == proxy else (df_tqqq if ticker == tqqq else df_splg))
        v = frame.at[d, field]
        if not np.isfinite(v):
            raise ValueError(f"Non-finite {field} for {ticker} on {d.date()}")
        return float(v)

    def portfolio_value(d: pd.Timestamp, *, field: str = "close") -> float:
        total = float(cash)
        for tk, sh in shares.items():
            if sh <= 0:
                continue
            # Safe ticker may be absent in early dates; if so, value should be zero anyway.
            if tk == safe and df_safe is None:
                continue
            try:
                px = price_at(d, tk, field)
            except Exception:
                continue
            total += sh * px
        return float(total)

    def apply_split_and_dividend(d: pd.Timestamp) -> None:
        nonlocal cash, tqqq_cost_basis
        for tk, sh in list(shares.items()):
            if sh <= 0:
                continue
            if tk == safe and df_safe is None:
                continue
            # Split: Apply split_ratio to shares for raw price data (merged data).
            # For split-adjusted prices (Yahoo), split_ratio should be 1.0.
            # If split_ratio != 1.0, multiply shares by the ratio.
            split_ratio = price_at(d, tk, "split_ratio")
            if split_ratio and np.isfinite(split_ratio) and float(split_ratio) != 1.0:
                shares[tk] = shares[tk] * float(split_ratio)
            # Dividend: cash += shares * dividend_per_share
            div = price_at(d, tk, "dividend")
            if div and np.isfinite(div) and float(div) != 0.0:
                cash += shares[tk] * float(div)

    def liquidate(tk: str, *, d: pd.Timestamp, price_field: str = "open") -> None:
        nonlocal cash, tqqq_cost_basis
        sh = shares.get(tk, 0.0)
        if sh <= 0:
            return
        px = price_at(d, tk, price_field)
        proceeds, fee = _sell(sh, price=px, fee_rate=fee_rate)
        cash += proceeds
        shares[tk] = 0.0
        trades.append(
            {
                "date": d,
                "action": "SELL",
                "ticker": tk,
                "shares": sh,
                "price": px,
                "fee": fee,
                "cash_delta": proceeds,
            }
        )
        if tk == tqqq:
            tqqq_cost_basis = 0.0
            take_profit_done.clear()

    def buy_with_cash(tk: str, *, d: pd.Timestamp, amount_cash: float, price_field: str = "open") -> None:
        nonlocal cash, tqqq_cost_basis
        if amount_cash <= 0:
            return
        px = price_at(d, tk, price_field)
        sh, fee = _buy(amount_cash, price=px, fee_rate=fee_rate)
        if sh <= 0:
            return
        cash -= amount_cash
        shares[tk] = shares.get(tk, 0.0) + sh
        trades.append(
            {
                "date": d,
                "action": "BUY",
                "ticker": tk,
                "shares": sh,
                "price": px,
                "fee": fee,
                "cash_delta": -amount_cash,
            }
        )
        if tk == tqqq:
            # Include fees in cost basis (avg cost reflects realized trading friction).
            tqqq_cost_basis += amount_cash

    def consolidate_to_safe(d: pd.Timestamp) -> None:
        # Sell all safe tickers to cash and buy the preferred safe ticker for date.
        preferred = safe_ticker_for(d)
        for tk in [safe, proxy]:
            if tk != preferred:
                liquidate(tk, d=d, price_field="open")
        # Buy preferred with all cash.
        buy_with_cash(preferred, d=d, amount_cash=cash, price_field="open")

    def enter_tqqq_from_safe(d: pd.Timestamp) -> None:
        # Sell safe tickers to cash, buy TQQQ with all cash (keep SPLG).
        liquidate(safe, d=d, price_field="open")
        liquidate(proxy, d=d, price_field="open")
        buy_with_cash(tqqq, d=d, amount_cash=cash, price_field="open")

    def invest_new_cash_by_state(d: pd.Timestamp, prev_state: State) -> None:
        # Uses available cash only (positions unchanged).
        nonlocal cash
        if cash <= 0:
            return
        if prev_state == "FOCUS" and not pending_entry_confirm:
            buy_with_cash(tqqq, d=d, amount_cash=cash, price_field="open")
            return
        if prev_state == "OVERHEAT":
            buy_with_cash(splg, d=d, amount_cash=cash, price_field="open")
            return
        # DOWN/NO_SIGNAL or confirmation wait: keep as safe
        consolidate_to_safe(d)

    # We hold "what to do at today's open" based on yesterday close.
    scheduled_rebalance: Optional[dict] = None
    scheduled_take_profit: list[dict] = []

    prev_close_state: State = "NO_SIGNAL"

    for i, d in enumerate(dates):
        # Pre-open: apply splits + dividends for holdings from previous close.
        apply_split_and_dividend(d)

        # Pre-open: monthly contribution (first trading day of month).
        if cfg.monthly_contribution > 0 and d in monthly_contrib_days:
            cash += float(cfg.monthly_contribution)

        # Open: execute scheduled take-profit first (if any).
        if scheduled_take_profit:
            # Execute in the stored order (ascending thresholds).
            for order in scheduled_take_profit:
                # Sell fraction of current TQQQ
                frac = float(order["sell_frac"])
                if shares[tqqq] <= 0 or frac <= 0:
                    continue
                sell_sh = shares[tqqq] * frac
                px = price_at(d, tqqq, "open")
                proceeds, fee = _sell(sell_sh, price=px, fee_rate=fee_rate)
                shares[tqqq] -= sell_sh
                cash += proceeds
                # Reduce cost basis proportionally (avg-cost assumption)
                if shares[tqqq] > 0 and tqqq_cost_basis > 0:
                    avg = tqqq_cost_basis / (shares[tqqq] + sell_sh)
                    tqqq_cost_basis -= avg * sell_sh
                else:
                    tqqq_cost_basis = 0.0
                    take_profit_done.clear()
                trades.append(
                    {
                        "date": d,
                        "action": "SELL_TAKE_PROFIT",
                        "ticker": tqqq,
                        "shares": sell_sh,
                        "price": px,
                        "fee": fee,
                        "cash_delta": proceeds,
                        "checkpoint": order["checkpoint"],
                    }
                )

                # Reinvest proceeds (rule 9)
                if cfg.take_profit_reinvest == "all_splg":
                    buy_with_cash(splg, d=d, amount_cash=proceeds, price_field="open")
                else:
                    r = float(order.get("roi_at_signal", 0.0))
                    principal = proceeds / (1.0 + r) if (1.0 + r) > 0 else proceeds
                    profit = proceeds - principal
                    if principal > 0:
                        buy_with_cash(splg, d=d, amount_cash=principal, price_field="open")
                    if profit > 0:
                        preferred_safe = safe_ticker_for(d)
                        buy_with_cash(preferred_safe, d=d, amount_cash=profit, price_field="open")

        scheduled_take_profit = []

        # Open: execute scheduled rebalance (if any).
        if scheduled_rebalance is not None:
            kind = scheduled_rebalance.get("kind")
            if kind == "EXIT_TO_SAFE":
                liquidate(tqqq, d=d, price_field="open")
                liquidate(splg, d=d, price_field="open")
                consolidate_to_safe(d)
            elif kind == "ENTER_TQQQ":
                enter_tqqq_from_safe(d)
            elif kind == "START_OVERHEAT_ALLOC":
                w = float(scheduled_rebalance.get("splg_weight", 1.0))
                w = min(1.0, max(0.0, w))
                # Start from whatever we currently have; target SPLG = w * total equity.
                total = portfolio_value(d, field="open")
                target_cash = total * w
                # Ensure we have enough cash: sell safe holdings as needed.
                preferred_safe = safe_ticker_for(d)
                safe_px = price_at(d, preferred_safe, "open")
                safe_value = shares[preferred_safe] * safe_px
                need = max(0.0, target_cash - cash)
                if need > 0 and safe_value > 0:
                    # Sell enough safe so that net proceeds cover 'need'
                    gross_need = need / (1.0 - fee_rate) if (1.0 - fee_rate) > 0 else need
                    sell_sh = min(shares[preferred_safe], gross_need / safe_px)
                    proceeds, fee = _sell(sell_sh, price=safe_px, fee_rate=fee_rate)
                    shares[preferred_safe] -= sell_sh
                    cash += proceeds
                    trades.append(
                        {
                            "date": d,
                            "action": "SELL",
                            "ticker": preferred_safe,
                            "shares": sell_sh,
                            "price": safe_px,
                            "fee": fee,
                            "cash_delta": proceeds,
                            "note": "start_overheat_alloc",
                        }
                    )
                buy_with_cash(splg, d=d, amount_cash=min(cash, target_cash), price_field="open")
            else:
                raise ValueError(f"Unknown scheduled rebalance kind: {kind}")
            scheduled_rebalance = None

        # Open: deploy any cash (dividends/monthly) per prev_close_state rules.
        invest_new_cash_by_state(d, prev_close_state)

        # Intraday: stop-loss on TQQQ (OHLC approximation).
        stopped_out_today = False
        if cfg.stop_loss_pct and cfg.stop_loss_pct > 0 and shares[tqqq] > 0 and tqqq_cost_basis > 0:
            avg_cost = tqqq_cost_basis / shares[tqqq]
            stop_price = avg_cost * (1.0 - float(cfg.stop_loss_pct))
            o = price_at(d, tqqq, "open")
            lo = price_at(d, tqqq, "low")
            if lo <= stop_price:
                fill = o if o <= stop_price else stop_price
                # Sell TQQQ at fill
                proceeds, fee = _sell(shares[tqqq], price=fill, fee_rate=fee_rate)
                trades.append(
                    {
                        "date": d,
                        "action": "SELL_STOP",
                        "ticker": tqqq,
                        "shares": shares[tqqq],
                        "price": fill,
                        "fee": fee,
                        "cash_delta": proceeds,
                        "stop_price": stop_price,
                    }
                )
                cash += proceeds
                shares[tqqq] = 0.0
                tqqq_cost_basis = 0.0
                take_profit_done.clear()
                stopped_out_today = True
                # Also liquidate SPLG (rule 7-B) at close (daily-bar approximation).
                # We'll schedule it by directly selling at close after valuation.

        # Close: portfolio valuation + state update
        equity_close = portfolio_value(d, field="close")

        row = {
            "date": d,
            "state": state.at[d],
            "equity": equity_close,
            "cash": cash,
            f"{tqqq}_shares": shares[tqqq],
            f"{splg}_shares": shares[splg],
            f"{safe}_shares": shares.get(safe, 0.0),
            f"{proxy}_shares": shares.get(proxy, 0.0),
        }
        daily_rows.append(row)

        # If stopped out intraday, also sell SPLG at close (approx), then decide next-day action.
        if stopped_out_today and shares[splg] > 0:
            px = price_at(d, splg, "close")
            proceeds, fee = _sell(shares[splg], price=px, fee_rate=fee_rate)
            trades.append(
                {
                    "date": d,
                    "action": "SELL_STOP_COMPANION",
                    "ticker": splg,
                    "shares": shares[splg],
                    "price": px,
                    "fee": fee,
                    "cash_delta": proceeds,
                }
            )
            cash += proceeds
            shares[splg] = 0.0

        # Determine next-day schedule based on today's close state.
        today_state: State = state.at[d]  # type: ignore[assignment]

        # Start-state special rule (one-time), when SMA becomes available.
        start_consumed_this_close = False
        if not start_state_consumed and today_state != "NO_SIGNAL":
            start_state_consumed = True
            start_consumed_this_close = True
            if today_state == "OVERHEAT":
                scheduled_rebalance = {
                    "kind": "START_OVERHEAT_ALLOC",
                    "splg_weight": float(cfg.overheat_start_splg_weight),
                }
            elif today_state == "FOCUS" and cfg.apply_entry_confirmation:
                pending_entry_confirm = True

        # Stop-loss re-entry override (rule 7-C): if stopped out and close above SMA, re-enter next open.
        if stopped_out_today:
            if today_state in ("FOCUS", "OVERHEAT"):
                scheduled_rebalance = {"kind": "ENTER_TQQQ"}
                pending_entry_confirm = False
            else:
                scheduled_rebalance = {"kind": "EXIT_TO_SAFE"}
                pending_entry_confirm = False
            prev_close_state = today_state
            continue

        # If we just consumed the first actionable start-state, don't apply normal scheduling on the
        # same close (e.g., FOCUS-start should wait 1 more day).
        if start_consumed_this_close:
            prev_close_state = today_state
            continue

        # Normal rules after close
        if today_state == "DOWN":
            scheduled_rebalance = {"kind": "EXIT_TO_SAFE"}
            pending_entry_confirm = False
        elif today_state == "FOCUS":
            # DOWN->FOCUS confirmation logic
            if cfg.apply_entry_confirmation and prev_close_state == "DOWN" and shares[tqqq] <= 0:
                pending_entry_confirm = True
            elif pending_entry_confirm:
                # Confirmed (second consecutive FOCUS), enter next open.
                scheduled_rebalance = {"kind": "ENTER_TQQQ"}
                pending_entry_confirm = False
            else:
                # If holding any safe, convert to TQQQ
                if shares[safe] > 0 or shares[proxy] > 0:
                    scheduled_rebalance = {"kind": "ENTER_TQQQ"}
        else:
            # OVERHEAT or NO_SIGNAL: no switching
            pass

        # Take-profit scheduling (evaluate on close, execute next open)
        if cfg.take_profit_mode != "none" and shares[tqqq] > 0 and tqqq_cost_basis > 0 and today_state != "DOWN":
            avg_cost = tqqq_cost_basis / shares[tqqq]
            px = price_at(d, tqqq, "close")
            roi = (px - avg_cost) / avg_cost if avg_cost > 0 else 0.0

            def enqueue(checkpoint: str, sell_frac: float) -> None:
                if checkpoint in take_profit_done:
                    return
                take_profit_done.add(checkpoint)
                scheduled_take_profit.append(
                    {"checkpoint": checkpoint, "sell_frac": float(sell_frac), "roi_at_signal": float(roi)}
                )

            if cfg.take_profit_mode == "official":
                if roi >= 0.10:
                    enqueue("+10%", 0.10)
                if roi >= 0.25:
                    enqueue("+25%", 0.10)
                if roi >= 0.50:
                    enqueue("+50%", 0.10)
            if cfg.take_profit_mode in ("official", "high"):
                # 100% multiples: +100%, +200%, ...
                if roi >= 1.0:
                    k = int(np.floor(roi / 1.0))
                    for m in range(1, k + 1):
                        enqueue(f"+{m*100}%", 0.50)

        prev_close_state = today_state

    # Build outputs
    daily = pd.DataFrame(daily_rows).set_index("date").sort_index()
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df = trades_df.sort_values("date").reset_index(drop=True)
    else:
        trades_df = pd.DataFrame(columns=["date", "action", "ticker", "shares", "price", "fee", "cash_delta"])

    equity = daily["equity"].astype(float)
    rets = equity.pct_change().fillna(0.0)
    n_days = len(equity)
    ann_factor = 252
    total_return = float(equity.iloc[-1] / float(cfg.initial_equity) - 1.0) if n_days > 0 else 0.0
    years = n_days / ann_factor
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    ann_vol = float(rets.std() * np.sqrt(ann_factor))
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0
    cagr_over_vol = cagr / ann_vol if ann_vol > 0 else 0.0

    metrics = {
        "total_return": total_return,
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "cagr_over_vol": float(cagr_over_vol),
        "max_drawdown": float(max_dd),
        "n_trades": int(len(trades_df)),
    }

    return {
        "daily": daily,
        "trades": trades_df,
        "metrics": metrics,
        "state": state_df,
    }
