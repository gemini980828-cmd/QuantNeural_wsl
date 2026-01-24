import os, json
import pandas as pd
import numpy as np

ROOT = "results/eval_9_0_0"
RUNS = ["A1_M_topk50", "A2_Q_topk50", "B3_M_topk200", "B4_Q_topk200", "C1_M_rank", "C2_Q_rank"]

def load_tw(path):
    df = pd.read_csv(path)
    w = df.drop(columns=["date"])
    w = w.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return w

def analyze(run_dir):
    tw_path = os.path.join(run_dir, "target_weights.csv")
    tr_path = os.path.join(run_dir, "trades.csv")
    
    w = load_tw(tw_path)
    arr = w.values
    
    # Holdings
    gt0 = (arr > 0.0).sum(axis=1)
    ge_01pct = (arr >= 0.001).sum(axis=1)
    ge_1pct = (arr >= 0.01).sum(axis=1)
    
    # Concentration
    max_w = np.max(arr, axis=1)
    top10 = np.sort(arr, axis=1)[:, -10:].sum(axis=1)
    eff_n = 1.0 / np.maximum((arr**2).sum(axis=1), 1e-18)
    
    # Trades
    t = pd.read_csv(tr_path)
    t = t[t["delta_weight"].abs() > 0.0]
    g = t.groupby("date")["ticker"].nunique()
    
    return {
        "avg_h>0": np.mean(gt0),
        "max_h>0": np.max(gt0),
        "avg_h>=0.1%": np.mean(ge_01pct),
        "avg_h>=1%": np.mean(ge_1pct),
        "avg_max_w": np.mean(max_w),
        "worst_max_w": np.max(max_w),
        "avg_top10": np.mean(top10),
        "worst_top10": np.max(top10),
        "avg_eff_n": np.mean(eff_n),
        "reb_dates": len(g),
        "avg_trades": g.mean(),
        "max_trades": g.max(),
    }

print("\n=== TRADABILITY DIAGNOSTICS ===\n")

for r in RUNS:
    run_dir = os.path.join(ROOT, r)
    s = analyze(run_dir)
    print(f"--- {r} ---")
    print(f"  Holdings: avg={s['avg_h>0']:.0f}, max={s['max_h>0']:.0f}, avg>=0.1%={s['avg_h>=0.1%']:.0f}, avg>=1%={s['avg_h>=1%']:.0f}")
    print(f"  Concentration: avg_max={s['avg_max_w']:.4f}, worst_max={s['worst_max_w']:.4f}, avg_top10={s['avg_top10']:.4f}, avg_eff_n={s['avg_eff_n']:.0f}")
    print(f"  Trades: dates={s['reb_dates']}, avg_tickers={s['avg_trades']:.0f}, max_tickers={s['max_trades']}")
    print()
