import os, json
import pandas as pd
import numpy as np

RUNS = {
    "B3_M_topk200": "results/eval_9_0_0/B3_M_topk200",
    "B4_Q_topk200": "results/eval_9_0_0/B4_Q_topk200",
    "R1_M_rank_topk200": "results/eval_9_1_0/R1_M_rank_topk200",
    "R2_Q_rank_topk200": "results/eval_9_1_0/R2_Q_rank_topk200",
}

def load_tw(path):
    df = pd.read_csv(path)
    w = df.drop(columns=["date"])
    w = w.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return w

def analyze(run_dir):
    tw_path = os.path.join(run_dir, "target_weights.csv")
    tr_path = os.path.join(run_dir, "trades.csv")
    sm_path = os.path.join(run_dir, "summary_metrics.json")
    
    w = load_tw(tw_path)
    arr = w.values
    
    # Holdings
    gt0 = (arr > 0.0).sum(axis=1)
    
    # Concentration
    max_w = np.max(arr, axis=1)
    top10 = np.sort(arr, axis=1)[:, -10:].sum(axis=1)
    eff_n = 1.0 / np.maximum((arr**2).sum(axis=1), 1e-18)
    
    # Trades
    t = pd.read_csv(tr_path)
    t = t[t["delta_weight"].abs() > 0.0]
    g = t.groupby("date")["ticker"].nunique()
    
    # Metrics
    j = json.load(open(sm_path))
    m = j["metrics"]
    
    return {
        "cagr": m.get("cagr"),
        "sharpe": m.get("sharpe"),
        "max_dd": m.get("max_drawdown"),
        "turnover": m.get("total_turnover"),
        "cost": m.get("total_cost"),
        "avg_holdings": np.mean(gt0),
        "avg_max_w": np.mean(max_w),
        "avg_eff_n": np.mean(eff_n),
        "avg_trades": g.mean(),
        "max_trades": g.max(),
    }

print("\n=== TRADABLE RANK VALIDATION (9.1.0) ===\n")
print("| Run | Method | CAGR | Sharpe | MaxDD | Turn | Cost | Holdings | EffN | AvgTrades | MaxTrades |")
print("|-----|--------|------|--------|-------|------|------|----------|------|-----------|-----------|")

for name, path in RUNS.items():
    s = analyze(path)
    method = "rank+topk" if "rank" in name else "topk"
    print(f"| {name} | {method} | {s['cagr']:.2%} | {s['sharpe']:.3f} | {s['max_dd']:.1%} | {s['turnover']:.1f} | {s['cost']:.1%} | {s['avg_holdings']:.0f} | {s['avg_eff_n']:.0f} | {s['avg_trades']:.0f} | {s['max_trades']} |")

# Comparison
print("\n=== COMPARISON (rank+topk vs topk, same top_k=200) ===\n")
b3 = analyze(RUNS["B3_M_topk200"])
r1 = analyze(RUNS["R1_M_rank_topk200"])
b4 = analyze(RUNS["B4_Q_topk200"])
r2 = analyze(RUNS["R2_Q_rank_topk200"])

print("Monthly (M): rank+topk vs topk")
print(f"  Sharpe: {r1['sharpe']:.3f} vs {b3['sharpe']:.3f} ({(r1['sharpe']/b3['sharpe']-1)*100:+.1f}%)")
print(f"  MaxDD:  {r1['max_dd']:.1%} vs {b3['max_dd']:.1%} ({(r1['max_dd']-b3['max_dd'])*100:+.1f}pp)")
print(f"  AvgTrades: {r1['avg_trades']:.0f} vs {b3['avg_trades']:.0f}")

print("\nQuarterly (Q): rank+topk vs topk")
print(f"  Sharpe: {r2['sharpe']:.3f} vs {b4['sharpe']:.3f} ({(r2['sharpe']/b4['sharpe']-1)*100:+.1f}%)")
print(f"  MaxDD:  {r2['max_dd']:.1%} vs {b4['max_dd']:.1%} ({(r2['max_dd']-b4['max_dd'])*100:+.1f}pp)")
print(f"  AvgTrades: {r2['avg_trades']:.0f} vs {b4['avg_trades']:.0f}")
