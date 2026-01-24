import json, os, glob

root = "results/eval_9_0_0"
rows = []
for d in sorted(glob.glob(os.path.join(root, "*"))):
    p = os.path.join(d, "summary_metrics.json")
    if not os.path.exists(p):
        continue
    j = json.load(open(p, "r"))
    m = j["metrics"]
    params = j["params"]
    rows.append({
        "run": os.path.basename(d),
        "rebalance": params.get("rebalance"),
        "method": params.get("method"),
        "top_k": params.get("top_k"),
        "temperature": params.get("temperature"),
        "max_weight": params.get("max_weight"),
        "cagr": m.get("cagr"),
        "sharpe": m.get("sharpe"),
        "max_dd": m.get("max_drawdown"),
        "turnover": m.get("total_turnover"),
        "total_cost": m.get("total_cost"),
    })

# Print as a markdown-friendly table
cols = ["run","rebalance","method","top_k","temperature","max_weight","cagr","sharpe","max_dd","turnover","total_cost"]
print("| " + " | ".join(cols) + " |")
print("|" + "|".join(["---"]*len(cols)) + "|")
for r in rows:
    def fmt(x):
        if x is None: return ""
        if isinstance(x, float): return f"{x:.4f}"
        return str(x)
    print("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
