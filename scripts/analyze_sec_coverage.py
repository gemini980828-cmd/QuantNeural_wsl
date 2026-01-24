"""
SEC V2.2 Drop Reason Diagnostics Script.

Analyzes recovery patterns for total_liabilities and other canonical items.
"""

import json
from pathlib import Path
from collections import defaultdict
import re


# Period validation config
QUARTER_MIN, QUARTER_MAX = 80, 100
ANNUAL_MIN, ANNUAL_MAX = 350, 380
QUARTER_MIN_RELAXED, QUARTER_MAX_RELAXED = 70, 120
ANNUAL_MIN_RELAXED, ANNUAL_MAX_RELAXED = 320, 410

# Consolidated frame pattern
CONSOLIDATED_PATTERN = re.compile(r"^CY\d{4}(Q[1-4])?I?$")


def count_dimensions(frame: str) -> int:
    """Count dimensions in frame identifier."""
    if not frame:
        return 0
    if CONSOLIDATED_PATTERN.match(frame):
        return 0
    if "_" in frame:
        return frame.count("_")
    return 1


def analyze_total_liabilities_recovery(companyfacts_dir: str, max_files: int = 100):
    """Analyze which rules caused total_liabilities recovery."""
    sample_path = Path(companyfacts_dir)
    files = list(sample_path.glob("CIK*.json"))[:max_files]
    
    tags_to_try = ["Liabilities", "LiabilitiesNoncurrent"]
    
    recovery_cases = []
    drop_reasons = defaultdict(int)
    
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        
        us_gaap = data.get("facts", {}).get("us-gaap", {})
        entity_name = data.get("entityName", fpath.stem)
        
        found = False
        for tag in tags_to_try:
            if tag not in us_gaap:
                continue
            
            entries = us_gaap[tag].get("units", {}).get("USD", [])
            if not entries:
                continue
            
            # Analyze entries
            zero_dim = [e for e in entries if count_dimensions(e.get("frame", "")) == 0]
            has_dim = [e for e in entries if count_dimensions(e.get("frame", "")) > 0]
            
            if zero_dim:
                rule = "DIM_ZERO" if tag == "Liabilities" else "TAG_FALLBACK+DIM_ZERO"
                recovery_cases.append({
                    "cik": fpath.stem,
                    "entity": entity_name[:30],
                    "tag": tag,
                    "rule": rule,
                    "entry_count": len(zero_dim),
                    "sample_val": zero_dim[0].get("val", 0),
                })
                found = True
                break  # Found consolidated data
            elif has_dim:
                # Would need dimension fallback
                drop_reasons[f"{tag}_DIM_REJECT"] += 1
        
        if not found:
            # Check why no valid entries found
            if "Liabilities" not in us_gaap and "LiabilitiesNoncurrent" not in us_gaap:
                drop_reasons["NO_LIAB_TAG"] += 1
            else:
                drop_reasons["OTHER"] += 1
    
    # Print results
    print("=" * 60)
    print("TOTAL_LIABILITIES RECOVERY ANALYSIS")
    print("=" * 60)
    print()
    
    # Summarize by rule
    rule_counts = defaultdict(int)
    for case in recovery_cases:
        rule_counts[case["rule"]] += 1
    
    print("=== Recovery Rule Summary ===")
    for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  {rule}: {count}")
    
    print()
    print("=== Recovery Cases (first 50) ===")
    for i, case in enumerate(recovery_cases[:50]):
        val_str = f"{case['sample_val']:,.0f}" if case['sample_val'] else "N/A"
        print(f"{i+1:2}. {case['cik']}: {case['tag']} via {case['rule']} (n={case['entry_count']}, val={val_str})")
    
    print()
    print("=== Drop Reasons (not recovered) ===")
    for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    print()
    print(f"Total recovered: {len(recovery_cases)}/{max_files} ({len(recovery_cases)/max_files*100:.1f}%)")
    
    return recovery_cases, drop_reasons


def analyze_drop_reasons_by_canonical(companyfacts_dir: str, max_files: int = 100):
    """Analyze drop reasons for all canonical items."""
    from src.sec_fundamentals_v2 import load_tag_mapping, PeriodValidationConfig
    
    sample_path = Path(companyfacts_dir)
    files = list(sample_path.glob("CIK*.json"))[:max_files]
    config = load_tag_mapping()
    period_cfg = PeriodValidationConfig()
    
    # Track drop reasons per canonical item
    canonical_drops = defaultdict(lambda: defaultdict(int))
    canonical_success = defaultdict(int)
    
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        
        for item_name, item_config in config.canonical_items.items():
            taxonomy = item_config.taxonomy
            found = False
            
            for tag in item_config.tags:
                facts = data.get("facts", {}).get(taxonomy, {}).get(tag, {})
                if not facts:
                    continue
                
                entries = facts.get("units", {}).get(item_config.unit, [])
                if not entries:
                    # Try any unit
                    units_data = facts.get("units", {})
                    if units_data:
                        entries = list(units_data.values())[0]
                
                if not entries:
                    canonical_drops[item_name]["NO_ENTRIES"] += 1
                    continue
                
                # Check dimension filter
                zero_dim = [e for e in entries if count_dimensions(e.get("frame", "")) == 0]
                if not zero_dim:
                    canonical_drops[item_name]["DIM_REJECT"] += 1
                    continue
                
                # Check period filter for duration items
                if item_config.is_duration:
                    valid_period = []
                    for e in zero_dim:
                        start = e.get("start")
                        end = e.get("end")
                        if start and end:
                            try:
                                from datetime import datetime
                                days = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
                                # Strict or relaxed
                                if (QUARTER_MIN_RELAXED <= days <= QUARTER_MAX_RELAXED or
                                    ANNUAL_MIN_RELAXED <= days <= ANNUAL_MAX_RELAXED):
                                    valid_period.append(e)
                            except Exception:
                                pass
                    
                    if not valid_period:
                        canonical_drops[item_name]["PERIOD_REJECT"] += 1
                        continue
                
                # Success
                canonical_success[item_name] += 1
                found = True
                break
            
            if not found and item_name not in canonical_success:
                if all(
                    data.get("facts", {}).get(item_config.taxonomy, {}).get(tag) is None
                    for tag in item_config.tags
                ):
                    canonical_drops[item_name]["NO_TAG"] += 1
    
    print()
    print("=" * 60)
    print("DROP REASONS BY CANONICAL ITEM")
    print("=" * 60)
    
    for item_name in sorted(config.canonical_items.keys()):
        success = canonical_success.get(item_name, 0)
        drops = canonical_drops.get(item_name, {})
        total_drops = sum(drops.values())
        
        if total_drops > 0:
            top_drop = max(drops.items(), key=lambda x: x[1]) if drops else ("N/A", 0)
            print(f"\n{item_name}: {success}/{max_files} recovered")
            print(f"  Top drop: {top_drop[0]} ({top_drop[1]})")
            for reason, count in sorted(drops.items(), key=lambda x: -x[1])[:3]:
                print(f"    - {reason}: {count}")


if __name__ == "__main__":
    import sys
    
    companyfacts_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sec_bulk/"
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Run analysis
    analyze_total_liabilities_recovery(companyfacts_dir, max_files)
    analyze_drop_reasons_by_canonical(companyfacts_dir, max_files)
