r"""
SEC Fundamentals V2.3: Canonical Tag Mapping Layer.

Layer 1 implementation for mapping raw XBRL tags to canonical financial line items.
Uses configs/sec_tag_mapping.yaml for configuration-driven tag fallback logic.

V2.3 Enhancements (over V2.2):
- Secondary computed fallback: Assets - StockholdersEquity for total_liabilities
- Balance sheet identity check: abs(assets - (liab + equity)) <= tolerance
- Enhanced diagnostics: stockholders_equity coverage, C\(A∪B) analysis

V2.2 Enhancements:
- Computed fallback: LiabilitiesAndStockholdersEquity - StockholdersEquity for total_liabilities
- Quality gates: sanity checks (0 <= total_liab <= total_assets * 1.01)
- CF period relaxation: wider period ranges for CFO/CAPEX items
- Enhanced diagnostics: gate reject counts

Architecture:
    Layer 0: Raw companyfacts JSON parsing (via extract_companyfacts_tag_entries)
    Layer 1: Canonical tag mapping with fallback, validation, quality gates (this module)
    Layer 2: Derived financial ratios (downstream in build_alpha_dataset)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.sec_fundamentals import (
    extract_companyfacts_tag_entries,
    pit_latest_snapshot,
)


logger = logging.getLogger(__name__)

DEFAULT_TAG_MAPPING_PATH = Path(__file__).parent.parent / "configs" / "sec_tag_mapping.yaml"


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class PeriodValidationConfig:
    """Configuration for period length validation (V2.2: item-specific relaxation)."""
    # Strict range (score +10)
    quarter_min: int = 80
    quarter_max: int = 100
    annual_min: int = 350
    annual_max: int = 380
    # Relaxed range (score +3) - V2.2: wider for CF items
    quarter_min_relaxed: int = 70
    quarter_max_relaxed: int = 120
    annual_min_relaxed: int = 320
    annual_max_relaxed: int = 410
    # CF-specific even wider relaxation
    cf_quarter_min_relaxed: int = 60
    cf_quarter_max_relaxed: int = 130
    cf_annual_min_relaxed: int = 300
    cf_annual_max_relaxed: int = 420
    # Hard reject thresholds
    reject_below: int = 45
    reject_between_min: int = 150
    reject_between_max: int = 200


@dataclass
class QualityGateConfig:
    """Configuration for quality gates (V2.2 + V2.3)."""
    require_positive_assets: bool = True
    require_positive_shares: bool = True
    liabilities_max_ratio_to_assets: float = 1.01  # Allow 1% tolerance
    # V2.3: Balance sheet identity check tolerance
    balance_identity_tolerance: float = 0.05  # Allow 5% deviation from A = L + E


@dataclass
class CanonicalItemConfig:
    """Configuration for a single canonical financial line item."""
    name: str
    taxonomy: str
    unit: str
    tags: list[str]
    fallback_taxonomy: str | None = None
    fallback_tags: list[str] = field(default_factory=list)
    is_duration: bool = False
    safety_tier: int = 2
    label_whitelist: list[str] = field(default_factory=list)
    allow_dimension_fallback: bool = False
    # V2.2: CF-specific period relaxation
    use_cf_period_relaxation: bool = False
    # V2.2: computed fallback support
    allow_computed_fallback: bool = False
    
    def all_tag_sources(self) -> list[tuple[str, str]]:
        sources = [(self.taxonomy, tag) for tag in self.tags]
        if self.fallback_taxonomy and self.fallback_tags:
            sources.extend([(self.fallback_taxonomy, tag) for tag in self.fallback_tags])
        return sources


@dataclass
class TagMappingConfig:
    """Full tag mapping configuration loaded from YAML."""
    version: str
    canonical_items: dict[str, CanonicalItemConfig]
    period_validation: PeriodValidationConfig = field(default_factory=PeriodValidationConfig)
    quality_gates: QualityGateConfig = field(default_factory=QualityGateConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "TagMappingConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tag mapping config not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid YAML structure: expected dict, got {type(raw)}")
        
        version = raw.get("version", "unknown")
        raw_items = raw.get("canonical_items", {})
        
        if not isinstance(raw_items, dict):
            raise ValueError("canonical_items must be a dict")
        
        # Parse period validation
        pv_raw = raw.get("period_validation", {})
        period_validation = PeriodValidationConfig(
            quarter_min=pv_raw.get("quarter_min", 80),
            quarter_max=pv_raw.get("quarter_max", 100),
            annual_min=pv_raw.get("annual_min", 350),
            annual_max=pv_raw.get("annual_max", 380),
        )
        
        # V2.2: CF items and computed fallback items
        cf_items = {"operating_cash_flow", "investing_cash_flow", "financing_cash_flow", "capital_expenditures"}
        computed_items = {"total_liabilities"}
        
        items = {}
        for name, item_config in raw_items.items():
            if not isinstance(item_config, dict):
                logger.warning("Skipping invalid item config for '%s'", name)
                continue
            
            items[name] = CanonicalItemConfig(
                name=name,
                taxonomy=item_config.get("taxonomy", "us-gaap"),
                unit=item_config.get("unit", "USD"),
                tags=item_config.get("tags", []),
                fallback_taxonomy=item_config.get("fallback_taxonomy"),
                fallback_tags=item_config.get("fallback_tags", []),
                is_duration=item_config.get("is_duration", False),
                safety_tier=item_config.get("safety_tier", 2),
                label_whitelist=item_config.get("label_whitelist", []),
                allow_dimension_fallback=name in ("total_liabilities",),
                use_cf_period_relaxation=name in cf_items,
                allow_computed_fallback=name in computed_items,
            )
        
        return cls(version=version, canonical_items=items, period_validation=period_validation)


# ==============================================================================
# Validation Functions
# ==============================================================================

CONSOLIDATED_PATTERN = re.compile(r"^CY\d{4}(Q[1-4])?I?$")


def _count_dimensions(frame: str) -> int:
    if not frame:
        return 0
    if CONSOLIDATED_PATTERN.match(frame):
        return 0
    if "_" in frame:
        return frame.count("_")
    return 1


def _score_period_length(
    start: str | None,
    end: str | None,
    config: PeriodValidationConfig,
    use_cf_relaxation: bool = False,
) -> tuple[int, str]:
    """Score period length with V2.2 CF-specific relaxation."""
    if not start or not end:
        return -1, "no_dates"
    
    try:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        days = (end_dt - start_dt).days
        
        # Hard reject
        if days < config.reject_below:
            return -1, "reject_too_short"
        if config.reject_between_min <= days <= config.reject_between_max:
            return -1, "reject_mid_range"
        
        # Strict ranges
        if config.quarter_min <= days <= config.quarter_max:
            return 10, "quarterly_strict"
        if config.annual_min <= days <= config.annual_max:
            return 10, "annual_strict"
        
        # V2.2: CF-specific wider relaxation
        if use_cf_relaxation:
            if config.cf_quarter_min_relaxed <= days <= config.cf_quarter_max_relaxed:
                return 3, "quarterly_cf_relaxed"
            if config.cf_annual_min_relaxed <= days <= config.cf_annual_max_relaxed:
                return 3, "annual_cf_relaxed"
        
        # Standard relaxed
        if config.quarter_min_relaxed <= days <= config.quarter_max_relaxed:
            return 3, "quarterly_relaxed"
        if config.annual_min_relaxed <= days <= config.annual_max_relaxed:
            return 3, "annual_relaxed"
        
        return -1, "reject_out_of_range"
        
    except Exception:
        return -1, "parse_error"


# ==============================================================================
# Quality Gates (V2.2 + V2.3 Balance Identity)
# ==============================================================================


def apply_quality_gates(
    canonical_values: dict[str, float],
    config: QualityGateConfig,
) -> dict[str, bool]:
    """
    Apply quality gates to canonical values.
    
    V2.3: Added balance sheet identity check.
    
    Returns dict of {item_name: is_low_confidence} flags.
    """
    flags = {}
    
    total_assets = canonical_values.get("total_assets")
    total_liab = canonical_values.get("total_liabilities")
    stockholders_eq = canonical_values.get("stockholders_equity")
    shares = canonical_values.get("shares_outstanding")
    
    # Gate 1: total_assets > 0
    if config.require_positive_assets and total_assets is not None:
        if not (total_assets > 0):
            flags["total_assets"] = True
    
    # Gate 2: shares_outstanding > 0
    if config.require_positive_shares and shares is not None:
        if not (shares > 0):
            flags["shares_outstanding"] = True
    
    # Gate 3: 0 <= total_liabilities <= total_assets * (1 + eps)
    if total_liab is not None and total_assets is not None and total_assets > 0:
        if total_liab < 0 or total_liab > total_assets * config.liabilities_max_ratio_to_assets:
            flags["total_liabilities"] = True
    
    # Gate 4 (V2.3): Balance sheet identity check
    # abs(assets - (liab + equity)) <= tolerance * assets
    if (total_assets is not None and total_liab is not None and 
        stockholders_eq is not None and total_assets > 0):
        balance_diff = abs(total_assets - (total_liab + stockholders_eq))
        if balance_diff > config.balance_identity_tolerance * total_assets:
            flags["balance_identity_fail"] = True
    
    return flags


# ==============================================================================
# Computed Fallback (V2.2)
# ==============================================================================


def _compute_liabilities_from_lse_minus_se(
    companyfacts_path: str,
    period_config: PeriodValidationConfig,
) -> tuple[pd.DataFrame, str | None, bool]:
    """
    Compute total_liabilities as LiabilitiesAndStockholdersEquity - StockholdersEquity.
    
    Returns
    -------
    tuple[pd.DataFrame, str | None, bool]
        - DataFrame with columns [end, filed, val]
        - Provenance tag if successful ("COMPUTED_LSE_MINUS_SE")
        - low_confidence flag (True for computed values)
    """
    try:
        with open(companyfacts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        us_gaap = data.get("facts", {}).get("us-gaap", {})
        
        # Get LiabilitiesAndStockholdersEquity
        lse_facts = us_gaap.get("LiabilitiesAndStockholdersEquity", {})
        lse_entries = lse_facts.get("units", {}).get("USD", [])
        
        # Get StockholdersEquity
        se_tags = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
        se_entries = []
        for se_tag in se_tags:
            se_facts = us_gaap.get(se_tag, {})
            se_entries = se_facts.get("units", {}).get("USD", [])
            if se_entries:
                break
        
        if not lse_entries or not se_entries:
            return pd.DataFrame(columns=["end", "filed", "val"]), None, False
        
        # Filter to consolidated (0 dimensions)
        lse_consolidated = [e for e in lse_entries if _count_dimensions(e.get("frame", "")) == 0]
        se_consolidated = [e for e in se_entries if _count_dimensions(e.get("frame", "")) == 0]
        
        if not lse_consolidated or not se_consolidated:
            return pd.DataFrame(columns=["end", "filed", "val"]), None, False
        
        # Build lookup by (end, filed)
        se_lookup = {}
        for e in se_consolidated:
            key = (e.get("end"), e.get("filed"))
            se_lookup[key] = e.get("val", 0)
        
        # Compute Liabilities = LSE - SE for matching contexts
        computed_entries = []
        for lse in lse_consolidated:
            key = (lse.get("end"), lse.get("filed"))
            if key in se_lookup:
                lse_val = lse.get("val", 0)
                se_val = se_lookup[key]
                if lse_val is not None and se_val is not None:
                    liab_val = lse_val - se_val
                    
                    # Sanity check: liabilities should be >= 0
                    if liab_val >= 0:
                        computed_entries.append({
                            "end": pd.to_datetime(lse.get("end")),
                            "filed": pd.to_datetime(lse.get("filed")),
                            "val": liab_val,
                        })
        
        if not computed_entries:
            return pd.DataFrame(columns=["end", "filed", "val"]), None, False
        
        df = pd.DataFrame(computed_entries)
        return df, "COMPUTED_LSE_MINUS_SE", True
        
    except Exception as e:
        logger.debug("Computed fallback (LSE-SE) failed: %s", e)
        return pd.DataFrame(columns=["end", "filed", "val"]), None, False


def _compute_liabilities_from_assets_minus_se(
    companyfacts_path: str,
    period_config: PeriodValidationConfig,
    assets_max_ratio: float = 1.01,
) -> tuple[pd.DataFrame, str | None, bool]:
    """
    V2.3: Compute total_liabilities as Assets - StockholdersEquity.
    
    Secondary fallback when LSE-SE fails. Uses the accounting identity:
    Assets = Liabilities + Equity => Liabilities = Assets - Equity
    
    Conditions:
    - Both Assets and StockholdersEquity must have DIM_ZERO entries
    - Same (end, filed) context required
    - Result must pass sanity gates: liab >= 0 and liab <= assets * 1.01
    
    Returns
    -------
    tuple[pd.DataFrame, str | None, bool]
        - DataFrame with columns [end, filed, val]
        - Provenance tag if successful ("COMPUTED_ASSETS_MINUS_SE")
        - low_confidence flag (True for computed values)
    """
    try:
        with open(companyfacts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        us_gaap = data.get("facts", {}).get("us-gaap", {})
        
        # Get Assets (primary tag only for safety)
        assets_facts = us_gaap.get("Assets", {})
        assets_entries = assets_facts.get("units", {}).get("USD", [])
        
        # Get StockholdersEquity (with fallback)
        se_tags = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
        se_entries = []
        for se_tag in se_tags:
            se_facts = us_gaap.get(se_tag, {})
            se_entries = se_facts.get("units", {}).get("USD", [])
            if se_entries:
                break
        
        if not assets_entries or not se_entries:
            return pd.DataFrame(columns=["end", "filed", "val"]), None, False
        
        # Filter to consolidated (0 dimensions)
        assets_consolidated = [e for e in assets_entries if _count_dimensions(e.get("frame", "")) == 0]
        se_consolidated = [e for e in se_entries if _count_dimensions(e.get("frame", "")) == 0]
        
        if not assets_consolidated or not se_consolidated:
            return pd.DataFrame(columns=["end", "filed", "val"]), None, False
        
        # Build lookup by (end, filed)
        se_lookup = {}
        for e in se_consolidated:
            key = (e.get("end"), e.get("filed"))
            se_lookup[key] = e.get("val", 0)
        
        # Compute Liabilities = Assets - SE for matching contexts
        computed_entries = []
        for assets_entry in assets_consolidated:
            key = (assets_entry.get("end"), assets_entry.get("filed"))
            if key in se_lookup:
                assets_val = assets_entry.get("val", 0)
                se_val = se_lookup[key]
                if assets_val is not None and se_val is not None and assets_val > 0:
                    liab_val = assets_val - se_val
                    
                    # Quality gates: liab >= 0 and liab <= assets * 1.01
                    if liab_val >= 0 and liab_val <= assets_val * assets_max_ratio:
                        computed_entries.append({
                            "end": pd.to_datetime(assets_entry.get("end")),
                            "filed": pd.to_datetime(assets_entry.get("filed")),
                            "val": liab_val,
                        })
        
        if not computed_entries:
            return pd.DataFrame(columns=["end", "filed", "val"]), None, False
        
        df = pd.DataFrame(computed_entries)
        return df, "COMPUTED_ASSETS_MINUS_SE", True
        
    except Exception as e:
        logger.debug("Computed fallback (Assets-SE) failed: %s", e)
        return pd.DataFrame(columns=["end", "filed", "val"]), None, False


# ==============================================================================
# Core Functions
# ==============================================================================


def load_tag_mapping(path: str | Path | None = None) -> TagMappingConfig:
    if path is None:
        path = DEFAULT_TAG_MAPPING_PATH
    return TagMappingConfig.from_yaml(path)


@dataclass
class ExtractionResult:
    """Result from canonical entry extraction with metadata."""
    df: pd.DataFrame
    used_tag: str | None
    low_confidence: bool = False
    computed: bool = False
    gate_rejected: bool = False


def extract_canonical_entry_v22(
    companyfacts_path: str,
    item_config: CanonicalItemConfig,
    period_config: PeriodValidationConfig,
) -> ExtractionResult:
    """
    Extract data for a canonical item with V2.2 enhancements.
    
    V2.2 features:
    - CF-specific period relaxation
    - Computed fallback for total_liabilities
    - Quality gate integration
    """
    unit_pref = item_config.unit
    unit_mapping = {"USD": "USD", "shares": "shares", "USD/shares": "USD/shares"}
    unit_to_try = unit_mapping.get(unit_pref, unit_pref)
    
    empty_result = ExtractionResult(
        df=pd.DataFrame(columns=["end", "filed", "val"]),
        used_tag=None,
    )
    
    # Step 1: Try regular tag extraction
    for taxonomy, tag in item_config.all_tag_sources():
        try:
            with open(companyfacts_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            facts = raw_data.get("facts", {}).get(taxonomy, {}).get(tag, {})
            units_data = facts.get("units", {})
            
            raw_entries = []
            for unit_key, entries in units_data.items():
                if unit_key == unit_to_try or (unit_pref == "USD/shares" and "/" in unit_key):
                    raw_entries = entries
                    break
            if not raw_entries and units_data:
                raw_entries = list(units_data.values())[0]
            
            if not raw_entries:
                continue
            
            # Score and filter entries
            scored_entries = []
            for entry in raw_entries:
                dim_count = _count_dimensions(entry.get("frame", ""))
                
                period_score = 10
                if item_config.is_duration:
                    period_score, _ = _score_period_length(
                        entry.get("start"), entry.get("end"), period_config,
                        use_cf_relaxation=item_config.use_cf_period_relaxation,
                    )
                    if period_score < 0:
                        continue
                
                form = entry.get("form", "")
                form_score = 5 if form in ("10-K", "10-Q") else 0
                total_score = (period_score + form_score) * 100 - dim_count * 1000
                
                scored_entries.append((total_score, dim_count, entry))
            
            if not scored_entries:
                continue
            
            scored_entries.sort(key=lambda x: x[0], reverse=True)
            zero_dim_entries = [(s, d, e) for s, d, e in scored_entries if d == 0]
            
            if zero_dim_entries:
                selected_entries = [e for _, _, e in zero_dim_entries]
                low_confidence = False
            elif item_config.allow_dimension_fallback:
                by_period: dict[tuple, list] = {}
                for _, dim, entry in scored_entries:
                    key = (entry.get("end"), entry.get("filed"))
                    if key not in by_period:
                        by_period[key] = []
                    by_period[key].append(entry)
                
                selected_entries = []
                for key, entries in by_period.items():
                    max_entry = max(entries, key=lambda e: abs(e.get("val", 0) or 0))
                    selected_entries.append(max_entry)
                low_confidence = True
            else:
                continue
            
            if not selected_entries:
                continue
            
            filtered_df = pd.DataFrame([
                {
                    "end": pd.to_datetime(e.get("end")),
                    "filed": pd.to_datetime(e.get("filed")),
                    "val": e.get("val"),
                }
                for e in selected_entries
                if e.get("end") and e.get("filed") and e.get("val") is not None
            ])
            
            if not filtered_df.empty:
                return ExtractionResult(
                    df=filtered_df,
                    used_tag=tag,
                    low_confidence=low_confidence,
                )
            
        except Exception as e:
            logger.debug("V2.2 extraction failed for %s/%s: %s", taxonomy, tag, e)
            continue
    
    # Step 2: V2.2 Computed fallback for total_liabilities (LSE - SE)
    if item_config.allow_computed_fallback:
        df, tag, low_conf = _compute_liabilities_from_lse_minus_se(
            companyfacts_path, period_config
        )
        if not df.empty:
            return ExtractionResult(
                df=df,
                used_tag=tag,
                low_confidence=True,  # Computed values are always low confidence
                computed=True,
            )
        
        # Step 3: V2.3 Secondary computed fallback (Assets - SE)
        df, tag, low_conf = _compute_liabilities_from_assets_minus_se(
            companyfacts_path, period_config
        )
        if not df.empty:
            return ExtractionResult(
                df=df,
                used_tag=tag,
                low_confidence=True,  # Computed values are always low confidence
                computed=True,
            )
    
    return empty_result


def extract_canonical_entry(
    companyfacts_path: str,
    item_config: CanonicalItemConfig,
) -> tuple[pd.DataFrame, str | None]:
    """V1 compatibility wrapper."""
    result = extract_canonical_entry_v22(
        companyfacts_path,
        item_config,
        PeriodValidationConfig(),
    )
    return result.df, result.used_tag


def build_canonical_wide_table(
    companyfacts_path: str,
    as_of_dates: pd.DatetimeIndex,
    *,
    tag_mapping_path: str | Path | None = None,
    include_provenance: bool = True,
    include_confidence_flags: bool = True,
) -> pd.DataFrame:
    """Build a canonical wide table with V2.2 enhancements."""
    n_dates = len(as_of_dates)
    result_data: dict[str, Any] = {}
    provenance_data: dict[str, str | None] = {}
    confidence_data: dict[str, bool] = {}
    
    try:
        config = load_tag_mapping(tag_mapping_path)
        
        if not Path(companyfacts_path).exists():
            logger.warning("Companyfacts file not found: %s", companyfacts_path)
            for item_name in config.canonical_items.keys():
                result_data[item_name] = np.full(n_dates, np.nan, dtype=np.float32)
                if include_provenance:
                    provenance_data[f"{item_name}_tag"] = None
                if include_confidence_flags:
                    confidence_data[f"{item_name}_low_confidence"] = False
            
            df = pd.DataFrame(result_data, index=as_of_dates)
            return df
        
        # First pass: extract all values
        extraction_results: dict[str, ExtractionResult] = {}
        for item_name, item_config in config.canonical_items.items():
            result = extract_canonical_entry_v22(
                companyfacts_path, item_config, config.period_validation
            )
            extraction_results[item_name] = result
        
        # Get sample values for quality gate checking
        sample_values = {}
        for item_name, result in extraction_results.items():
            if not result.df.empty:
                sample_values[item_name] = result.df["val"].iloc[-1]
        
        # Apply quality gates
        quality_flags = apply_quality_gates(sample_values, config.quality_gates)
        
        # Build final output
        for item_name, result in extraction_results.items():
            if result.df.empty:
                values = np.full(n_dates, np.nan, dtype=np.float32)
            else:
                values = pit_latest_snapshot(result.df, as_of_dates)
                values = values.astype(np.float32)
            
            result_data[item_name] = values
            provenance_data[f"{item_name}_tag"] = result.used_tag
            
            # Combine extraction low_confidence with quality gate flags
            is_low_conf = result.low_confidence or quality_flags.get(item_name, False)
            confidence_data[f"{item_name}_low_confidence"] = is_low_conf
        
        df = pd.DataFrame(result_data, index=as_of_dates)
        
        if include_provenance:
            for tag_col, used_tag in provenance_data.items():
                df[tag_col] = used_tag
        
        if include_confidence_flags:
            for conf_col, is_low in confidence_data.items():
                df[conf_col] = is_low
        
        return df
        
    except Exception as e:
        logger.error("Error building canonical wide table: %s", e)
        
        try:
            config = load_tag_mapping(tag_mapping_path)
            for item_name in config.canonical_items.keys():
                result_data[item_name] = np.full(n_dates, np.nan, dtype=np.float32)
        except Exception:
            pass
        
        df = pd.DataFrame(result_data, index=as_of_dates)
        return df


# ==============================================================================
# Coverage Diagnostics (V2.3 Enhanced)
# ==============================================================================


@dataclass
class CoverageDiagnostics:
    """Results from coverage analysis with V2.3 gate tracking."""
    total_tickers: int
    items_coverage: dict[str, float]
    tag_usage_distribution: dict[str, dict[str, int]]
    overall_coverage: float
    low_confidence_counts: dict[str, int] = field(default_factory=dict)
    computed_fallback_counts: dict[str, int] = field(default_factory=dict)
    gate_reject_counts: dict[str, int] = field(default_factory=dict)
    # V2.3: Breakdown of computed fallback sources
    primary_lse_se_counts: dict[str, int] = field(default_factory=dict)
    secondary_assets_se_counts: dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tickers": self.total_tickers,
            "items_coverage": self.items_coverage,
            "tag_usage_distribution": self.tag_usage_distribution,
            "overall_coverage": self.overall_coverage,
            "low_confidence_counts": self.low_confidence_counts,
            "computed_fallback_counts": self.computed_fallback_counts,
            "gate_reject_counts": self.gate_reject_counts,
            "primary_lse_se_counts": self.primary_lse_se_counts,
            "secondary_assets_se_counts": self.secondary_assets_se_counts,
        }
    
    def summary(self) -> str:
        lines = [
            f"Coverage Diagnostics V2.3 (n={self.total_tickers} tickers)",
            f"Overall Coverage: {self.overall_coverage:.1%}",
            "",
            "Per-Item Coverage:",
        ]
        
        for item, coverage in sorted(self.items_coverage.items(), key=lambda x: -x[1]):
            tag_dist = self.tag_usage_distribution.get(item, {})
            top_tag = max(tag_dist.items(), key=lambda x: x[1])[0] if tag_dist else "N/A"
            
            computed = self.computed_fallback_counts.get(item, 0)
            low_conf = self.low_confidence_counts.get(item, 0)
            lse_se = self.primary_lse_se_counts.get(item, 0)
            assets_se = self.secondary_assets_se_counts.get(item, 0)
            
            suffix_parts = []
            if computed > 0:
                computed_detail = f"🔧{computed} computed"
                if lse_se > 0 or assets_se > 0:
                    computed_detail += f" [LSE-SE:{lse_se}, Assets-SE:{assets_se}]"
                suffix_parts.append(computed_detail)
            if low_conf > 0:
                suffix_parts.append(f"⚠{low_conf} low-conf")
            
            suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
            lines.append(f"  {item}: {coverage:.1%} (most used: {top_tag}){suffix}")
        
        return "\n".join(lines)


def compute_coverage_diagnostics(
    companyfacts_dir: str,
    manifest_path: str | None = None,
    *,
    tag_mapping_path: str | Path | None = None,
    max_tickers: int | None = None,
) -> CoverageDiagnostics:
    """Compute coverage diagnostics with V2.2 tracking."""
    config = load_tag_mapping(tag_mapping_path)
    
    if manifest_path:
        manifest = pd.read_csv(manifest_path)
        if "companyfacts_path" not in manifest.columns:
            raise ValueError("Manifest must have 'companyfacts_path' column")
        paths = manifest["companyfacts_path"].dropna().tolist()
    else:
        companyfacts_dir_path = Path(companyfacts_dir)
        paths = [str(p) for p in companyfacts_dir_path.glob("CIK*.json")]
    
    if max_tickers:
        paths = paths[:max_tickers]
    
    total_tickers = len(paths)
    
    if total_tickers == 0:
        return CoverageDiagnostics(
            total_tickers=0,
            items_coverage={},
            tag_usage_distribution={},
            overall_coverage=0.0,
        )
    
    items_found: dict[str, int] = {name: 0 for name in config.canonical_items.keys()}
    tag_usage: dict[str, dict[str, int]] = {name: {} for name in config.canonical_items.keys()}
    low_conf_counts: dict[str, int] = {name: 0 for name in config.canonical_items.keys()}
    computed_counts: dict[str, int] = {name: 0 for name in config.canonical_items.keys()}
    # V2.3: Track computed fallback sources
    lse_se_counts: dict[str, int] = {name: 0 for name in config.canonical_items.keys()}
    assets_se_counts: dict[str, int] = {name: 0 for name in config.canonical_items.keys()}
    
    for path in paths:
        if not Path(path).exists():
            continue
        
        for item_name, item_config in config.canonical_items.items():
            result = extract_canonical_entry_v22(path, item_config, config.period_validation)
            
            if result.used_tag is not None:
                items_found[item_name] += 1
                
                if result.used_tag not in tag_usage[item_name]:
                    tag_usage[item_name][result.used_tag] = 0
                tag_usage[item_name][result.used_tag] += 1
                
                if result.low_confidence:
                    low_conf_counts[item_name] += 1
                
                if result.computed:
                    computed_counts[item_name] += 1
                    # V2.3: Track which computed method was used
                    if result.used_tag == "COMPUTED_LSE_MINUS_SE":
                        lse_se_counts[item_name] += 1
                    elif result.used_tag == "COMPUTED_ASSETS_MINUS_SE":
                        assets_se_counts[item_name] += 1
    
    items_coverage = {
        name: count / total_tickers if total_tickers > 0 else 0.0
        for name, count in items_found.items()
    }
    
    overall_coverage = sum(items_coverage.values()) / len(items_coverage) if items_coverage else 0.0
    
    return CoverageDiagnostics(
        total_tickers=total_tickers,
        items_coverage=items_coverage,
        tag_usage_distribution=tag_usage,
        overall_coverage=overall_coverage,
        low_confidence_counts=low_conf_counts,
        computed_fallback_counts=computed_counts,
        primary_lse_se_counts=lse_se_counts,
        secondary_assets_se_counts=assets_se_counts,
    )


def run_coverage_diagnostics(
    companyfacts_dir: str,
    manifest_path: str | None = None,
    output_path: str | None = None,
) -> None:
    """Run coverage diagnostics and print/save results."""
    print(f"Running V2.3 coverage diagnostics on {companyfacts_dir}...")
    
    diagnostics = compute_coverage_diagnostics(companyfacts_dir, manifest_path)
    print(diagnostics.summary())
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")


def analyze_secondary_fallback_potential(
    companyfacts_dir: str,
    max_tickers: int = 200,
) -> dict[str, Any]:
    r"""
    V2.3: Analyze the potential impact of secondary computed fallback.
    
    Measures C \ (A ∪ B) - the incremental coverage from Assets-SE fallback:
    - A: Primary tags (Liabilities) success
    - B: COMPUTED_LSE_MINUS_SE success
    - C: COMPUTED_ASSETS_MINUS_SE success
    
    Returns
    -------
    dict with keys:
        - primary_only: tickers covered by primary tags
        - lse_se_only: tickers covered by LSE-SE (excluding primary)
        - assets_se_only: tickers covered ONLY by Assets-SE (= C \ (A ∪ B))
        - total_covered: total unique tickers with liabilities data
        - total_tickers: total tickers analyzed
    """
    companyfacts_dir_path = Path(companyfacts_dir)
    paths = [str(p) for p in companyfacts_dir_path.glob("CIK*.json")][:max_tickers]
    
    primary_set = set()  # Covered by primary tags
    lse_se_set = set()   # Covered by LSE-SE
    assets_se_set = set() # Covered by Assets-SE
    
    config = load_tag_mapping()
    liab_config = config.canonical_items.get("total_liabilities")
    
    if not liab_config:
        return {"error": "total_liabilities not found in config"}
    
    period_config = config.period_validation
    
    for i, path in enumerate(paths):
        if not Path(path).exists():
            continue
        
        cik = Path(path).stem
        
        # Check primary tag extraction (without computed fallback)
        primary_found = False
        for taxonomy, tag in liab_config.all_tag_sources():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                facts = raw_data.get("facts", {}).get(taxonomy, {}).get(tag, {})
                units_data = facts.get("units", {}).get("USD", [])
                if units_data:
                    zero_dim = [e for e in units_data if _count_dimensions(e.get("frame", "")) == 0]
                    if zero_dim:
                        primary_found = True
                        break
            except Exception:
                continue
        
        if primary_found:
            primary_set.add(cik)
            continue  # Primary found, no need for fallback
        
        # Check LSE-SE fallback
        df_lse, tag_lse, _ = _compute_liabilities_from_lse_minus_se(path, period_config)
        if not df_lse.empty:
            lse_se_set.add(cik)
            continue
        
        # Check Assets-SE fallback
        df_assets, tag_assets, _ = _compute_liabilities_from_assets_minus_se(path, period_config)
        if not df_assets.empty:
            assets_se_set.add(cik)
    
    return {
        "primary_only": len(primary_set),
        "lse_se_only": len(lse_se_set),
        "assets_se_only": len(assets_se_set),  # This is C \ (A ∪ B)
        "total_covered": len(primary_set) + len(lse_se_set) + len(assets_se_set),
        "total_tickers": len(paths),
        "assets_se_ciks": list(assets_se_set)[:10],  # Sample for debugging
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 2:
        run_coverage_diagnostics(
            companyfacts_dir=sys.argv[1],
            manifest_path=sys.argv[2] if len(sys.argv) >= 3 else None,
            output_path=sys.argv[3] if len(sys.argv) >= 4 else None,
        )
    else:
        print("Usage: python sec_fundamentals_v2.py <companyfacts_dir> [manifest_csv] [output_json]")
