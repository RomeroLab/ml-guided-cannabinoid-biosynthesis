#!/usr/bin/env python
"""Audit summary measurements or analyze explicitly paired metabolite replicates.

This module never treats numbered assay conditions as replicate measurements.
Concentrations must already share a comparable calibrated concentration basis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def coefficient_of_variation(mean: float, sd: float, min_mean: float = 0.0) -> float:
    """CV is undefined at zero or below the user-selected near-zero threshold."""
    if not np.isfinite(mean) or not np.isfinite(sd) or sd < 0 or abs(mean) <= min_mean:
        return np.nan
    return float(sd / abs(mean))


def summary_audit(data: pd.DataFrame, min_mean: float = 0.0, ratio_min_mean: float = 0.0) -> tuple[dict, pd.DataFrame]:
    """Describe reported SD/mean only; do not reconstruct raw replicates."""
    rows = []
    for column in data.columns:
        match = re.fullmatch(r"(\d+)_(DVA|OA)", column)
        if not match:
            continue
        condition, product = match.groups()
        other = "DVO" if product == "DVA" else "OL"
        ratio_col = f"{condition}_{product}/({product}+{other})"
        if any(c not in data for c in [column + "_std", ratio_col, ratio_col + "_std"]):
            continue
        for index, record in data.iterrows():
            vals = pd.to_numeric(record[[column, column + "_std", ratio_col, ratio_col + "_std"]], errors="coerce")
            if not np.all(np.isfinite(vals)):
                continue
            if vals.iloc[1] < 0 or vals.iloc[3] < 0:
                raise ValueError("Reported SD cannot be negative")
            rows.append({"variant": record.get("Enzyme", str(index)), "condition": condition,
                         "product": product, "ratio_definition": ratio_col.split("_", 1)[1],
                         "product_mean": vals.iloc[0], "product_reported_sd": vals.iloc[1],
                         "ratio_mean": vals.iloc[2], "ratio_reported_sd": vals.iloc[3],
                         "product_cv": coefficient_of_variation(vals.iloc[0], vals.iloc[1], min_mean),
                         "ratio_cv": coefficient_of_variation(vals.iloc[2], vals.iloc[3], ratio_min_mean)})
    report = {
        "rows": len(data), "columns": len(data.columns), "matched_summary_pairs": len(rows),
        "numbered_prefixes_are": "assay-condition labels; not independent replicates",
        "available": ["metabolite summary means", "reported standard deviations", "reported ratio summary statistics"] if rows else [],
        "not_established": ["replicate identities/counts and pairing", "biological versus technical replication",
                            "molar calibration/comparable metabolite response factors", "LOD/LOQ and measurement units",
                            "replicate covariance needed for ratio error propagation"],
        "limitations": ["Summary CVs are descriptive; sample sizes and uncertainty are unavailable.",
                        "The stored ratio DVA/(DVA+DVO) or OA/(OA+OL) differs from a denominator including VDAL/PDAL.",
                        "No claim of noise suppression or molar selectivity follows from model prediction accuracy.",
                        "Near-zero means make CV unstable; min_mean is a user threshold, not an inferred analytical limit."],
        "minimum_absolute_product_mean_for_cv": min_mean,
        "minimum_absolute_ratio_mean_for_cv": ratio_min_mean,
    }
    return report, pd.DataFrame(rows)


def read_limits(path: Path | None, components: list[str]) -> dict:
    limits = {} if path is None else json.loads(path.read_text())
    unknown = set(limits) - set(components)
    if unknown:
        raise ValueError(f"Limits contain unknown analytes: {sorted(unknown)}")
    for name, values in limits.items():
        if set(values) - {"lod", "loq"}:
            raise ValueError(f"Unknown limits for {name}; use lod and/or loq")
        for threshold in values.values():
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("LOD/LOQ values must be finite and nonnegative")
        if values.get("lod", 0) > values.get("loq", float("inf")):
            raise ValueError("LOD cannot exceed LOQ")
    return limits


def delta_ratio_se(values: np.ndarray) -> float:
    """Paired covariance delta-method SE for mean(product)/mean(total).

    Columns contain product first then byproducts; rows are paired replicates.
    This is the SE for a ratio of means, not for the mean of replicate ratios.
    """
    if len(values) < 2:
        return np.nan
    mean = values.mean(axis=0)
    total = mean.sum()
    if total <= 0:
        return np.nan
    gradient = np.full(len(mean), -mean[0] / total**2)
    gradient[0] += 1 / total
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1)) / len(values)
    return float(np.sqrt(max(0.0, gradient @ covariance @ gradient)))


def apply_censoring(values: np.ndarray, components: list[str], limits: dict, rule: str) -> np.ndarray:
    """Keep complete paired rows; never independently resample metabolite columns."""
    result = values.copy()
    valid = np.isfinite(result).all(axis=1)
    for j, component in enumerate(components):
        component_limits = limits.get(component, {})
        if rule == "exclude_below_loq":
            limit = component_limits.get("loq", component_limits.get("lod", 0))
            valid &= result[:, j] >= limit
        elif rule == "half_lod_sensitivity" and "lod" in component_limits:
            lod = component_limits["lod"]
            result[result[:, j] < lod, j] = lod / 2
    result = result[valid]
    return result[result.sum(axis=1) > 0]


def analyze_replicates(data: pd.DataFrame, components: list[str], variant_col: str,
                       condition_col: str, replicate_col: str, limits: dict,
                       seed: int = 777, bootstrap: int = 2000,
                       min_mean: float = 0.0, ratio_min_mean: float = 0.0) -> pd.DataFrame:
    required = [variant_col, condition_col, replicate_col] + components
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(set(required)) != len(required):
        raise ValueError("Identifiers and analytes must use distinct column names")
    if data[required[:3]].isna().any().any():
        raise ValueError("Variant, condition and replicate identifiers cannot be missing")
    if data.duplicated(required[:3]).any():
        raise ValueError("Duplicate variant/condition/replicate keys; specify experimental units explicitly")
    converted = data[components].apply(pd.to_numeric, errors="raise")
    if (converted < 0).any().any() or np.isinf(converted).any().any():
        raise ValueError("Concentrations must be nonnegative and finite or missing")
    if bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    rng = np.random.default_rng(seed)
    rows = []
    rules = ["unfiltered"]
    if limits:
        rules += ["exclude_below_loq", "half_lod_sensitivity"]
    for key, group in data.groupby([variant_col, condition_col], sort=True):
        original = group[components].apply(pd.to_numeric, errors="raise").to_numpy(float)
        for rule in rules:
            values = apply_censoring(original, components, limits, rule)
            n = len(values)
            row = {"variant": key[0], "condition": key[1], "rule": rule,
                   "n_input": len(original), "n_complete_positive_total": n,
                   "n_dropped": len(original) - n,
                   "ratio_definition": f"{components[0]}/({' + '.join(components)})"}
            for j, name in enumerate(components):
                row[f"n_below_lod_{name}"] = int(np.sum(original[:, j] < limits.get(name, {}).get("lod", 0)))
                row[f"n_below_loq_{name}"] = int(np.sum(original[:, j] < limits.get(name, {}).get("loq", 0)))
            if n:
                product = values[:, 0]
                ratios = product / values.sum(axis=1)
                p_sd = float(product.std(ddof=1)) if n > 1 else np.nan
                r_sd = float(ratios.std(ddof=1)) if n > 1 else np.nan
                row.update(product_mean=float(product.mean()), product_sd=p_sd,
                           product_cv=coefficient_of_variation(product.mean(), p_sd, min_mean),
                           mean_replicate_ratio=float(ratios.mean()), replicate_ratio_sd=r_sd,
                           replicate_ratio_cv=coefficient_of_variation(ratios.mean(), r_sd, ratio_min_mean),
                           ratio_of_means=float(product.mean() / values.mean(axis=0).sum()),
                           ratio_of_means_delta_se=delta_ratio_se(values))
                if n > 1:
                    # Row bootstrap preserves product/byproduct covariance.
                    indices = rng.integers(0, n, size=(bootstrap, n))
                    means = values[indices].mean(axis=1)
                    ratio_draws = means[:, 0] / means.sum(axis=1)
                    mean_ratio_draws = ratios[indices].mean(axis=1)
                    for name, draws in [("ratio_of_means", ratio_draws), ("mean_replicate_ratio", mean_ratio_draws)]:
                        row[name + "_ci_low"], row[name + "_ci_high"] = np.quantile(draws, [.025, .975])
            rows.append(row)
    return pd.DataFrame(rows)


def paired_cv_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Within-condition descriptive comparison on the same variants only.

    CV compares relative variability of different estimands and does not on its
    own establish absolute measurement accuracy or biological reproducibility.
    """
    if data.empty:
        return pd.DataFrame()
    ratio_col = "replicate_ratio_cv" if "replicate_ratio_cv" in data else "ratio_cv"
    if ratio_col not in data or "product_cv" not in data:
        return pd.DataFrame()
    keys = [c for c in ["condition", "product", "rule"] if c in data]
    rows = []
    for key, group in data.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        pair = group[["product_cv", ratio_col]].dropna()
        row = dict(zip(keys, key))
        row.update(n_variants=len(group), n_defined_cv_pairs=len(pair),
                   n_undefined_cv_pairs=len(group) - len(pair))
        if len(pair):
            row.update(product_median_cv=float(pair.product_cv.median()),
                       ratio_median_cv=float(pair[ratio_col].median()),
                       median_paired_cv_difference=float((pair[ratio_col] - pair.product_cv).median()),
                       fraction_ratio_cv_lower=float((pair[ratio_col] < pair.product_cv).mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "replicates"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant-col", default="variant")
    parser.add_argument("--condition-col", default="condition")
    parser.add_argument("--replicate-col", default="replicate")
    parser.add_argument("--product", default="DVA")
    parser.add_argument("--byproducts", nargs="+", default=["DVO", "VDAL"])
    parser.add_argument("--units", help="Common calibrated concentration unit, e.g. umol/L")
    parser.add_argument("--comparable-units", action="store_true", help="Acknowledge all analytes already use a directly comparable calibrated concentration basis")
    parser.add_argument("--limits-json", type=Path, help='Optional analyte limits in --units: {"DVA":{"lod":0.1,"loq":0.3},...}')
    parser.add_argument("--minimum-product-mean-for-cv", type=float, default=0.0, help="Near-zero threshold in --units")
    parser.add_argument("--minimum-ratio-mean-for-cv", type=float, default=0.0, help="Separate dimensionless near-zero threshold")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    thresholds = [args.minimum_product_mean_for_cv, args.minimum_ratio_mean_for_cv]
    if any(t < 0 or not np.isfinite(t) for t in thresholds):
        parser.error("CV near-zero thresholds must be finite and nonnegative")
    data = pd.read_csv(args.input)
    metadata = {"input": str(args.input.resolve()), "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "mode": args.mode, "seed": args.seed, "bootstrap": args.bootstrap,
                "numpy": np.__version__, "pandas": pd.__version__}
    if args.mode == "audit":
        report, output = summary_audit(data, *thresholds)
        metadata.update(report)
        filename = "summary_cv.csv"
    else:
        if not args.units or not args.comparable_units:
            parser.error("replicates requires --units and --comparable-units; do not calculate molar selectivity from uncalibrated signals")
        components = [args.product] + args.byproducts
        limits = read_limits(args.limits_json, components)
        output = analyze_replicates(data, components, args.variant_col, args.condition_col,
                                    args.replicate_col, limits, args.seed, args.bootstrap,
                                    *thresholds)
        metadata.update(units=args.units, components=components, limits=limits,
                        comparable_units_acknowledged=True,
                        minimum_absolute_product_mean_for_cv=args.minimum_product_mean_for_cv,
                        minimum_absolute_ratio_mean_for_cv=args.minimum_ratio_mean_for_cv,
                        uncertainty_unit="Independent paired replicate row within variant and condition; technical replicates alone do not estimate biological variation.",
                        limits_status="Provided externally" if limits else "Unavailable; no analytical-limit filtering or censoring inferred",
                        interpretation=["Unfiltered and half-LOD results are sensitivity analyses, not corrected ground truth.",
                                        "The ratio of means and mean of replicate ratios are different estimands.",
                                        "Delta SE uses paired covariance; bootstrap CIs resample paired replicate rows.",
                                        "Within-condition CVs are descriptive and unstable near zero; compare on matched variants.",
                                        "Predictive multi-task performance does not establish measurement error suppression."])
        filename = "replicate_reliability.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / filename, index=False)
    paired_cv_summary(output).to_csv(args.output_dir / "paired_cv_summary.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(output)} records to {args.output_dir / filename}")


if __name__ == "__main__":
    main()
