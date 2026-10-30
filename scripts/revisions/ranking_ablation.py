#!/usr/bin/env python
"""Retrospective, matched-pool ranking ablations; no unmeasured hit claims.

All ranks are higher-is-better. Hits use an explicitly chosen assay and its
matched WT control. Supervised-training sequences are excluded by default.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__:
    from .common import ROOT, canonical_id, json_write, load_snapshot_ids, load_snapshot_sequences, parent_id, sha256
else:
    from common import ROOT, canonical_id, json_write, load_snapshot_ids, load_snapshot_sequences, parent_id, sha256


def pareto_layers(x, y):
    """Nondominated layers for maximization, including tied points together."""
    values = np.column_stack((x, y)).astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Pareto objectives must be finite")
    remaining = np.arange(len(values))
    layers = np.zeros(len(values), dtype=int)
    layer = 0
    while len(remaining):
        v = values[remaining]
        dominates = (v[:, None, :] >= v[None, :, :]).all(2) & (v[:, None, :] > v[None, :, :]).any(2)
        front = ~dominates.any(0)
        layers[remaining[front]] = layer
        remaining = remaining[~front]
        layer += 1
    return layers


def top_indices(score, ids, k):
    # Stable, predeclared tie break independent of measured outcomes.
    return np.lexsort((np.asarray(ids, dtype=str), -np.asarray(score)))[:k]


def fixed_selection_difference(hit, left, right, units, draws, seed):
    """Paired cluster bootstrap for the two fixed selected sets.

    Reweights shared candidates together. This conditions on the observed pool
    and fitted scores; it does not represent model-training or design uncertainty.
    """
    unique, inverse = np.unique(units, return_inverse=True)
    if len(unique) < 2:
        return (np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        weights = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[inverse]
        a, b = weights[left].sum(), weights[right].sum()
        if a and b:
            diffs.append(np.dot(weights[left], hit[left]) / a - np.dot(weights[right], hit[right]) / b)
    if not diffs:
        return (np.nan, np.nan, 0)
    low, high = np.quantile(diffs, [0.025, 0.975])
    return float(low), float(high), len(diffs)


def holm(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return out


def analyze(pool, score_columns, outcome, wt_mean, wt_sd, hit_sd, k, draws, unit, seed):
    y = pool[outcome].to_numpy(float)
    hit = y > wt_mean + hit_sd * wt_sd
    loss = y < wt_mean - wt_sd
    ranks = {name: pool[col].rank(pct=True, method="average").to_numpy() for name, col in score_columns.items()}
    ranks["combined_rank"] = (ranks["VAE"] + ranks["MLP"]) / 2
    layer = pareto_layers(ranks["VAE"], ranks["MLP"])
    # Layer has priority; average ranks only break ties within a layer.
    ranks["pareto"] = -layer + ranks["combined_rank"] / 2
    ids = pool.Enzyme.to_numpy()
    selected = {name: top_indices(score, ids, k) for name, score in ranks.items()}
    units = pool.parent.to_numpy() if unit == "parent" else ids
    n, nhit = len(pool), int(hit.sum())
    rows, curves, comparisons = [], [], []
    for name, score in ranks.items():
        idx = selected[name]
        precision = float(hit[idx].mean())
        rows.append(dict(method=name, n=n, k=k, pool_hits=nhit, hits=int(hit[idx].sum()),
                         precision_at_k=precision, recall_at_k=float(hit[idx].sum()/nhit) if nhit else np.nan,
                         enrichment=precision / hit.mean() if nhit else np.nan,
                         below_WT_minus_SD_fraction_at_k=float(loss[idx].mean()),
                         auroc_hit=roc_auc_score(hit, score) if 0 < nhit < n else np.nan,
                         average_precision_hit=average_precision_score(hit, score) if 0 < nhit < n else np.nan,
                         auroc_low_outcome=roc_auc_score(loss, -score) if 0 < loss.sum() < n else np.nan,
                         spearman=float(spearmanr(y, score).statistic) if np.ptp(y) and np.ptp(score) else np.nan,
                         random_pool_tail_p=float(hypergeom.sf(hit[idx].sum()-1, n, nhit, k))))
        order = top_indices(score, ids, n)
        for budget in range(1, n + 1):
            h = int(hit[order[:budget]].sum())
            curves.append(dict(method=name, budget=budget, hits=h, precision=h/budget,
                               recall=h/nhit if nhit else np.nan))
    for row, adjusted in zip(rows, holm([r["random_pool_tail_p"] for r in rows])):
        row["random_pool_tail_p_holm"] = adjusted
    for name in ranks:
        if name == "combined_rank":
            continue
        left, right = selected["combined_rank"], selected[name]
        low, high, valid = fixed_selection_difference(hit, left, right, units, draws, seed)
        comparisons.append(dict(left="combined_rank", right=name,
                                precision_difference=float(hit[left].mean()-hit[right].mean()),
                                ci_low=low, ci_high=high, valid_bootstraps=valid,
                                bootstrap_unit=unit, n_units=len(set(units))))
    null = dict(method="random", n=n, k=k, pool_hits=nhit, precision_at_k=nhit/n,
                recall_at_k=k/n if nhit else np.nan, enrichment=1 if nhit else np.nan,
                below_WT_minus_SD_fraction_at_k=float(loss.mean()),
                random_precision_low=float(hypergeom.ppf(.025,n,nhit,k)/k),
                random_precision_high=float(hypergeom.ppf(.975,n,nhit,k)/k))
    rows.append(null)
    annotated = pool.copy()
    annotated["hit"] = hit
    annotated["below_WT_minus_SD"] = loss
    annotated["pareto_layer"] = layer
    for name, score in ranks.items():
        annotated[f"rank_{name}"] = score
        annotated[f"selected_{name}"] = np.isin(np.arange(n), selected[name])
    return pd.DataFrame(rows), pd.DataFrame(curves), pd.DataFrame(comparisons), annotated


def plot_results(annotated, metrics, curves, path, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    points = ax[0].scatter(annotated.rank_VAE, annotated.rank_MLP,
                           c=annotated.hit.astype(int), cmap="coolwarm", vmin=0, vmax=1, alpha=.7)
    top = annotated[annotated.selected_combined_rank]
    ax[0].scatter(top.rank_VAE, top.rank_MLP, s=90, facecolors="none", edgecolors="black", label="Combined top k")
    random = annotated.sample(n=int(metrics.k.iloc[0]), random_state=seed)
    ax[0].scatter(random.rank_VAE, random.rank_MLP, marker="x", color="green", label="One random subset")
    front = annotated[annotated.pareto_layer.eq(0)].sort_values("rank_VAE")
    ax[0].plot(front.rank_VAE, front.rank_MLP, color="gray", alpha=.6, label="First Pareto front")
    ax[0].set(xlabel="VAE percentile rank", ylabel="MLP percentile rank", title="Measured candidate pool (red = hit)")
    ax[0].legend(fontsize=7)
    for method, d in curves.groupby("method", sort=False):
        ax[1].plot(d.budget, d.precision, label=method)
    ax[1].axhline(annotated.hit.mean(), color="gray", linestyle="--", label="Random expectation")
    ax[1].set(xlabel="Selection budget", ylabel="Precision", ylim=(0, 1.05))
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT/"data/all_SeqFxnData_with_predicted_scores.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--outcome", required=True, help="Exact measured assay column; e.g. 6_DVA/(DVA+DVO)")
    p.add_argument("--supervised-score", required=True, help="Exact deposited score column")
    p.add_argument("--vae-score", default="vae_score")
    p.add_argument("--libraries", required=True, help="Comma-separated measured libraries to compare")
    p.add_argument("--wt-id", default="PKC001.000")
    p.add_argument("--hit-sd", type=float, nargs="+", default=[0., 1.])
    p.add_argument("--training-snapshot", type=Path, help="Trusted pickle for supervised score training population")
    p.add_argument("--allow-training-overlap", action="store_true", help="Descriptive only; disables automatic training exclusion")
    p.add_argument("--extra-scores", type=Path, help="CSV with unique Enzyme and optional external scores")
    p.add_argument("--extra-score-columns", nargs="*", default=[])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    p.add_argument("--bootstrap-unit", choices=["parent", "sequence"], default="parent")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--preflight-only", action="store_true", help="Validate the matched pool without writing results")
    a = p.parse_args(argv)
    if a.top_k < 1 or a.bootstrap_draws < 100 or any(s < 0 or not np.isfinite(s) for s in a.hit_sd):
        p.error("top-k must be positive; bootstrap-draws >=100; hit-sd finite and nonnegative")
    df = pd.read_csv(a.data)
    if df.Enzyme.isna().any() or df.Enzyme.duplicated().any():
        p.error("Enzyme must be nonmissing and unique")
    paths = {"data": {"path": str(a.data), "sha256": sha256(a.data)}}
    if a.extra_scores:
        extra = pd.read_csv(a.extra_scores)
        if extra.Enzyme.isna().any() or extra.Enzyme.duplicated().any():
            p.error("External Enzyme must be nonmissing and unique")
        if set(a.extra_score_columns) & set(df.columns):
            p.error("External score column names must not overwrite existing columns")
        df = df.merge(extra[["Enzyme"] + a.extra_score_columns], on="Enzyme", how="left", validate="one_to_one")
        paths["external"] = {"path": str(a.extra_scores), "sha256": sha256(a.extra_scores)}
    cols = {"VAE": a.vae_score, "MLP": a.supervised_score}
    if set(a.extra_score_columns) & {"VAE", "MLP", "combined_rank", "pareto", "random"}:
        p.error("External scores must not use reserved method names")
    cols.update({c: c for c in a.extra_score_columns})
    missing = set([a.outcome, a.outcome+"_std", "Aligned_Sequence", "library"] + list(cols.values())) - set(df)
    if missing:
        p.error(f"Missing columns: {sorted(missing)}")
    wt = df[df.Enzyme.eq(a.wt_id)]
    if len(wt) != 1:
        p.error("Exactly one WT row required")
    wt_mean, wt_sd = float(wt[a.outcome].iloc[0]), float(wt[a.outcome+"_std"].iloc[0])
    if not np.isfinite([wt_mean, wt_sd]).all() or wt_sd < 0:
        p.error("Matched-assay WT mean and nonnegative SD required")
    libraries = a.libraries.split(",")
    if set(libraries) - set(df.library):
        p.error("Unknown requested library")
    pool = df[df.library.isin(libraries) & ~df.Enzyme.eq(a.wt_id)].copy()
    n_requested = len(pool)
    # Use aligned sequences as model inputs, stripping gaps for overlap detection.
    pool["ungapped"] = pool.Aligned_Sequence.str.replace("-", "", regex=False)
    snapshot = a.training_snapshot
    if not a.allow_training_overlap:
        if snapshot is None:
            prefix = a.supervised_score.split("_ens_")[0]
            if prefix not in ("round_1", "round_2", "round_3"):
                p.error("Supply --training-snapshot for custom supervised scores, or explicitly --allow-training-overlap")
            snapshot = ROOT/"models"/prefix/f"{prefix}_seq_fxn_data.pkl"
        training = load_snapshot_sequences(snapshot)
        paths["training_snapshot"] = {"path": str(snapshot), "sha256": sha256(snapshot)}
        overlap = pool.ungapped.isin(training) | pool.Enzyme.map(canonical_id).isin(load_snapshot_ids(snapshot))
        excluded_training = pool.loc[overlap, "Enzyme"].tolist()
        pool = pool[~overlap].copy()
    else:
        excluded_training = []
    for c in [a.outcome] + list(cols.values()):
        pool[c] = pd.to_numeric(pool[c], errors="coerce")
    valid = np.isfinite(pool[[a.outcome]+list(cols.values())].to_numpy(float)).all(axis=1)
    excluded_missing = pool.loc[~valid, "Enzyme"].tolist()
    pool = pool[valid].sort_values("Enzyme").reset_index(drop=True)
    if pool.ungapped.isna().any() or pool.ungapped.duplicated().any():
        p.error("Candidate pool has missing or duplicate biological sequences; resolve before ranking")
    if len(pool) < a.top_k:
        p.error(f"Only {len(pool)} matched candidates; requested top-k={a.top_k}. Choose a smaller matched budget explicitly.")
    pool["parent"] = pool.Enzyme.map(parent_id)
    if a.preflight_only:
        print(f"Validated {len(pool)} matched candidates for {a.outcome}")
        return
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Output directory must be empty; preserve prior runs")
    a.out.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for sd in sorted(set(a.hit_sd)):
        result = a.out/f"hit_sd_{sd:g}"
        result.mkdir()
        metrics, curves, diffs, annotated = analyze(pool, cols, a.outcome, wt_mean, wt_sd, sd, a.top_k,
                                                  a.bootstrap_draws, a.bootstrap_unit, a.seed)
        metrics.to_csv(result/"metrics.csv", index=False)
        curves.to_csv(result/"budget_curves.csv", index=False)
        diffs.to_csv(result/"paired_comparisons.csv", index=False)
        annotated.to_csv(result/"candidates.csv", index=False)
        plot_results(annotated, metrics, curves, result/"ranking.png", a.seed)
        all_metrics.append(dict(hit_sd=sd, threshold=wt_mean+sd*wt_sd, pool_hits=int(annotated.hit.sum())))
    json_write(a.out/"metadata.json", dict(arguments=vars(a), inputs=paths, command=sys.argv,
        versions={pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "pandas", "scipy", "scikit-learn")},
        code_hashes={str(f):sha256(f) for f in (Path(__file__),Path(__file__).with_name('common.py'))},
        requested_candidates=n_requested, final_candidates=len(pool), excluded_training=excluded_training,
        excluded_missing=excluded_missing, wt_mean=wt_mean, wt_sd=wt_sd, thresholds=all_metrics,
        interpretation="Retrospective within experimentally measured, selected candidates only. Not a prospective random-mutation comparison or causal ML benefit estimate.",
        score_provenance="Deposited CSV scores; training overlap checked by canonical variant ID OR ungapped aligned sequence unless explicitly disabled. Early training snapshot sequences differ from current CSV for some IDs; verify historic score provenance independently.",
        bootstrap="Fixed-selection paired bootstrap; parent clusters by default. No CI if fewer than 2 units. Does not include training, WT measurement, or threshold uncertainty. Sequence bootstrap assumes independent variants.",
        multiplicity="Holm correction across methods vs finite-pool random null within each threshold/run only; not across assays or runs.",
        ties="Average percentile ranks; lexicographic Enzyme tie-break independent of outcome.",
        ratio_warning="Assay column used exactly as supplied; two-product ratios are not three-product fractions; no molar calibration inferred."))
    print(f"Wrote ranking ablations for {len(pool)} candidates to {a.out}")


if __name__ == "__main__":
    main()
