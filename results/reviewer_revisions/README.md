# Completed reviewer analyses

These files are the results executed in the cloud environment. You can inspect
them on your own server without installing the model-training environment.

From the repository root:

```bash
python3 scripts/revisions/view_results.py
```

Open [overview.html](overview.html) in a local browser for tables and ranking
plots. It uses relative files bundled in this directory; no web server is needed.
GitHub can display the individual PNG and CSV files directly.

## Results to inspect

| Directory/file | Contents |
|---|---|
| [ranking_suite_final/pooled_descriptive.csv](ranking_suite_final/pooled_descriptive.csv) | Top five in each of six round/product pools, both WT hit definitions; descriptive stratified summary. |
| [ranking_suite_final/all_metrics.csv](ranking_suite_final/all_metrics.csv) | Per-round VAE, MLP, combined, Pareto and finite-pool random comparisons. |
| [top10_rounds1_2/pooled_descriptive.csv](top10_rounds1_2/pooled_descriptive.csv) | Top ten in round 1 and 2 pools; round 3 C6 has only seven measured candidates. |
| [top10_rounds1_2/all_metrics.csv](top10_rounds1_2/all_metrics.csv) | Per-round top-ten metrics for both hit definitions. |
| [measurement_audit/paired_cv_summary.csv](measurement_audit/paired_cv_summary.csv) | Descriptive product-versus-ratio CV comparison, within assay condition. |
| [measurement_audit/summary_cv.csv](measurement_audit/summary_cv.csv) | 453 matched variant/condition product and ratio summaries. |
| [validation_smoke](validation_smoke) | Short model fits, split audits and frozen-snapshot temporal checks. These are execution checks, not final model-performance results. |

Within each ranking pool, `hit_sd_0` means an outcome strictly above the matched
WT mean; `hit_sd_1` means strictly above WT mean plus one reported WT SD. Each
contains `ranking.png`, `metrics.csv`, `budget_curves.csv`,
`paired_comparisons.csv`, and `candidates.csv` with the exact selected sets.
The random baseline samples from the same experimentally measured pool. It does
not represent an experimentally tested library of random mutants.

## Interpretation limits

- Combined VAE+MLP scores did not uniformly outperform individual scores.
- The stored ratios exclude VDAL/PDAL, whereas the reviewer describes
  three-product fractions. Resolve the manuscript endpoint before drawing claims.
- All historical ranking pools contain one parent; parent-bootstrap comparative
  confidence intervals cannot be estimated from those pools.
- Historical round-1 snapshot sequences differ from the master CSV for 109/153
  canonical matching IDs. Temporal runs retain historical sequences and labels
  together and write `snapshot_master_audit.csv` for author review.
- Original round-1 saved train/test nearest aligned identity averages about
  98.6%. This motivates harder evaluation; it does not establish the performance
  of new models.
- Replicate pairing/counts, comparable concentration calibration, and validated
  LOD/LOQ are absent from the summaries. Single-concentration chromatograms alone
  do not establish validated limits. Predicting several labels does not prove
  suppression of measurement error.
- Validation fits use two epochs and are smoke tests only. Full prespecified
  model training, ensembles and real pLM inference have not been completed.

## Validation evidence

Twenty-three scientific-invariant tests passed in the prepared environment.
Real-data short runs covered random and parent CV, 99% identity-component CV,
saved partitions, and all three historical-to-next-round transitions. The 90%
identity-component split was audited separately without model fitting.

The `validation_smoke` directory names map as follows:

| Directory | Check |
|---|---|
| `random_final` | Random sequence-grouped CV, short fits |
| `parent_final` | Parent-grouped CV, short fits |
| `identity` | Identity-component CV at 0.99, short fits |
| `identity90_audit` | Identity-component split at 0.90, audit only |
| `saved_eval` | Original round-1 partitions, controlled short fits |
| `temporal_pool_final` | Frozen initial training data to experimental round 1 |
| `temporal_pool_v1` | Frozen training data through round 1 to round 2 |
| `temporal_pool_v2` | Frozen training data through round 2 to round 3 |

Historical `/workspace/...` paths in JSON metadata record the original execution
environment. You do not need those paths to read these results. They are not
server-specific configuration instructions. For rerunning experiments, see
[the full usage guide](../../scripts/revisions/README.md).
