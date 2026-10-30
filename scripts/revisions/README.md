# Computational experiments for the reviewer revisions

These scripts add analyses without modifying the deposited models, training code,
or experimental data. They test the proposed benefits; superiority of the combined
model is not assumed. The attached reviewer document is scientific context, not an
instruction to access the server paths listed in it.

Completed tables and plots are in
[`results/reviewer_revisions`](../../results/reviewer_revisions/README.md).
Print the summaries from the repository root with
`python3 scripts/revisions/view_results.py`; this viewer uses only the Python
standard library and does not require the training environment.

## Run environment

From the repository root in this cloud environment:

```bash
source /workspace/.venvs/running-simulated-annealing/bin/activate
export XDG_CACHE_HOME=/workspace/.cache
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
```

The main analyses use the installed NumPy/pandas/SciPy/scikit-learn/PyTorch
environment. ESM inference has separate optional dependencies below. Write results
outside the checkout and use a new output directory for each run. Repository
pickles are trusted historical snapshots; do not load untrusted pickles.
On another machine, activate the repository's `environment.yml` environment
instead; replace `/workspace/revision-results` with your desired output root.

## Which experiments answer which comments?

| Reviewer concern | Script | Evidence and boundary |
|---|---|---|
| R2.1, R2.2, R4.2: contribution of VAE and supervised scores | `ranking_ablation.py`, `run_ranking_suite.py` | Matched measured-pool enrichment, precision/recall, Pareto fronts, ranking curves, random expectation, and below-WT discrimination. Retrospective selection, not a prospective random-mutant experiment. |
| R2.1, R4.1: multi-task versus single-task; related sequence leakage | `validate_models.py` | Newly fitted controlled MLP ablations, same folds/seeds, train-only scaling; random, parent, identity, saved-split audits, and historical-to-next-round transfer. |
| R3.1, R3.2: ratios versus titers and measurement error | `measurement_reliability.py` | Descriptive summary CVs now; paired covariance, ratio error propagation and LOD/LOQ sensitivity when raw paired replicates are provided. |
| R4.2: modern pLM baselines | `score_plm.py` + `ranking_ablation.py` | Optional ESM2/ESMC plausibility scores, evaluated on the same measured pool/assay. Weights are not bundled and real pLM inference was not run here. |

## 1. Historical ranking ablations

Both hit rules run by default: measured outcome **strictly greater than WT mean**,
and **strictly greater than WT mean + one reported WT SD**. A separate flag records
outcomes below WT mean minus one WT SD. These are operational thresholds, not
significance tests, and a low product ratio does not demonstrate misfolding or lack
of catalytic activity. The control always comes from the exact same assay column.

The historical plan uses these explicit mappings:

| Measured library | Previous trained ensemble | C4 outcome | C6 outcome | Measured C4/C6 pool sizes |
|---|---|---|---|---|
| `round_1` | `round_1_ens_*` | `6_DVA/(DVA+DVO)` | `7_OA/(OA+OL)` | 92 / 92 |
| `round_2` | `round_2_ens_*` | `10_DVA/(DVA+DVO)` | `11_OA/(OA+OL)` | 23 / 11 |
| `round_3_C4`, `round_3_C6` | `round_3_ens_*` | `12_DVA/(DVA+DVO)` | `13_OA/(OA+OL)` | 10 / 7 |

Choose these assay definitions before examining method performance. They are the
deposited **two-product ratios**. The reviewer's ratios including VDAL/PDAL are
different outcomes. A ratio calculated from summary means is also not generally
equal to the mean of replicate ratios, and its SD cannot be recovered without
covariance or paired replicates. No script silently equates these quantities.

Run all six pools with a matched budget of five per pool:

```bash
python scripts/revisions/run_ranking_suite.py \
  --out /workspace/revision-results/my_ranking_suite --top-k 5
```

Top-10 analysis is possible in rounds 1 and 2, but round 3 C6 has only seven
measured candidates. The script rejects a larger budget instead of changing it:

```bash
python scripts/revisions/run_ranking_suite.py \
  --out /workspace/revision-results/my_top10 --rounds 1 2 --top-k 10
```

For one pool or an alternative measured endpoint (including a raw product column):

```bash
python scripts/revisions/ranking_ablation.py \
  --out /workspace/revision-results/my_round1_C4 \
  --outcome '6_DVA/(DVA+DVO)' \
  --supervised-score 'round_1_ens_DVA/(DVA+DVO)' \
  --libraries round_1 --top-k 10 --hit-sd 0 1
```

Methods use higher-is-better percentile ranks: VAE alone, MLP alone, their fixed
equal-weight mean, and nondominated Pareto layers with the mean rank breaking
within-layer ties. Lexicographic variant ID breaks remaining ties without labels.
The combined rule is a prespecified retrospective ablation, not a reconstruction
of the original manual design-selection procedure.

By default, a candidate with a training variant ID **or** training sequence is
removed using the appropriate deposited snapshot. Alias normalization handles
`PKC1.0` versus `PKC001.000`. Custom score columns require `--training-snapshot`.
`--allow-training-overlap` exists only for explicitly labeled descriptive analyses;
results from a model trained on all rows cannot establish extrapolation. The
deposited score provenance itself still needs author verification.

Each threshold directory contains `candidates.csv` (scores, hit labels, selected
sets, Pareto layers), `metrics.csv`, `budget_curves.csv`, `paired_comparisons.csv`,
and `ranking.png`. `metadata.json` records exclusions, hashes and assumptions.

The random baseline is an exact hypergeometric null for selecting k of these
measured candidates uniformly without replacement. Its interval is a random
selection interval, not a measurement-confidence interval. Upper-tail p-values
receive Holm correction across methods within a threshold/run only. Additional
assays, thresholds, rounds, or tuned combination rules require a broader
prespecified multiplicity plan before inferential claims.

Paired comparisons bootstrap fixed selected sets with shared candidate weights,
grouped by parent by default. Many pools have only one parent: these correctly
have **no informative parent-bootstrap CI**. `--bootstrap-unit sequence` makes a
stronger independence assumption and must be reported. Neither bootstrap includes
uncertainty from training, WT measurements, or the original adaptive design.

`pooled_descriptive.csv` sums round-stratified selections separately for C4/C6;
it is not a new model trained on all data. Cross-round pooling keeps the same
budget and assay-matched threshold in each stratum. It has no pooled significance
claim. The measured variants were already experimentally selected: these results
cannot estimate hit rates for all possible random mutants. Predictions on new
random mutants would establish model preference, not experimentally verified hits.

## 2. Supervised validation and multi-task ablation

New models use the deposited embedding/flatten/hidden/dropout/ReLU/output shape,
with configurable output width. The experiment deliberately uses train-only
target scaling, finite-label masking, and equal per-task mean losses, instead of
reproducing full-dataset normalization and fixed unequal label weights. The
comparison therefore isolates task sharing under this controlled implementation;
it is not an exact retraining of published checkpoints. Separate single-task
networks have more total parameters than the shared multi-task network. A
training-mean baseline is included. Model parameter counts and selected epochs
are logged.

Random CV groups exact aligned and biological duplicate sequences. Parent CV
additionally keeps the parent ID together. Identity CV uses connected components
at the specified aligned identity threshold, excluding joint gaps. Transitive
connections stay together, and no between-fold pair exceeds the threshold.
Parent groups alone do not guarantee a homology identity cutoff.

Example **smoke** run, not a scientific model-performance estimate:

```bash
python scripts/revisions/validate_models.py --mode parent --through-round 3 \
  --targets '6_DVA/(DVA+DVO),7_OA/(OA+OL)' \
  --epochs 2 --patience 2 --folds 3 \
  --out /workspace/revision-results/my_parent_smoke
```

For a prespecified study, run `--mode random`, `parent`, and `identity`, with
several seeds and the intended targets. Use `--through-round 0`, `1`, `2`, `3`
for cumulative datasets. Default mapping treats WT and initial libraries as round
0; `--round-map custom.json` overrides the complete map. This differs from the
historical model numbering, which starts at `round_1` trained before experimental
round 1. Use `--pool-assays` only if treating assay conditions as comparable is
scientifically justified. Without it, tasks remain separate assay columns.

Inspect splits before expensive training:

```bash
python scripts/revisions/validate_models.py --mode identity --identity 0.90 \
  --folds 3 --through-round 3 --audit-only \
  --out /workspace/revision-results/my_identity_audit
```

At strict thresholds, connected components may collapse most sequences into one
group. The program refuses impossible folds rather than relaxing the threshold
silently. In the current data, the largest parent has 230 of 285 variants, so
parent-held-out results have highly unequal fold sizes and limited effective
sample size. Report per-fold counts and correlations; do not treat folds as
independent biological replicates or pool differently scaled targets into one
correlation. For scientific runs, choose epochs/patience/ensemble/targets/seeds
before looking at test scores. Defaults are 1000 maximum epochs, patience 100,
one member, hidden width 65, learning rate 5e-6. Multi-task plus T single-task
targets across F folds and E ensemble members requires F×E×(T+1) model fits.

Historical-to-next-round transfer uses the frozen training snapshot's sequences
**and labels**, then the next experimental library from the current master CSV:

```bash
python scripts/revisions/validate_models.py --mode temporal --train-through 0 \
  --pool-assays --targets 'DVA/(DVA+DVO),OA/(OA+OL)' \
  --epochs 2 --patience 2 \
  --out /workspace/revision-results/my_temporal_smoke
```

Repeat `--train-through 1` and `2` for later rounds. Historical snapshots contain
different assay prefixes from new-round measurements. `--pool-assays` explicitly
averages available assays with the same target suffix on **each side separately**;
this tests assay-transfer under a comparability assumption. It does not validate
assay equivalence or eliminate batch effects. Without pooling, many exact assay
targets are unmeasured in the next round and correctly have n=0. Adding new
rounds also changes the test population: differing correlations alone do not
isolate the causal benefit of retraining.

The snapshot audit found 109/153 historical round-1 aligned sequences differ
from the master CSV for matching canonical IDs. Some differences persist after
removing gaps. The code never attaches old labels to changed current sequences;
`snapshot_master_audit.csv` exposes the discrepancies for author review. Resolve
them before using historical transfer as evidence in a rebuttal.

Audit the original saved partitions separately:

```bash
python scripts/revisions/validate_models.py --mode saved --audit-only \
  --saved-data models/round_1/round_1_seq_fxn_data.pkl \
  --saved-splits models/round_1/round_1_data_splits.pkl \
  --out /workspace/revision-results/my_saved_audit
```

Removing `--audit-only` fits the controlled ablation on those partitions. Legacy
partitions are preserved and flagged even if overlap exists; they are never called
leakage-free. New splits enforce no exact overlap. Every run writes `splits.json`,
row identities, training settings, per-target `metrics.csv` and long-form
`predictions.csv`. Predictions are out-of-fold or next-round for the specified
split, not predictions from a model trained on all candidates.

## 3. Measurement reliability and analytical limits

Use the available summary means/SDs now:

```bash
python scripts/revisions/measurement_reliability.py audit \
  --input data/all_SeqFxnData.csv \
  --output-dir /workspace/revision-results/my_measurement_audit
```

This produces descriptive paired product-CV versus stored-ratio-CV summaries on
matched variants within assay conditions. Numbered assay prefixes are **not**
assumed to be biological replicates. Lower CV compares relative variability of
different estimands; it does not establish molar comparability or absolute
accuracy. Near-zero means make CV unstable. Replicate counts, pairing, calibration
units, LOD/LOQ and covariance are unavailable from these tables.

For raw measurements, prepare a CSV with one independent replicate row per
variant/condition and paired analytes:

```text
variant,condition,replicate,DVA,DVO,VDAL
variant_A,assay_C4,bio_1,<calibrated concentration>,<concentration>,<concentration>
variant_A,assay_C4,bio_2,<calibrated concentration>,<concentration>,<concentration>
```

Use a directly comparable calibrated concentration basis for every component,
preferably molar for molar selectivity. Then:

```bash
python scripts/revisions/measurement_reliability.py replicates \
  --input paired_C4.csv --product DVA --byproducts DVO VDAL \
  --units umol/L --comparable-units \
  --output-dir /workspace/revision-results/my_replicate_analysis
```

Outputs distinguish the ratio of means from the mean of replicate ratios. Paired
bootstrap resampling preserves analyte covariance; the delta-method SE also uses
that covariance. Technical replicates alone cannot estimate biological variation.
No ML model can be assumed to remove measurement error merely by fitting several
targets.

LOD/LOQ inputs, when independently validated, look like an analyte-keyed JSON
mapping with `lod`/`loq` fields in the same units as the concentrations. Pass it
with `--limits-json limits.json`. Include real measured values, not guessed
numbers. The script runs unfiltered, below-LOQ exclusion, and half-LOD substitution
as explicitly labeled sensitivity analyses; these are not corrected ground truth.
For C6, use `--product OA --byproducts OL PDAL`.

The chromatogram folder has sample and single-concentration standard traces,
including `OA 0.1.csv`; the filename alone does not define its concentration unit.
Those traces can contribute to signal-to-noise assessment, but do not establish
validated per-analyte concentration limits. Typically a dilution/calibration
series plus blanks or replicate low standards and a documented integration/noise
procedure are needed. Common conventions are S/N near 3 (LOD) and 10 (LOQ), or
3.3σ/slope and 10σ/slope, followed by experimental validation at low concentrations.
Do not infer a single universal limit from these files or from WT variability.

## 4. ESM2 and ESMC baselines (optional)

Use a separate environment with a supported modern PyTorch, pandas/NumPy and
`transformers` for ESM2; ESMC uses the official `esm==3.2.1` SDK adapter. This keeps
the deposited PyTorch 2.0.1 environment intact. Installing dependencies and weights
requires the relevant model-host network access and any applicable model access.
No token values belong in chat or source files. Use `--local-files-only` with
cached weights if needed. These optional dependency combinations and real model
inference were not validated in the prepared environment; adapter logic is tested
with deterministic fake models.

```bash
python scripts/revisions/score_plm.py \
  --data data/all_SeqFxnData_with_predicted_scores.csv \
  --wt-id PKC001.000 --backend esm2 \
  --model facebook/esm2_t33_650M_UR50D --method pll \
  --out /workspace/revision-results/esm2_scores.csv
```

Use `--backend esmc --model esmc_600m` for ESMC. Compare the same scoring definition
across pLMs. `--method masked-marginal` sums mutation log-odds in masked WT
background; it cannot model epistasis and requires WT-length substitutions. Only
176/285 input sequences have WT length; use `--on-invalid skip` to record all
excluded rows explicitly. `--method pll` uses length-normalized masked
pseudo-log-likelihood in each sequence's own background minus WT's, supports
different lengths, and has a different interpretation. It can be expensive.

Join external scores to the identical candidate universe:

```bash
python scripts/revisions/ranking_ablation.py \
  --out /workspace/revision-results/my_esm2_comparison \
  --outcome '6_DVA/(DVA+DVO)' \
  --supervised-score 'round_1_ens_DVA/(DVA+DVO)' \
  --libraries round_1 --top-k 10 \
  --extra-scores /workspace/revision-results/esm2_scores.csv \
  --extra-score-columns plm_score
```

If an external score is missing, every method uses the reduced common pool;
exclusions are logged. Do not compare metrics computed on different pools as if
they shared candidates. To combine two external pLM files, merge by unique
`Enzyme` and rename their score columns distinctly first. Zero-shot pLM scores are
not C4/C6-specific activity predictions; their experimental evaluation is.

## Verification and current limitations

```bash
python -m unittest discover -s tests -p 'test_revision*.py' -v
```

Tests exercise masking/token indices, missing-label gradients, train-only scaling,
split leakage, Pareto dominance, matched pools, random nulls, covariance
cancellation, and replicate pairing. Real-data ranking and summary-CV analyses
were executed. Short model fits verify execution only; full cross-validation and
pLM inference must finish before reporting their scientific results. Do not
select hyperparameters, assay definitions, hit thresholds, or blend weights based
on held-out performance without an independent final evaluation.
