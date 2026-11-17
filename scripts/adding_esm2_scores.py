#!/usr/bin/env python3
"""ESM-2 masked pseudo-log-likelihood and retrospective Figure 3d/e comparison.

Create the supplied Linux/NVIDIA server environment (nothing is installed here):
    conda env create -f esm2.yml
    conda activate esm2-scoring
Run from the repository root:
    python scripts/adding_esm2_scores.py --device cuda
An existing all_SeqFxnData_with_esm2_scores.csv is reused without any model loading.
Use --scored-input PATH to choose it explicitly. Incomplete/inconsistent files fail
without rescoring. When no scored CSV exists, completed pairs resume from cache.
    python scripts/adding_esm2_scores.py --analysis-only

Scoring: mask one residue at a time, log P(original residue | all OTHER residues),
then average over residues (natural log; higher is better). This is length-normalized
masked pseudo-log-likelihood, NOT an unmasked reconstruction likelihood or a
WT-marginal mutation score. Gaps '-' and '.' and whitespace are removed. Nonstandard
residues fail explicitly. Identical ungapped sequences share one calculation.

Figure 3d/e preparation reproduced from cell 52 of
Figure_3_updated_for_units_and_dots.ipynb, inspected 2026-11-17:
OA primary prefixes 7/11/13, DVA 6/10/12; WT PKC001.000; hd 1..8.
Fraction = stored OA/C6 or DVA/C4, not recomputed from mean titers.
Delta fraction = 100*(variant - same-screen WT), in percentage points.
Delta titer = (variant - same-screen WT)*MW/1000 in mg/L assuming input uM.
No dilution correction. Prefixes 8/9 are validation, not extra round-1 variants.

Rounds 1, 2 and 3 are analyzed SEPARATELY because strains/assays differ.
No cross-round pooling or averaging is performed. Within each analyte/round,
k counts unique enzymes and comparisons use a common finite-score cohort.
Mean and maximum change curves are reported. With score ties the plotted values
are exact expectations under uniformly randomized tie breaking, including the
maximum (not the maximum over every partially selected tied candidate).
Hit = outcome strictly above same-screen WT (delta > 0 by default); this is an
operational improvement, NOT a significance test. Enrichment = top-k hit rate / full
eligible cohort hit rate. Top-k score ties receive fractional membership, so results
are the exact expected means/hits for uniformly broken ties, not arbitrary row order.
Correlation p-values are nominal/descriptive, with no multiple-testing correction. This is retrospective
ranking of experimentally selected variants, not prospective validation.

Utopia scores combine VAE with each ESM separately using equal-weight Euclidean
closeness to (1,1): -sqrt((1-VAE_norm)^2 + (1-ESM_norm)^2); higher is better.
Each component is min-max normalized within the analyte/round finite-base-score
cohort, before endpoint-specific missing-outcome filtering. Constant components
are assigned 1 (uninformative). Bounds use model scores only, not measured outcomes.
These retrospective cohort bounds are not the calibrated simulated-annealing
endpoints of the paper. Bounds and per-variant utopia scores/ranks are exported.
Spearman summary bars are unweighted arithmetic means of valid round-level
coefficients, with each round shown as a point; no measurements are pooled.
Pearson coefficients remain in CSV tables; no Pearson figures are generated.
Top-k figures have rounds as rows and metabolites as columns, one per endpoint.
Enrichment is saved only as numeric tables, not plotted. All outputs are saved
directly in results/esm2_comparison (or --output-dir); no round subfolders are created.
Combined CSVs retain the Round column so observations remain separate.
Figures are PNG and SVG only. Existing output files are left untouched unless rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_IDS = {
    'esm2_35M': 'facebook/esm2_t12_35M_UR50D',
    'esm2_650M': 'facebook/esm2_t33_650M_UR50D',
}
BASE_SCORES = ['vae_score', 'esm2_35M', 'esm2_650M']
SCORES = BASE_SCORES + ['utopia_35M', 'utopia_650M']
PREFIXES = {'OA': [7, 11, 13], 'DVA': [6, 10, 12]}
MW = {'OA': 224.25, 'DVA': 196.2}
TARGETS = {'Titer_delta_mg_L': 'Titer change relative to CsOAC (mg/L)',
           'Fraction_change_pp': 'Fraction change relative to CsOAC (percentage points)'}
# Cellibre paper families: VAE blue, model-only magenta, utopia green.
COLORS = {'vae_score': '#4477AA', 'esm2_35M': '#AA3377', 'esm2_650M': '#CC79A7',
          'utopia_35M': '#228833', 'utopia_650M': '#66AA55'}
LABELS = {'vae_score': 'VAE-only', 'esm2_35M': 'ESM-2 35M-only',
          'esm2_650M': 'ESM-2 650M-only', 'utopia_35M': 'VAE + ESM-2 35M utopia',
          'utopia_650M': 'VAE + ESM-2 650M utopia'}
ALGORITHM = 'masked_pll_mean_natural_log_v1'


def clean_sequence(value):
    if pd.isna(value):
        raise ValueError('Missing sequence')
    sequence = ''.join(str(value).upper().split()).replace('-', '').replace('.', '')
    invalid = set(sequence) - set('ACDEFGHIKLMNPQRSTVWY')
    if not sequence or invalid:
        raise ValueError(f'Empty/noncanonical sequence; invalid residues: {sorted(invalid)}')
    return sequence


def read_input(path, sequence_column):
    frame = pd.read_csv(path)
    required = {'Enzyme', 'hd', 'vae_score', sequence_column}
    if not required.issubset(frame):
        raise ValueError(f'Missing columns: {sorted(required - set(frame))}')
    if frame['Enzyme'].isna().any() or frame['Enzyme'].duplicated().any():
        raise ValueError('Expected exactly one nonempty source row per Enzyme.')
    frame['sequence_ungapped'] = frame[sequence_column].map(clean_sequence)
    if frame['sequence_ungapped'].str.len().gt(1022).any():
        raise ValueError('Sequences longer than 1022 residues are unsupported; no truncation.')
    frame['hd'] = pd.to_numeric(frame['hd'], errors='raise')
    frame['vae_score'] = pd.to_numeric(frame['vae_score'], errors='raise')
    return frame


def cache_key(model_id, sequence):
    return hashlib.sha256(f'{ALGORITHM}|{model_id}|{sequence}'.encode()).hexdigest()


def read_cache(path):
    cache = {}
    if path.exists():
        # Do not silently ignore damaged/incomplete records: preserve the cache for review.
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row['algorithm'] != ALGORITHM or not math.isfinite(row['score']):
                    raise ValueError('Incompatible/nonfinite score')
                if row['key'] != cache_key(row['model_id'], row['sequence']):
                    raise ValueError('Cache key mismatch')
                cache[row['key']] = row['score']
            except (ValueError, KeyError) as exc:
                raise ValueError(f'Invalid cache record {path}:{line_no}: {exc}') from exc
    return cache


def score_sequences(frame, args, output):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback.')
    torch.set_num_threads(args.cpu_threads)
    path = output / 'esm2_score_cache.jsonl'
    cache = read_cache(path)
    sequences = frame['sequence_ungapped'].drop_duplicates().tolist()
    for column, model_id in MODEL_IDS.items():
        missing = [s for s in sequences if cache_key(model_id, s) not in cache]
        if missing:
            print(f'Loading {model_id}: {len(missing)} unique sequences remaining', flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForMaskedLM.from_pretrained(model_id).eval().to(args.device)
            started = time.monotonic()
            for number, sequence in enumerate(missing, 1):
                ids = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)['input_ids']
                if ids.shape[1] != len(sequence) + 2:
                    raise ValueError('Expected one token per residue plus BOS/EOS.')
                # ESM checkpoints support at most 1022 amino acids; never truncate silently.
                if len(sequence) > 1022:
                    raise ValueError(f'Sequence length {len(sequence)} exceeds supported 1022.')
                ids = ids.to(args.device)
                total = 0.0
                with torch.inference_mode():
                    for start in range(1, len(sequence) + 1, args.mask_batch_size):
                        positions = torch.arange(start, min(start + args.mask_batch_size, len(sequence) + 1), device=args.device)
                        batch = ids.repeat(len(positions), 1)
                        row = torch.arange(len(positions), device=args.device)
                        original = batch[row, positions].clone()
                        batch[row, positions] = tokenizer.mask_token_id
                        logits = model(input_ids=batch, attention_mask=torch.ones_like(batch)).logits
                        logp = torch.log_softmax(logits[row, positions].float(), dim=-1)
                        total += logp[row, original].double().sum().item()
                score = total / len(sequence)
                if not math.isfinite(score):
                    raise ValueError('Nonfinite ESM score')
                key = cache_key(model_id, sequence)
                record = {'key': key, 'algorithm': ALGORITHM, 'model_id': model_id,
                          'sequence': sequence, 'score': score}
                with path.open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                    handle.flush()
                cache[key] = score
                elapsed = time.monotonic() - started
                remaining = elapsed / number * (len(missing) - number)
                print(f'{column}: {number}/{len(missing)}; elapsed {elapsed/60:.1f} min; '
                      f'ETA {remaining/60:.1f} min (this model only)', flush=True)
            del model
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
        frame[column] = [cache[cache_key(model_id, s)] for s in frame['sequence_ungapped']]
    return frame


def attach_cached_scores(frame, output):
    cache = read_cache(output / 'esm2_score_cache.jsonl')
    for column, model_id in MODEL_IDS.items():
        keys = [cache_key(model_id, s) for s in frame['sequence_ungapped']]
        if any(k not in cache for k in keys):
            raise ValueError(f'Incomplete cache for {model_id}; run scoring first.')
        frame[column] = [cache[k] for k in keys]
    return frame


def reuse_scored_csv(frame, path, sequence_column):
    """Reuse by enzyme + exact ungapped sequence; never fill missing scores by inference."""
    scored = read_input(path, sequence_column)
    missing = set(MODEL_IDS) - set(scored)
    if missing:
        raise ValueError(f'{path}: missing score columns {sorted(missing)}; no rescoring performed.')
    for col in MODEL_IDS:
        scored[col] = pd.to_numeric(scored[col], errors='raise')
    if set(frame.Enzyme) != set(scored.Enzyme):
        raise ValueError('Scored CSV enzyme IDs differ from input; choose a matching --input/--scored-input. No rescoring.')
    indexed = scored.set_index('Enzyme').loc[frame.Enzyme]
    if not np.array_equal(indexed.sequence_ungapped.to_numpy(), frame.sequence_ungapped.to_numpy()):
        raise ValueError('Scored CSV sequences differ from input; no rescoring performed.')
    if not np.isfinite(indexed[list(MODEL_IDS)].to_numpy(float)).all():
        raise ValueError('Scored CSV contains missing/nonfinite ESM scores; no rescoring performed.')
    for col in MODEL_IDS:
        frame[col] = indexed[col].to_numpy()
    return frame


def find_scored_csv(args, output):
    if args.scored_input is not None:
        if not args.scored_input.is_file():
            raise FileNotFoundError(f'Explicit scored CSV missing: {args.scored_input}; no inference.')
        return args.scored_input
    name = 'all_SeqFxnData_with_esm2_scores.csv'
    candidates = [args.input] if args.input.name == name else []
    candidates += [output / name, args.input.parent / name, Path.cwd() / name]
    return next((path for path in candidates if path.is_file()), None)


def add_utopia_scores(observations):
    """Model-only min-max bounds per analyte/round, independent of measured endpoint."""
    result = observations.copy()
    bounds = []
    for (analyte, round_number), indices in result.groupby(['Analyte', 'Round']).groups.items():
        group = result.loc[indices]
        finite = np.isfinite(group[BASE_SCORES].to_numpy(float)).all(axis=1)
        good = group.index[finite]
        normalized = {}
        for score in BASE_SCORES:
            values = group.loc[good, score].to_numpy(float)
            low, high = (float(values.min()), float(values.max())) if len(values) else (np.nan, np.nan)
            normalized[score] = (values - low) / (high - low) if high > low else np.ones(len(values))
            bounds.append(dict(Analyte=analyte, Round=round_number, score=score,
                               minimum=low, maximum=high, n=len(values), constant=bool(high == low)))
        for size in ['35M', '650M']:
            col = f'utopia_{size}'
            result.loc[good, col] = -np.sqrt((1 - normalized['vae_score']) ** 2
                                           + (1 - normalized[f'esm2_{size}']) ** 2)
            result.loc[indices, col + '_rank'] = result.loc[indices, col].rank(ascending=False, method='min')
    return result, pd.DataFrame(bounds)


def prepare_observations(frame, args):
    wt = frame.loc[frame['Enzyme'].eq(args.wt_id)]
    if len(wt) != 1:
        raise ValueError(f'Expected one WT row {args.wt_id}')
    wt = wt.iloc[0]
    parts, baselines = [], []
    for analyte, prefixes in PREFIXES.items():
        for round_number, prefix in enumerate(prefixes, 1):
            titer, fraction = f'{prefix}_{analyte}', f'{prefix}_{analyte}/C{6 if analyte == "OA" else 4}'
            wt_t, wt_f = float(wt[titer]), float(wt[fraction])
            if not (np.isfinite(wt_t) and np.isfinite(wt_f) and 0 < wt_f <= 1):
                raise ValueError(f'Invalid WT baseline for prefix {prefix}')
            scale = MW[analyte] / 1000 if args.input_titer_unit == 'uM' else 1.0
            part = frame[['Enzyme', 'hd', 'sequence_ungapped'] + BASE_SCORES + [titer, fraction]].copy()
            part = part.rename(columns={titer: 'Titer_input', fraction: 'Fraction'})
            for col in ['Titer_input', 'Fraction']:
                part[col] = pd.to_numeric(part[col], errors='raise')
                if np.isinf(part[col]).any():
                    raise ValueError(f'Infinite measurement in {col}, prefix {prefix}')
            part = part.loc[part['Enzyme'].ne(args.wt_id) & part['hd'].between(args.hd_min, args.hd_max)
                            & part[['Titer_input', 'Fraction']].notna().any(axis=1)].copy()
            if not part['Fraction'].dropna().between(0, 1).all() or not part['hd'].eq(part['hd'].round()).all():
                raise ValueError('Invalid fractions or noninteger hd')
            part['Analyte'], part['Round'], part['Prefix'] = analyte, round_number, prefix
            part['Fraction_change_pp'] = 100 * (part['Fraction'] - wt_f)
            part['Titer_delta_mg_L'] = (part['Titer_input'] - wt_t) * scale
            parts.append(part)
            baselines.append({'Analyte': analyte, 'Round': round_number, 'Prefix': prefix,
                              'WT_fraction': wt_f, 'WT_titer_input': wt_t,
                              'WT_titer_mg_L': wt_t * scale, 'Input_titer_unit': args.input_titer_unit})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(baselines)


def topk_weights(scores, k):
    """Expected membership of top k under uniformly randomized score ties."""
    scores = np.asarray(scores, float)
    cutoff = np.sort(scores)[::-1][k - 1]
    above, equal = scores > cutoff, scores == cutoff
    return above.astype(float) + equal * ((k - above.sum()) / equal.sum())


def topk_expected_max(scores, outcomes, k):
    """Exact expected maximum for top k when the cutoff score has ties."""
    scores, outcomes = np.asarray(scores, float), np.asarray(outcomes, float)
    cutoff = np.sort(scores)[::-1][k - 1]
    above = scores > cutoff
    tied_y = np.sort(outcomes[scores == cutoff])
    r = int(k - above.sum())
    m = len(tied_y)
    fixed_max = float(outcomes[above].max()) if above.any() else -np.inf
    denominator = math.comb(m, r)
    # Highest selected tied index j has probability C(j, r-1)/C(m, r).
    return float(sum(max(fixed_max, tied_y[j]) * math.comb(j, r - 1) / denominator
                     for j in range(r - 1, m)))


def evaluate(data, args):
    from scipy.stats import pearsonr, spearmanr
    correlations, enrichment, curves, selections = [], [], [], []
    for (analyte, round_number), group in data.groupby(['Analyte', 'Round'], sort=True):
        for target in TARGETS:
            # All methods compared on the identical finite-score/outcome population.
            eligible = group.loc[np.isfinite(group[SCORES + [target]].to_numpy(float)).all(axis=1)].copy()
            n = len(eligible)
            if not n:
                continue
            y = eligible[target].to_numpy(float)
            threshold = args.titer_hit_delta if target == 'Titer_delta_mg_L' else args.fraction_hit_delta
            hits = y > threshold
            base = float(hits.mean())
            for score_col in SCORES:
                x = eligible[score_col].to_numpy(float)
                for method, fn in [('spearman', spearmanr), ('pearson', pearsonr)]:
                    result = fn(x, y) if n >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 else (np.nan, np.nan)
                    correlations.append(dict(Analyte=analyte, Round=round_number, target=target, score=score_col,
                        method=method, n=n, coefficient=float(result[0]), pvalue=float(result[1])))
                selected_k = min(args.top_k, n)
                for k in range(1, n + 1):
                    weights = topk_weights(x, k)
                    avg = float(weights @ y / k)
                    hit_count = float(weights @ hits)
                    rate = hit_count / k
                    curves.append(dict(Analyte=analyte, Round=round_number, target=target, score=score_col, k=k,
                        mean_delta=avg, max_delta=topk_expected_max(x, y, k),
                        population_mean_delta=float(y.mean()), expected_hits=hit_count,
                        hit_rate=rate, enrichment=rate / base if base else np.nan))
                    if k == selected_k:
                        enrichment.append(dict(Analyte=analyte, Round=round_number, target=target, score=score_col, n=n,
                            requested_k=args.top_k, effective_k=k, hit_delta_threshold=threshold,
                            expected_hits=hit_count, hit_rate=rate, population_hit_rate=base,
                            enrichment=rate / base if base else np.nan))
                        selected = eligible.loc[weights > 0].copy()
                        selected['selection_weight'] = weights[weights > 0]
                        selected['ranking_score'], selected['target'] = score_col, target
                        selections.append(selected)
    if not curves:
        raise ValueError('No common finite-score/outcome cohort remains.')
    return pd.DataFrame(correlations), pd.DataFrame(enrichment), pd.DataFrame(curves), pd.concat(selections, ignore_index=True)


def plot_results(correlations, curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none'})
    method_handles = [Line2D([0], [0], color=COLORS[s], lw=2, label=LABELS[s]) for s in SCORES]

    def save(fig, stem):
        for extension in ['png', 'svg']:
            fig.savefig(stem.with_suffix('.' + extension), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Average coefficients, never pool observations from different rounds.
    for metric in ['pearson', 'spearman']:
        table = correlations.loc[correlations.method == metric]
        if metric == 'pearson':
            summary = table.groupby(['Analyte', 'target', 'score'], as_index=False).agg(
                mean_coefficient=('coefficient', 'mean'), valid_rounds=('coefficient', 'count'))
            summary['method'] = metric
            summary.to_csv(output / f'{metric}_round_mean_summary.csv', index=False)
            continue
        summary_rows = []
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        markers = {1: 'o', 2: 's', 3: '^'}
        for i, analyte in enumerate(['OA', 'DVA']):
            for j, target in enumerate(TARGETS):
                ax = axes[i, j]
                for x, score in enumerate(SCORES):
                    part = table.loc[(table.Analyte == analyte) & (table.target == target) & (table.score == score)]
                    values = part.coefficient.to_numpy(float)
                    finite = values[np.isfinite(values)]
                    mean = float(finite.mean()) if len(finite) else np.nan
                    summary_rows.append(dict(Analyte=analyte, target=target, score=score,
                                             method=metric, mean_coefficient=mean, valid_rounds=len(finite)))
                    ax.bar(x, mean, color=COLORS[score], alpha=.45, width=.66, zorder=2)
                    for row in part.itertuples():
                        if np.isfinite(row.coefficient):
                            ax.scatter(x + (row.Round - 2)*.13, row.coefficient,
                                       marker=markers[row.Round], s=34, color=COLORS[score],
                                       edgecolor='black', linewidth=.45, zorder=3)
                ax.axhline(0, color='grey', lw=.7)
                ax.set(title=f'{analyte}: {"titer" if j == 0 else "fraction"} change',
                       ylabel=f'{metric.title()} correlation', ylim=(-1.05, 1.05))
                ax.set_xticks(range(len(SCORES)), [LABELS[s].replace(' utopia', '\nutopia') for s in SCORES],
                              rotation=25, ha='right')
                ax.spines[['top', 'right']].set_visible(False)
        round_handles = [Line2D([0], [0], marker=markers[r], color='black', linestyle='', label=f'Round {r}') for r in [1,2,3]]
        fig.legend(handles=round_handles, loc='lower center', ncol=3, frameon=False)
        fig.suptitle('Bars: unweighted mean across available rounds; points: individual rounds')
        fig.tight_layout(rect=(0, .055, 1, .96))
        save(fig, output / f'{metric}_all_rounds')
        pd.DataFrame(summary_rows).to_csv(output / f'{metric}_round_mean_summary.csv', index=False)

    # One figure per endpoint: rows are separate rounds, columns are metabolites.
    for target, label in TARGETS.items():
        fig, axes = plt.subplots(3, 2, figsize=(12, 12), squeeze=False)
        for row_index, round_number in enumerate([1, 2, 3]):
            for column_index, analyte in enumerate(['OA', 'DVA']):
                ax = axes[row_index, column_index]
                subset = curves.loc[(curves.Round == round_number) & (curves.Analyte == analyte)
                                    & (curves.target == target)]
                for score in SCORES:
                    part = subset.loc[subset.score == score]
                    for stat, style, marker in [('max', '-', '^'), ('mean', (0, (5, 3)), 'o')]:
                        ax.plot(part.k, part[f'{stat}_delta'], color=COLORS[score], linestyle=style,
                                marker=marker, markersize=3, markevery=max(1, len(part)//15), linewidth=1.8)
                if len(subset):
                    ax.axhline(subset.population_mean_delta.iloc[0], color='grey', ls=':', lw=1.3)
                ax.axhline(0, color='black', ls='--', lw=.8)
                ax.set(title=f'Round {round_number}: {analyte}',
                       xlabel='Number of selected variants (k)',
                       ylabel=label.replace(' (', '\n('))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.spines[['top', 'right']].set_visible(False)
                ax.grid(axis='y', alpha=.2)
        stat_handles = [Line2D([0], [0], color='#444444', lw=1.8, linestyle=style,
                              marker=marker, label=stat) for stat, style, marker in
                        [('Maximum', '-', '^'), ('Mean', (0, (5, 3)), 'o')]]
        baseline_handles = [Line2D([0], [0], color='grey', ls=':', lw=1.3,
                                   label='Random: expected mean'),
                            Line2D([0], [0], color='black', ls='--', lw=.8,
                                   label='CsOAC baseline')]
        fig.legend(handles=method_handles + stat_handles + baseline_handles,
                   loc='lower center', ncol=3, frameon=False, fontsize=9,
                   bbox_to_anchor=(.5, .005))
        fig.suptitle('Mean and maximum WT-relative changes; rounds analyzed separately')
        fig.tight_layout(rect=(0, .09, 1, .97))
        suffix = 'titer' if target == 'Titer_delta_mg_L' else 'fraction'
        save(fig, output / f'topk_max_mean_{suffix}_all_rounds')



def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, default=root / 'data/all_SeqFxnData_with_predicted_scores.csv')
    p.add_argument('--output-dir', type=Path, default=root / 'results/esm2_comparison')
    p.add_argument('--sequence-column', default='Sequence')
    p.add_argument('--scored-input', type=Path, help='Existing scored CSV to reuse; never triggers inference if invalid')
    p.add_argument('--device', default='cuda')
    p.add_argument('--mask-batch-size', type=int, default=8)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--analysis-only', action='store_true')
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--wt-id', default='PKC001.000')
    p.add_argument('--hd-min', type=int, default=1)
    p.add_argument('--hd-max', type=int, default=8)
    p.add_argument('--input-titer-unit', choices=['uM', 'mg/L'], default='uM')
    p.add_argument('--titer-hit-delta', type=float, default=0.0, help='Strict hit threshold in mg/L above WT')
    p.add_argument('--fraction-hit-delta', type=float, default=0.0, help='Strict hit threshold in percentage points above WT')
    args = p.parse_args()
    if min(args.mask_batch_size, args.cpu_threads, args.top_k) < 1 or args.hd_min > args.hd_max:
        p.error('Batch size, threads, k must be positive and hd-min <= hd-max.')
    return args


def main():
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame = read_input(args.input, args.sequence_column)
    print(f'{len(frame)} rows; {frame.sequence_ungapped.nunique()} unique ungapped sequences', flush=True)
    # Fail on assay schema/baseline problems before downloading or loading models.
    validation_frame = frame.assign(esm2_35M=np.nan, esm2_650M=np.nan)
    prepare_observations(validation_frame, args)
    del validation_frame
    scored_path = find_scored_csv(args, output)
    if scored_path is not None:
        print(f'Reusing {scored_path}; no model loading or rescoring.', flush=True)
        frame = reuse_scored_csv(frame, scored_path, args.sequence_column)
    else:
        frame = attach_cached_scores(frame, output) if args.analysis_only else score_sequences(frame, args, output)
        frame.to_csv(output / 'all_SeqFxnData_with_esm2_scores.csv', index=False)
    observations, baselines = prepare_observations(frame, args)
    observations.to_csv(output / 'figure3de_normalized_observations.csv', index=False)
    baselines.to_csv(output / 'wt_baselines.csv', index=False)
    data, utopia_bounds = add_utopia_scores(observations)
    utopia_bounds.to_csv(output / 'utopia_normalization_bounds.csv', index=False)
    data.to_csv(output / 'round_specific_analysis_data.csv', index=False)
    corr, enrich, curves, selected = evaluate(data, args)
    for name, table in [('correlations', corr), (f'top{args.top_k}_hit_enrichment', enrich),
                        ('topk_curves', curves), (f'top{args.top_k}_selected_memberships', selected)]:
        table.to_csv(output / f'{name}.csv', index=False)
    plot_results(corr, curves, output)
    metadata = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'algorithm': ALGORITHM, 'models': MODEL_IDS, 'prefixes': PREFIXES,
                'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'reused_scored_csv': str(scored_path) if scored_path is not None else None,
                'scored_csv_sha256': hashlib.sha256(scored_path.read_bytes()).hexdigest() if scored_path else None,
                'definitions_and_caveats': __doc__}
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(enrich.to_string(index=False))
    print(f'Finished. Input unchanged. Results: {output}', flush=True)


if __name__ == '__main__':
    main()
