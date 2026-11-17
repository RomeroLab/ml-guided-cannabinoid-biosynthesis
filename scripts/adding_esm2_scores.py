#!/usr/bin/env python3
"""ESM-2 masked pseudo-log-likelihood and retrospective Figure 3d/e comparison.

Create the supplied Linux/NVIDIA server environment (nothing is installed here):
    conda env create -f esm2.yml
    conda activate esm2-scoring
Run from the repository root:
    python scripts/adding_esm2_scores.py --device cuda
Resume by rerunning the same command; completed sequence/model pairs are cached.
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
SCORES = ['vae_score', 'esm2_35M', 'esm2_650M']
PREFIXES = {'OA': [7, 11, 13], 'DVA': [6, 10, 12]}
MW = {'OA': 224.25, 'DVA': 196.2}
TARGETS = {'Titer_delta_mg_L': 'Titer change relative to CsOAC (mg/L)',
           'Fraction_change_pp': 'Fraction change relative to CsOAC (percentage points)'}
COLORS = {'vae_score': '#009E73', 'esm2_35M': '#0072B2', 'esm2_650M': '#D55E00'}
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
            part = frame[['Enzyme', 'hd', 'sequence_ungapped'] + SCORES + [titer, fraction]].copy()
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


def plot_results(correlations, enrichment, curves, args, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    # Separate files per round avoid conflating strain/assay differences.
    for round_number in [1, 2, 3]:
        round_dir = output / f'round_{round_number}'
        round_dir.mkdir(parents=True, exist_ok=True)
        round_curves = curves.loc[curves.Round == round_number]
        for stat in ['mean', 'max']:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
            for i, analyte in enumerate(['OA', 'DVA']):
                for j, (target, label) in enumerate(TARGETS.items()):
                    ax = axes[i, j]
                    subset = round_curves.loc[(round_curves.Analyte == analyte) & (round_curves.target == target)]
                    for score in SCORES:
                        part = subset.loc[subset.score == score]
                        ax.plot(part.k, part[f'{stat}_delta'], color=COLORS[score], label=score)
                    if stat == 'mean' and len(subset):
                        ax.axhline(subset.population_mean_delta.iloc[0], color='grey', ls=':',
                                   label='Random-selection expectation')
                    ax.axhline(0, color='black', lw=.6)
                    ax.set(title=f'{analyte}, Round {round_number}', xlabel='Number of selected variants (k)',
                           ylabel=('Mean ' if stat == 'mean' else 'Maximum ') + label.lower())
                    ax.spines[['top', 'right']].set_visible(False)
            axes[0, 0].legend(frameon=False, fontsize=8)
            fig.tight_layout()
            for ext in ['png', 'pdf', 'svg']:
                fig.savefig(round_dir / f'topk_{stat}_changes.{ext}', dpi=300)
            plt.close(fig)
        for metric in ['spearman', 'pearson', 'enrichment']:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
            for ax, analyte in zip(axes, ['OA', 'DVA']):
                for j, score in enumerate(SCORES):
                    vals = []
                    for target in TARGETS:
                        table = enrichment if metric == 'enrichment' else correlations.loc[correlations.method == metric]
                        row = table.loc[(table.Analyte == analyte) & (table.Round == round_number)
                                        & (table.target == target) & (table.score == score)]
                        vals.append(float(row.iloc[0]['enrichment' if metric == 'enrichment' else 'coefficient']) if len(row) else np.nan)
                    ax.bar(np.arange(2) + (j-1)*.24, vals, width=.24, color=COLORS[score], label=score)
                ax.set_xticks([0, 1], ['Titer change', 'Fraction change'])
                ax.set(title=f'{analyte}, Round {round_number}',
                       ylabel=f'Top-{args.top_k} enrichment' if metric == 'enrichment' else f'{metric.title()} correlation')
                ax.axhline(1 if metric == 'enrichment' else 0, color='grey', ls=':', lw=.8)
                ax.spines[['top', 'right']].set_visible(False)
            axes[0].legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(round_dir / f'{metric}.png', dpi=300)
            fig.savefig(round_dir / f'{metric}.pdf')
            plt.close(fig)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, default=root / 'data/all_SeqFxnData_with_predicted_scores.csv')
    p.add_argument('--output-dir', type=Path, default=root / 'results/esm2_comparison')
    p.add_argument('--sequence-column', default='Sequence')
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
    frame = attach_cached_scores(frame, output) if args.analysis_only else score_sequences(frame, args, output)
    frame.to_csv(output / 'all_SeqFxnData_with_esm2_scores.csv', index=False)
    observations, baselines = prepare_observations(frame, args)
    analysis_dir = output / 'by_round'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(analysis_dir / 'figure3de_normalized_observations.csv', index=False)
    baselines.to_csv(analysis_dir / 'wt_baselines.csv', index=False)
    data = observations
    data.to_csv(analysis_dir / 'round_specific_analysis_data.csv', index=False)
    corr, enrich, curves, selected = evaluate(data, args)
    for name, table in [('correlations', corr), (f'top{args.top_k}_hit_enrichment', enrich),
                        ('topk_curves', curves), (f'top{args.top_k}_selected_memberships', selected)]:
        table.to_csv(analysis_dir / f'{name}.csv', index=False)
        for round_number, part in table.groupby('Round'):
            round_dir = analysis_dir / f'round_{round_number}'
            round_dir.mkdir(parents=True, exist_ok=True)
            part.to_csv(round_dir / f'{name}.csv', index=False)
    plot_results(corr, enrich, curves, args, analysis_dir)
    metadata = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'algorithm': ALGORITHM, 'models': MODEL_IDS, 'prefixes': PREFIXES,
                'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'definitions_and_caveats': __doc__}
    (analysis_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(enrich.to_string(index=False))
    print(f'Finished. Input unchanged. Results: {analysis_dir}', flush=True)


if __name__ == '__main__':
    main()
