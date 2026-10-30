#!/usr/bin/env python
"""Run the explicit historical round/assay ranking plan on deposited scores."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

if __package__:
    from .common import ROOT, json_write
    from .ranking_ablation import main as run_ranking
else:
    from common import ROOT, json_write
    from ranking_ablation import main as run_ranking


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data/all_SeqFxnData_with_predicted_scores.csv')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', nargs='+', type=int, choices=[1,2,3], default=[1,2,3])
    p.add_argument('--top-k', type=int, default=5, help='Same explicit budget in each round/product pool; round 3 C6 has only 7 measured variants')
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--bootstrap-draws', type=int, default=2000)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Output must be empty')
    if a.top_k < 1 or a.bootstrap_draws < 100:
        p.error('Budgets must be positive and bootstrap draws >=100')
    # Explicit assay mapping is an analysis plan, not inferred replicate structure.
    jobs = []
    for r in sorted(set(a.rounds)):
        for product, prefix, suffix in [('C4', {1:6,2:10,3:12}[r], 'DVA/(DVA+DVO)'),
                                         ('C6', {1:7,2:11,3:13}[r], 'OA/(OA+OL)')]:
            jobs.append(dict(round=r, product=product, libraries=f'round_{r}' if r<3 else f'round_3_{product}',
                             outcome=f'{prefix}_{suffix}', supervised_score=f'round_{r}_ens_{suffix}'))
    # Preflight every job before launching any analyses.
    def job_arguments(job):
        return ['--data',str(a.data),'--out',str(a.out/f"round{job['round']}_{job['product']}"),
                '--outcome',job['outcome'],'--supervised-score',job['supervised_score'],
                '--libraries',job['libraries'],'--top-k',str(a.top_k),'--seed',str(a.seed),
                '--bootstrap-draws',str(a.bootstrap_draws)]
    for job in jobs:
        run_ranking(job_arguments(job)+['--preflight-only'])
    a.out.mkdir(parents=True, exist_ok=True)
    json_write(a.out/'analysis_plan.json', dict(jobs=jobs, top_k=a.top_k, seed=a.seed,
        warning='Per-round assays and prior-round deposited models. Pooled results sum stratified selections, not predictions from an all-data trained model.'))
    metrics = []
    for job in jobs:
        out = a.out/f"round{job['round']}_{job['product']}"
        run_ranking(job_arguments(job))
        for sd in (0,1):
            d = pd.read_csv(out/f'hit_sd_{sd}/metrics.csv')
            for key in ['round','product','libraries','outcome']:
                d[key] = job[key]
            d['hit_sd'] = sd
            metrics.append(d)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_metrics.to_csv(a.out/'all_metrics.csv', index=False)
    rows = []
    # Pool separately by product; C4 and C6 targets are distinct scientific outcomes.
    for (product, sd, method), d in all_metrics.groupby(['product','hit_sd','method']):
        budget = int(d.k.sum())
        expected = float((d.k * d.pool_hits / d.n).sum())
        hits = expected if method == 'random' else int(d.hits.sum())
        rows.append(dict(product=product,hit_sd=sd,method=method,n_round_pools=len(d),
                         total_selected=budget,hits_or_expected_random_hits=hits,
                         stratified_precision=hits/budget,expected_random_hits=expected,
                         enrichment_over_stratified_random=hits/expected if expected else None))
    pd.DataFrame(rows).to_csv(a.out/'pooled_descriptive.csv', index=False)
    print(f'Saved six-pool plan (or requested subset) and descriptive stratified summary to {a.out}')


if __name__ == '__main__':
    main()
