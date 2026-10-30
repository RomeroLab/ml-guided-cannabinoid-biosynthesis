#!/usr/bin/env python3
"""Print the completed descriptive results; uses Python standard library only."""
import csv
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'results/reviewer_revisions'
for label, sub in [('Top 5 in each of six round/product pools','ranking_suite_final'),('Top 10 in round 1 and 2 pools','top10_rounds1_2')]:
    print('\n'+label)
    path=root/sub/'pooled_descriptive.csv'
    with path.open(newline='') as fh:
        rows=list(csv.DictReader(fh))
    print('Product  WT_SD  Method          Selected  Hits*    Precision  Enrichment')
    for r in rows:
        enrichment = '{:.3f}'.format(float(r['enrichment_over_stratified_random'])) if r['enrichment_over_stratified_random'] else 'NA'
        print('{:<8} {:<6} {:<15} {:<9} {:<8.2f} {:<10.3f} {}'.format(r['product'],r['hit_sd'],r['method'],r['total_selected'],float(r['hits_or_expected_random_hits']),float(r['stratified_precision']),enrichment))
print('\n*For random, Hits is the expected number under random selection from the measured pool.')
print('WT_SD=0: above WT mean; WT_SD=1: above WT mean plus one reported WT SD.')
print('These are retrospective measured-pool results, not prospective random-mutant hit rates.')
print('Full CV training and real pLM inference have not been run; validation_smoke fits use 2 epochs.')
print('Read results/reviewer_revisions/README.md before interpreting results.')
