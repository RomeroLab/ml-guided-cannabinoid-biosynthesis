#!/usr/bin/env python3
"""REUSE-FIRST revision: existing CV predictions; optional opt-in all-data reference.

python scripts/generalization-analysis.py --reuse-run results/generalization-analysis/run_TIMESTAMP
Pearson/Spearman/top10 recall figures use0..1 axes; sample SD across OOF seeds.
Top10 recall ranks UNIQUE sequences by mean WT-relative ratio across their
observed heads. This differs from Pearson/Spearman, which retain assay-level
concatenation. Recall=expected overlap of true and predicted top-k /k, with
independently uniform ties and k=min(10,n); n<=10 is flagged trivial.
Optional --in-sample-predictions plus --in-sample-provenance overlays saved
all-data model predictions on the identical finite OOF cohort. No baseline model
is created by default. Explicit --train-all-data-baseline --baseline-epochs N
trains a NEW fixed-epoch all-data reference (10members, oneGPU sequentially),
without tuning on OOF scores. It saves models and raw predictions for future reuse. All-data metadata is supplied provenance, not proof
of the training process. No dashed line is fabricated if unavailable.

Historical training methodology below describes SOURCE RUNS ONLY. The command
line in this revision no longer accepts --run or repeats CV training. Only the
explicit --train-all-data-baseline flag enables new in-sample reference training.

Original description: Four-fold generalization analysis; no training without --run.

Environment (Linux NVIDIA server): conda env create -f generalization.yml
  conda activate generalization-analysis
  python scripts/generalization-analysis.py
  python scripts/generalization-analysis.py --run --device cuda:0 \
      --output-dir results/generalization-analysis

Defaults: snapshots1/2/3 x cluster/random/homology x4folds x3seeds x10models
=1080 individual fits before any skipped/infeasible folds. One selected GPU is
used sequentially, never10GPUs. --run is required; without it only split plans
are produced. All models, CSVs and figures go under results/generalization-analysis by default.
Each invocation creates a unique timestamped plan_/run_ subfolder, preserving
prior runs. --output-dir changes only the base directory. Each best ensemble
member is saved with architecture, target scaling, labels, seed and split IDs.

CLUSTER: evenly distribute each existing Cluster across4 test folds (as far as
integer cluster sizes allow), preserving original within-cluster sampling idea,
NOT old saved fixed train/val/test indices. Each entity is held out exactly once
per seed. Clusters smaller than4 cannot appear in all folds. Inner validation
samples ~15% of each cluster's remaining sequence groups; singleton groups stay
in training. Identical ungapped sequences stay together. Cross-cluster identical
sequences cause an explicit skip, not leakage or arbitrary cluster reassignment.
RANDOM: seeded balanced4fold sequence-group split, ~15% inner group validation.
HOMOLOGY (default third protocol):98% global-alignment connected components.
See identity_components docstring for exact alignment scoring and tie convention.
Groups never cross train/validation/test; threshold/fold count are never relaxed.
All pair identities and memberships are exported before splitting, including in
plan-only mode. Fewer than4 components causes an explicit skip.
PARENT (optional): canonical PKC### lineage groups held intact across outer/inner partitions,
including identical sequence connected components. This is lineage grouping, NOT
sequence-identity homology clustering. Large PKC001 makes equal-sized folds
impossible; tiny/unsupported heads and infeasible splits are reported, not filled.
Three seeds repeat each entire4fold procedure. All seeds/folds use a10-member
independently initialized ensemble; median prediction matches the supplied plot.

Round datasets are cumulative model-availability snapshots, not assay rounds:
24/40/48 ratio heads for screens0..5/0..9/0..11. The architecture is reimplemented
without importing existing training scripts or pretrained checkpoints:
21x16 embedding -> flatten -> Linear -> Dropout(.2) -> ReLU -> Linear;
hidden widths2000/65/65; Adam(lr5e-6,wd.001), masked weighted MSE, early stopping.
Use --hidden-width 65 for an architecture-controlled sensitivity analysis.
Max-absolute target scaling is fit on training ONLY; validation chooses checkpoint.
True zeros remain observations (unlike original zero-to-missing plotting).
Unsupported heads produce NaN, never untrained predictions. Missing/nonfinite
labels are masked. No test labels tune model fitting, scaling or stopping.
Canonical-ID aliases retain every measurement row and original ID plus row_id.
They remain in the same group; conflicting sequences fail rather than merge.
PKC001.26 and PKC001.026 in snapshot3 have identical sequences and complementary
screen4/5 versus10/11 observations; neither row is deleted or averaged.

Compiled plots reproduce the reference's8 ratio panels: ensemble median outputs
are converted back to raw ratios, and BOTH predictions and observations are
divided by each assay head's PKC001.000 raw WT baseline. Baselines are a fixed
post-hoc reporting reference from the full snapshot, NOT input to model training
or checkpoint selection. Missing/zero WT baselines are omitted and audited.
Heads with the same base ratio are concatenated across assay prefixes; no per-
enzyme averaging. Every point is held-out OOF, once per enzyme/head/seed, not
training or validation points. Each seed has its own plots/correlations; never
pool repeated seed predictions. Repeated assays on one enzyme are not independent
biological replicates. Report Pearson, Spearman, MSE, coverage and sample counts
both per head and for compiled WT-relative base ratios. Across-round comparisons
are descriptive because cohorts, assay heads and native architectures differ.

Optional --protocols forward: source1->snapshot2 and2->snapshot3 comparisons,
not included in default1080fits. Half of novel sequence groups are locked test;
train source only versus retrain next snapshot FROM SCRATCH, never using locked
test entities or parent groups in parent mode. Unseen-screen source predictions
are an explicit historical same-ratio head-average heuristic, not a trained new
head. Shared-head and new-screen results stay separate;8/9 are validation screens.
No snapshot4 exists, so12/13 not evaluated. Infeasible forward groups are skipped.

Split plans are saved before training, progress/results after each fold. Outputs
are CSV/JSON plus PNG/SVG, with source hashes, package versions and seeds. No
checkpoint reuse/resume or automatic GPU launch; saved models are inference
checkpoints, not optimizer-state training-resume checkpoints. Only trusted local pickles load.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import uuid
import hashlib
import importlib.metadata
import platform
import json
import os
import pickle
import re
from pathlib import Path
import numpy as np
import pandas as pd

AA = 'ACDEFGHIKLMNPQRSTVWY-'


def canonical(value):
    m = re.fullmatch(r'PKC0*(\d+)\.0*(\d+)', str(value).strip())
    if not m:
        raise ValueError(f'Unexpected enzyme ID: {value}')
    return f'PKC{int(m[1]):03d}.{int(m[2]):03d}'


def load_snapshot(root, number):
    folder = root / 'models' / f'round_{number}'
    path = folder / f'round_{number}_seq_fxn_data.pkl'
    df = pd.read_pickle(path).copy().reset_index(drop=True)
    df = df.rename(columns={'name': 'Enzyme', 'sequence': 'Aligned_Sequence'})
    df['original_id'] = df.Enzyme.astype(str)
    df['row_id'] = [f'r{number}_row{i}' for i in range(len(df))]
    df['Enzyme'] = df.Enzyme.map(canonical)
    df['parent'] = df.Enzyme.str.split('.').str[0]
    df['sequence_key'] = df.Aligned_Sequence.str.replace('-', '', regex=False)
    labels = [c for c in df if re.match(r'^\d+_', c)]
    assert len(labels) == {1:24, 2:40, 3:48}[number], 'Unexpected label schema'
    if df.Aligned_Sequence.isna().any():
        raise ValueError('Missing aligned sequence')
    for enzyme, part in df.groupby('Enzyme'):
        if part.Aligned_Sequence.nunique()!=1:
            raise ValueError(f'Canonical identity has conflicting sequences: {enzyme}')
    if df.Aligned_Sequence.str.len().nunique() != 1:
        raise ValueError('Aligned sequence lengths differ')
    if any(set(s) - set(AA) for s in df.Aligned_Sequence):
        raise ValueError('Noncanonical aligned sequence')
    df[labels] = df[labels].apply(pd.to_numeric, errors='raise').replace([np.inf,-np.inf], np.nan)
    splitpath = folder / f'round_{number}_data_splits.pkl'
    with splitpath.open('rb') as handle:
        split = pickle.load(handle)
    if not isinstance(split, (tuple,list)) or len(split) != 3:
        raise ValueError('Expected one saved train/val/test positional partition')
    split = [np.asarray(x, dtype=int) for x in split]
    flat = np.concatenate(split)
    if len(flat) != len(df) or sorted(flat.tolist()) != list(range(len(df))):
        raise ValueError('Original partition overlaps, omits rows, or is out of bounds')
    return df, labels, split, [path, splitpath, folder/f'round_{number}_models.py', folder/f'round_{number}_EnsMLPs.py']


def identity_components(df):
    """98% edges from global affine alignment of ungapped sequences.

    Biopython PairwiseAligner, global, match2/mismatch-1/open-5/extend-.5,
    terminal gaps penalized identically. Identity=exact matches/all alignment
    columns, including terminal/internal gaps. First optimal alignment in
    deterministic input order breaks ties. This is NOT local alignment identity.
    Connected components are single-linkage: endpoints need not be98% identical.
    """
    from Bio.Align import PairwiseAligner
    aligner=PairwiseAligner(mode='global',match_score=2,mismatch_score=-1,
                          open_gap_score=-5,extend_gap_score=-.5)
    seqs=sorted(df.sequence_key.unique());parents=list(range(len(seqs)));edges=[]
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):parents[find(j)]=find(i)
    for i,a in enumerate(seqs):
        for j in range(i+1,len(seqs)):
            b=seqs[j]
            alignment=aligner.align(a,b)[0]
            coords=alignment.coordinates;matches=0;columns=0
            for k in range(coords.shape[1]-1):
                a0,a1=map(int,coords[0,k:k+2]);b0,b1=map(int,coords[1,k:k+2])
                da,db=a1-a0,b1-b0;columns+=max(da,db)
                if da and db:
                    if da!=db:raise RuntimeError('Unexpected alignment coordinate block')
                    matches+=sum(x==y for x,y in zip(a[a0:a1],b[b0:b1]))
            identity=matches/columns
            edge=100*matches>=98*columns
            edges.append(dict(sequence_a=i,sequence_b=j,matches=matches,
                              alignment_columns=columns,identity=identity,edge98=edge))
            if edge:union(i,j)
    # Conservative identity aliases are already required to have equal sequences.
    lookup={s:find(i) for i,s in enumerate(seqs)}
    df['identity98_group']=df.sequence_key.map(lookup)
    nodes=pd.DataFrame([dict(sequence_index=i,sequence=s,component=find(i)) for i,s in enumerate(seqs)])
    return nodes,pd.DataFrame(edges)


def groups(df, mode):
    if mode=='homology':
        return df.identity98_group.to_numpy()
    # Connected components prevent identical sequences from bridging parent groups.
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for column in ['Enzyme', 'sequence_key'] + (['parent'] if mode == 'parent' else (['identity98_group'] if mode=='homology' else [])):
        seen = {}
        for i, value in enumerate(df[column]):
            if value in seen:
                parent[find(i)] = find(seen[value])
            else:
                seen[value] = i
    return np.asarray([find(i) for i in range(len(df))])


def partition(indices, group, fraction, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(group[indices]); rng.shuffle(unique)
    if len(unique) < 2:
        raise ValueError('Fewer than two independent groups for inner split')
    n = min(len(unique)-1, max(1, int(np.ceil(fraction*len(unique)))))
    hold = np.isin(group[indices], unique[:n])
    return indices[~hold], indices[hold]


def cluster_holdout(df, indices, g, fraction, seed):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for _, part in df.iloc[indices].groupby('Cluster', sort=True):
        ids = part.index.to_numpy(); unique = np.unique(g[ids]); rng.shuffle(unique)
        count = min(len(unique)-1, max(1, round(len(unique)*fraction))) if len(unique)>1 else 0
        isval = np.isin(g[ids], unique[:count])
        train.extend(ids[~isval]);val.extend(ids[isval])
    if not val: raise ValueError('No inner validation groups: all remaining clusters are singletons')
    return np.asarray(train,dtype=int),np.asarray(val,dtype=int)


def cv_splits(df, mode, folds, seed):
    g = groups(df, mode); unique = np.unique(g)
    if len(unique) < folds:
        raise ValueError(f'{len(unique)} groups cannot support {folds} folds')
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(folds)]; sizes = np.zeros(folds, int)
    if mode == 'cluster':
        if df.Cluster.isna().any(): raise ValueError('Cluster contains missing values')
        for value in unique:
            if df.loc[g==value,'Cluster'].nunique()!=1:
                raise ValueError('Identical sequence group spans multiple Cluster labels')
        for _, part in df.groupby('Cluster',sort=True):
            values = np.unique(g[part.index]);rng.shuffle(values)
            cluster_sizes = np.zeros(folds,int)
            for value in sorted(values,key=lambda v:-int((g==v).sum())):
                choices=np.flatnonzero(cluster_sizes==cluster_sizes.min())
                choices=choices[sizes[choices]==sizes[choices].min()]
                k=int(rng.choice(choices));count=int((g==value).sum())
                buckets[k].append(value);sizes[k]+=count;cluster_sizes[k]+=count
    else:
        rng.shuffle(unique)
        for value in sorted(unique,key=lambda v:-int((g==v).sum())):
            choices=np.flatnonzero(sizes==sizes.min());k=int(rng.choice(choices))
            buckets[k].append(value);sizes[k]+=(g==value).sum()
    for fold, bucket in enumerate(buckets):
        test=np.flatnonzero(np.isin(g,bucket));remain=np.flatnonzero(~np.isin(g,bucket))
        try:
            if mode=='cluster':train,val=cluster_holdout(df,remain,g,.15,seed+fold+1)
            else:train,val=partition(remain,g,.15,seed+fold+1)
        except ValueError as exc:
            # Preserve later feasible folds rather than aborting the generator.
            yield fold, None, None, test, str(exc)
            continue
        yield fold,train,val,test,None


def audit(df, splits, mode):
    train, val, test = splits
    if any(len(x) == 0 for x in splits):
        raise ValueError('Empty train/validation/test partition')
    for field in ['Enzyme', 'sequence_key'] + (['parent'] if mode == 'parent' else (['identity98_group'] if mode=='homology' else [])):
        sets = [set(df.iloc[x][field]) for x in splits]
        if any(sets[i] & sets[j] for i,j in [(0,1),(0,2),(1,2)]):
            raise ValueError(f'{field} crosses partitions')
    return {'n_train':len(train),'n_val':len(val),'n_test':len(test),
            'train_parents':df.iloc[train].parent.nunique(),
            'test_parents':df.iloc[test].parent.nunique(),
            'parent_overlap':len(set(df.iloc[train].parent)&set(df.iloc[test].parent))}


def fit_predict(df, labels, train, val, query, number, seed, args, checkpoint_dir):
    import torch
    from torch import nn
    checkpoint_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    torch.use_deterministic_algorithms(True)
    x = torch.tensor([[AA.index(a) for a in s] for s in df.Aligned_Sequence],dtype=torch.long)
    q = torch.tensor([[AA.index(a) for a in s] for s in query.Aligned_Sequence],dtype=torch.long)
    y = df[labels].to_numpy(float)
    counts = np.isfinite(y[train]).sum(axis=0)
    active = counts > 0
    scale = np.array([max(float(np.nanmax(np.abs(y[train,j]))),1e-12) if active[j] else 1.
                      for j in range(len(labels))])
    mask = np.isfinite(y) & active
    target = torch.tensor(np.where(mask, y/scale, 0.), dtype=torch.float32)
    mask = torch.tensor(mask)
    if not mask[val].any():
        raise ValueError('No observed validation targets supported by training')
    weights = torch.tensor(([.2,1,1,.1] if number == 1 else [1,1,1,.1])*(len(labels)//4), device=args.device)
    width = args.hidden_width or {1:2000,2:65,3:65}[number]
    predictions, history = [], []
    for member in range(args.ensemble_size):
        member_seed = seed + member
        torch.manual_seed(member_seed); np.random.seed(member_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(member_seed)
        model = nn.Sequential(nn.Embedding(21,16), nn.Flatten(), nn.Linear(x.shape[1]*16,width),
                              nn.Dropout(.2), nn.ReLU(), nn.Linear(width,len(labels))).to(args.device)
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=.001)
        generator = torch.Generator().manual_seed(member_seed)
        best, state, stale, best_epoch = float('inf'), None, 0, 0
        for epoch in range(args.epochs):
            model.train()
            order = torch.tensor(train)[torch.randperm(len(train), generator=generator)]
            for idx in order.split(args.batch_size):
                observed = mask[idx].to(args.device)
                if not observed.any(): continue
                pred = model(x[idx].to(args.device))
                loss = (((pred-target[idx].to(args.device))**2)*weights*observed).sum()/observed.sum()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                observed = mask[val].to(args.device)
                vloss = ((((model(x[val].to(args.device))-target[val].to(args.device))**2)*weights*observed).sum()/observed.sum()).item()
            if vloss < best:
                best, stale, best_epoch = vloss, 0, epoch+1
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else: stale += 1
            if stale >= args.patience: break
        if state is None: raise RuntimeError('No finite validation checkpoint')
        checkpoint_path=checkpoint_dir/f'member_{member:02d}.pt'
        checkpoint={'state_dict':state,'architecture':'Embedding-Flatten-Linear-Dropout-ReLU-Linear',
                    'alphabet':AA,'embedding_dim':16,'sequence_length':int(x.shape[1]),
                    'hidden_width':width,'dropout':.2,'labels':list(labels),
                    'target_scale':scale.tolist(),'active_heads':active.tolist(),
                    'training_observation_counts':counts.tolist(),'snapshot':number,
                    'seed':member_seed,'member':member,'best_epoch':best_epoch,
                    'validation_loss':best,'learning_rate':args.learning_rate,'weight_decay':.001,
                    'label_weights':weights.detach().cpu().tolist(),
                    'training_row_ids':df.iloc[train].row_id.tolist(),
                    'validation_row_ids':df.iloc[val].row_id.tolist(),
                    'ensemble_aggregation':'median','prediction_units':'normalized; multiply by target_scale'}
        # Exclusive create: never replace a previously saved member.
        with checkpoint_path.open('xb') as handle: torch.save(checkpoint,handle)
        model.load_state_dict(state); model.eval()
        with torch.no_grad():
            pred = np.concatenate([model(b.to(args.device)).cpu().numpy() for b in q.split(args.batch_size)])*scale
        pred[:,~active] = np.nan
        predictions.append(pred)
        history.append({'member':member,'seed':member_seed,'best_epoch':best_epoch,'validation_loss':best,'checkpoint':str(checkpoint_path)})
    return np.median(predictions,axis=0), history, counts, scale


def metrics(y, pred):
    from scipy.stats import pearsonr, spearmanr
    good = np.isfinite(y)&np.isfinite(pred); y=y[good]; pred=pred[good]; n=len(y)
    r = {'n':n,'pearson':np.nan,'spearman':np.nan,'rmse':np.nan,'mae':np.nan,'r2':np.nan,'top10_hit_enrichment':np.nan,'mse':np.nan}
    if not n: return r
    r['mse']=float(np.mean((y-pred)**2))
    r.update(rmse=float(np.sqrt(np.mean((y-pred)**2))),mae=float(np.mean(np.abs(y-pred))))
    if n>=2 and np.ptp(y)>0: r['r2']=float(1-((y-pred)**2).sum()/((y-y.mean())**2).sum())
    if n>=3 and np.ptp(y)>0 and np.ptp(pred)>0:
        r['pearson']=float(pearsonr(y,pred)[0]);r['spearman']=float(spearmanr(y,pred)[0])
    # Operational hit: top observed decile WITHIN test cohort, not significance/WT.
    hit=y>=np.quantile(y,.9); k=min(10,n); threshold=np.sort(pred)[-k]
    above=pred>threshold; tied=pred==threshold
    w=above.astype(float);w[tied]=(k-above.sum())/tied.sum()
    r['top10_hit_enrichment']=float((w@hit/k)/hit.mean());r['effective_k']=k
    return r


def record(pred, query, labels, context, rows, metric_rows):
    for j,label in enumerate(labels):
        actual=query[label].to_numpy(float)
        valid=np.isfinite(actual)&np.isfinite(pred[:,j])
        metric_rows.append(dict(context,target=label,n_observed=int(np.isfinite(actual).sum()),
                                coverage=float(valid.sum()/max(1,np.isfinite(actual).sum())),**metrics(actual,pred[:,j])))
        for i,row in enumerate(query.itertuples()):
            rows.append(dict(context,Enzyme=row.Enzyme,row_id=row.row_id,original_id=row.original_id,parent=row.parent,target=label,
                             observed=actual[i],prediction=pred[i,j]))


BASE_RATIOS = ['DVA/C4','DVA/DVO','DVA/(DVA+DVO)','(DVA+DVO)/VDAL',
               'OA/C6','OA/OL','OA/(OA+OL)','(OA+OL)/PDAL']


def top10_recall(subset, actual='actual_relative', predicted='predicted_relative'):
    """Unique-sequence top10 overlap, with independent uniform boundary ties."""
    finite=subset[np.isfinite(subset[actual])&np.isfinite(subset[predicted])]
    # One value per sequence/head, then equal-weight available heads per sequence.
    units=finite.groupby(['sequence_key','target'])[[actual,predicted]].mean().groupby('sequence_key').mean()
    n=len(units);k=min(10,n)
    if not n:return {'top10_recall':np.nan,'recall_n_sequences':0,'recall_effective_k':0,'recall_trivial':True}
    def membership(x):
        boundary=np.sort(x)[-k];w=(x>boundary).astype(float);tie=x==boundary
        w[tie]=(k-w.sum())/tie.sum();return w
    a=membership(units[actual].to_numpy());b=membership(units[predicted].to_numpy())
    return {'top10_recall':float(a@b/k),'recall_n_sequences':n,'recall_effective_k':k,'recall_trivial':n<=10}


def compiled_reports(predictions, snapshots, out, reference=None):
    """Test-only, per-seed OOF reference-style WT-relative ratio comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    within=predictions[predictions.experiment=='within_snapshot'].copy()
    if within.empty:return
    keys=['round','protocol','repeat','row_id','target']
    if within.duplicated(keys).any():
        raise ValueError('Duplicate OOF predictions within a seed; refusing pooled correlations')
    baselines={}; baseline_rows=[]
    for number,(df,labels,_,_) in snapshots.items():
        wt=df[df.Enzyme=='PKC001.000']
        if len(wt)!=1:raise ValueError(f'Expected exactly one PKC001.000 in snapshot{number}')
        for label in labels:
            value=float(wt.iloc[0][label]);usable=np.isfinite(value) and value!=0
            baselines[number,label]=value if usable else np.nan
            baseline_rows.append(dict(round=number,target=label,wt_raw=value,usable=usable))
    pd.DataFrame(baseline_rows).to_csv(out/'compiled_wt_baselines.csv',index=False)
    sequence_lookup={(n,r.row_id):r.sequence_key for n,(df,_,_,_) in snapshots.items() for r in df.itertuples()}
    within['sequence_key']=[sequence_lookup[int(n),rid] for n,rid in zip(within['round'],within.row_id)]
    if reference is not None:
        within=within.merge(reference[['round','row_id','target','prediction']].rename(columns={'prediction':'reference_prediction'}),on=['round','row_id','target'],how='left',validate='many_to_one')
        needed=np.isfinite(within.observed)&np.isfinite(within.prediction)
        if not np.isfinite(within.loc[needed,'reference_prediction']).all():raise ValueError('Reference must cover EVERY finite OOF observation; no silent cohort shrinkage')
    within['base_ratio']=within.target.str.split('_',n=1).str[1]
    within['wt_baseline']=[baselines[int(r),t] for r,t in zip(within['round'],within.target)]
    within['actual_relative']=within.observed/within.wt_baseline
    within['predicted_relative']=within.prediction/within.wt_baseline
    within.to_csv(out/'compiled_oof_points.csv',index=False)
    result=[];reference_rows=[]
    for (number,protocol,repeat),part in within.groupby(['round','protocol','repeat']):
        df,labels,_,_=snapshots[number]
        fig,axes=plt.subplots(1,8,figsize=(19,3.7),squeeze=False)
        for j,base in enumerate(BASE_RATIOS):
            subset=part[part.base_ratio==base]
            x=subset.actual_relative.to_numpy(float);y=subset.predicted_relative.to_numpy(float)
            good=np.isfinite(x)&np.isfinite(y)
            relevant=[l for l in labels if l.split('_',1)[1]==base]
            valid_heads=[l for l in relevant if np.isfinite(baselines[number,l])]
            expected_raw=int(np.isfinite(df[relevant].to_numpy(float)).sum())
            expected_relative=int(np.isfinite(df[valid_heads].to_numpy(float)).sum())
            context=dict(round=number,protocol=protocol,repeat=repeat,base_ratio=base,
                         seed=int(part.split_seed.iloc[0]))
            stats=metrics(x,y)
            stats.update(top10_recall(subset))
            if reference is not None:
                refpart=subset.loc[good].copy()
                refpart['reference_relative']=refpart.reference_prediction/refpart.wt_baseline
                refstats=metrics(refpart.actual_relative.to_numpy(float),refpart.reference_relative.to_numpy(float))
                refstats.update(top10_recall(refpart,predicted='reference_relative'))
                reference_rows.append(dict(context,interpretation='IN_SAMPLE_NOT_GENERALIZATION_NOT_CEILING',**refstats))
            result.append(dict(context,scope='pooled_oof',fold=-1,expected_raw=expected_raw,
                               expected_wt_relative=expected_relative,
                               coverage=stats['n']/expected_relative if expected_relative else np.nan,
                               n_unique_enzymes=subset.loc[good,'Enzyme'].nunique(),**stats))
            for fold,foldpart in subset.groupby('fold'):
                result.append(dict(context,scope='fold',fold=int(fold),
                                   n_unique_enzymes=foldpart.loc[np.isfinite(foldpart.actual_relative)&np.isfinite(foldpart.predicted_relative),'Enzyme'].nunique(),
                                   **metrics(foldpart.actual_relative.to_numpy(float),foldpart.predicted_relative.to_numpy(float))))
            ax=axes[0,j];color='#c51b7d' if j<4 else '#762a83'
            ax.scatter(x[good],y[good],s=9,color=color,alpha=.65)
            # Unlike reference fixed0..2, retain outliers rather than silently clipping.
            if good.any():
                low=min(0.,float(min(x[good].min(),y[good].min())))
                high=max(2.,float(max(x[good].max(),y[good].max())))
            else:low,high=0.,2.
            pad=(high-low)*.04
            ax.plot([low,high],[low,high],color='grey',lw=.8)
            ax.set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad),title=base,xlabel='Actual / WT')
            if j==0:ax.set_ylabel('Predicted / WT')
            ax.text(.04,.96,f"n={stats['n']}\nMSE={stats['mse']:.2f}\nR={stats['pearson']:.2f}\nRho={stats['spearman']:.2f}",
                    transform=ax.transAxes,va='top',fontsize=8)
            for spine in ax.spines.values():spine.set_edgecolor(color)
        fig.suptitle(f'Round{number} | {protocol} | seed{int(part.split_seed.iloc[0])} | held-out OOF only')
        fig.tight_layout(rect=(0,0,1,.93))
        for ext in ['png','svg']:
            fig.savefig(out/f'compiled_round_{number}_{protocol}_seed_{int(part.split_seed.iloc[0])}_EnsMLP_Test.{ext}',dpi=300)
        plt.close(fig)
    compiled=pd.DataFrame(result);compiled.to_csv(out/'compiled_ratio_metrics.csv',index=False)
    pooled=compiled[compiled.scope=='pooled_oof']
    pd.DataFrame(reference_rows).to_csv(out/'in_sample_reference_metrics.csv',index=False)
    summary=[]
    for metric in ['pearson','spearman','top10_recall']:
        fig,axes=plt.subplots(2,4,figsize=(14,7),squeeze=False)
        for ax,base in zip(axes.flat,BASE_RATIOS):
            for offset,(protocol,color) in enumerate([('cluster','#4477AA'),('random','#AA3377'),('homology','#228833')]):
                for number in [1,2,3]:
                    values=pooled.loc[(pooled.base_ratio==base)&(pooled.protocol==protocol)&(pooled['round']==number),metric].dropna()
                    if len(values):
                        x=number+(offset-1)*.2
                        ax.scatter(x+np.linspace(-.035,.035,len(values)),values,s=20,color=color,alpha=.7)
                        sd=float(values.std(ddof=1)) if len(values)>1 else np.nan
                        ax.errorbar(x,values.mean(),yerr=sd if np.isfinite(sd) else None,fmt='_',markersize=13,color=color,capsize=4)
                        clipped=int(((values<0)|(values>1)).sum())
                        if clipped:ax.scatter(x,0,marker='v',color=color,s=35,clip_on=False)
                        trivial=int(pooled.loc[(pooled.base_ratio==base)&(pooled.protocol==protocol)&(pooled['round']==number),'recall_trivial'].fillna(False).sum()) if metric=='top10_recall' else 0
                        if trivial:ax.scatter(x,values.mean(),marker='D',facecolors='none',edgecolors=color,s=65)
                        summary.append(dict(trivial_recall_seeds=trivial,metric=metric,base_ratio=base,protocol=protocol,round=number,n_seeds=len(values),mean=float(values.mean()),sample_sd=sd,clipped_seed_values=clipped))
                        if reference_rows:
                            refs=pd.DataFrame(reference_rows)
                            ref=refs.loc[(refs.base_ratio==base)&(refs.protocol==protocol)&(refs['round']==number),metric].dropna()
                            if len(ref):
                                ax.plot([x-.075,x+.075],[ref.mean(),ref.mean()],ls='--',color='black',lw=1.5)
                                summary[-1]['clipped_reference_values']=int(((ref<0)|(ref>1)).sum())
                                if (ref<0).any():ax.scatter(x,0,marker='v',color='black',s=40,clip_on=False)
            ax.set(title=base,xticks=[1,2,3],xlabel='Model snapshot',ylabel=metric.title(),ylim=(0,1))
            ax.axhline(0,color='grey',lw=.6)
        handles=[plt.Line2D([],[],marker='o',ls='',color=c,label=n) for n,c in
                 [('cluster','#4477AA'),('random','#AA3377'),('homology','#228833')]]
        if reference_rows:handles.append(plt.Line2D([],[],ls='--',color='black',label='All-data in-sample reference (not a ceiling)'))
        fig.legend(handles=handles,loc='lower center',ncol=2 if reference_rows else 3,frameon=False)
        fig.suptitle('Points: seed OOF metrics; error bars: mean ± sample SD across seeds\n0–1 display clips negative values/whiskers; triangles flag negatives; hollow diamonds: trivial recall (n≤10)')
        fig.tight_layout(rect=(0,.055,1,.95))
        for ext in ['png','svg']:fig.savefig(out/f'compiled_ratio_{metric}_comparison.{ext}',dpi=300)
        plt.close(fig)
    pd.DataFrame(summary).to_csv(out/'compiled_metric_seed_summary.csv',index=False)


def train_all_data_reference(snapshots,out,args):
    """Explicit opt-in fixed-epoch training on every labeled source row.

    No OOF/test result selects epochs, architecture, member or checkpoint. This
    intentionally measures resubstitution fit, never generalization/upper bound.
    """
    import torch
    from torch import nn
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if args.baseline_device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    torch.set_num_threads(args.baseline_cpu_threads);torch.use_deterministic_algorithms(True)
    rows=[];paths=[];hashes={}
    for number,(df,labels,_,sources) in snapshots.items():
        y=df[labels].to_numpy(float);mask=np.isfinite(y);active=mask.any(axis=0)
        scale=np.array([max(np.nanmax(np.abs(y[:,j])),1e-12) if active[j] else 1. for j in range(len(labels))])
        targets=torch.tensor(np.where(mask,y/scale,0),dtype=torch.float32)
        observed=torch.tensor(mask);x=torch.tensor([[AA.index(a) for a in seq] for seq in df.Aligned_Sequence],dtype=torch.long)
        weights=torch.tensor(([.2,1,1,.1] if number==1 else [1,1,1,.1])*(len(labels)//4),device=args.baseline_device)
        width=args.source_hyperparams.get('hidden_width') or {1:2000,2:65,3:65}[number]
        learning_rate=float(args.source_hyperparams.get('learning_rate',5e-6));batch_size=int(args.source_hyperparams.get('batch_size',32))
        member_predictions=[]
        for member in range(args.baseline_members):
            seed=args.baseline_seed+1000*number+member;torch.manual_seed(seed)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
            model=nn.Sequential(nn.Embedding(21,16),nn.Flatten(),nn.Linear(x.shape[1]*16,width),nn.Dropout(.2),nn.ReLU(),nn.Linear(width,len(labels))).to(args.baseline_device)
            optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=.001)
            generator=torch.Generator().manual_seed(seed)
            print(f'OPT-IN all-data IN-SAMPLE model: round{number} member{member+1}/{args.baseline_members}',flush=True)
            for epoch in range(args.baseline_epochs):
                model.train()
                for idx in torch.randperm(len(df),generator=generator).split(batch_size):
                    valid=observed[idx].to(args.baseline_device)
                    if not valid.any():continue
                    loss=(((model(x[idx].to(args.baseline_device))-targets[idx].to(args.baseline_device))**2)*weights*valid).sum()/valid.sum()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite all-data loss')
                    optimizer.zero_grad();loss.backward();optimizer.step()
            model.eval()
            with torch.no_grad():prediction=np.concatenate([model(b.to(args.baseline_device)).cpu().numpy() for b in x.split(batch_size)])*scale
            prediction[:,~active]=np.nan;member_predictions.append(prediction)
            path=out/'all_data_models'/f'round{number}_member{member:02d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
            checkpoint={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'training_scope':'all_data',
                        'epochs':args.baseline_epochs,'learning_rate':learning_rate,'batch_size':batch_size,'seed':seed,'labels':labels,'target_scale':scale.tolist(),
                        'active_heads':active.tolist(),'hidden_width':width,'sequence_length':int(x.shape[1]),
                        'training_row_ids':df.row_id.tolist(),'source_snapshot_sha256':hashlib.sha256(sources[0].read_bytes()).hexdigest()}
            with path.open('xb') as handle:torch.save(checkpoint,handle)
            paths.append(str(path));del model
        pred=np.median(member_predictions,axis=0)
        for i,row in enumerate(df.itertuples()):
            rows.extend(dict(round=number,row_id=row.row_id,target=label,prediction=pred[i,j]) for j,label in enumerate(labels))
        hashes[str(number)]=hashlib.sha256(sources[0].read_bytes()).hexdigest()
        pd.DataFrame(rows).to_csv(out/'all_data_predictions.csv',index=False)
    meta={'training_scope':'all_data','checkpoint_paths':paths,'source_snapshot_sha256':hashes,
          'source_training_hyperparameters':args.source_hyperparams,'epochs_fixed_in_advance':args.baseline_epochs,'ensemble_members':args.baseline_members,'aggregation':'median',
          'interpretation':'resubstitution/in-sample reference, NOT a ceiling or generalization'}
    (out/'all_data_provenance.json').write_text(json.dumps(meta,indent=2))
    return pd.DataFrame(rows),meta


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--reuse-run',type=Path,required=True,help='Completed run containing predictions.csv and provenance.json')
    p.add_argument('--output-dir',type=Path,help='Output base; new report subfolder, never overwrite source')
    p.add_argument('--in-sample-predictions',type=Path,help='Optional RAW-ratio predictions CSV from an already trained all-data model')
    p.add_argument('--in-sample-provenance',type=Path,help='Required with baseline CSV: JSON training_scope=all_data; checkpoint_paths; source_snapshot_sha256 mapping round numbers to hashes')
    p.add_argument('--train-all-data-baseline',action='store_true',help='Explicitly train NEW all-data in-sample reference; never enabled by reuse-run alone')
    p.add_argument('--baseline-epochs',type=int,help='Required fixed epoch budget when all-data training explicitly requested; no OOF tuning or early stopping')
    p.add_argument('--baseline-members',type=int,default=10)
    p.add_argument('--baseline-seed',type=int,default=777)
    p.add_argument('--baseline-device',default='cuda:0')
    p.add_argument('--baseline-cpu-threads',type=int,default=4)
    args=p.parse_args()
    if args.train_all_data_baseline and args.in_sample_predictions:p.error('Choose supplied baseline OR explicit all-data training, not both')
    if args.train_all_data_baseline and (args.baseline_epochs is None or args.baseline_epochs<1 or args.baseline_members<1 or args.baseline_cpu_threads<1):p.error('Explicit all-data training requires positive --baseline-epochs, member and thread counts')
    if bool(args.in_sample_predictions)!=bool(args.in_sample_provenance):p.error('Baseline predictions and provenance must be supplied together')
    source=args.reuse_run
    provenance=json.loads((source/'provenance.json').read_text())
    args.source_hyperparams=provenance.get('arguments',{})
    snapshots={n:load_snapshot(args.root,n) for n in [1,2,3]}
    # Prevent mixing saved OOF predictions with modified assay tables.
    for n,item in snapshots.items():
        snapshot=item[3][0];actual=hashlib.sha256(snapshot.read_bytes()).hexdigest()
        matches=[v for k,v in provenance.get('sha256',{}).items() if Path(k).name==snapshot.name]
        if not matches or any(x!=actual for x in matches):raise ValueError(f'Snapshot{n} hash does not match completed run provenance')
    predictions=pd.read_csv(source/'predictions.csv')
    required={'experiment','round','protocol','repeat','fold','split_seed','row_id','Enzyme','target','observed','prediction'}
    if not required.issubset(predictions):raise ValueError(f'Missing saved prediction columns: {required-set(predictions)}')
    for n,(df,labels,_,_) in snapshots.items():
        truth=df.set_index('row_id')
        part=predictions[(predictions.experiment=='within_snapshot')&(predictions['round']==n)]
        for row in part.itertuples():
            if row.target not in labels or row.row_id not in truth.index:raise ValueError('Prediction row/head absent from source snapshot')
            if not np.isclose(row.observed,truth.loc[row.row_id,row.target],equal_nan=True):raise ValueError('Saved observed target differs from source')
    audit=pd.read_csv(source/'split_audit.csv')
    split_table=pd.read_csv(source/'splits.csv')
    audit=audit[audit.experiment=='within_snapshot']
    if not audit.status.isin(['completed','skipped']).all():raise ValueError('Source run has incomplete/planned CV jobs; refusing partial OOF report')
    jobkeys=['round','protocol','repeat','fold']
    if audit.duplicated(jobkeys).any():raise ValueError('Ambiguous source split audit jobs')
    expected=set()
    for job in audit[audit.status=='completed'].itertuples():
        selected=split_table[(split_table.experiment=='within_snapshot')&(split_table['round']==job.round)&(split_table.protocol==job.protocol)&(split_table.repeat==job.repeat)&(split_table.fold==job.fold)&(split_table.role=='test')]
        if selected.empty:raise ValueError('Completed job lacks saved test membership')
        for rid in selected.row_id:
            expected.update((job.round,job.protocol,job.repeat,job.fold,rid,label) for label in snapshots[job.round][1])
    actual_predictions=predictions[predictions.experiment=='within_snapshot']
    actual=set(actual_predictions[jobkeys+['row_id','target']].itertuples(index=False,name=None))
    if actual!=expected or len(actual)!=len(actual_predictions):raise ValueError('Saved predictions do not exactly cover completed-fold test rows/heads')
    reference=None;baseline_meta=None
    if args.in_sample_predictions:
        baseline_meta=json.loads(args.in_sample_provenance.read_text())
        if baseline_meta.get('training_scope')!='all_data' or not baseline_meta.get('checkpoint_paths'):raise ValueError('Need documented already-trained ALL-DATA reference; original held-out ensembles are not an all-data model')
        for n,item in snapshots.items():
            digest=hashlib.sha256(item[3][0].read_bytes()).hexdigest()
            if baseline_meta.get('source_snapshot_sha256',{}).get(str(n))!=digest:raise ValueError('All-data reference snapshot provenance mismatch')
        reference=pd.read_csv(args.in_sample_predictions)
        required={'round','row_id','target','prediction'}
        if not required.issubset(reference) or reference.duplicated(['round','row_id','target']).any():raise ValueError('Baseline requires one raw prediction per round,row_id,target')
    base=args.output_dir or args.root/'results/generalization-analysis'
    out=base/('report_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    if args.train_all_data_baseline:reference,baseline_meta=train_all_data_reference(snapshots,out,args)
    meta={'source_run':str(source),'source_prediction_sha256':hashlib.sha256((source/'predictions.csv').read_bytes()).hexdigest(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'training_executed':args.train_all_data_baseline,'inference_executed':args.train_all_data_baseline,
          'baseline_provenance':baseline_meta,'baseline_label':'in-sample reference, NOT generalization or theoretical ceiling',
          'recall_definition':'unique sequence; mean WT-relative ratio across observed heads; expected independent-tie top-k overlap/k; k=min(10,n); n<=10 trivial',
          'sd_definition':'sample SD(ddof=1) across seed-level pooled OOF metrics, not folds or ensemble members'}
    if reference is not None:meta['baseline_prediction_sha256']=hashlib.sha256((args.in_sample_predictions if args.in_sample_predictions else out/'all_data_predictions.csv').read_bytes()).hexdigest()
    (out/'report_provenance.json').write_text(json.dumps(meta,indent=2))
    compiled_reports(predictions,snapshots,out,reference)
    print(f'Reused existing OOF predictions. New all-data baseline trained: {args.train_all_data_baseline}. Reports: {out}',flush=True)


if __name__=='__main__':main()
