#!/usr/bin/env python3
"""Four-fold generalization analysis; no training without --run.

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


def compiled_reports(predictions, snapshots, out):
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
    within['base_ratio']=within.target.str.split('_',n=1).str[1]
    within['wt_baseline']=[baselines[int(r),t] for r,t in zip(within['round'],within.target)]
    within['actual_relative']=within.observed/within.wt_baseline
    within['predicted_relative']=within.prediction/within.wt_baseline
    within.to_csv(out/'compiled_oof_points.csv',index=False)
    result=[]
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
    for metric in ['pearson','spearman']:
        fig,axes=plt.subplots(2,4,figsize=(14,7),squeeze=False)
        for ax,base in zip(axes.flat,BASE_RATIOS):
            for offset,(protocol,color) in enumerate([('cluster','#4477AA'),('random','#AA3377'),('homology','#228833')]):
                for number in [1,2,3]:
                    values=pooled.loc[(pooled.base_ratio==base)&(pooled.protocol==protocol)&(pooled['round']==number),metric].dropna()
                    if len(values):
                        x=number+(offset-1)*.2
                        ax.scatter(x+np.linspace(-.035,.035,len(values)),values,s=20,color=color,alpha=.7)
                        ax.scatter(x,values.mean(),marker='_',s=250,color=color)
            ax.set(title=base,xticks=[1,2,3],xlabel='Model snapshot',ylabel=metric.title(),ylim=(-1.05,1.05))
            ax.axhline(0,color='grey',lw=.6)
        handles=[plt.Line2D([],[],marker='o',ls='',color=c,label=n) for n,c in
                 [('cluster','#4477AA'),('random','#AA3377'),('homology','#228833')]]
        fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False)
        fig.suptitle('WT-relative compiled ratios: points=seed OOF correlations; bars=seed means')
        fig.tight_layout(rect=(0,.055,1,.95))
        for ext in ['png','svg']:fig.savefig(out/f'compiled_ratio_{metric}_comparison.{ext}',dpi=300)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output-dir',type=Path,help='Output base; default ROOT/results/generalization-analysis. Each invocation creates a new run subfolder.')
    p.add_argument('--run',action='store_true',help='Explicitly enable model training')
    p.add_argument('--device',default='cuda');p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--folds',type=int,default=4);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--ensemble-size',type=int,default=10);p.add_argument('--epochs',type=int,default=10000)
    p.add_argument('--patience',type=int,default=200);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--learning-rate',type=float,default=5e-6);p.add_argument('--hidden-width',type=int)
    p.add_argument('--seed',type=int,default=777)
    p.add_argument('--protocols',nargs='+',choices=['cluster','random','homology','parent','forward'],default=['cluster','random','homology'])
    args=p.parse_args()
    if min(args.cpu_threads,args.folds,args.repeats,args.ensemble_size,args.epochs,args.patience,args.batch_size)<1 or args.folds<2:
        p.error('Positive sizes and at least two folds required')
    if args.hidden_width is not None and args.hidden_width<1:p.error('Hidden width must be positive')
    if args.learning_rate<=0:p.error('Learning rate must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    snapshots={n:load_snapshot(args.root,n) for n in [1,2,3]}
    if args.output_dir is None:args.output_dir=args.root/'results/generalization-analysis'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    run_name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8]
    out=args.output_dir/('run_'+run_name if args.run else 'plan_'+run_name)
    out.mkdir(exist_ok=False)
    print(f'All outputs and models will be saved under: {out}',flush=True)
    for number,(df,labels,_,_) in snapshots.items():
        nodes,edges=identity_components(df)
        nodes.to_csv(out/f'round_{number}_identity98_sequences.csv',index=False)
        edges.to_csv(out/f'round_{number}_identity98_pair_audit.csv',index=False)
        df[['row_id','original_id','Enzyme','identity98_group']].to_csv(out/f'round_{number}_identity98_membership.csv',index=False)
        df[df.Enzyme.duplicated(keep=False)].to_csv(out/f'round_{number}_identity_alias_measurements.csv',index=False)
    audit_rows=[];split_rows=[];prediction_rows=[];metric_rows=[];training_rows=[];coverage=[]
    paths=[p for snapshot in snapshots.values() for p in snapshot[3]]
    provenance={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'methodology':__doc__}
    provenance['python'] = platform.python_version()
    provenance['packages'] = {}
    for package in ['numpy','pandas','scipy','torch','matplotlib','biopython']:
        try: provenance['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: provenance['packages'][package] = 'not installed'
    provenance['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))

    jobs=[]
    def flush():
        for name,rows in [('split_audit',audit_rows),('splits',split_rows),('predictions',prediction_rows),
                          ('metrics',metric_rows),('training_history',training_rows),('target_coverage',coverage)]:
            pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)

    for n,(df,labels,original,_) in snapshots.items():
        for protocol in [x for x in args.protocols if x!='forward']:
            for repeat in range(args.repeats):
                seed=args.seed+10000*n+1000*repeat
                context={'experiment':'within_snapshot','round':n,'protocol':protocol,'repeat':repeat,'split_seed':seed}
                try:
                    for fold,tr,va,te,error in cv_splits(df,protocol,args.folds,seed):
                        ctx=dict(context,fold=fold,seed=seed+100*fold)
                        try:
                            if error:raise ValueError(error)
                            info=audit(df,(tr,va,te),protocol)
                        except ValueError as exc:
                            audit_rows.append(dict(ctx,status='skipped',reason=str(exc)));continue
                        row=dict(ctx,status='planned',**info);audit_rows.append(row)
                        for role,indices in zip(['train','validation','test'],[tr,va,te]):
                            for i in indices:
                                split_rows.append(dict(ctx,role=role,position=int(i),Enzyme=df.iloc[i].Enzyme,row_id=df.iloc[i].row_id,original_id=df.iloc[i].original_id,
                                                       parent=df.iloc[i].parent,Cluster=df.iloc[i].Cluster))
                        jobs.append((n,df,labels,tr,va,te,ctx,row))
                except ValueError as exc:audit_rows.append(dict(context,status='skipped',reason=str(exc)))
    flush()
    print(f'{len(jobs)} feasible CV ensembles x{args.ensemble_size} models; training={args.run}',flush=True)
    if args.run:
        for job_index,(n,df,labels,tr,va,te,ctx,status) in enumerate(jobs):
            print(f'CV ensemble {job_index+1}/{len(jobs)}: {ctx}',flush=True)
            try:pred,hist,counts,scale=fit_predict(df,labels,tr,va,df.iloc[te],n,ctx['seed'],args,out/'models'/f'round_{n}_{ctx["protocol"]}_repeat_{ctx["repeat"]}_fold_{ctx["fold"]}')
            except ValueError as exc:
                status.update(status='skipped',reason=str(exc));flush();continue
            training_rows.extend(dict(ctx,**h) for h in hist)
            coverage.extend(dict(ctx,target=l,training_observations=int(c),scale=float(sc)) for l,c,sc in zip(labels,counts,scale))
            record(pred,df.iloc[te],labels,ctx,prediction_rows,metric_rows)
            status['status']='completed';flush()
    if 'forward' in args.protocols:
        for n in [1,2]:
            source,source_labels,_,_=snapshots[n]; nxt,next_labels,_,_=snapshots[n+1]
            novel=~nxt.Enzyme.isin(source.Enzyme)&~nxt.sequence_key.isin(source.sequence_key)
            for mode in ['random','parent']:
                for repeat in range(args.repeats):
                    seed=args.seed+100000*n+1000*repeat
                    context={'experiment':f'forward_{n}_to_{n+1}','round':n+1,'protocol':mode,'repeat':repeat,'fold':0,'seed':seed}
                    try:
                        g=groups(nxt,mode)
                        _,test=partition(np.flatnonzero(novel),g,.5,seed)
                        # Lock all entities in selected test groups, including old relatives.
                        testgroups=set(g[test]); remaining=np.flatnonzero(~np.isin(g,list(testgroups)))
                        tr_next,va_next=partition(remaining,g,.15,seed+1)
                        audit(nxt,(tr_next,va_next,test),mode)
                        # Remove source identities/sequences and (parent mode) lineages held out.
                        banned=nxt.iloc[np.flatnonzero(np.isin(g,list(testgroups)))]
                        keep=~source.Enzyme.isin(banned.Enzyme)&~source.sequence_key.isin(banned.sequence_key)
                        if mode=='parent':keep &= ~source.parent.isin(banned.parent)
                        tr_src,va_src=partition(np.flatnonzero(keep),groups(source,mode),.15,seed+2)
                        if not len(test):raise ValueError('No novel test entities')
                    except ValueError as exc:
                        audit_rows.append(dict(context,status='skipped',reason=str(exc)));continue
                    query=nxt.iloc[test]
                    for arm,df,labels,tr,va,number in [('frozen_source',source,source_labels,tr_src,va_src,n),('retrained_next',nxt,next_labels,tr_next,va_next,n+1)]:
                        ctx=dict(context,arm=arm)
                        audit_rows.append(dict(ctx,status='planned' if not args.run else 'run',n_train=len(tr),n_val=len(va),n_test=len(test)))
                        for role,frame in [('train',df.iloc[tr]),('validation',df.iloc[va]),('test',query)]:
                            split_rows.extend(dict(ctx,role=role,Enzyme=r.Enzyme,parent=r.parent) for r in frame.itertuples())
                        if not args.run:continue
                        try:pred,hist,counts,scale=fit_predict(df,labels,tr,va,query,number,seed,args,out/'models'/f'forward_{n}_to_{n+1}_{mode}_repeat_{repeat}_{arm}')
                        except ValueError as exc:
                            audit_rows.append(dict(ctx,status='skipped',reason=str(exc)));continue
                        training_rows.extend(dict(ctx,**h) for h in hist)
                        coverage.extend(dict(ctx,target=l,training_observations=int(c),scale=float(s)) for l,c,s in zip(labels,counts,scale))
                        mapped=np.full((len(query),len(next_labels)),np.nan)
                        for j,label in enumerate(next_labels):
                            if label in labels:mapped[:,j]=pred[:,labels.index(label)]
                            elif arm=='frozen_source':
                                matches=[k for k,l in enumerate(labels) if l.split('_',1)[1]==label.split('_',1)[1] and counts[k]>0]
                                if matches:mapped[:,j]=pred[:,matches].mean(axis=1)
                        for kind,chosen in [('shared_head',[l for l in next_labels if l in source_labels]),
                                            ('new_screen_transfer',[l for l in next_labels if l not in source_labels and int(l.split('_',1)[0]) not in [8,9]]),
                                            ('new_screen_validation',[l for l in next_labels if l not in source_labels and int(l.split('_',1)[0]) in [8,9]])]:
                            record(mapped[:,[next_labels.index(l) for l in chosen]],query,chosen,dict(ctx,evaluation=kind),prediction_rows,metric_rows)
                        flush()
    for name,rows in [('split_audit',audit_rows),('splits',split_rows),('predictions',prediction_rows),('metrics',metric_rows),('training_history',training_rows),('target_coverage',coverage)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    if args.run and prediction_rows:
        predictions=pd.DataFrame(prediction_rows)
        compiled_reports(predictions,snapshots,out)
        forward=predictions[predictions.experiment.str.startswith('forward')]
        paired=[]
        keys=['experiment','round','protocol','repeat','fold','evaluation','target']
        if len(forward):
            for values,part in forward.groupby(keys):
                pivot=part.pivot(index='row_id',columns='arm',values='prediction')
                if not {'frozen_source','retrained_next'}.issubset(pivot):continue
                obs=part.groupby('row_id').observed.first().reindex(pivot.index)
                valid=np.isfinite(obs)&np.isfinite(pivot).all(axis=1)
                for arm in ['frozen_source','retrained_next']:
                    paired.append(dict(zip(keys,values),arm=arm,**metrics(obs[valid].to_numpy(),pivot.loc[valid,arm].to_numpy())))
        pd.DataFrame(paired).to_csv(out/'paired_forward_metrics.csv',index=False)
        table=pd.DataFrame(metric_rows)
        summary_keys=['experiment','round','protocol','repeat','fold']
        for extra in ['arm','evaluation']:
            if extra in table:table[extra]=table[extra].fillna('not_applicable');summary_keys.append(extra)
        macro=table.groupby(summary_keys,dropna=False).agg(spearman=('spearman','mean'),pearson=('pearson','mean'),rmse=('rmse','mean'),valid_spearman_heads=('spearman','count'),observations=('n','sum')).reset_index()
        macro.to_csv(out/'macro_metrics.csv',index=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        within=macro[macro.experiment=='within_snapshot']
        fig,ax=plt.subplots(figsize=(8,5))
        for offset,(protocol,color) in enumerate([('cluster','#4477AA'),('random','#AA3377'),('homology','#228833')]):
            part=within[within.protocol==protocol]
            for n in [1,2,3]:
                values=part[part['round']==n].spearman.dropna()
                if len(values):
                    x=n+(offset-1)*.2
                    ax.scatter(np.full(len(values),x),values,color=color,alpha=.45,s=15)
                    ax.scatter(x,values.mean(),color=color,marker='_',s=220,label=protocol if n==1 else None)
        ax.set(xlabel='Model availability snapshot',ylabel='Macro per-head Spearman',xticks=[1,2,3])
        ax.legend();fig.tight_layout()
        for ext in ['png','svg']:fig.savefig(out/f'within_snapshot_comparison.{ext}',dpi=300)
        plt.close(fig)
    print(f'Saved {"analysis" if args.run else "split plan only; no models trained"}: {out}')


if __name__=='__main__':main()
