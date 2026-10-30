#!/usr/bin/env python3
"""Leakage-aware MLP ablations. See scripts/revisions/README.md before inference.

This is a controlled reimplementation of the deposited embedding/MLP architecture,
not an evaluation of pretrained checkpoints. Outputs include exact split membership.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

AAS = 'ACDEFGHIKLMNPQRSTVWY-'
DEFAULT_ROUNDS = {'WT': 0, 'initial_library_1': 0, 'initial_library_2': 0,
                  'initial_library_3': 0, 'round_1': 1, 'round_2': 2,
                  'round_3_C4': 3, 'round_3_C6': 3}


def parent_id(name):
    m = re.match(r'^PKC(\d+)(?:\.|$)', str(name), flags=re.I)
    if not m:
        raise ValueError(f'Cannot derive parent from {name!r}; expected PKC###.variant')
    return f'PKC{int(m.group(1))}'


def variant_id(name):
    m = re.fullmatch(r'PKC(\d+)\.(\d+)', str(name), flags=re.I)
    return f'PKC{int(m.group(1))}.{int(m.group(2))}' if m else str(name)


def canonical_frame(df):
    df = df.rename(columns={'name': 'Enzyme', 'sequence': 'Aligned_Sequence'}).copy()
    for col in ['Enzyme', 'Aligned_Sequence']:
        if col not in df or df[col].isna().any():
            raise ValueError(f'{col} must be present and nonmissing')
    if not len(df):
        raise ValueError('Input table is empty')
    seqs = df.Aligned_Sequence.astype(str).str.upper()
    if seqs.str.len().nunique() != 1 or not seqs.str.len().iloc[0]:
        raise ValueError('Provide equal-length nonempty aligned protein sequences')
    bad = set(''.join(seqs)) - set(AAS)
    if bad:
        raise ValueError(f'Unsupported sequence characters: {sorted(bad)}')
    df['Aligned_Sequence'] = seqs
    df['parent'] = df.Enzyme.map(parent_id)
    df['variant_id'] = df.Enzyme.map(variant_id)
    df['bio_sequence'] = df['Sequence'].astype(str).str.upper() if 'Sequence' in df else seqs.str.replace('-', '', regex=False)
    df = df.reset_index(drop=True)
    df['row_id'] = np.arange(len(df))
    return df


def encode(seqs):
    lookup = {a: i for i, a in enumerate(AAS)}
    return np.asarray([[lookup[a] for a in seq] for seq in seqs], dtype=np.int64)


def identity_matrix(seqs):
    """Aligned identity, excluding joint gaps; gap-vs-residue is a mismatch."""
    a = encode(seqs)
    out = np.empty((len(a), len(a)), dtype=float)
    for i, s in enumerate(a):
        active = (a != 20) | (s != 20)
        out[i] = ((a == s) & active).sum(1) / np.maximum(active.sum(1), 1)
    return out


def make_groups(df, mode, identity=0.9):
    """Connected components ensure exact sequences never cross new CV splits."""
    n = len(df)
    roots = list(range(n))
    def find(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i
    def union(i, j):
        roots[find(i)] = find(j)
    lookup = {}
    for i, seq in enumerate(df.Aligned_Sequence):
        if seq in lookup:
            union(i, lookup[seq])
        lookup[seq] = i
    lookup = {}
    for i, seq in enumerate(df.bio_sequence):
        if seq in lookup:
            union(i, lookup[seq])
        lookup[seq] = i
    if mode == 'parent':
        lookup = {}
        for i, p in enumerate(df.parent):
            if p in lookup:
                union(i, lookup[p])
            lookup[p] = i
    if mode == 'identity':
        ids = identity_matrix(df.Aligned_Sequence)
        for i, j in zip(*np.where(np.triu(ids >= identity, k=1))):
            union(int(i), int(j))
    return np.asarray([find(i) for i in range(n)])


def group_folds(groups, folds, seed):
    """Randomized, size-balanced group folds; no target values are inspected."""
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < folds:
        raise ValueError(f'{folds} folds require >= {folds} groups; found {len(unique)}. '
                         'Reduce --folds or revise the prespecified split threshold.')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-counts[order], kind='stable')]
    assigned = [[] for _ in range(folds)]
    sizes = np.zeros(folds, dtype=int)
    for j in order:
        f = int(np.argmin(sizes))
        assigned[f].append(unique[j])
        sizes[f] += counts[j]
    idx = np.arange(len(groups))
    return [(idx[~np.isin(groups, test)], idx[np.isin(groups, test)]) for test in assigned]


def inner_split(indices, groups, seed, fraction=0.2):
    unique = np.unique(groups[indices])
    if len(unique) < 2:
        raise ValueError('Outer training partition has <2 groups; cannot form independent validation.')
    rng = np.random.default_rng(seed)
    counts = {g:int(np.sum(groups[indices] == g)) for g in unique}
    shuffled = rng.permutation(unique)
    order = sorted(shuffled,key=lambda g:-counts[g])
    target = len(indices)*fraction
    selected, size = [], 0
    for g in order:
        if abs(size+counts[g]-target) < abs(size-target) and len(selected)<len(unique)-1:
            selected.append(g)
            size += counts[g]
    if not selected:
        selected = [min(order,key=lambda g:abs(counts[g]-target))]
    val_groups = np.asarray(selected)

    mask = np.isin(groups[indices], val_groups)
    return indices[~mask], indices[mask]


def scale_from_train(y):
    """Max-absolute scaling from finite training labels only, never held-out rows."""
    finite = np.isfinite(y)
    count = finite.sum(0)
    scale = np.max(np.where(finite, np.abs(y), 0), axis=0)
    scale[scale == 0] = 1
    return scale, count


class MLP(nn.Module):
    def __init__(self, length, outputs, hidden=65, embedding=16, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(len(AAS), embedding)
        self.linear_1 = nn.Linear(length*embedding, hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden, outputs)
    def forward(self, x):
        return self.linear_2(torch.relu(self.dropout(self.linear_1(self.embed(x).flatten(1)))))


def masked_task_mse(pred, true):
    mask = torch.isfinite(true)
    per_target = ((pred - torch.nan_to_num(true))**2 * mask).sum(0) / mask.sum(0).clamp(min=1)
    active = mask.sum(0) > 0
    return per_target[active].mean() if active.any() else None


def fit_predict(x, y, train, val, test, args, seed):
    scale, count = scale_from_train(y[train])
    active = count >= args.min_labels
    pred = np.full((len(test), y.shape[1]), np.nan)
    if not active.any():
        return pred, {'status': 'no_targets_with_enough_training_labels', 'train_counts': count.tolist()}
    yt = torch.tensor(y[:, active]/scale[active], dtype=torch.float32)
    xt = torch.tensor(x, dtype=torch.long)
    torch.manual_seed(seed)
    model = MLP(x.shape[1], int(active.sum()), args.hidden, args.embedding, args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_loss, best_state, best_epoch, stalled = np.inf, None, 0, 0
    generator = torch.Generator().manual_seed(seed)
    has_val = bool(torch.isfinite(yt[val]).any())
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train), generator=generator).numpy()
        for offset in range(0, len(train), args.batch_size):
            batch = train[order[offset:offset+args.batch_size]]
            loss = masked_task_mse(model(xt[batch]), yt[batch])
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; inspect labels or lower learning rate')
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            loss = masked_task_mse(model(xt[val]), yt[val]) if has_val else None
        score = float(loss) if loss is not None else 0.0
        # With no measured validation labels, train all prespecified epochs.
        if not has_val or score < best_loss:
            best_loss, best_state, best_epoch, stalled = score, copy.deepcopy(model.state_dict()), epoch+1, 0
        else:
            stalled += 1
        if has_val and stalled >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred[:, active] = model(xt[test]).numpy()*scale[active]
    return pred, {'status': 'ok', 'best_epoch': best_epoch,
                  'validation_loss': best_loss if has_val else None,
                  'validation_labels_present': has_val,
                  'train_counts': count.tolist(), 'scale': scale.tolist(),
                  'parameters': sum(p.numel() for p in model.parameters())}


def metrics(true, pred):
    mask = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[mask], pred[mask]
    n = len(true)
    if not n:
        return {'n': 0, 'mae': None, 'rmse': None, 'r2': None, 'spearman': None}
    rho = float(spearmanr(true, pred).statistic) if n > 2 and np.ptp(true) and np.ptp(pred) else None
    return {'n': n, 'mae': float(mean_absolute_error(true,pred)),
            'rmse': float(np.sqrt(mean_squared_error(true,pred))),
            'r2': float(r2_score(true,pred)) if n > 1 and np.ptp(true) else None,
            'spearman': rho}


def split_audit(df, train, val, test):
    ids = identity_matrix(df.Aligned_Sequence)
    report = {}
    for name, a, b in [('train_test',train,test),('train_val',train,val),('val_test',val,test)]:
        near = ids[np.ix_(b,a)].max(1) if len(a) and len(b) else np.array([])
        report[name] = {'row_overlap': len(set(a)&set(b)),
            'sequence_overlap': len(set(df.Aligned_Sequence.iloc[a]) & set(df.Aligned_Sequence.iloc[b])),
            'biological_sequence_overlap': len(set(df.bio_sequence.iloc[a]) & set(df.bio_sequence.iloc[b])),
            'parent_overlap': sorted(set(df.parent.iloc[a])&set(df.parent.iloc[b])),
            'nearest_identity_mean': float(near.mean()) if len(near) else None,
            'nearest_identity_max': float(near.max()) if len(near) else None}
    return report


def target_columns(df):
    return [c for c in df if re.match(r'^\d+_', c) and '/' in c and not c.endswith('_std')]


def pool_assays(df):
    """Explicitly average assay means; does not estimate measurement reliability."""
    df = df.copy()
    groups = {}
    for c in target_columns(df):
        groups.setdefault(c.split('_',1)[1], []).append(c)
    for suffix, cols in groups.items():
        values = df[cols].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf,-1],np.nan)
        df[suffix] = values.mean(axis=1,skipna=True)
    df.attrs['targets'] = list(groups)
    return df


def read_table(path):
    return pd.read_pickle(path) if str(path).endswith('.pkl') else pd.read_csv(path)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/all_SeqFxnData.csv'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['random','parent','identity','temporal','saved'], default='parent')
    p.add_argument('--through-round', type=int, default=3)
    p.add_argument('--train-through', type=int, default=0)
    p.add_argument('--round-map', type=Path, help='JSON library -> integer round; replaces default mapping')
    p.add_argument('--training-snapshot', type=Path, help='Historical training labels for temporal mode')
    p.add_argument('--saved-data', type=Path)
    p.add_argument('--saved-splits', type=Path)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--identity', type=float, default=0.9)
    p.add_argument('--pool-assays', action='store_true', help='Explicitly mean-pool matching target suffixes across assay conditions; assumes comparable conditions')
    p.add_argument('--targets', help='Comma-separated exact target columns; default all ratio targets')
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--hidden', type=int, default=65)
    p.add_argument('--embedding', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--learning-rate', type=float, default=5e-6)
    p.add_argument('--weight-decay', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--ensemble', type=int, default=1)
    p.add_argument('--min-labels', type=int, default=3)
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args(argv)
    if a.through_round < 0 or a.train_through < 0:
        p.error('Round numbers must be nonnegative')
    if a.mode == 'temporal' and a.train_through >= a.through_round:
        p.error('--through-round must include the round after --train-through')
    if a.folds < 2:
        p.error('--folds must be at least 2')
    if min(a.epochs,a.patience,a.ensemble,a.batch_size,a.threads,a.min_labels,a.folds) < 1:
        p.error('Counts and thread settings must be positive')
    if not 0 < a.identity <= 1:
        p.error('--identity must be in (0,1]')
    if a.mode == 'saved' and (not a.saved_data or not a.saved_splits):
        p.error('saved mode requires --saved-data and --saved-splits')
    return a


def prepare(a):
    source_files = [a.data,Path(__file__)]
    round_map = json.loads(a.round_map.read_text()) if a.round_map else DEFAULT_ROUNDS
    if a.mode == 'saved':
        df = canonical_frame(read_table(a.saved_data)); source_files = [a.saved_data,a.saved_splits,Path(__file__)]
        raw = pickle.load(a.saved_splits.open('rb'))
        if len(raw) != 3:
            raise ValueError('Saved split must contain train, validation, test indices')
        train,val,test = [np.asarray(v,dtype=int) for v in raw]
        for v in [train,val,test]:
            if not len(v) or v.min() < 0 or v.max() >= len(df) or len(set(v)) != len(v):
                raise ValueError('Invalid or duplicate indices within saved partition')
        # Legacy partitions are retained exactly and flagged, never called leakage-free.
        if a.pool_assays:
            df = pool_assays(df)
        return df, [('legacy_saved',train,val,test)], source_files
    df = canonical_frame(read_table(a.data))
    if 'library' not in df:
        raise ValueError('Non-saved modes require a library column for round provenance')
    df['round'] = df.library.map(round_map)
    if df['round'].isna().any():
        raise ValueError(f'Unmapped libraries: {df.loc[df["round"].isna(),"library"].unique().tolist()}')
    df = df[df['round'] <= a.through_round].reset_index(drop=True)
    if a.mode == 'temporal':
        if a.train_through not in [0,1,2] and not a.training_snapshot:
            raise ValueError('Provide --training-snapshot for this historical stage')
        snapshot = a.training_snapshot or Path(f'models/round_{a.train_through+1}/round_{a.train_through+1}_seq_fxn_data.pkl')
        old = canonical_frame(read_table(snapshot)); source_files.append(snapshot)
        # Use the frozen snapshot AS IT IS. Several deposited historical sequences
        # differ from the later master CSV under the same name; do not transfer
        # labels across those sequence discrepancies or reconstitute old training
        # from the master CSV.
        test_frame = df[df['round'] == a.train_through+1].copy()
        if not len(test_frame):
            raise ValueError('No rows in the next experimental round')
        targets = target_columns(old)
        old['library'] = f'historical_snapshot_through_{a.train_through}'
        old['round'] = a.train_through
        # Preserve a per-ID audit of historical/master changes, including aliases.
        master_by_id = df.set_index('variant_id')
        audit_records = []
        for _, r in old.iterrows():
            if r.variant_id in master_by_id.index:
                current = master_by_id.loc[r.variant_id]
                if isinstance(current,pd.DataFrame):
                    raise ValueError('Master CSV has ambiguous canonical variant identifiers')
                audit_records.append({'variant_id':r.variant_id,'historical_name':r.Enzyme,
                    'master_name':current.Enzyme,'aligned_match':r.Aligned_Sequence==current.Aligned_Sequence,
                    'biological_match':r.bio_sequence==current.bio_sequence,
                    'historical_sequence':r.Aligned_Sequence,'master_sequence':current.Aligned_Sequence})
        test_seqs = set(test_frame.Aligned_Sequence)
        test_bio = set(test_frame.bio_sequence) | set(test_frame.Aligned_Sequence.str.replace('-','',regex=False))
        test_ids = set(test_frame.variant_id)
        purge = (old.Aligned_Sequence.isin(test_seqs) | old.bio_sequence.isin(test_bio)
                 | old.Aligned_Sequence.str.replace('-','',regex=False).isin(test_bio)
                 | old.variant_id.isin(test_ids))
        purged_ids = old.loc[purge,'Enzyme'].tolist()
        old = old[~purge].copy()
        if a.pool_assays:
            old = pool_assays(old)
            test_frame = pool_assays(test_frame)
            targets = old.attrs['targets']
        # Pool before concatenating: each side sees only its own recorded assays.
        n_old = len(old)
        df = pd.concat([old,test_frame],ignore_index=True)
        if not a.pool_assays:
            for c in target_columns(df):
                if c not in targets:
                    df = df.drop(columns=c)
        df['row_id'] = np.arange(len(df))
        df.attrs['targets'] = targets
        df.attrs['snapshot_audit'] = audit_records
        df.attrs['purged_historical_names'] = purged_ids
        train_rows = np.arange(n_old)
        test = np.arange(n_old,len(df))
        groups = make_groups(df,'random')
        train,val = inner_split(train_rows,groups,a.seed)
        return df,[(f'through_{a.train_through}_to_{a.train_through+1}',train,val,test)],source_files
    if a.pool_assays:
        df = pool_assays(df)
    groups = make_groups(df,a.mode,a.identity)
    folds = []
    for i,(trainval,test) in enumerate(group_folds(groups,a.folds,a.seed)):
        train,val = inner_split(trainval,groups,a.seed+i)
        folds.append((f'fold_{i}',train,val,test))
    return df,folds,source_files


def main(argv=None):
    a = arguments(argv)
    torch.set_num_threads(a.threads)
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError(f'Output directory is nonempty: {a.out}; choose a new directory to preserve earlier results')
    a.out.mkdir(parents=True,exist_ok=True)
    df,folds,files = prepare(a)
    targets = a.targets.split(',') if a.targets else df.attrs.get('targets',target_columns(df))
    if not targets or any(c not in df for c in targets):
        raise ValueError('No targets selected or a requested target is unavailable at this stage')
    y = df[targets].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
    y[~np.isfinite(y) | (y == -1)] = np.nan
    x = encode(df.Aligned_Sequence)
    manifest, predictions, scores, logs = [],[],[],[]
    for fold,train,val,test in folds:
        audit = split_audit(df,train,val,test)
        item = {'fold':fold,'train_rows':train.tolist(),'validation_rows':val.tolist(),
                'test_rows':test.tolist(),'audit':audit}
        if a.mode != 'saved':
            if any(v['row_overlap'] or v['sequence_overlap'] or v['biological_sequence_overlap'] for v in audit.values()):
                raise AssertionError('New split leaked exact rows or sequences')
        item['legacy_leakage_warning'] = a.mode == 'saved'
        manifest.append(item)
        if a.audit_only:
            continue
        for method in ['multi_task','single_task','train_mean']:
            all_preds = []
            if method == 'train_mean':
                count = np.isfinite(y[train]).sum(0)
                means = np.nansum(y[train],axis=0)/np.maximum(count,1)
                means[count<a.min_labels] = np.nan
                all_preds = [np.tile(means,(len(test),1))]
            else:
                for member in range(a.ensemble):
                    if method == 'multi_task':
                        pred,log = fit_predict(x,y,train,val,test,a,a.seed+member)
                        log.update(fold=fold,method=method,member=member)
                        logs.append(log)
                    else:
                        pred = np.full((len(test),len(targets)),np.nan)
                        for j,c in enumerate(targets):
                            one,log = fit_predict(x,y[:,[j]],train,val,test,a,a.seed+member)
                            pred[:,j] = one[:,0]
                            log.update(fold=fold,method=method,member=member,target=c)
                            logs.append(log)
                    all_preds.append(pred)
            stack = np.asarray(all_preds)
            valid = np.isfinite(stack).sum(0)
            pred = np.nansum(stack,axis=0)/np.maximum(valid,1)
            pred[valid==0] = np.nan
            for j,c in enumerate(targets):
                result = metrics(y[test,j],pred[:,j])
                result.update(fold=fold,method=method,target=c,
                              train_n=int(np.isfinite(y[train,j]).sum()),
                              validation_n=int(np.isfinite(y[val,j]).sum()))
                scores.append(result)
                for k,idx in enumerate(test):
                    predictions.append({'fold':fold,'method':method,'target':c,
                        'row_id':int(df.row_id.iloc[idx]),'Enzyme':df.Enzyme.iloc[idx],
                        'Aligned_Sequence':df.Aligned_Sequence.iloc[idx],
                        'library':df.library.iloc[idx] if 'library' in df else None,
                        'observed':y[idx,j], 'predicted':pred[k,j]})
        print(f'{fold}: train={len(train)} validation={len(val)} test={len(test)}',flush=True)
    pd.DataFrame(predictions).to_csv(a.out/'predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(a.out/'metrics.csv',index=False)
    row_table = df[['row_id','Enzyme','Aligned_Sequence','parent']+[c for c in ['library','round'] if c in df]].copy()
    row_table.insert(0,'position',np.arange(len(df)))
    row_table.to_csv(a.out/'rows.csv',index=False)
    if 'snapshot_audit' in df.attrs:
        pd.DataFrame(df.attrs['snapshot_audit']).to_csv(a.out/'snapshot_master_audit.csv',index=False)
    (a.out/'splits.json').write_text(json.dumps(manifest,indent=2))
    (a.out/'training_log.json').write_text(json.dumps(logs,indent=2))
    metadata = {'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'round_mapping':json.loads(a.round_map.read_text()) if a.round_map else DEFAULT_ROUNDS,
        'temporal_purged_historical_names':df.attrs.get('purged_historical_names'),
        'snapshot_master_mismatches':sum(not r['aligned_match'] for r in df.attrs.get('snapshot_audit',[])),
        'targets':targets,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
        'split_index_convention':'Indices in splits.json are positions in rows.csv; row_id preserves original input row where applicable',
        'architecture':'embedding -> flatten -> linear -> dropout -> ReLU -> linear',
        'scaling':'per-target max absolute TRAIN value only; missing labels masked; equal task mean loss',
        'pooling_assumption': 'Different assay conditions treated as comparable; unweighted mean across observed conditions only' if a.pool_assays else None,
        'temporal_provenance': 'Frozen historical snapshot sequences AND labels retained; master CSV used only for next-round outcomes' if a.mode=='temporal' else None,
        'comparison':'same trunk and optimizer hyperparameters, independent heads; single-task total parameters exceed shared multi-task',
        'legacy_saved_warning':'Original partitions may contain close neighbors or exact overlap; inspect splits.json',
        'interpretation':'Smoke epochs are not scientific results. No test labels are used for selection or scaling.'}
    (a.out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f'Wrote {len(scores)} target evaluations to {a.out}')


if __name__ == '__main__':
    main()
