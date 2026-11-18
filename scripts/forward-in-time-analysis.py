#!/usr/bin/env python3
"""INFERENCE ONLY: original OAC ensembles on later-round novel variants.

No training code; default audits inputs/cohorts only. --infer explicitly enables
model inference, which the user launches after choosing resources.
  python scripts/forward-in-time-analysis.py --original-model-root ../Cellibre_PKC/models
  CUDA_VISIBLE_DEVICES=7 python scripts/forward-in-time-analysis.py --infer --device cuda:0 --original-model-root ../Cellibre_PKC/models

Original model1 contains screens0..5 (round0), model2 contains0..9 (round0+1,
INCLUDING validation screens8/9), model3 contains0..11 (round0+1+2). Targets are
primary screens6/7,10/11,12/13. Every later-round comparison uses the same candidates
novel to ALL source rows of the newest available original snapshot. Exclude WT,
hd outside1..8, previously seen canonical IDs and exact ungapped sequences.
Model1->test1; models1/2->same test2; models1/2/3->same test3. No CV substitutes.
All100 existing original members by default, median aggregation; --members N uses
firstN numbered members, fixed before viewing outcomes (not performance selection).

Recover original preprocessing exactly: full ORIGINAL matching snapshot maxima
per named ratio head, not train-only/new-data maxima. This reproduces the original
within-snapshot normalization leakage; future test data never enter these scales.
Saved train+validation+test rows are all embargoed from future test to protect
against checkpoint selection and preprocessing use. File hashes/labels/scales and
split indices are exported. A filename is not proof of provenance: exact supplied
historical snapshots/sources are assumed to be the originals; conflicting schema
or weights fail explicitly. Never import auto-training original source scripts.

Transfer score: mean historical OA/C6 or DVA/C4 prediction divided by its own
historical WT fraction. NOT a trained future head or predicted titer. Historical
ratios/scales/WT only determine ranking. Future same-screen WT is used ONLY to
compute observed changes: delta titer(uM)*MW/1000 in mg/L (OA224.25,DVA196.2),
delta fraction*100 percentage points. Top-k displays observed mean/expected maximum
under randomized score ties. No dilution correction. Identical finite outcome and
all-arm score cohorts are used for paired comparisons; assay records stay separate.

Architecture changes2000->65 hidden units confound data-only retraining claims.
This is an ORIGINAL MODEL UPDATE comparison, not a controlled data-only experiment.
No new seeds or pseudo-replicates are invented: one original ensemble per snapshot.
Three top10 analyses are exported separately:
(1) Shared observed-best pooled sequences, separately for each metabolite/endpoint:
select by mean observed WT-relative change across available rounds. All models
score the SAME outcome-selected set. This is selection-biased, range-restricted
retrospective description, NOT validation. Outcome benefits are identical by design. Shared-set plots compare score percentile
and conditional Pearson/Spearman instead; correlations are undefined when a
boundary tie requires fractional membership, rather than arbitrarily picking rows.
(2) Each original model selects its own pooled top10 by its historical fraction
score. Earlier-round sequences may have been used to train later models; explicit
seen/unseen flags refer conservatively to ANY original snapshot row, not just fit.
(3) Each time-eligible model selects within each future-round cohort, with existing
novelty/chronology. Never apply later models backward and call it prospective.
Pooled unit is exact full protein sequence. First average duplicate observations
within a sequence/screen, then equally average available round-relative changes.
No raw assay values are pooled. Different assay/strain conditions and missing-round
coverage still make pooled comparisons descriptive, not condition-adjusted effects.
No observations dropped from the exported long table. Shared and model-selected
pools use identical finite all-model scores/outcomes per endpoint. Exact ties use
fractional membership, including unbiased expected maxima; lists can have more
than10 rows but weights sum to min(10,n). Never pad short cohorts. Export effectivek.
No inference/training happens on import. All outputs go to unique forward_original_
subfolder within results/generalization-analysis. Existing files remain untouched.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import uuid
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SCREENS={1:[6,7],2:[10,11],3:[12,13]}
FAMILY={6:'DVA',7:'OA',10:'DVA',11:'OA',12:'DVA',13:'OA'}
MW={'OA':224.25,'DVA':196.2}
FRACTION={'OA':'OA/C6','DVA':'DVA/C4'}
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def predict_original(source,labels,query,paths,device,batch_size):
    import torch
    from torch import nn
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    aa='ACDEFGHIKLMNPQRSTVWY-'
    scale=source[labels].max().to_numpy(float)
    if not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid historical scaling maxima')
    q=torch.tensor([[aa.index(a) for a in seq] for seq in query.Aligned_Sequence],dtype=torch.long)
    class OriginalMLP(nn.Module):
        def __init__(self,length,width,outputs):
            super().__init__();self.embed=nn.Embedding(21,16);self.linear_1=nn.Linear(length*16,width)
            self.dropout=nn.Dropout(.2);self.linear_2=nn.Linear(width,outputs)
        def forward(self,x):return self.linear_2(torch.relu(self.dropout(self.linear_1(self.embed(x).flatten(1)))))
    predictions=[];details=[]
    for path in paths:
        loaded=torch.load(path,map_location='cpu',weights_only=True)
        state=loaded.get('state_dict',loaded)
        state={k.removeprefix('model.'):v for k,v in state.items()}
        width=state['linear_1.weight'].shape[0]
        model=OriginalMLP(q.shape[1],width,len(labels))
        # label_weights is a loss-only buffer in some Lightning versions.
        if 'label_weights' in state:state.pop('label_weights')
        model.load_state_dict(state,strict=True);model.to(device);model.eval()
        with torch.no_grad():pred=np.concatenate([model(b.to(device)).cpu().numpy() for b in q.split(batch_size)])*scale
        predictions.append(pred);details.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hidden_width=width))
        del model
    return np.median(predictions,axis=0),scale,details

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',type=Path,default=ROOT/'data/all_SeqFxnData.csv')
    p.add_argument('--original-model-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/generalization-analysis')
    p.add_argument('--infer',action='store_true');p.add_argument('--device',default='cuda:0')
    p.add_argument('--members',type=int,default=100);p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args()
    if not 1<=args.members<=100 or args.batch_size<1:p.error('members1..100 and positive batch size required')
    helper=ROOT/'scripts/generalization-analysis.py';tiepath=ROOT/'scripts/adding_esm2_scores.py'
    h=module('genhelpers',helper);ties=module('ties',tiepath)
    snapshots={n:h.load_snapshot(args.original_model_root.parent,n) for n in [1,2,3]}
    # helper expects ROOT/models/round_N; require explicit models directory.
    if args.original_model_root.name!='models':raise ValueError('--original-model-root must name the original models directory')
    checkpoints={}
    for n in [1,2,3]:
        ext='pt' if n==1 else 'ckpt'
        checkpoints[n]=[args.original_model_root/f'round_{n}/model_ensemble/round_{n}_EnsMLP_{i}.{ext}' for i in range(args.members)]
        missing=[str(x) for x in checkpoints[n] if not x.is_file()]
        if missing:raise FileNotFoundError(f'Missing ORIGINAL checkpoints; no substitute or training: {missing}')
        source,labels,split,_=snapshots[n]
        allowed=set(range({1:6,2:10,3:12}[n]))
        if {int(l.split('_',1)[0]) for l in labels}!=allowed:raise ValueError('Unexpected chronological label schema')
    df=pd.read_csv(args.input);df['original_id']=df.Enzyme.astype(str);df.Enzyme=df.Enzyme.map(h.canonical)
    df['row_id']=[f'future_row{i}' for i in range(len(df))];df['parent']=df.Enzyme.str.split('.').str[0]
    if df.Aligned_Sequence.isna().any():raise ValueError('Missing aligned sequence')
    if 'Sequence' not in df or df.Sequence.isna().any():raise ValueError('Full Sequence required for exact-sequence embargo')
    df['sequence_key']=df.Sequence.astype(str).str.upper().str.replace(r'\s+','',regex=True).str.replace('-','',regex=False).str.replace('.','',regex=False)
    if any(not x or set(x)-set(h.AA[:-1]) for x in df.sequence_key):raise ValueError('Invalid full Sequence for exact-sequence embargo')
    if any(set(x)-set(h.AA) for x in df.Aligned_Sequence):raise ValueError('Invalid aligned sequence alphabet')
    needed=[f'{s}_{a}' for screens in SCREENS.values() for s in screens for a in [FAMILY[s],FRACTION[FAMILY[s]]]]
    df[needed]=df[needed].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan)
    df.hd=pd.to_numeric(df.hd,errors='raise');wt=df[df.Enzyme=='PKC001.000']
    if len(wt)!=1:raise ValueError('Expected one future WT row')
    wt=wt.iloc[0]
    out=args.output_dir/('forward_original_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    paths=[args.input,helper,tiepath,Path(__file__)]+[path for item in snapshots.values() for path in item[3]]
    provenance={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'methodology':__doc__,
                'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    manifest=[];plans=[];exclusions=[];predictions=[];coverage=[];baselines=[]
    for n,(source,labels,split,_) in snapshots.items():
        scale_check=source[labels].max().to_numpy(float)
        if not np.isfinite(scale_check).all() or (scale_check<=0).any():raise ValueError(f'Invalid historical maxima in model{n}')
        old_wt_rows=source[source.Enzyme=='PKC001.000']
        if len(old_wt_rows)!=1:raise ValueError(f'Expected one historical WT in model{n}')
        old_wt=old_wt_rows.iloc[0]
        head_audit=[]
        for analyte in ['OA','DVA']:
            usable=[l for l in labels if l.split('_',1)[1]==FRACTION[analyte] and np.isfinite(old_wt[l]) and old_wt[l]!=0 and source.iloc[split[0]][l].notna().any()]
            if not usable:raise ValueError(f'No trained/calibrated {analyte} fraction heads in model{n}')
            head_audit.extend(dict(analyte=analyte,target=l,historical_wt=float(old_wt[l]),training_observations=int(source.iloc[split[0]][l].notna().sum())) for l in usable)
        pd.DataFrame(head_audit).to_csv(out/f'original_snapshot_{n}_score_head_audit.csv',index=False)
        source.to_csv(out/f'original_snapshot_{n}_source_rows.csv',index=False)
        split_rows=[]
        for role,positions in zip(['train','validation','test'],split):
            for position in positions:
                row=source.iloc[int(position)]
                split_rows.append(dict(role=role,position=int(position),row_id=row.row_id,Enzyme=row.Enzyme,original_id=row.original_id,sequence_key=row.sequence_key))
        pd.DataFrame(split_rows).to_csv(out/f'original_snapshot_{n}_saved_splits.csv',index=False)
        pd.DataFrame({'target':labels,'original_full_snapshot_max':source[labels].max().to_numpy(float)}).to_csv(out/f'original_snapshot_{n}_scales.csv',index=False)
        for path in checkpoints[n]:manifest.append(dict(snapshot=n,path=str(path),bytes=path.stat().st_size))
    pd.DataFrame(manifest).to_csv(out/'checkpoint_manifest.csv',index=False)
    for test_round in [1,2,3]:
        newest=snapshots[test_round][0]
        observed=df[[f'{s}_{a}' for s in SCREENS[test_round] for a in [FAMILY[s],FRACTION[FAMILY[s]]]]].notna().any(axis=1)
        eligible=observed&df.hd.between(1,8)&(df.Enzyme!='PKC001.000')
        # Union all available snapshots, not just latest: robust to removed historical rows.
        seen=np.zeros(len(df),dtype=bool)
        for model_round in range(1,test_round+1):
            source=snapshots[model_round][0];seen|=df.Enzyme.isin(source.Enzyme)|df.sequence_key.isin(source.sequence_key)
        test=df[eligible&~seen].copy()
        malformed=test.Aligned_Sequence.str.replace('-','',regex=False)!=test.sequence_key
        if malformed.any():raise ValueError(f'Eligible query alignment does not reproduce full Sequence; fix upstream: {test.loc[malformed,"original_id"].tolist()}')
        exclusions.extend(dict(test_round=test_round,row_id=r.row_id,Enzyme=r.Enzyme,reason='prior model snapshot ID or exact sequence') for r in df[eligible&seen].itertuples())
        test.to_csv(out/f'test_round_{test_round}_novel_cohort.csv',index=False)
        for analyte in ['OA','DVA']:
            s=next(s for s in SCREENS[test_round] if FAMILY[s]==analyte)
            if not np.isfinite(wt[f'{s}_{analyte}']) or not np.isfinite(wt[f'{s}_{FRACTION[analyte]}']):raise ValueError('Missing future WT reference')
            baselines.append(dict(test_round=test_round,analyte=analyte,titer_uM=float(wt[f'{s}_{analyte}']),fraction=float(wt[f'{s}_{FRACTION[analyte]}'])))
        for model_round in range(1,test_round+1):
            ctx=dict(test_round=test_round,train_through_round=model_round-1,repeat=0,original_model_round=model_round)
            plan=dict(ctx,n_test=len(test),members=args.members,status='planned' if len(test) else 'skipped_empty');plans.append(plan)
            source,labels,split,_=snapshots[model_round]
            bad_length=test.Aligned_Sequence.str.len()!=int(source.Aligned_Sequence.str.len().iloc[0])
            if bad_length.any():raise ValueError(f'Query alignment length mismatch: {test.loc[bad_length,"original_id"].tolist()}')
            if not args.infer or test.empty:continue
            oldwt=source[source.Enzyme=='PKC001.000']
            if len(oldwt)!=1:raise ValueError('Missing historical WT calibration')
            oldwt=oldwt.iloc[0]
            pred,scale,details=predict_original(source,labels,test,checkpoints[model_round],args.device,args.batch_size)
            (out/f'test{test_round}_model{model_round}_loaded_checkpoints.json').write_text(json.dumps(details,indent=2))
            for analyte in ['OA','DVA']:
                chosen=[j for j,l in enumerate(labels) if l.split('_',1)[1]==FRACTION[analyte] and np.isfinite(oldwt[l]) and oldwt[l]!=0 and source.iloc[split[0]][l].notna().any()]
                scores=np.mean(pred[:,chosen]/np.array([oldwt[labels[j]] for j in chosen]),axis=1) if chosen else np.full(len(test),np.nan)
                coverage.append(dict(ctx,analyte=analyte,heads=json.dumps([labels[j] for j in chosen]),n_heads=len(chosen)))
                s=next(s for s in SCREENS[test_round] if FAMILY[s]==analyte)
                for i,row in enumerate(test.itertuples()):
                    raw=test.iloc[i];predictions.append(dict(ctx,analyte=analyte,row_id=row.row_id,Enzyme=row.Enzyme,sequence_key=row.sequence_key,score=scores[i],
                        titer_delta_mg_L=(raw[f'{s}_{analyte}']-wt[f'{s}_{analyte}'])*MW[analyte]/1000,
                        fraction_delta_pp=100*(raw[f'{s}_{FRACTION[analyte]}']-wt[f'{s}_{FRACTION[analyte]}'])))
            plan['status']='completed'
            pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    for name,rows in [('plan',plans),('excluded_prior_variants',exclusions),('future_wt_baselines',baselines),('score_head_coverage',coverage)]:pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    if predictions:report(pd.DataFrame(predictions),out,h,ties)
    if args.infer:pooled_top10(df,wt,snapshots,checkpoints,args,out,h,ties)
    print(f'Saved original-model {"inference" if args.infer else "audit only"}: {out}',flush=True)

def pooled_top10(df,wt,snapshots,checkpoints,args,out,h,ties):
    """Explicitly retrospective pools; not an extension of prospective metrics."""
    long=[]
    for test_round,screens in SCREENS.items():
        for screen in screens:
            analyte=FAMILY[screen]
            for row in df[df.hd.between(1,8)&(df.Enzyme!='PKC001.000')].itertuples():
                raw=df.loc[df.row_id==row.row_id].iloc[0]
                t=raw[f'{screen}_{analyte}'];f=raw[f'{screen}_{FRACTION[analyte]}']
                if not np.isfinite(t) and not np.isfinite(f):continue
                long.append(dict(row_id=row.row_id,Enzyme=row.Enzyme,sequence_key=row.sequence_key,round=test_round,
                                 analyte=analyte,titer_delta_mg_L=(t-wt[f'{screen}_{analyte}'])*MW[analyte]/1000,
                                 fraction_delta_pp=100*(f-wt[f'{screen}_{FRACTION[analyte]}'])))
    observations=pd.DataFrame(long)
    if observations.empty:return
    observations.to_csv(out/'pooled_observations_long.csv',index=False)
    candidates=df[df.sequence_key.isin(observations.sequence_key)].copy()
    bad=candidates.Aligned_Sequence.str.replace('-','',regex=False)!=candidates.sequence_key
    if bad.any():raise ValueError(f'Pooled query alignment mismatch: {candidates.loc[bad,"original_id"].tolist()}')
    ambiguous=candidates.groupby('sequence_key').Aligned_Sequence.nunique()
    if (ambiguous>1).any():raise ValueError('Same pooled full sequence has multiple aligned representations; resolve upstream')
    query=candidates.drop_duplicates('sequence_key').copy()
    aliases=candidates.groupby('sequence_key').Enzyme.agg(lambda x:json.dumps(sorted(set(x))))
    scores={};flags={}
    for model_round,(source,labels,split,_) in snapshots.items():
        badlen=query.Aligned_Sequence.str.len()!=int(source.Aligned_Sequence.str.len().iloc[0])
        if badlen.any():raise ValueError('Pooled query alignment lengths differ from original model')
        pred,_,details=predict_original(source,labels,query,checkpoints[model_round],args.device,args.batch_size)
        (out/f'pooled_model{model_round}_checkpoints.json').write_text(json.dumps(details,indent=2))
        historical_wt=source[source.Enzyme=='PKC001.000'].iloc[0]
        seen_sequences=set(source.sequence_key)
        alias_seen=set(candidates[candidates.Enzyme.isin(source.Enzyme)].sequence_key)
        flags[model_round]={seq:seq in seen_sequences or seq in alias_seen for seq in query.sequence_key}
        for analyte in ['OA','DVA']:
            chosen=[j for j,l in enumerate(labels) if l.split('_',1)[1]==FRACTION[analyte] and np.isfinite(historical_wt[l]) and historical_wt[l]!=0 and source.iloc[split[0]][l].notna().any()]
            if not chosen:raise ValueError('No calibrated pooled scoring heads')
            score=np.mean(pred[:,chosen]/np.array([historical_wt[labels[j]] for j in chosen]),axis=1)
            scores[model_round,analyte]=dict(zip(query.sequence_key,score))
    selected=[];summary=[];all_candidates=[];pool_coverage=[]
    for analyte in ['OA','DVA']:
        for target in ['titer_delta_mg_L','fraction_delta_pp']:
            # One observation per sequence/round, then equal-weight available rounds.
            within=observations[observations.analyte==analyte].groupby(['sequence_key','round'])[target].mean().dropna()
            means=within.groupby('sequence_key').mean();counts=within.groupby('sequence_key').count()
            table=pd.DataFrame({'observed_mean_delta':means,'rounds_observed':counts})
            for model_round in [1,2,3]:
                table[f'model{model_round}_score']=table.index.map(scores[model_round,analyte])
                table[f'model{model_round}_seen']=table.index.map(flags[model_round])
            finite=np.isfinite(table[['observed_mean_delta']+[f'model{i}_score' for i in [1,2,3]]].to_numpy(float)).all(axis=1)
            pool_coverage.append(dict(analyte=analyte,target=target,n_observed_sequences=len(table),n_common_finite=int(finite.sum()),
                                      **{f'model{i}_nonfinite_scores':int((~np.isfinite(table[f"model{i}_score"])).sum()) for i in [1,2,3]}))
            table=table[finite];n=len(table);k=min(10,n)
            for seq,row in table.iterrows():all_candidates.append(dict(analyte=analyte,target=target,sequence_key=seq,Enzymes=aliases.loc[seq],**row.to_dict()))
            if not n:continue
            y=table.observed_mean_delta.to_numpy(float)
            for analysis in ['shared_observed_top10','per_model_predicted_top10']:
                for model_round in [1,2,3]:
                    x=table[f'model{model_round}_score'].to_numpy(float)
                    rank=y if analysis=='shared_observed_top10' else x
                    w=ties.topk_weights(rank,k)
                    ctx=dict(analysis=analysis,analyte=analyte,target=target,original_model_round=model_round)
                    fractional=bool(((w>0)&(w<1)).any())
                    conditional=h.metrics(y[w>0],x[w>0]) if not fractional else {'pearson':np.nan,'spearman':np.nan}
                    ctx.update(conditional_pearson=conditional['pearson'],conditional_spearman=conditional['spearman'],
                               conditional_correlation_status='undefined_fractional_boundary_tie' if fractional else ('outcome_conditioned_descriptive_not_validation' if analysis=='shared_observed_top10' else 'model_selected_descriptive_not_validation'))
                    # No conditional correlation pretending fractional tied rows are independent.
                    summary.append(dict(ctx,interpretation='retrospective_outcome_selected' if analysis=='shared_observed_top10' else 'retrospective_mixed_seen_unseen',
                                        n_candidates=n,requested_k=10,effective_k=k,mean_observed_delta=float(w@y/k),
                                        max_observed_delta=ties.topk_expected_max(rank,y,k),
                                        mean_model_score=float(w@x/k),mean_model_score_percentile=float(w@pd.Series(x).rank(method='average',pct=True).to_numpy()/k),selected_seen_weight=float(w@table[f'model{model_round}_seen'].to_numpy(float))))
                    for i,seq in enumerate(table.index):
                        if w[i]>0:selected.append(dict(ctx,sequence_key=seq,Enzymes=aliases.loc[seq],observed_mean_delta=y[i],score=x[i],selection_weight=w[i],
                                                       seen_in_original_snapshot=bool(table.iloc[i][f'model{model_round}_seen']),rounds_observed=int(table.iloc[i].rounds_observed)))
    pd.DataFrame(pool_coverage).to_csv(out/'pooled_candidate_coverage.csv',index=False)
    pd.DataFrame(all_candidates).to_csv(out/'pooled_common_candidate_scores.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'pooled_top10_summary.csv',index=False)
    pd.DataFrame(selected).to_csv(out/'pooled_top10_memberships.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sums=pd.DataFrame(summary)
    if sums.empty:return
    for analysis in ['shared_observed_top10','per_model_predicted_top10']:
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for r,analyte in enumerate(['OA','DVA']):
            for c,target in enumerate(['titer_delta_mg_L','fraction_delta_pp']):
                part=sums[(sums.analysis==analysis)&(sums.analyte==analyte)&(sums.target==target)]
                ax=axes[r,c]
                if analysis=='shared_observed_top10':
                    ax.plot(part.original_model_round,part.conditional_pearson,'o-',label='Conditional Pearson')
                    ax.plot(part.original_model_round,part.conditional_spearman,'s-',label='Conditional Spearman')
                    ax.plot(part.original_model_round,part.mean_model_score_percentile,'^--',label='Mean score percentile')
                    ax.set_ylim(-1.05,1.05)
                else:
                    ax.plot(part.original_model_round,part.mean_observed_delta,'o--',label='Mean observed change')
                    ax.plot(part.original_model_round,part.max_observed_delta,'^-',label='Expected maximum change')
                k=int(part.effective_k.iloc[0]) if len(part) else 0
                ax.set(title=f'{analyte}: {"Titer" if target=="titer_delta_mg_L" else "Fraction"}; effective k={k}',xlabel='Original model',ylabel='Conditional correlation / percentile' if analysis=='shared_observed_top10' else target,xticks=[1,2,3]);ax.legend(fontsize=8)
        fig.suptitle(analysis+' (retrospective, not prospective validation)');fig.tight_layout(rect=(0,0,1,.94))
        for ext in ['png','svg']:fig.savefig(out/f'{analysis}.{ext}',dpi=300)
        plt.close(fig)

def report(data,out,h,ties):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # Collapse repeated measurement rows only within the same sequence/screen.
    data=data.groupby(['test_round','repeat','analyte','train_through_round','sequence_key'],as_index=False).agg(Enzyme=('Enzyme',lambda x:json.dumps(sorted(set(x)))),score=('score','mean'),titer_delta_mg_L=('titer_delta_mg_L','mean'),fraction_delta_pp=('fraction_delta_pp','mean'))
    data['row_id']=data.sequence_key
    stats=[];curves=[];cohorts=[];top10=[];memberships=[]
    colors=['#4477AA','#AA3377','#228833']
    for (test_round,repeat,analyte),part in data.groupby(['test_round','repeat','analyte']):
        required=list(range(test_round))
        pivot=part.pivot(index='row_id',columns='train_through_round',values='score')
        for target in ['titer_delta_mg_L','fraction_delta_pp']:
            y=part.groupby('row_id')[target].first().reindex(pivot.index)
            if not set(required).issubset(pivot.columns):
                cohorts.append(dict(test_round=test_round,repeat=repeat,analyte=analyte,target=target,status='incomplete arms'));continue
            mask=np.isfinite(y)&np.isfinite(pivot[required]).all(axis=1)
            cohort=pivot.loc[mask];outcomes=y[mask].to_numpy(float)
            cohorts.append(dict(test_round=test_round,repeat=repeat,analyte=analyte,target=target,status='evaluated',n=len(cohort),n_observed=int(np.isfinite(y).sum())))
            for cutoff in required:
                x=cohort[cutoff].to_numpy(float)
                ctx=dict(test_round=test_round,repeat=repeat,analyte=analyte,target=target,train_through_round=cutoff)
                stats.append(dict(ctx,**h.metrics(outcomes,x)))
                if len(cohort):
                    effective=min(10,len(cohort));w=ties.topk_weights(x,effective)
                    top10.append(dict(ctx,analysis='prospective_per_round',requested_k=10,effective_k=effective,n_candidates=len(cohort),
                                      mean_delta=float(w@outcomes/effective),max_delta=ties.topk_expected_max(x,outcomes,effective)))
                    identities=part.groupby('row_id').Enzyme.first()
                    memberships.extend(dict(ctx,row_id=rid,Enzyme=identities.loc[rid],score=float(x[i]),observed=float(outcomes[i]),selection_weight=float(w[i])) for i,rid in enumerate(cohort.index) if w[i]>0)
                for k in range(1,min(10,len(cohort))+1):
                    weights=ties.topk_weights(x,k)
                    curves.append(dict(ctx,k=k,mean_delta=float(weights@outcomes/k),max_delta=ties.topk_expected_max(x,outcomes,k),random_mean=float(outcomes.mean())))
    # Ranking-score scale differs from endpoint units: only correlations are meaningful here.
    metric=pd.DataFrame(stats).drop(columns=['rmse','mae','r2','mse','top10_hit_enrichment','effective_k'],errors='ignore')
    metric.to_csv(out/'forward_correlations.csv',index=False);pd.DataFrame(cohorts).to_csv(out/'paired_cohort_coverage.csv',index=False)
    pd.DataFrame(top10).to_csv(out/'prospective_top10_summary.csv',index=False)
    pd.DataFrame(memberships).to_csv(out/'prospective_top10_memberships.csv',index=False)
    curve=pd.DataFrame(curves);curve.to_csv(out/'topk_observed_changes.csv',index=False)
    if curve.empty:return
    for repeat in sorted(curve.repeat.unique()):
        for target,label in [('titer_delta_mg_L','Titer change vs WT (mg/L)'),('fraction_delta_pp','Fraction change vs WT (percentage points)')]:
            fig,axes=plt.subplots(3,2,figsize=(12,12))
            for test_round in [1,2,3]:
                for col,analyte in enumerate(['OA','DVA']):
                    ax=axes[test_round-1,col]
                    subset=curve[(curve.repeat==repeat)&(curve.target==target)&(curve.test_round==test_round)&(curve.analyte==analyte)]
                    for cutoff,part in subset.groupby('train_through_round'):
                        ax.plot(part.k,part.mean_delta,ls='--',color=colors[cutoff]);ax.plot(part.k,part.max_delta,color=colors[cutoff])
                    if len(subset):ax.axhline(subset.random_mean.iloc[0],ls=':',color='grey')
                    ax.axhline(0,color='black',lw=.6);ax.set(title=f'Test round{test_round}: {analyte}',xlabel='Number selected (k)',ylabel=label.replace(' (','\n('))
            handles=[Line2D([],[],color=colors[i],label=f'Original model{i+1}') for i in range(3)]+[Line2D([],[],color='black',ls='--',label='Mean'),Line2D([],[],color='black',label='Maximum'),Line2D([],[],color='grey',ls=':',label='Random expected mean')]
            fig.legend(handles=handles,loc='lower center',ncol=3);fig.tight_layout(rect=(0,.07,1,1))
            for ext in ['png','svg']:fig.savefig(out/f'topk_{target}_repeat{repeat}.{ext}',dpi=300)
            plt.close(fig)
    for method in ['spearman','pearson']:
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for row,analyte in enumerate(['OA','DVA']):
            for col,target in enumerate(['titer_delta_mg_L','fraction_delta_pp']):
                ax=axes[row,col]
                for cutoff in [0,1,2]:
                    for test_round in [1,2,3]:
                        values=metric.loc[(metric.analyte==analyte)&(metric.target==target)&(metric.train_through_round==cutoff)&(metric.test_round==test_round),method].dropna()
                        if len(values):
                            x=test_round+(cutoff-1)*.18;ax.scatter(np.full(len(values),x),values,color=colors[cutoff],s=20);ax.scatter(x,values.mean(),color=colors[cutoff],marker='_',s=200)
                ax.set(title=f'{analyte}: {target}',xlabel='Test round',ylabel=method,xticks=[1,2,3],ylim=(-1.05,1.05))
        fig.legend(handles=[Line2D([],[],color=colors[i],label=f'Original model{i+1}') for i in range(3)],loc='lower center',ncol=3)
        fig.tight_layout(rect=(0,.06,1,1))
        for ext in ['png','svg']:fig.savefig(out/f'forward_{method}_comparison.{ext}',dpi=300)
        plt.close(fig)


if __name__=='__main__':main()
