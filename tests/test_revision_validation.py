"""Scientific invariants for leakage-aware validation, independent of repository data."""
import importlib.util
import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

SPEC = importlib.util.spec_from_file_location('validate_models',Path(__file__).parents[1]/'scripts/revisions/validate_models.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class ValidationTests(unittest.TestCase):
    def frame(self):
        return m.canonical_frame(pd.DataFrame({'Enzyme':['PKC001.0','PKC1.1','PKC4.0','PKC7.0','PKC8.0','PKC9.0'],
            'Aligned_Sequence':['AAA-','AAA-','AAC-','ACC-','CCC-','DDD-']}))

    def test_duplicate_sequences_and_parent_aliases_never_split(self):
        d=self.frame()
        self.assertEqual(m.variant_id('PKC001.000'),'PKC1.0')
        for mode in ['random','parent','identity']:
            groups=m.make_groups(d,mode,identity=0.99)
            self.assertEqual(groups[0],groups[1])
            for train,test in m.group_folds(groups,2,11):
                self.assertFalse(set(groups[train]) & set(groups[test]))
                inner, val = m.inner_split(train,groups,12)
                self.assertFalse(set(groups[inner]) & set(groups[val]))

    def test_identity_transitive_connections_and_joint_gaps(self):
        d=self.frame().iloc[[0,2,3,4,5]].reset_index(drop=True)
        groups=m.make_groups(d,'identity',identity=0.66)
        self.assertEqual(groups[0],groups[3])  # AAA->AAC->ACC->CCC transitive
        self.assertNotEqual(groups[0],groups[4])
        self.assertAlmostEqual(m.identity_matrix(['AAA-','AAC-'])[0,1],2/3)
        with self.assertRaises(ValueError):
            m.group_folds(groups,3,1)

    def test_scale_uses_training_labels_only_and_handles_empty_targets(self):
        y=np.array([[1,np.nan],[2,np.nan],[999,100]])
        scale,count=m.scale_from_train(y[:2])
        np.testing.assert_array_equal(scale,[2,1])
        np.testing.assert_array_equal(count,[2,0])

    def test_missing_targets_have_zero_gradient(self):
        pred=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
        true=torch.tensor([[1.,float('nan')],[1.,float('nan')]])
        loss=m.masked_task_mse(pred,true)
        self.assertAlmostEqual(loss.item(),2)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        np.testing.assert_array_equal(pred.grad[:,1].numpy(),[0,0])

    def test_fit_does_not_learn_unobserved_target_or_test_extreme(self):
        x=m.encode(['AAA','AAC','ACC','CCC','DDD','EEE'])
        y=np.array([[1,np.nan],[2,np.nan],[3,np.nan],[4,np.nan],[5,np.nan],[999,99]],float)
        a=SimpleNamespace(min_labels=2,hidden=3,embedding=2,dropout=0.,learning_rate=.001,
                          weight_decay=0.,epochs=2,batch_size=2,patience=2)
        pred,log=m.fit_predict(x,y,np.array([0,1,2]),np.array([3,4]),np.array([5]),a,1)
        self.assertTrue(np.isfinite(pred[0,0]))
        self.assertTrue(np.isnan(pred[0,1]))
        self.assertEqual(log['scale'],[3.,1.])

    def test_temporal_frozen_labels_id_purge_and_independent_pooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            historical=pd.DataFrame({'Enzyme':['PKC1.0','PKC1.1','PKC1.2','PKC1.3'],
                'Aligned_Sequence':['AAAA','AACC','AADD','AAEE'],
                '0_A/B':[1,2,3,4],'1_A/B':[3,4,5,6]})
            master=pd.DataFrame({'Enzyme':['PKC1.1','PKC1.2','PKC1.3','PKC001.000','PKC1.9'],
                'Aligned_Sequence':['AACC','AADD','AAEE','DDDD','EEEE'],
                'library':['initial_library_1']*3+['round_1']*2,
                '0_A/B':[999,999,999,np.nan,np.nan],
                '2_A/B':[888,888,888,90,100]})
            historical.to_pickle(tmp/'old.pkl');master.to_csv(tmp/'master.csv',index=False)
            a=m.arguments(['--out',str(tmp/'out'),'--data',str(tmp/'master.csv'),
                '--mode','temporal','--train-through','0','--pool-assays',
                '--training-snapshot',str(tmp/'old.pkl')])
            frame,folds,_=m.prepare(a)
            _,train,val,test=folds[0]
            self.assertEqual(len(train)+len(val),3)
            self.assertIn('PKC1.0',frame.attrs['purged_historical_names'])
            np.testing.assert_allclose(sorted(frame.iloc[np.r_[train,val]]['A/B']),[3,4,5])
            np.testing.assert_allclose(frame.iloc[test]['A/B'],[90,100])
            self.assertEqual(frame.attrs['targets'],['A/B'])
            changed=[r for r in frame.attrs['snapshot_audit'] if r['variant_id']=='PKC1.0']
            self.assertEqual(len(changed),1)
            self.assertFalse(changed[0]['aligned_match'])
            # The future-only assay never becomes a separate training task.
            self.assertNotIn('2_A/B',frame.attrs['targets'])

    def test_pooling_is_explicit_and_preserves_missing(self):
        d=self.frame()
        d['0_A/B']=[1,2,np.nan,np.nan,3,4]
        d['1_A/B']=[3,np.nan,np.nan,np.nan,5,6]
        pooled=m.pool_assays(d)
        self.assertEqual(pooled['A/B'].iloc[0],2)
        self.assertTrue(np.isnan(pooled['A/B'].iloc[2]))
        self.assertNotIn('A/B',d)

if __name__=='__main__':
    unittest.main()
