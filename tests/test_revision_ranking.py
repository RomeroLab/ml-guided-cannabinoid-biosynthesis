"""Scientific checks for retrospective ranking comparisons."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.revisions.common import canonical_id, parent_id
from scripts.revisions.ranking_ablation import analyze, fixed_selection_difference, holm, main, pareto_layers, top_indices


class RankingTests(unittest.TestCase):
    def test_dominance_and_ties(self):
        np.testing.assert_array_equal(pareto_layers([3, 2, 1, 3, 0], [1, 2, 0, 1, 0]), [0, 0, 1, 0, 2])
        np.testing.assert_array_equal(top_indices([1, 1, 0], ['b', 'a', 'c'], 1), [1])

    def test_exact_random_and_sign_of_loss(self):
        d = pd.DataFrame({'Enzyme': ['a','b','c','d'], 'parent': ['p','p','q','q'],
                          'y': [2., 1.5, .5, 0.], 'vae': [4,3,2,1], 'mlp': [4,3,2,1]})
        m, c, diff, ann = analyze(d, {'VAE':'vae', 'MLP':'mlp'}, 'y', 1., .2, 1., 2, 100, 'parent', 1)
        x = m.set_index('method')
        self.assertEqual(x.loc['VAE', 'precision_at_k'], 1.)
        self.assertEqual(x.loc['VAE', 'random_pool_tail_p'], 1/6)
        self.assertEqual(x.loc['random', 'precision_at_k'], .5)
        self.assertEqual(x.loc['VAE', 'auroc_low_outcome'], 1.)
        self.assertEqual(len(c), 4 * 4)
        self.assertTrue((diff.precision_difference == 0).all())

    def test_bootstrap_shared_sets_and_insufficient_clusters(self):
        low, high, valid = fixed_selection_difference(np.array([0,1,1]), np.array([0,1]), np.array([0,1]),
                                                      np.array(['a','b','c']), 100, 1)
        self.assertEqual((low, high), (0.,0.))
        self.assertGreater(valid, 0)
        low, high, valid = fixed_selection_difference(np.array([0,1]), [0], [1], ['p','p'], 100, 1)
        self.assertTrue(np.isnan(low))
        self.assertEqual(valid, 0)

    def test_identifier_normalization(self):
        self.assertEqual(canonical_id('PKC001.000'), 'PKC1.0')
        self.assertEqual(parent_id('PKC001.100'), parent_id('PKC1.7'))
        np.testing.assert_allclose(holm([.01,.04,.03]), [.03,.06,.06])

    def test_invalid_threshold_rejected_before_loading_data(self):
        for threshold in ('nan','inf','-1'):
            with self.assertRaises(SystemExit):
                main(['--out','/tmp/unused-revision-output','--outcome','y','--supervised-score','s',
                      '--libraries','r','--hit-sd',threshold])

    def test_cli_excludes_snapshot_by_id_and_sequence_and_matches_missing(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            d = pd.DataFrame({'Enzyme':['PKC001.000','PKC001.1','PKC001.2','PKC001.3','PKC001.4'],
                              'Aligned_Sequence':['AAA','AAC','AAD','AAE','AAF'],
                              'library':['WT','round_1','round_1','round_1','round_1'],
                              'y':[1.,0.,2.,3.,4.], 'y_std':[.2]*5,
                              'vae_score':[0,1,2,3,4], 'custom_mlp':[0,1,2,3,np.nan]})
            d.to_csv(t/'data.csv', index=False)
            # Changed old sequence still excluded by canonical ID.
            pd.DataFrame({'name':['PKC1.1'], 'sequence':['QQQ']}).to_pickle(t/'train.pkl')
            main(['--data',str(t/'data.csv'),'--out',str(t/'result'),'--outcome','y',
                  '--supervised-score','custom_mlp','--libraries','round_1','--training-snapshot',str(t/'train.pkl'),
                  '--top-k','1','--bootstrap-draws','100'])
            meta = json.loads((t/'result/metadata.json').read_text())
            self.assertEqual(meta['final_candidates'],2)
            self.assertEqual(meta['excluded_training'],['PKC001.1'])
            self.assertEqual(meta['excluded_missing'],['PKC001.4'])
            self.assertEqual(meta['thresholds'][1]['threshold'],1.2)


if __name__ == '__main__':
    unittest.main()
