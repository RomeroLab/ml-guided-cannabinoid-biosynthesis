import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_plm", ROOT / "scripts/revisions/score_plm.py")
plm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plm)


class ToyTokenizer:
    mask_token_id = 23

    def encode(self, sequence, add_special_tokens=True):
        tokens = ["ACDEFGHIKLMNPQRSTVWY".index(aa) + 1 for aa in sequence]
        return [0] + tokens + [22] if add_special_tokens else tokens


class PLMTests(unittest.TestCase):
    def adapter(self):
        self.calls = []

        def forward(inputs):
            self.assertTrue(torch.all((inputs == 23).sum(1) == 1))
            self.assertTrue(torch.all(inputs[:, 0] == 0))
            self.assertTrue(torch.all(inputs[:, -1] == 22))
            self.calls.append(inputs.clone())
            # Position-dependent logits catch shifted indexing errors.
            positions = torch.arange(inputs.shape[1]).reshape(1, -1, 1)
            vocabulary = torch.arange(24).reshape(1, 1, -1)
            return (positions * vocabulary).float().expand(inputs.shape[0], -1, -1)

        return plm.MaskedAdapter(ToyTokenizer(), forward)

    def test_single_and_multi_site_masked_marginals(self):
        adapter = self.adapter()
        scores = plm.score_sequences(["ACDE", "ACDA", "ADDA", "ACDA"], "ACDE", adapter, batch_size=1)
        # E->A at pos 3: (1-4)*4; C->D at pos1: (3-2)*2.
        np.testing.assert_allclose(scores, [0., -12., -10., -12.], atol=1e-6)
        self.assertEqual(len(self.calls), 2)  # One mask per unique mutated WT site.
        for inputs in self.calls:
            for row in inputs:
                unmasked = row != 23
                expected = torch.tensor([0, 1, 2, 3, 4, 22])
                self.assertTrue(torch.equal(row[unmasked], expected[unmasked]))

    def test_wt_requires_no_forward_calls(self):
        adapter = self.adapter()
        self.assertEqual(plm.score_sequences(["ACD"], "ACD", adapter), [0.])
        self.assertEqual(len(self.calls), 0)

    def test_pll_supports_indels_and_reuses_duplicates(self):
        adapter = self.adapter()
        scores = plm.score_sequences(["ACD", "ACDE", "ACD"], "ACDE", adapter, method="pll", batch_size=2)
        self.assertEqual(scores[1], 0.)
        self.assertEqual(scores[0], scores[2])
        self.assertEqual(len(self.calls), 4)  # ceil(4/2) + ceil(3/2)

    def test_rejects_indels_and_ambiguous_residues(self):
        adapter = self.adapter()
        with self.assertRaisesRegex(ValueError, "WT-length"):
            plm.score_sequences(["ACD"], "ACDE", adapter)
        for invalid in ["AC-D", "ACDX", "", float("nan")]:
            with self.assertRaises(ValueError):
                plm.canonical_sequence(invalid)

    def test_token_layout_and_length_guard(self):
        adapter = self.adapter()
        adapter.max_tokens = 4
        with self.assertRaisesRegex(ValueError, "token limit"):
            adapter.log_probabilities("ACD", [0])
        class WrongTokenizer(ToyTokenizer):
            def encode(self, sequence, add_special_tokens=True):
                base = super().encode(sequence, add_special_tokens)
                return [0] + base if add_special_tokens else base
        adapter = plm.MaskedAdapter(WrongTokenizer(), lambda _: None)
        with self.assertRaisesRegex(ValueError, "token layout"):
            adapter.log_probabilities("ACD", [0])

    def test_cli_skips_invalid_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "variants.csv"
            out = Path(temp) / "scores.csv"
            pd.DataFrame({"Enzyme": ["WT", "mutant", "indel"],
                          "Sequence": ["ACDE", "ACDA", "ACD"]}).to_csv(data, index=False)
            with patch.object(plm, "load_backend", return_value=self.adapter()):
                plm.main(["--data", str(data), "--out", str(out), "--wt-id", "WT", "--on-invalid", "skip"])
            result = pd.read_csv(out)
            self.assertEqual(result.plm_score.iloc[1], -12.)
            self.assertTrue(np.isnan(result.plm_score.iloc[2]))
            provenance = json.loads(out.with_suffix(".csv.json").read_text())
            self.assertEqual(provenance["n_scored"], 2)
            self.assertEqual(provenance["n_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
