#!/usr/bin/env python
"""Optional ESM2/ESMC zero-shot scores, without fitting to assay labels.

ESM2 uses Hugging Face Transformers. ESMC uses the released ``esm==3.2.1``
SDK (ESMC.from_pretrained, tokenizer, forward(sequence_tokens=...)); its API
was checked against that release. Install either backend in a separate
environment: the deposited torch==2.0.1 environment is deliberately untouched.

Masked marginals sum mutation log-odds in the masked WT background; they do
not model epistasis. ``pll`` instead computes the mean masked-residue log
probability in each sequence's own background, minus that of WT. PLL supports
indels/other homolog lengths but is more expensive and is a different score.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def canonical_sequence(value: str) -> str:
    """Accept explicitly ungapped sequences; never remove gaps silently."""
    if not isinstance(value, str) or not value:
        raise ValueError("Sequence must be a nonempty ungapped string")
    value = value.upper()
    invalid = sorted(set(value) - AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Noncanonical residues/gaps in sequence: {invalid}")
    return value


class MaskedAdapter:
    """Small shared adapter; validates residue-to-token indexing before inference."""

    def __init__(self, tokenizer, forward, device="cpu", max_tokens=None):
        self.tokenizer = tokenizer
        self.forward = forward
        self.device = device
        self.max_tokens = max_tokens

    def log_probabilities(self, sequence, positions, batch_size=8):
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        tokens = self.tokenizer.encode(sequence, add_special_tokens=True)
        residue_tokens = self.tokenizer.encode(sequence, add_special_tokens=False)
        if len(residue_tokens) != len(sequence):
            raise ValueError("Tokenizer must map each residue to exactly one token")
        # Both supported ESM tokenizers add one BOS and one EOS. Fail explicitly
        # if a different tokenizer layout is supplied, instead of shifting sites.
        if len(tokens) != len(sequence) + 2 or tokens[1:-1] != residue_tokens:
            raise ValueError("Expected ESM BOS + residues + EOS token layout")
        if self.max_tokens and len(tokens) > self.max_tokens:
            raise ValueError(f"Sequence exceeds model token limit {self.max_tokens}")
        if self.tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer has no mask token")
        positions = list(positions)
        if len(set(positions)) != len(positions) or any(
            p < 0 or p >= len(sequence) for p in positions
        ):
            raise ValueError("Mask positions must be unique valid residue indices")
        result = {}
        with torch.inference_mode():
            for start in range(0, len(positions), batch_size):
                selected = positions[start:start + batch_size]
                inputs = torch.tensor(tokens, device=self.device).repeat(len(selected), 1)
                for row, pos in enumerate(selected):
                    inputs[row, pos + 1] = self.tokenizer.mask_token_id
                logits = self.forward(inputs)
                if logits.shape[:2] != inputs.shape:
                    raise ValueError("Model logits do not match the token batch shape")
                for row, pos in enumerate(selected):
                    result[pos] = torch.log_softmax(logits[row, pos + 1].float(), -1).cpu().numpy()
        return result

    def residue_id(self, residue):
        ids = self.tokenizer.encode(residue, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("Residue does not map to one token")
        return ids[0]


def score_sequences(sequences, wt, adapter, method="masked-marginal", batch_size=8):
    """Return scores in input order; raises on invalid input without fitting labels."""
    wt = canonical_sequence(wt)
    sequences = [canonical_sequence(seq) for seq in sequences]
    if method == "masked-marginal":
        if any(len(seq) != len(wt) for seq in sequences):
            raise ValueError("Masked marginals require WT-length substitutions; use --method pll for indels")
        needed = sorted({i for seq in sequences for i, (a, b) in enumerate(zip(wt, seq)) if a != b})
        probabilities = adapter.log_probabilities(wt, needed, batch_size)
        return [float(sum(probabilities[i][adapter.residue_id(mut)] -
                          probabilities[i][adapter.residue_id(ref)]
                          for i, (ref, mut) in enumerate(zip(wt, seq)) if ref != mut))
                for seq in sequences]
    if method != "pll":
        raise ValueError("Unknown method")

    def mean_pll(seq):
        probabilities = adapter.log_probabilities(seq, range(len(seq)), batch_size)
        return float(np.mean([probabilities[i][adapter.residue_id(aa)] for i, aa in enumerate(seq)]))

    cache = {wt: mean_pll(wt)}
    for seq in dict.fromkeys(sequences):
        if seq not in cache:
            cache[seq] = mean_pll(seq)
    return [cache[seq] - cache[wt] for seq in sequences]


def load_backend(backend, model_name, device, local_files_only, cache_dir):
    import torch

    if backend == "esm2":
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
        except ImportError as exc:
            raise RuntimeError("ESM2 needs transformers in a separate pLM environment") from exc
        kwargs = dict(local_files_only=local_files_only, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs).to(device).eval()
        max_tokens = getattr(model.config, "max_position_embeddings", None)
        return MaskedAdapter(tokenizer, lambda ids: model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits,
                             device, max_tokens)
    # Stable SDK version. Newer esm versions changed class/model names; fail with
    # a concrete dependency instruction rather than silently using a new model.
    try:
        version = importlib.metadata.version("esm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("ESMC requires esm==3.2.1 in a separate pLM environment") from exc
    if version != "3.2.1":
        raise RuntimeError(f"ESMC adapter requires esm==3.2.1; installed {version}")
    import os
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if cache_dir:
        os.environ["HF_HUB_CACHE"] = str(Path(cache_dir).resolve())
    from esm.models.esmc import ESMC
    model = ESMC.from_pretrained(model_name, device=torch.device(device)).eval()
    return MaskedAdapter(model.tokenizer, lambda ids: model(sequence_tokens=ids).sequence_logits, device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wt-id", required=True, help="Exactly one reference row, e.g. PKC001.000")
    parser.add_argument("--id-column", default="Enzyme")
    parser.add_argument("--sequence-column", default="Sequence", help="Must contain ungapped canonical sequences")
    parser.add_argument("--backend", choices=["esm2", "esmc"], default="esm2")
    parser.add_argument("--model", help="ESM2 HF ID/local path; ESMC alias esmc_300m or esmc_600m")
    parser.add_argument("--method", choices=["masked-marginal", "pll"], default="masked-marginal")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true", help="Require previously cached/local weights")
    parser.add_argument("--on-invalid", choices=["error", "skip"], default="error")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    data = pd.read_csv(args.data)
    if data[args.id_column].isna().any() or data[args.id_column].duplicated().any():
        raise ValueError("Input identifiers must be present and unique")
    wt_rows = data.loc[data[args.id_column] == args.wt_id, args.sequence_column]
    if len(wt_rows) != 1:
        raise ValueError("--wt-id must identify exactly one reference row")
    wt = canonical_sequence(wt_rows.iloc[0])
    valid, reasons, canonical = [], [], []
    for index, seq in enumerate(data[args.sequence_column]):
        try:
            seq = canonical_sequence(seq)
            if args.method == "masked-marginal" and len(seq) != len(wt):
                raise ValueError("Different WT length: indel/homolog unsupported by masked marginals; use pll")
            valid.append(index)
            reasons.append("")
            canonical.append(seq)
        except ValueError as exc:
            if args.on_invalid == "error":
                raise ValueError(f"{data[args.id_column].iloc[index]}: {exc}") from exc
            reasons.append(str(exc))
    model_name = args.model or ("facebook/esm2_t33_650M_UR50D" if args.backend == "esm2" else "esmc_600m")
    adapter = load_backend(args.backend, model_name, args.device, args.local_files_only, args.cache_dir)
    scores = score_sequences(canonical, wt, adapter, args.method, args.batch_size)
    result = data[[args.id_column, args.sequence_column]].copy()
    result["plm_score"] = np.nan
    result.loc[valid, "plm_score"] = scores
    if not all(math.isfinite(x) for x in scores):
        raise ValueError("Model produced a nonfinite score")
    result["plm_status"] = ["ok" if not x else x for x in reasons]
    result["plm_model"] = model_name
    result["plm_method"] = args.method
    result["plm_wt_id"] = args.wt_id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    versions = {}
    for name in ["torch", "transformers", "esm", "numpy", "pandas"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    provenance = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "resolved_model": model_name,
                  "input_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  "wt_sequence_sha256": hashlib.sha256(wt.encode()).hexdigest(),
                  "versions": versions, "n_scored": len(valid), "n_skipped": len(data) - len(valid),
                  "higher_is_better": True,
                  "interpretation": "Zero-shot sequence plausibility, not predicted C4/C6-specific activity"}
    args.out.with_suffix(args.out.suffix + ".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"scored": len(valid), "skipped": len(data) - len(valid), "out": str(args.out)}))


if __name__ == "__main__":
    main()
