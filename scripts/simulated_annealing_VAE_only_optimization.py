#!/usr/bin/env python
# coding: utf-8

"""Single-objective simulated annealing for normalization calibration.

Run from the repository root or from ``scripts/``. The script optimizes one
objective during annealing and evaluates the other objective only for each
trial's final best sequence. This provides the paired endpoint scores needed
to normalize subsequent Pareto or utopia optimization.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchtext import vocab

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.round_3.round_3_models import PTLModule
from models.VAE.ConvVAE import ConvVAE, compute_scores_from_batch
from SA_utils import (
    generate_all_point_mutants,
    generate_random_mut_non_gap_indices,
    get_non_gap_indices,
    mut2seq,
)


AAS = "ACDEFGHIKLMNPQRSTVWY-"
WT = ('-------MAVKHLIVLKFKDEITEAQKEEFFKTYVNLVN--IIPAMKDVYW----GK-DVTQKNKEEGYTHIVEVTFESVETIQDYII-HPAHVGFGDVYRSFWEKLLIFDY-----TPRK-------')
WT_NO_GAPS = ('MAVKHLIVLKFKDEITEAQKEEFFKTYVNLVNIIPAMKDVYWGKDVTQKNKEEGYTHIVEVTFESVETIQDYIIHPAHVGFGDVYRSFWEKLLIFDYTPRK')

AA_TO_INDEX = vocab.vocab(OrderedDict((aa, 1) for aa in AAS))
AA_TO_INDEX.set_default_index(20)

RESIDUES_WITHIN_4A = [
    5, 7, 9, 23, 24, 27, 28, 30, 40, 49,
    59, 72, 73, 78, 81, 82, 89, 92, 94, 96,
]
FIXED_WINDOW_SIZE = 19
START_POSITION = 6
START_TEMP = 3.25
FINAL_TEMP = -2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-type", choices=("C4", "C6"), required=True)
    parser.add_argument("--num-mut", type=int, choices=(3, 6), required=True)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--nsteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--num-models", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "SA_single_objective")
    return parser.parse_args()


def mutation_settings(num_mut: int, nsteps: int | None) -> tuple[int, int]:
    if num_mut == 3:
        return 2, nsteps if nsteps is not None else 50_000
    return 3, nsteps if nsteps is not None else 50_000


def build_aa_options(
    non_gap_indices: list[int],
    aa_weights: dict[str, float],
) -> list[tuple[str, ...] | str | list[str]]:
    options: list[tuple[str, ...] | str | list[str]] = [
        tuple("ACDEFGHIKLMNPQRSTVWY") for _ in WT
    ]

    # Keep the initiating methionine fixed.
    options[non_gap_indices[0]] = WT_NO_GAPS[0]

    mutating_window_size = len(WT_NO_GAPS) - FIXED_WINDOW_SIZE
    window_end = START_POSITION + mutating_window_size

    for sequence_index, wt_aa in enumerate(WT_NO_GAPS):
        aligned_index = non_gap_indices[sequence_index]

        if sequence_index <= START_POSITION or sequence_index > window_end:
            options[aligned_index] = wt_aa
        elif sequence_index + 1 in RESIDUES_WITHIN_4A:
            wt_weight = aa_weights[wt_aa]
            options[aligned_index] = [
                aa for aa in "ACDEFGHIKLMNPQRSTVWY"
                if aa_weights[aa] <= wt_weight
            ]

    return options


def load_vae() -> ConvVAE:
    model = ConvVAE(
        slen=len(WT),
        ks=4,
        nlatent=40,
        learning_rate=7.5e-6,
    )
    weights_path = REPO_ROOT / "models" / "VAE" / "VAE_weights.pt"
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_ensemble(num_models: int) -> list[PTLModule]:
    n_exps = 10
    n_labels = n_exps * 4
    pattern = [1, 1, 1, 0.1]
    repeats = n_labels // len(pattern)
    remainder = n_labels % len(pattern)
    label_weights = pattern * repeats + pattern[:remainder]

    models: list[PTLModule] = []
    checkpoint_dir = REPO_ROOT / "models" / "round_3" / "model_ensemble"

    for model_index in range(num_models):
        model = PTLModule(
            len(WT),
            learning_rate=5e-6,
            epochs=10_000,
            weights=label_weights,
        )
        checkpoint_path = checkpoint_dir / f"round_3_EnsMLP_{model_index}.ckpt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = {
            key.replace("model.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    return models


@torch.no_grad()
def score_vae(sequence: str, model: ConvVAE) -> float:
    encoded = torch.tensor(
        AA_TO_INDEX(list(sequence)),
        dtype=torch.long,
    ).unsqueeze(0)
    score = -compute_scores_from_batch(encoded, model)
    return float(np.asarray(score).reshape(-1)[0])


@torch.no_grad()
def score_ensemble(
    sequence: str,
    models: list[PTLModule],
    product_type: str,
) -> float:
    predictions = [
        np.asarray(model.predict(sequence), dtype=float)
        for model in models
    ]
    lower_confidence_bound = np.quantile(predictions, q=0.05, axis=0)

    # Preserve the deposited round 3 product-specific objective definitions.
    if product_type == "C4":
        indices = (18, 26)
    else:
        indices = (22, 30)

    flattened = np.asarray(lower_confidence_bound).reshape(-1)
    return float(np.mean(flattened[list(indices)]))


class SimulatedAnnealingOptimizer:
    def __init__(
        self,
        score_function: Callable[[str], float],
        aa_options: list,
        num_mut: int,
        mut_rate: int,
        nsteps: int,
        non_gap_indices: list[int],
    ) -> None:
        self.score_function = score_function
        self.aa_options = aa_options
        self.num_mut = num_mut
        self.mut_rate = mut_rate
        self.nsteps = nsteps
        self.non_gap_indices = non_gap_indices
        self.fitness_trajectory: list[tuple[float, float]] = []

    def optimize(self) -> tuple[tuple[str, ...], float]:
        start_mutations = generate_random_mut_non_gap_indices(
            WT,
            self.aa_options,
            self.num_mut,
            self.non_gap_indices,
            len(WT_NO_GAPS) - FIXED_WINDOW_SIZE,
        ).split(",")

        all_point_mutants = generate_all_point_mutants(
            WT,
            self.non_gap_indices,
            self.aa_options,
        )
        temperatures = np.logspace(START_TEMP, FINAL_TEMP, self.nsteps)

        current_mutations = tuple(start_mutations)
        current_fitness = self.score_function(mut2seq(WT, current_mutations))
        best_mutations = current_mutations
        best_fitness = current_fitness
        self.fitness_trajectory = [(best_fitness, current_fitness)]

        for step, temperature in enumerate(temperatures, start=1):
            candidate = list(current_mutations)

            replacements = np.random.poisson(self.mut_rate)
            replacements = min(self.num_mut - 1, max(1, replacements))

            while len(candidate) > self.num_mut - replacements:
                candidate.pop(random.randrange(len(candidate)))

            occupied_positions = {mutation[1:-1] for mutation in candidate}
            mutation_options = [
                mutation
                for mutation in all_point_mutants
                if mutation[1:-1] not in occupied_positions
            ]

            while len(candidate) < self.num_mut:
                selected = random.choice(mutation_options)
                candidate.append(selected)
                occupied_positions.add(selected[1:-1])
                mutation_options = [
                    mutation
                    for mutation in all_point_mutants
                    if mutation[1:-1] not in occupied_positions
                ]

            candidate_tuple = tuple(
                mutation
                for _, mutation in sorted(
                    (int(mutation[1:-1]), mutation)
                    for mutation in candidate
                )
            )
            candidate_fitness = self.score_function(
                mut2seq(WT, candidate_tuple)
            )

            if candidate_fitness > best_fitness:
                best_mutations = candidate_tuple
                best_fitness = candidate_fitness

            delta = candidate_fitness - current_fitness
            acceptance_probability = np.exp(min(0.0, delta / temperature))

            if acceptance_probability > random.random():
                current_mutations = candidate_tuple
                current_fitness = candidate_fitness

            self.fitness_trajectory.append(
                (best_fitness, current_fitness)
            )

            if step == 1 or step % 1000 == 0 or step == self.nsteps:
                print(
                    f"step={step:,}/{self.nsteps:,} "
                    f"best={best_fitness:.6f} "
                    f"current={current_fitness:.6f}",
                    flush=True,
                )

        return best_mutations, best_fitness


def save_trajectory(
    trajectory: list[tuple[float, float]],
    csv_path: Path,
    plot_path: Path,
) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Step", "BestFitness", "CurrentFitness"])
        for step, (best, current) in enumerate(trajectory):
            writer.writerow([step, best, current])

    values = np.asarray(trajectory, dtype=float)
    plt.figure(figsize=(8, 5))
    plt.plot(values[:, 0], label="Best")
    plt.plot(values[:, 1], label="Current")
    plt.xlabel("Step")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    mut_rate, nsteps = mutation_settings(args.num_mut, args.nsteps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    non_gap_indices = get_non_gap_indices(WT)
    with (SCRIPT_DIR / "amino_acid_weight_lookup.pkl").open("rb") as handle:
        aa_weights = pickle.load(handle)
    aa_options = build_aa_options(non_gap_indices, aa_weights)

    print("Loading VAE...", flush=True)
    vae = load_vae()

    # The ensemble is not used during annealing. It is loaded only to score
    # each final VAE-optimal sequence and obtain the paired MTFCNN endpoint.
    print("Loading MLP ensemble for final endpoint scoring...", flush=True)
    ensemble = load_ensemble(args.num_models)

    output_dir = (
        args.output_root
        / "VAE_only"
        / f"{args.product_type}_{args.num_mut}mut_{nsteps}steps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_rows = []

    for trial in range(args.num_trials):
        trial_seed = args.seed + trial
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)

        print(
            f"Starting VAE-only trial {trial + 1}/{args.num_trials}; "
            f"seed={trial_seed}",
            flush=True,
        )

        optimizer = SimulatedAnnealingOptimizer(
            score_function=lambda sequence: score_vae(sequence, vae),
            aa_options=aa_options,
            num_mut=args.num_mut,
            mut_rate=mut_rate,
            nsteps=nsteps,
            non_gap_indices=non_gap_indices,
        )
        best_mutations, vae_score = optimizer.optimize()
        best_sequence = mut2seq(WT, best_mutations)
        mtlp_score = score_ensemble(
            best_sequence,
            ensemble,
            args.product_type,
        )

        row = {
            "trial": trial,
            "seed": trial_seed,
            "product_type": args.product_type,
            "num_mut": args.num_mut,
            "mutations": list(best_mutations),
            "VAE_score": vae_score,
            "MTFCNN_score": mtlp_score,
        }
        trial_rows.append(row)

        stem = f"trial_{trial:03d}"
        with (output_dir / f"{stem}_best.json").open("w") as handle:
            json.dump(row, handle, indent=2)

        save_trajectory(
            optimizer.fitness_trajectory,
            output_dir / f"{stem}_trajectory.csv",
            output_dir / f"{stem}_trajectory.png",
        )

    best_row = max(trial_rows, key=lambda row: row["VAE_score"])

    summary = {
        "objective": "VAE",
        "product_type": args.product_type,
        "num_mut": args.num_mut,
        "num_trials": args.num_trials,
        "nsteps": nsteps,
        "best_trial": best_row,
        "normalization_endpoint": {
            "VAE_Max": best_row["VAE_score"],
            "MTFCNN_Min": best_row["MTFCNN_score"],
        },
    }

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
