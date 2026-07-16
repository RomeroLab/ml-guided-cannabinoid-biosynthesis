## Running simulated annealing

This repository includes two simulated annealing workflows:

1. `scripts/simulated_annealing_w_pareto_optimization.py`
2. `scripts/simulated_annealing_w_utopia_optimization.py`

### 1. Create the conda environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate running-simulated-annealing
````

### 2. Run pareto simulated annealing

From the repository root:

```bash
cd scripts
nohup python simulated_annealing_w_pareto_optimization.py > simulated_annealing_w_pareto_optimization.out 2>&1 &
```

Monitor the run:

```bash
tail -f simulated_annealing_w_pareto_optimization.out
```

### 3. Run utopia simulated annealing

From the repository root:

```bash
cd scripts
nohup python simulated_annealing_w_utopia_optimization.py > simulated_annealing_w_utopia_optimization.out 2>&1 &
```

Monitor the run:

```bash
tail -f simulated_annealing_w_utopia_optimization.out
```

### Notes

The scripts save outputs to workflow-specific output directories. Outputs include parameter files, best-mutant pickle files, close-sequence pickle files, fitness trajectory CSV files, and trajectory plots.

---

## Training the variational autoencoder

### 1. Create the conda environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate running-simulated-annealing
```

### 2. Prepare the training dataset

To reproduce the deposited model, use:

```text
models/VAE/model_data/syn_query_clean_55.fasta
models/VAE/model_data/syn_query_cleaned_reweights_55.npy
```

To train on a different dataset, provide:

- a FASTA file containing aligned protein sequences of equal length; and
- a NumPy `.npy` file containing one sequence weight per FASTA record, in the same order.

### 3. Configure the training script

In `models/VAE/Training_VAE.py`, set the dataset paths:

```python
FASTA_PATH = "model_data/syn_query_clean_55.fasta"
WEIGHTS_PATH = "model_data/syn_query_cleaned_reweights_55.npy"
```

For a custom dataset, replace these paths with your FASTA and weight files. Also update the `WT` sequence so that its length matches the aligned sequences.

The default training hyperparameters are:

```python
BATCH_SIZE = 64
KERNEL_SIZE = 4
N_LATENT = 40
EPOCHS = 50
LEARNING_RATE = 7.5e-6 # this is an important hyperparameter to optimize if using your own MSA
```

Modify these values in `Training_VAE.py` as needed.

### 4. Train the model

From the repository root:

```bash
cd models/VAE
nohup python Training_VAE.py > training_vae.out 2>&1 &
tail -f training_vae.out
```

### 5. Locate the trained model

Training writes the model checkpoint and state dictionary to `models/VAE/`, with training logs under:

```text
models/VAE/logs/ConvVAE/version_<N>/
```

---

## Training the round 3 MLP ensemble (predicts metabolite ratios)

### 1. Create the conda environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate running-simulated-annealing
```

### 2. Prepare the training data

To reproduce the deposited round 3 ensemble, use the supplied files:

```text
models/round_3/round_3_seq_fxn_data.pkl
models/round_3/round_3_data_splits.pkl
```

The sequence–function dataframe must contain:

- an `Aligned_Sequence` column containing equal-length aligned protein sequences
- the response columns listed in `columns_w_labels` in `round_3_EnsMLPs.py`

Missing response values may be stored as `NaN`. The training script replaces them with `-1` and masks them during loss calculation.

To train with a new dataset, save the dataframe as a pickle file and update `round_num`, the input filename, `WT`, `columns_w_labels`, and the number of model outputs as needed. To generate new splits instead of using the deposited split file, initialize `ProtDataModule` without a split path and call `save_splits(...)` as shown in the commented lines of `round_3_EnsMLPs.py`.

### 3. Configure ensemble training

The deposited round 3 configuration using `models/round_3/round_3_EnsMLPs.py` uses:

```python
round_num = "round_3"
seed = 777
batch_size = 32 # important to optimize using validation loss if using your custom dataset
learning_rate = 5e-6 # important to optimize using validation loss if using your custom dataset
epochs = 10000
num_models = 100
```

### 4. Train the ensemble

From the repository root:

```bash
cd models/round_3
nohup python round_3_EnsMLPs.py > round_3_ensemble.out 2>&1 &
tail -f round_3_ensemble.out
```

### 5. Locate the trained ensemble

The best checkpoint for each model, selected by minimum validation loss, is written to:

```text
models/round_3/model_ensemble/round_3_EnsMLP_<N>.ckpt
```

where `<N>` ranges from `0` to `num_models - 1`.

Training logs are written under:

```text
models/round_3/logs/ensemble_results/version_<N>/
```

