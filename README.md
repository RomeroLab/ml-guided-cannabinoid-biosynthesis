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
LEARNING_RATE = 7.5e-6 # this is an important hyperparameter is using a custom sequence dataset
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
