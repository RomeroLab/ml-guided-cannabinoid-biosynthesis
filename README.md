Use this:

````markdown
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

### 2. Run Pareto simulated annealing

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

```
```
