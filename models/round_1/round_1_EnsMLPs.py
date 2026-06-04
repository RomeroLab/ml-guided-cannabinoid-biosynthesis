#!/usr/bin/env python
# coding: utf-8

# Import packages
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import pytorch_lightning as pl
from collections import OrderedDict
from torchtext import vocab
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import metrics
from argparse import ArgumentParser
import torchmetrics
import pickle
import random
import statistics
import math
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import spearmanr
from round_1_models import (SeqFcnDataset, ProtDataModule, PTLModule)

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MDAAKSQMAVKHLIVLKFKDEITEAQKEEFFKTYVNLVNKCIIPAMKDVYWLRSSGKLDVTQKNKEEGYTHIVEVTFESVETIQDYIIEHPAHVGFGDVYRSFWEKLLIFDYPSVLVTPRKIQLNSSY' # Synthetic Query Sequence
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap

# parameters
round_num = 'round_1'
seed = 777
save_filepath = './model_ensemble'
log_filepath = './ensemble_results'
train_models = False
plot_loss_curve = True
perform_evals = True
columns_w_labels = ['0_DVA/C4', '0_DVA/DVO','0_DVA/(DVA+DVO)', '0_(DVA+DVO)/VDAL',
                        '1_DVA/C4', '1_DVA/DVO','1_DVA/(DVA+DVO)', '1_(DVA+DVO)/VDAL',
                        '2_OA/C6', '2_OA/OL', '2_OA/(OA+OL)', '2_(OA+OL)/PDAL',
                        '3_OA/C6', '3_OA/OL', '3_OA/(OA+OL)', '3_(OA+OL)/PDAL',
                        '4_DVA/C4', '4_DVA/DVO', '4_DVA/(DVA+DVO)', '4_(DVA+DVO)/VDAL',
                        '5_OA/C6', '5_OA/OL', '5_OA/(OA+OL)', '5_(OA+OL)/PDAL']

batch_size = 32 # typically powers of 2: 32, 64, 128, 256, ...
slen = len(WT) # length of protein
learning_rate = 5e-6 # important to optimize this
epochs = 100 # rounds of training
num_models = 100 # 100 # number of models in ensemble

n_exps = 6
n_labels = n_exps*4
pattern = [0.2, 1 ,1 ,0.1]
full_repeats = n_labels // len(pattern)
additional_elements = n_labels % len(pattern)
weights = pattern * full_repeats + pattern[:additional_elements]

DVA_COLOR = "#c51b7d"
OA_COLOR  = "#762a83"
MARKERS   = {"train": "s", "val": "^", "test": "o"}
DVA_PREFIXES = ["0", "1", "4"]
OA_PREFIXES  = ["2", "3", "5"]

# load data
df = pd.read_pickle(f"{round_num}_seq_fxn_data.pkl")

# process data
max_values = df.iloc[:, 3:].max() # Find max value in each column after the first three columns
df.iloc[:, 3:] = df.iloc[:, 3:].div(max_values) # Divide each column by its respective max value to avoid upweighting labels
df = df.fillna(-1) # Fill NaN values with -1, I will mask these values later

# dm = ProtDataModule(df, batch_size) # dm an instance of the class defined above, see notes above for its purpose
# dm.save_splits(f'{round_num}_data_splits.pkl')
dm = ProtDataModule(df, batch_size, f'{round_num}_data_splits.pkl') # dm an instance of the class defined above, see notes above for its purpose

# Training Models
if train_models:
    torch.manual_seed(seed)
    random.seed(seed)
    for i in range(num_models):
        model = PTLModule(slen, learning_rate, epochs) # Instantiate the model previously defined
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_filepath,
            filename=f"{round_num}_EnsMLP_{i}",
            monitor="val_loss",
            mode="min",
            save_top_k=1)
        early_stopping = EarlyStopping(monitor="val_loss", patience=200, mode="min")
        logger = CSVLogger('logs', name=log_filepath) # logger is a class instance that stores performance data to a csv after each epoch
        trainer = pl.Trainer(logger=logger, max_epochs=epochs, callbacks=[checkpoint_callback, early_stopping], enable_progress_bar=True) # trainer is the class PTL uses for fitting a model and saving checkpoints
        trainer.fit(model, dm) ### errors will arise here if the dimensions are incorrectly defined

# Plot loss curves
if plot_loss_curve:
    try:
        # create an empty list to store the loss curves
        train_losses = []
        val_losses = []

        # loop over the versions 0 to 99 and append the loss curves to the lists
        for i in range(num_models):
            version = i
            try:
                pt_metrics = pd.read_csv(f'./logs/{log_filepath}/version_{i}/metrics.csv')
                train = pt_metrics[~pt_metrics.train_loss.isna()]
                val = pt_metrics[~pt_metrics.val_loss.isna()]
                train_losses.append(train.train_loss)
                val_losses.append(val.val_loss)
            except:
                pass

        # Assuming the length of the training might vary, find the max length
        max_length = max([max(len(tl), len(vl)) for tl, vl in zip(train_losses, val_losses)])

        # Pad the loss arrays to have the same length
        train_losses_padded = [np.pad(tl, (0, max_length - len(tl)), 'constant', constant_values=np.nan) for tl in train_losses]
        val_losses_padded = [np.pad(vl, (0, max_length - len(vl)), 'constant', constant_values=np.nan) for vl in val_losses]

        # calculate the mean and standard deviation of the loss curves
        train_mean = np.mean(train_losses_padded, axis=0)
        val_mean = np.mean(val_losses_padded, axis=0)
        train_std = np.std(train_losses_padded, axis=0)
        val_std = np.std(val_losses_padded, axis=0)
        epochs = np.arange(1, max_length + 1)

        # plot the mean loss curves with shaded standard deviation regions
        plt.plot(epochs, train_mean, label='training loss')
        plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)
        plt.plot(epochs, val_mean, label='validation loss')
        plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)
        plt.title('loss vs. epoch')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()

        # save the plot to a file
        if not os.path.exists(save_filepath):
            os.makedirs(save_filepath)
        file_path = os.path.join(save_filepath, f'{round_num}_EnsMLP_Loss.svg')
        plt.savefig(file_path)

        file_path = os.path.join(save_filepath, f'{round_num}_EnsMLP_Loss.svg')
        plt.savefig(file_path)
    except:
        print('No loss data found')
        pass

# Plot model predictions
if perform_evals:

    # Empty list to store all Y values
    all_Y_values = []
    all_Y_train_values = []
    all_Y_val_values = []

    # Scores Sequences for Models
    for i in range(num_models):
        model = PTLModule(slen, learning_rate, epochs) # Instantiate the model previously defined
        model.load_state_dict(torch.load(f'{save_filepath}/{round_num}_EnsMLP_{i}.pt')) # Load the saved parameters into the model
        model.eval()
        test_data_frame = df.iloc[list(dm.test_idx)].copy() # Define the test data frame using the dm splits
        train_data_frame = df.iloc[list(dm.train_idx)].copy() # Define the test data frame using the dm splits
        val_data_frame = df.iloc[list(dm.val_idx)].copy() # Define the test data frame using the dm splits

        X = test_data_frame[columns_w_labels].values
        Y = [model.predict(i).astype(float) for i in test_data_frame['sequence']] # Predict the scores for each sequence in the test data using the `model` and create a list of arrays where each array corresponds to the predicted scores for one sequence. The `astype(float)` converts the predicted scores to floats.
        Y = np.concatenate(Y, axis=0) # Concatenate the arrays of predicted scores into a single array

        X_train = train_data_frame[columns_w_labels].values
        Y_train = [model.predict(i).astype(float) for i in train_data_frame['sequence']] # Predict the scores for each sequence in the test data using the `model` and create a list of arrays where each array corresponds to the predicted scores for one sequence. The `astype(float)` converts the predicted scores to floats.
        Y_train = np.concatenate(Y_train, axis=0) # Concatenate the arrays of predicted scores into a single array

        X_val = val_data_frame[columns_w_labels].values
        Y_val = [model.predict(i).astype(float) for i in val_data_frame['sequence']] # Predict the scores for each sequence in the test data using the `model` and create a list of arrays where each array corresponds to the predicted scores for one sequence. The `astype(float)` converts the predicted scores to floats.
        Y_val = np.concatenate(Y_val, axis=0) # Concatenate the arrays of predicted scores into a single array

        # Remove Missing Labels and Corresponding Scores
        X[X == -1] = 0
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if X[i][j] == 0:
                    Y[i][j] = 0

        X_train[X_train == -1] = 0
        for i in range(len(Y_train)):
            for j in range(len(Y_train[i])):
                if X_train[i][j] == 0:
                    Y_train[i][j] = 0

        X_val[X_val == -1] = 0
        for i in range(len(Y_val)):
            for j in range(len(Y_val[i])):
                if X_val[i][j] == 0:
                    Y_val[i][j] = 0
        
        # Store Y values
        Y = Y.tolist()
        all_Y_values.extend(Y)

        # Store Y values
        Y_train = Y_train.tolist()
        all_Y_train_values.extend(Y_train)

        # Store Y values
        Y_val = Y_val.tolist()
        all_Y_val_values.extend(Y_val)

    # Set X values for each test sequence as actual label scores 
    X = X.tolist()
    for i in range(len(test_data_frame)):
        globals()[f"test_seq{i+1}_actual"] = X[i]

    X_train = X_train.tolist()
    for i in range(len(train_data_frame)):
        globals()[f"train_seq{i+1}_actual"] = X_train[i]

    X_val = X_val.tolist()
    for i in range(len(val_data_frame)):
        globals()[f"val_seq{i+1}_actual"] = X_val[i]

    # loop through test_seq1_actual to test_seq15_actual and replace 0 values with NaN
    for i in range(1, len(test_data_frame)+1):
        seq_name = f"test_seq{i}_actual"
        globals()[seq_name] = np.array(globals()[seq_name])
        globals()[seq_name][globals()[seq_name] == 0] = np.nan

    for i in range(1, len(train_data_frame)+1):
        seq_name = f"train_seq{i}_actual"
        globals()[seq_name] = np.array(globals()[seq_name])
        globals()[seq_name][globals()[seq_name] == 0] = np.nan

    for i in range(1, len(val_data_frame)+1):
        seq_name = f"val_seq{i}_actual"
        globals()[seq_name] = np.array(globals()[seq_name])
        globals()[seq_name][globals()[seq_name] == 0] = np.nan

    # print(test_seq1_actual)

    test_sequences = [all_Y_values[i::len(test_data_frame)] for i in range(len(test_data_frame))] # Split all_Y_values into test sequences, each corresponding to a single test sequence.
    train_sequences = [all_Y_train_values[i::len(train_data_frame)] for i in range(len(train_data_frame))]
    val_sequences = [all_Y_val_values[i::len(val_data_frame)] for i in range(len(val_data_frame))]

    label_scores_dict = {} # Initialize an empty dictionary to store the label scores for each test sequence.
    for i, test_seq in enumerate(test_sequences): # Loop over each of the 15 test sequences.
        label_scores = [[seq[j] for seq in test_seq] for j in range(len(weights))] # For each test sequence, extract the scores for each label and store them in a list of 24 lists.
        label_scores_dict[f"Test sequence {i+1}"] = label_scores # Add the label scores for the current test sequence to the dictionary with a key of "Test sequence {i+1}".

    for i, train_seq in enumerate(train_sequences):
        label_scores = [[seq[j] for seq in train_seq] for j in range(len(weights))]
        label_scores_dict[f"Train sequence {i+1}"] = label_scores

    for i, val_seq in enumerate(val_sequences):
        label_scores = [[seq[j] for seq in val_seq] for j in range(len(weights))]
        label_scores_dict[f"Val sequence {i+1}"] = label_scores

    test_seq_medians = [(test_seq, [statistics.median(label_scores_dict[f"Test sequence {i+1}"][j]) for j in range(len(weights))]) for i, test_seq in enumerate(test_sequences)] # For each test sequence, calculate the median score for each label and store the results in a list of tuples, where each tuple contains the test sequence number and a list of median scores.
    train_seq_medians = [(train_seq, [statistics.median(label_scores_dict[f"Train sequence {i+1}"][j]) for j in range(len(weights))]) for i, train_seq in enumerate(train_sequences)]
    val_seq_medians = [(val_seq, [statistics.median(label_scores_dict[f"Val sequence {i+1}"][j]) for j in range(len(weights))]) for i, val_seq in enumerate(val_sequences)]

    # Extract the median scores for each label for each test sequence and store them in individual variables.
    for i in range(len(test_seq_medians)):
        exec(f"test_seq{i+1}_medians = test_seq_medians[{i}][1]")

    for i in range(len(train_seq_medians)):
        exec(f"train_seq{i+1}_medians = train_seq_medians[{i}][1]")

    for i in range(len(val_seq_medians)):
        exec(f"val_seq{i+1}_medians = val_seq_medians[{i}][1]")

    # print(test_seq1_medians) # Print the median scores for the first test sequence.

    # loop through test_seq1_actual to test_seq15_actual and replace 0 values with NaN
    for i in range(1, len(test_data_frame)+1):
        seq_name = f"test_seq{i}_medians"
        globals()[seq_name] = np.array(globals()[seq_name])
        globals()[seq_name][globals()[seq_name] == 0] = np.nan

    for i in range(1, len(train_data_frame)+1):
        seq_name = f"train_seq{i}_medians"
        globals()[seq_name] = np.array(globals()[seq_name])
        globals()[seq_name][globals()[seq_name] == 0] = np.nan

    for i in range(1, len(val_data_frame)+1):
        seq_name = f"val_seq{i}_medians"
        globals()[seq_name] = np.array(globals()[seq_name])
        globals()[seq_name][globals()[seq_name] == 0] = np.nan
        
    # print(test_seq1_medians)

    num_sequences = len(test_data_frame)
    num_sequences_train = len(train_data_frame)
    num_sequences_val = len(val_data_frame)
    num_values = len(weights)

    x_values = [[] for _ in range(num_values)]
    y_values = [[] for _ in range(num_values)]
    x_train_values = [[] for _ in range(num_values)]
    y_train_values = [[] for _ in range(num_values)]
    x_val_values = [[] for _ in range(num_values)]
    y_val_values = [[] for _ in range(num_values)]

    # Loop over the test sequences and create scatter plots for the given label
    for i in range(num_sequences):
        test_seq_name = f"test_seq{i+1}_medians"
        actual_seq_name = f"test_seq{i+1}_actual"
        
        for j in range(num_values):
            x_values[j].append(globals()[actual_seq_name][j])
            y_values[j].append(globals()[test_seq_name][j])

    for i in range(num_sequences_train):
        train_seq_name = f"train_seq{i+1}_medians"
        actual_seq_name = f"train_seq{i+1}_actual"
        
        for j in range(num_values):
            x_train_values[j].append(globals()[actual_seq_name][j])
            y_train_values[j].append(globals()[train_seq_name][j])

    for i in range(num_sequences_val):
        val_seq_name = f"val_seq{i+1}_medians"
        actual_seq_name = f"val_seq{i+1}_actual"
        
        for j in range(num_values):
            x_val_values[j].append(globals()[actual_seq_name][j])
            y_val_values[j].append(globals()[val_seq_name][j])
            
    # print(test_data_frame)
    # print(x_values)

    fig, axs = plt.subplots(10, 4, figsize=(10, 20))
    fig.tight_layout()

    for i in range(len(weights)):
        axs[i//4, i%4].scatter(x_train_values[i], y_train_values[i], color='blue', s = 5) # Plot and annotate the test scatterplot
        axs[i//4, i%4].scatter(x_val_values[i], y_val_values[i], color='orange', s = 5) # Plot and annotate the test scatterplot
        axs[i//4, i%4].scatter(x_values[i], y_values[i], color='red', s = 5) # Plot and annotate the test scatterplot
        axs[i//4, i%4].plot([0, 1], [0, 1], color='black')
        axs[i//4, i%4].set_xlim([0, 1])
        axs[i//4, i%4].set_ylim([0, 1])
        axs[i//4, i%4].set_ylabel("Predicted Score", fontsize=8)
        axs[i//4, i%4].set_xlabel(columns_w_labels[i], fontsize=8)
        
        x_values_formatted = [x for x in x_values[i] if not math.isnan(x)]
        y_values_formatted = [y for y in y_values[i] if not math.isnan(y)]
        
        mse = metrics.mean_squared_error(x_values_formatted, y_values_formatted)
        r = np.corrcoef(x_values_formatted, y_values_formatted)[0][1]  # Extract the correlation coefficient
        rho, _ = spearmanr(x_values_formatted, y_values_formatted)
        

        # Add annotations to the plot
        axs[i//4, i%4].text(0.05, 0.9, f"MSE = {mse:.2f}", fontsize=8)
        axs[i//4, i%4].text(0.05, 0.8, f"R = {r :.2f}", fontsize=8)
        axs[i//4, i%4].text(0.05, 0.7, f"Rho = {rho:.2f}", fontsize=8)

    # Save the plot as a file
    file_path = os.path.join(save_filepath, f'{round_num}_EnsMLP_Test.svg')
    plt.savefig(file_path)

    # Save the plot as a file
    file_path = os.path.join(save_filepath, f'{round_num}_EnsMLP_Test.png')
    plt.savefig(file_path)

    # --- Save per-label metrics (train/val/test) to CSVs ---
    def _safe_metrics(x_list, y_list):
        x = np.asarray(x_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        n = x.size
        if n < 2:
            return np.nan, np.nan, np.nan, n
        mse = metrics.mean_squared_error(x, y)
        r = np.nan if (np.std(x) == 0 or np.std(y) == 0) else np.corrcoef(x, y)[0, 1]
        rho = spearmanr(x, y, nan_policy='omit')[0]
        return mse, r, rho, n

    train_rows, val_rows, test_rows = [], [], []

    for i, label in enumerate(columns_w_labels):
        # train
        mse, r, rho, n = _safe_metrics(x_train_values[i], y_train_values[i])
        train_rows.append({"label": label, "mse": mse, "r": r, "rho": rho, "n_points": n})

        # val
        mse, r, rho, n = _safe_metrics(x_val_values[i], y_val_values[i])
        val_rows.append({"label": label, "mse": mse, "r": r, "rho": rho, "n_points": n})

        # test
        mse, r, rho, n = _safe_metrics(x_values[i], y_values[i])
        test_rows.append({"label": label, "mse": mse, "r": r, "rho": rho, "n_points": n})

    train_df = pd.DataFrame(train_rows)
    val_df   = pd.DataFrame(val_rows)
    test_df  = pd.DataFrame(test_rows)

    os.makedirs(save_filepath, exist_ok=True)
    train_csv = os.path.join(save_filepath, f"{round_num}_EnsMLP_metrics_train.csv")
    val_csv   = os.path.join(save_filepath, f"{round_num}_EnsMLP_metrics_val.csv")
    test_csv  = os.path.join(save_filepath, f"{round_num}_EnsMLP_metrics_test.csv")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Saved: {train_csv}")
    print(f"Saved: {val_csv}")
    print(f"Saved: {test_csv}")

if perform_evals:
    # Aggregated family plots (DVA top row, OA bottom row) =====

    # Families and base labels
    DVA_BASE = ["DVA/C4", "DVA/DVO", "DVA/(DVA+DVO)", "(DVA+DVO)/VDAL"]
    OA_BASE  = ["OA/C6",  "OA/OL",   "OA/(OA+OL)",    "(OA+OL)/PDAL"]

    # Build helper maps
    col_index_by_name = {c: i for i, c in enumerate(columns_w_labels)}
    def col_idxs(prefixes, base):
        names = [f"{p}_{base}" for p in prefixes]
        return [col_index_by_name[n] for n in names if n in col_index_by_name]

    DVA_GROUPS = {base: col_idxs(DVA_PREFIXES, base) for base in DVA_BASE}
    OA_GROUPS  = {base: col_idxs(OA_PREFIXES,  base) for base in OA_BASE}

    # ---- Unnormalize (× max) then normalize by PKC (÷ pkc_raw) ----
    # max per label (aligned to columns_w_labels)
    max_vec = np.array([float(max_values[c]) for c in columns_w_labels], dtype=float)

    # PKC row (normalized in df) -> back to raw by × max_vec
    if "name" in df.columns and (df["name"] == "PKC1.0").any():
        pkc_norm = df.loc[df["name"] == "PKC1.0", columns_w_labels].iloc[0].to_numpy(dtype=float)
        pkc_raw  = pkc_norm * max_vec
    else:
        raise RuntimeError("PKC001.000 row not found in df['name'].")

    # --- Compute relative std of pkc_norm across prefixes for each base label ---
    if "name" in df.columns and (df["name"] == "PKC1.0").any():
        pkc_rows = df.loc[df["name"] == "PKC1.0", columns_w_labels].to_numpy(dtype=float)
    else:
        raise RuntimeError("PKC001.000 row not found in df['name'].")

    if pkc_rows.ndim == 1:
        pkc_rows = pkc_rows[None, :]  # ensure 2D

    def compute_base_rel_std(group_dict):
        rel_std = {}
        for base, idxs in group_dict.items():
            if not idxs:
                rel_std[base] = np.nan
                continue
            vals = pkc_rows[:, idxs]              # shape: (n_reps, n_prefixes)
            means = np.nanmean(vals, axis=0)      # one value per prefix
            mu = np.nanmean(means)
            sigma = np.nanstd(means, ddof=1) if means.size > 1 else 0.0
            rel_std[base] = sigma / mu if mu != 0 else np.nan
        return rel_std

    DVA_REL_STD = compute_base_rel_std(DVA_GROUPS)
    OA_REL_STD  = compute_base_rel_std(OA_GROUPS)
    BASE_REL_STD = {**DVA_REL_STD, **OA_REL_STD}

    def _transform_lists(lists_by_label):
        """Take list-of-arrays per label; unnormalize ×max, then ÷PKC per-label."""
        out = []
        for i, arr in enumerate(lists_by_label):
            a = np.asarray(arr, dtype=float)
            a = a * max_vec[i]
            den = pkc_raw[i]
            if np.isnan(den) or den == 0.0:
                a = np.full_like(a, np.nan)
            else:
                a = a / den
            out.append(a)
        return out

    # Apply transform to actual (x*) and predicted (y*) across splits
    x_train_t = _transform_lists(x_train_values)
    y_train_t = _transform_lists(y_train_values)
    x_val_t   = _transform_lists(x_val_values)
    y_val_t   = _transform_lists(y_val_values)
    x_test_t  = _transform_lists(x_values)
    y_test_t  = _transform_lists(y_values)

    # ---- Plot 1 row × 8 cols ----
    fig, axs = plt.subplots(1, 8, figsize=(16, 4))
    axs = np.atleast_1d(axs)

    def _aggregate(idxs, xl, yl):
        if not idxs:
            return np.array([]), np.array([])
        x = np.concatenate([xl[i] for i in idxs]) if len(idxs) > 1 else xl[idxs[0]]
        y = np.concatenate([yl[i] for i in idxs]) if len(idxs) > 1 else yl[idxs[0]]
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    # Order: DVA (4) then OA (4)
    ordered = [(DVA_COLOR, DVA_BASE[j], list(DVA_GROUPS[DVA_BASE[j]])) for j in range(4)] + \
              [(OA_COLOR,  OA_BASE[j],  list(OA_GROUPS[OA_BASE[j]]))   for j in range(4)]

    for j, (color, base, idxs) in enumerate(ordered):
        ax = axs[j]

        # aggregate across prefixes for this base label
        xt, yt = _aggregate(idxs, x_train_t, y_train_t)
        xv, yv = _aggregate(idxs, x_val_t,   y_val_t)
        x , y  = _aggregate(idxs, x_test_t,  y_test_t)

        if xt.size: ax.scatter(xt, yt, s=12, marker=MARKERS["train"], color=color, label="train")
        if xv.size: ax.scatter(xv, yv, s=12, marker=MARKERS["val"],   color=color, label="val")
        if x.size:  ax.scatter(x,  y,  s=12, marker=MARKERS["test"],  color=color, label="test")

        ax.plot([0, 2], [0, 2])
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        rel_std = BASE_REL_STD.get(base, np.nan)
        if np.isfinite(rel_std) and rel_std > 0:
            ax.axvspan(1.0 - rel_std, 1.0 + rel_std, alpha=0.15, color='gray')
        ax.axvline(1.0, ls='--', lw=1, color='k')
        ax.set_xlabel(base, fontsize=9)
        if j == 0:
            ax.set_ylabel("Pred / PKC", fontsize=9)

        # Metrics on TEST only
        if x.size > 1:
            mse = metrics.mean_squared_error(x, y)
            r   = np.corrcoef(x, y)[0, 1]
            rho = spearmanr(x, y, nan_policy='omit')[0]
            ax.text(0.05, 0.92, f"MSE={mse:.2f}", transform=ax.transAxes, fontsize=8)
            ax.text(0.05, 0.84, f"R={r:.2f}",     transform=ax.transAxes, fontsize=8)
            ax.text(0.05, 0.76, f"Rho={rho:.2f}", transform=ax.transAxes, fontsize=8)

        # tint frame by family color
        for sp in ax.spines.values():
            sp.set_edgecolor(color)

    # Legend for splits (shape encodes split; color encodes label family)
    handles = [
        plt.Line2D([], [], marker=MARKERS["train"], linestyle='None', label='train', markersize=6, color='k'),
        plt.Line2D([], [], marker=MARKERS["val"],   linestyle='None', label='val',   markersize=6, color='k'),
        plt.Line2D([], [], marker=MARKERS["test"],  linestyle='None', label='test',  markersize=6, color='k'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=9, title="Split")

    os.makedirs(save_filepath, exist_ok=True)
    plt.savefig(os.path.join(save_filepath, f'compiled_{round_num}_EnsMLP_Test.svg'))
    plt.savefig(os.path.join(save_filepath, f'compiled_{round_num}_EnsMLP_Test.png'))

    # --- collect metrics into a CSV ---
    def _safe_metrics(x, y):
        n = x.size
        if n < 2:
            return np.nan, np.nan, np.nan, n
        mse = metrics.mean_squared_error(x, y)
        # Pearson r: guard against zero variance
        if np.all(x == x[0]) or np.all(y == y[0]):
            r = np.nan
        else:
            r = np.corrcoef(x, y)[0, 1]
        rho = spearmanr(x, y, nan_policy='omit')[0]
        return mse, r, rho, n

    metrics_rows = []
    for j, (color, base, idxs) in enumerate(ordered):
        family = "DVA" if base in DVA_BASE else "OA"

        # re-aggregate (same as above)
        xt, yt = _aggregate(idxs, x_train_t, y_train_t)
        xv, yv = _aggregate(idxs, x_val_t,   y_val_t)
        x , y  = _aggregate(idxs, x_test_t,  y_test_t)

        for split_name, xs, ys in (("train", xt, yt), ("val", xv, yv), ("test", x, y)):
            mse, r, rho, n = _safe_metrics(xs, ys)
            metrics_rows.append({
                "family": family,              # DVA or OA
                "base_label": base,            # e.g., "DVA/C4"
                "split": split_name,           # train/val/test
                "mse": mse,
                "r": r,
                "rho": rho,
                "n_points": n
            })

    metrics_df = pd.DataFrame(metrics_rows)
    os.makedirs(save_filepath, exist_ok=True)
    metrics_csv_path = os.path.join(save_filepath, f'compiled_{round_num}_EnsMLP_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics to {metrics_csv_path}")

    # --- Score VAE single-point mutants with the ensemble and add aggregated base scores ---
    vae_savepath = '../VAE/vae_single_point_mutations.csv'
    vae_df = pd.read_csv(vae_savepath)

    # Robustly find the sequence column
    seq_col_candidates = [c for c in vae_df.columns if c.lower() in {"sequence","seq","mutant_sequence","mut_seq"}]
    if not seq_col_candidates:
        raise ValueError("No sequence column found in VAE CSV. Expected one of: sequence, seq, mutant_sequence, mut_seq.")
    SEQ_COL = seq_col_candidates[0]
    vae_seqs = vae_df[SEQ_COL].astype(str).tolist()

    # Build ensemble (reload cleanly for inference)
    models = []
    for i in range(num_models):
        m = PTLModule(slen, learning_rate, epochs)
        m.load_state_dict(torch.load(f'{save_filepath}/{round_num}_EnsMLP_{i}.pt', map_location='cpu'))
        m.eval()
        models.append(m)

    # Predict per model -> stack -> median across ensemble
    def _predict_matrix(model, seqs):
        # shape: (n_seqs, n_labels)
        return np.stack([np.asarray(model.predict(s), dtype=float).ravel() for s in seqs], axis=0)

    pred_mats = [_predict_matrix(m, vae_seqs) for m in models]                    # list of (n_seqs, n_labels)
    pred_ens_med = np.median(np.stack(pred_mats, axis=0), axis=0)                 # (n_seqs, n_labels)

    # Unnormalize (× max per label) then normalize by PKC (÷ pkc_raw)
    # max_vec and pkc_raw must be aligned to columns_w_labels (you already created both above)
    pred_raw = pred_ens_med * max_vec                                            # (n_seqs, n_labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        pred_rel = pred_raw / pkc_raw                                            # relative to PKC1.0
        pred_rel[~np.isfinite(pred_rel)] = np.nan

    col_index_by_name = {c: i for i, c in enumerate(columns_w_labels)}
    def _col_idxs(prefixes, base):
        names = [f"{p}_{base}" for p in prefixes]
        return [col_index_by_name[n] for n in names if n in col_index_by_name]

    DVA_GROUPS = {base: _col_idxs(DVA_PREFIXES, base) for base in DVA_BASE}
    OA_GROUPS  = {base: _col_idxs(OA_PREFIXES,  base) for base in OA_BASE}

    # Aggregate per base label for each sequence (mean across prefixes)
    def _agg_row(y_rel_row, idxs):
        if not idxs:
            return np.nan
        vals = y_rel_row[idxs]
        return float(np.nanmean(vals)) if np.size(vals) else np.nan

    agg_cols = {}
    for base, idxs in {**DVA_GROUPS, **OA_GROUPS}.items():
        agg_vals = np.array([_agg_row(pred_rel[i, :], idxs) for i in range(pred_rel.shape[0])], dtype=float)
        # Column name; keep slashes to stay consistent with your schema
        agg_cols[f"{round_num}_ens_{base}"] = agg_vals

    # Attach aggregated columns to the VAE dataframe
    for k, v in agg_cols.items():
        vae_df[k] = v

    # Save
    vae_df.to_csv(vae_savepath, index=False)
    print(f"VAE mutants scored and saved: {vae_savepath}")

    # --- Score experimentally verified variants with the ensemble and add aggregated base scores
    experimental_data_savepath = '../../data/all_SeqFxnData_with_predicted_scores.csv'
    experimental_df = pd.read_csv(experimental_data_savepath)

    # Robustly find the sequence column
    seq_col_candidates = [c for c in experimental_df.columns if c.lower() in {"aligned_sequence"}]
    if not seq_col_candidates:
        raise ValueError("No sequence column found in CSV. Expected: Aligned_Sequence")
    SEQ_COL = seq_col_candidates[0]
    vae_seqs = experimental_df[SEQ_COL].astype(str).tolist()

    # Build ensemble (reload cleanly for inference)
    models = []
    for i in range(num_models):
        m = PTLModule(slen, learning_rate, epochs)
        m.load_state_dict(torch.load(f'{save_filepath}/{round_num}_EnsMLP_{i}.pt', map_location='cpu'))
        m.eval()
        models.append(m)

    # Predict per model -> stack -> median across ensemble
    def _predict_matrix(model, seqs):
        # shape: (n_seqs, n_labels)
        return np.stack([np.asarray(model.predict(s), dtype=float).ravel() for s in seqs], axis=0)

    pred_mats = [_predict_matrix(m, vae_seqs) for m in models]                    # list of (n_seqs, n_labels)
    pred_ens_med = np.median(np.stack(pred_mats, axis=0), axis=0)                 # (n_seqs, n_labels)

    # Unnormalize (× max per label) then normalize by PKC (÷ pkc_raw)
    # max_vec and pkc_raw must be aligned to columns_w_labels (you already created both above)
    pred_raw = pred_ens_med * max_vec                                            # (n_seqs, n_labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        pred_rel = pred_raw / pkc_raw                                            # relative to PKC1.0
        pred_rel[~np.isfinite(pred_rel)] = np.nan

    col_index_by_name = {c: i for i, c in enumerate(columns_w_labels)}
    def _col_idxs(prefixes, base):
        names = [f"{p}_{base}" for p in prefixes]
        return [col_index_by_name[n] for n in names if n in col_index_by_name]

    DVA_GROUPS = {base: _col_idxs(DVA_PREFIXES, base) for base in DVA_BASE}
    OA_GROUPS  = {base: _col_idxs(OA_PREFIXES,  base) for base in OA_BASE}

    # Aggregate per base label for each sequence (mean across prefixes)
    def _agg_row(y_rel_row, idxs):
        if not idxs:
            return np.nan
        vals = y_rel_row[idxs]
        return float(np.nanmean(vals)) if np.size(vals) else np.nan

    agg_cols = {}
    for base, idxs in {**DVA_GROUPS, **OA_GROUPS}.items():
        agg_vals = np.array([_agg_row(pred_rel[i, :], idxs) for i in range(pred_rel.shape[0])], dtype=float)
        # Column name; keep slashes to stay consistent with your schema
        agg_cols[f"{round_num}_ens_{base}"] = agg_vals

    # Attach aggregated columns to the VAE dataframe
    for k, v in agg_cols.items():
        experimental_df[k] = v

    # Save
    experimental_df.to_csv(experimental_data_savepath, index=False)
    print(f"Experimental mutants scored and saved: {experimental_data_savepath}")










