#!/usr/bin/env python
# coding: utf-8

import datetime
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import CSVLogger

from ConvVAE import ProtDataModule, ConvVAE, get_msa_from_fasta


# -----------------------------
# Parameters
# -----------------------------
torch.set_num_threads(4)

FASTA_PATH = "./model_data/syn_query_clean_55.fasta"
WEIGHTS_PATH = "./model_data/syn_query_cleaned_reweights_55.npy"

WT = (
    "MDAAKSQMAVKHLIVLKFKDEITEAQKEEFFKTYVNLVNKCIIPAMKDVYWLRSSGKLDVTQKNKEEGYTHIVEVTFESVETIQDYIIEHPAHVGFGDVYRSFWEKLLIFDYPSVLVTPRKIQLNSSY"
)

BATCH_SIZE = 64
KERNEL_SIZE = 4
N_LATENT = 40
EPOCHS = 50
LEARNING_RATE = 0.0000075
NUM_WORKERS = 0

LOG_DIR = "logs"
LOGGER_NAME = "ConvVAE"

CHECKPOINT_PATH = "VAE_weights.ckpt"
STATE_DICT_PREFIX = "VAE_weights"

PLOT_DIR = "ConvVAEGraphs"


# -----------------------------
# Load data
# -----------------------------
MSA = get_msa_from_fasta(FASTA_PATH)
sample_weights = np.load(WEIGHTS_PATH)

slen = len(WT)

data_module = ProtDataModule(
    MSA=MSA,
    batch_size=BATCH_SIZE,
    sample_weights=sample_weights,
    num_workers=NUM_WORKERS,
)


# -----------------------------
# Train model
# -----------------------------
model = ConvVAE(
    slen=slen,
    ks=KERNEL_SIZE,
    nlatent=N_LATENT,
    learning_rate=LEARNING_RATE,
)

logger = CSVLogger(LOG_DIR, name=LOGGER_NAME)

trainer = pl.Trainer(
    logger=logger,
    max_epochs=EPOCHS,
    accelerator="cpu",
)

trainer.fit(model, data_module)
trainer.save_checkpoint(CHECKPOINT_PATH)

# -----------------------------
# Save state dict
# -----------------------------
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
state_dict_path = f"{STATE_DICT_PREFIX}{timestamp}.pt"
torch.save(model.state_dict(), state_dict_path)

# -----------------------------
# Plot training curves
# -----------------------------
version = model.logger.version
metrics_path = os.path.join(LOG_DIR, LOGGER_NAME, f"version_{version}", "metrics.csv")

pt_metrics = pd.read_csv(metrics_path)

train = pt_metrics[~pt_metrics.train_ce_loss.isna()]
val = pt_metrics[~pt_metrics.val_ce_loss.isna()]

sns.lineplot(x=train.epoch, y=train.train_ce_loss / len(MSA))
sns.lineplot(x=val.epoch, y=val.val_ce_loss / len(MSA))

plt.title("loss vs. epoch")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["training loss", "validation loss"])

os.makedirs(PLOT_DIR, exist_ok=True)

plot_path = os.path.join(PLOT_DIR, f"ConvVAE_v_{version}.jpg")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
