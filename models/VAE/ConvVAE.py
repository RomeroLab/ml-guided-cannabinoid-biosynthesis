#!/usr/bin/env python
# coding: utf-8

# Import packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
from collections import OrderedDict
from torchtext import vocab
import matplotlib.pyplot as plt
import os
import seaborn as sns
from typing import Sequence, Optional, Tuple, Set

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MDAAKSQMAVKHLIVLKFKDEITEAQKEEFFKTYVNLVNKCIIPAMKDVYWLRSSGKLDVTQKNKEEGYTHIVEVTFESVETIQDYIIEHPAHVGFGDVYRSFWEKLLIFDYPSVLVTPRKIQLNSSY' #synthetic query seq
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap

# Load Data from MSA for ConvVAE format
def get_msa_from_fasta(filename):
    import Bio.SeqIO
    with open(filename, "rt") as fh: 
        return [r[1] for r in Bio.SeqIO.FastaIO.SimpleFastaParser(fh)]

# ProtMSA is a helper class for loading and formating the data from MSA to a torch tensor
class ProtMSA(torch.utils.data.Dataset):
   
    def __init__(self, MSA):
        self.MSA = MSA

    def __getitem__(self, idx):
        # index the MSA
        sequence = torch.tensor(aa2ind(list(self.MSA[idx])))
        return sequence

    def __len__(self):
        return len(self.MSA)

# ProtDataModule is a helper class for splitting data into Training, Validation, and Test splits
class ProtDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data splitting"""

    def __init__(self, MSA, batch_size, sample_weights, num_workers):
        super().__init__()
        self.MSA = MSA
        self.batch_size = batch_size
        train_val_test_split = [0.89314, 0.1, 0.00686] # 100 Test Sequences
        self.sample_weights = sample_weights
        self.num_workers = num_workers
        if self.sample_weights is not None:
            assert(len(self.MSA) == self.sample_weights.shape[0])
        n_train_val_test = np.round(np.array(train_val_test_split)*len(MSA)).astype(int)
        if sum(n_train_val_test)<len(MSA): n_train_val_test[0] += 1 # necesary when round is off by 1
        if sum(n_train_val_test)>len(MSA): n_train_val_test[0] -= 1 
        self.train_idx, self.val_idx, self.test_idx = data_utils.random_split(range(len(MSA)),n_train_val_test)

    def prepare_data(self):
        # prepare_data is called from a single GPU. Do not use it to assign state (self.x = y)
        # use this method to do things that might write to disk or that need to be done only from a single process
        # in distributed settings.
        pass
        
    def setup(self, stage=None):
              
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_MSA = [self.MSA[i] for i in self.train_idx]
            self.train_MSA = ProtMSA(train_MSA)
            self.train_sample_weights = self.sample_weights[self.train_idx]
            
            val_MSA = [self.MSA[i] for i in self.val_idx]
            self.val_MSA = ProtMSA(val_MSA)
            self.val_sample_weights = self.sample_weights[self.val_idx]
            
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            test_MSA = [self.MSA[i] for i in self.test_idx]
            self.test_MSA = ProtMSA(test_MSA)
            self.test_sample_weights = self.sample_weights[self.test_idx]

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.sample_weights is not None:
            sampler = data_utils.WeightedRandomSampler(
                         weights=self.train_sample_weights,
                         num_samples=len(self.train_sample_weights), 
                                 replacement=False)
            shuffle = False
        return data_utils.DataLoader(self.train_MSA, sampler=sampler,
                    batch_size=self.batch_size, shuffle=shuffle,
                    num_workers=self.num_workers)

    def val_dataloader(self):
        sampler=None
        shuffle = True
        if self.sample_weights is not None:
            sampler = data_utils.WeightedRandomSampler(
                         weights=self.val_sample_weights,
                         num_samples=len(self.val_sample_weights), 
                                 replacement=False)
            shuffle = False
        return data_utils.DataLoader(self.val_MSA, sampler=sampler,
                        batch_size=self.batch_size, shuffle=shuffle,
                        num_workers=self.num_workers)

    def test_dataloader(self):
        sampler = None
        return data_utils.DataLoader(self.test_MSA, sampler=sampler,
                batch_size=self.batch_size, 
                num_workers=self.num_workers)

# VAE Model
class ConvVAE(pl.LightningModule):
    """
    Adapted from: ...
    What this does...
    """
    def __init__(self, slen, ks, nlatent, learning_rate):
        super().__init__()
        
        # The VAE uses a probabilistic approach to encoding the input data, which is why it generates a mean and a
        # variance for each of the latent variables. This is done to ensure that the encoded representation is not
        # overfitting to the input data and is instead learning the underlying structure of the data.

        # During training, the model minimizes two types of loss: the reconstruction loss and the Kullback-Leibler
        # (KL) divergence. The reconstruction loss measures how well the model is able to reconstruct the input data
        # from the encoded representation, while the KL divergence measures how well the encoded representation
        # follows a standard normal distribution.

        # The mean and variance of the latent variables are used to generate samples from the latent space during
        # training. The mean is used as the center of a normal distribution, and the variance is used to scale the
        # distribution. This sampling is done to generate new, novel sequences that are similar to the ones seen 
        # during training.
      
        ## GENERAL INPUT PARAMETERS
        self.slen = slen # sequence length
        self.ks = ks # kernel size
        self.nlatent = nlatent # num latent vars
        self.learning_rate = learning_rate
    
        ## ENCODER
        self.embed = nn.Embedding(21, 16)
        # self.embed = nn.Embedding.from_pretrained(torch.eye(21),freeze=False)
        self.edim = self.embed.embedding_dim # dimensions of AA embedding
        self.enc_conv_1 = torch.nn.Conv1d(in_channels=  self.edim, out_channels=2*self.edim, kernel_size=ks)
        self.enc_conv_2 = torch.nn.Conv1d(in_channels=2*self.edim, out_channels=4*self.edim, kernel_size=ks) 
        self.nparam = (slen-2*(ks-1))*(4*self.edim) # each convolution reduces slen by ks-1, multiply by the # output channels
        self.linear_postConv = torch.nn.Linear(self.nparam,1000) ##### self.param = 6080
        self.z_mean = torch.nn.Linear(1000,nlatent)
        self.z_log_var = torch.nn.Linear(1000,nlatent)
    
        ## DECODER
        self.dec_linear_1 = torch.nn.Linear(nlatent,1000)
        self.dec_linear_2 = torch.nn.Linear(1000,self.nparam)
        self.dec_deconv_1 = torch.nn.ConvTranspose1d(in_channels=4*self.edim, out_channels=2*self.edim, kernel_size=ks)
        self.dec_deconv_2 = torch.nn.ConvTranspose1d(in_channels=2*self.edim, out_channels=  self.edim, kernel_size=ks)
        self.nembed = self.embed.num_embeddings
        self.rev_embed = torch.nn.Linear(self.edim,self.nembed)
        
        # record additional dimensions for reshaping
        # self.edim = edim
        # self.nparam = nparam
        # self.nembed = nembed
        
        # save hyperparameters for logging 
        self.save_hyperparameters()

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1), device=self.device)
        
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
    
    def encoder(self,x):
        x = self.embed(x)
        x = x.permute(0,2,1) # swap length and channel dims

        x = self.enc_conv_1(x)
        x = F.leaky_relu(x)

        x = self.enc_conv_2(x)
        x = F.leaky_relu(x)

        x = x.view(-1,self.nparam) # flatten
        
        x = self.linear_postConv(x) #####
        x = F.relu(x) #####
        
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        
        return z_mean, z_log_var, encoded


    def decoder(self, encoded):
        x = self.dec_linear_1(encoded)
        x = F.relu(x) #####
        x = self.dec_linear_2(x)
        
        x = x.view(-1,4*self.edim,(self.slen-2*(self.ks-1)))
        
        x = self.dec_deconv_1(x)
        x = F.leaky_relu(x)
        
        x = self.dec_deconv_2(x)
        x = F.leaky_relu(x)
        
        x = x.permute(0,2,1) # swap channel and length dims
        x = self.rev_embed(x)
        decoded = x.permute(0,2,1) # need to permute back
        
        return decoded
    
    def forward(self, x):
        z_mean, z_log_var, encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return z_mean, z_log_var, encoded, decoded
        
    def training_step(self, batch, batch_idx):
        # pass through network 
        z_mean, z_log_var, encoded, decoded = self(batch)

        # cost = reconstruction loss + Kullback-Leibler divergence
        kl_divergence = (0.5 * (z_mean**2 + torch.exp(z_log_var) - z_log_var - 1)).sum()
        ce_loss = F.cross_entropy(decoded,batch,reduction='sum')
        cost = kl_divergence + ce_loss
        
        # log 
        self.log("train_ce_loss", ce_loss, prog_bar=True, logger=True, on_step = False, on_epoch=True)

        return cost

    def validation_step(self, batch, batch_idx):
        # pass through network 
        z_mean, z_log_var, encoded, decoded = self(batch)

        # cost = reconstruction loss + Kullback-Leibler divergence
        kl_divergence = (0.5 * (z_mean**2 + torch.exp(z_log_var) - z_log_var - 1)).sum()
        ce_loss = F.cross_entropy(decoded,batch,reduction='sum')
        cost = kl_divergence + ce_loss 
        
        # log 
        self.log("val_ce_loss", ce_loss, prog_bar=True, logger=True, on_step = False, on_epoch=True)

        return cost
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Compute Log Probability
def compute_scores_from_batch(batch, model):
    """
    Adapted from: ...
    Computes the cross-entropy loss scores for a given batch of sequences using a pre-trained neural network model.
    Args:
        batch (torch.Tensor): A LongTensor of shape (b, L) with
            b = batch size
            L = sequence length.
        model (torch.nn.Module): A pre-trained neural network model that takes in a batch of sequences and returns
            four tensors: z_mean, z_log_var, encoded, and decoded.
    Returns:
        scores (torch.Tensor): A 1D tensor of shape (b,) containing the cross-entropy loss scores for each sequence in
            the batch.
    """
    with torch.no_grad():  # We do not want training to occur during scoring
        # Pass the batch through the model to get the z_mean, z_log_var, encoded, and decoded tensors
        z_mean, z_log_var, encoded, decoded = model(batch)
        
        # The encoded/decoded value above is from a stochastic reparameterization z_mean + random * z_var.
        # So we redo it below by decoding from the mean so it becomes deterministic.
        decoded_from_mean = model.decoder(z_mean)  # Use z_mean instead of encoded to remove stochastic reparameterization
        
        # Compute the cross-entropy loss scores for each sequence in the batch
        # The reduction='none' argument tells PyTorch to return a score for each element in the sequence
        # Sum scores for entire seq (length dim) with .sum(dim=-1) to get a single score for the entire protein sequence
        scores = F.cross_entropy(decoded_from_mean, batch, reduction='none').sum(dim=-1)
    
    return scores

def generate_single_point_mutants(sequence):
    AAs = 'ACDEFGHIKLMNPQRSTVWY'

    mutant_tensors = []
    
    for i, aa in enumerate(sequence):
        if aa != '-':  # Skip the gap character
            for new_aa in AAs:
                if new_aa != aa:  # Skip the wild-type amino acid
                    mutant = sequence[:i] + new_aa + sequence[i+1:]
                    mutant_tensor = torch.tensor([aa2ind[a] for a in mutant], dtype=torch.long).unsqueeze(0)  # Add a batch dimension
                    mutant_tensors.append(mutant_tensor)
    
    # Stack the list of tensors to create a batch
    mutant_batch = torch.cat(mutant_tensors, dim=0)
    
    return mutant_batch

import time
import numpy as np
import torch

def score_all_double_mutants(
    WT_sequence: str,
    model,
    *,
    aa2ind,
    AAs: str = "ACDEFGHIKLMNPQRSTVWY-",
    batch_size: int = 4096,  # m=361 usually <= batch_size
    device=None,
    seed: int = None,
    verbose: bool = True,
):
    """
    Enumerate + score ALL double mutants of WT_sequence.

    Rules:
      - mutate only positions where WT != '-'
      - never mutate to '-'
      - never mutate to WT residue at that position

    IMPORTANT: Uses a reusable batch buffer but correctly resets previously mutated columns
    to avoid accidentally accumulating >2 mutations across pairs.
    """
    # Optional: speed CPU scoring if device is cpu
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    if verbose:
        print(f"[INFO] torch num_threads={torch.get_num_threads()}, interop={torch.get_num_interop_threads()}")

    rng = np.random.default_rng(seed)

    # ---- device ----
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wt_chars = list(WT_sequence)
    L = len(wt_chars)

    # Identify mutable positions (WT != '-')
    non_gap_pos = np.array([i for i, ch in enumerate(wt_chars) if ch != "-"], dtype=np.int64)
    npos = int(non_gap_pos.size)
    if npos < 2:
        raise ValueError(f"Need at least 2 non-gap positions; found {npos}.")

    # Allowed AAs (exclude gap)
    aa_letters = [aa for aa in AAs if aa != "-"]  # typically 20
    aa_idx = torch.tensor(aa2ind(aa_letters), dtype=torch.long, device=device)  # (nAA,)
    wt_idx = torch.tensor(aa2ind(wt_chars), dtype=torch.long, device=device)    # (L,)

    nAA = int(aa_idx.numel())
    m1 = nAA - 1
    variants_per_pair = m1 * m1  # typically 19^2 = 361
    num_pairs = npos * (npos - 1) // 2
    total = num_pairs * variants_per_pair

    meta = dict(
        total=int(total),
        npos=int(npos),
        pairs=int(num_pairs),
        variants_per_pair=int(variants_per_pair),
        L=int(L),
        device=str(device),
    )

    if verbose:
        print(f"[INFO] WT length={L}, non-gap positions={npos}")
        print(f"[INFO] Total double mutants = C({npos},2)*({m1}^2) = {total:,}")

    scores_out = np.empty((total,), dtype=np.float32)

    # Precompute allowed AAs per position (exclude WT residue)
    allowed = {}
    for p in non_gap_pos:
        p = int(p)
        wt_p = wt_idx[p]
        ap = aa_idx[aa_idx != wt_p]  # (m1,)
        if ap.numel() != m1:
            raise ValueError(
                f"Position {p} WT='{wt_chars[p]}' not in aa_letters? "
                f"allowed size={ap.numel()} expected={m1}"
            )
        allowed[p] = ap

    # Precompute cartesian template indices once
    grid_i = torch.arange(m1, device=device).repeat_interleave(m1)  # (m1*m1,)
    grid_j = torch.arange(m1, device=device).repeat(m1)             # (m1*m1,)
    m = int(m1 * m1)

    # Reusable batch buffer
    base = wt_idx.unsqueeze(0).expand(m, -1).clone()

    model.eval()
    t0 = time.perf_counter()
    t_gen = 0.0
    t_score = 0.0
    done = 0
    last_report = time.perf_counter()

    prev_p = None
    prev_q = None

    with torch.inference_mode():
        for a_i in range(npos - 1):
            p = int(non_gap_pos[a_i])
            ap = allowed[p]
            for a_j in range(a_i + 1, npos):
                q = int(non_gap_pos[a_j])
                aq = allowed[q]

                tg = time.perf_counter()

                # ---- CRITICAL FIX: reset previously mutated columns back to WT ----
                if prev_p is not None:
                    base[:, prev_p] = wt_idx[prev_p]
                    base[:, prev_q] = wt_idx[prev_q]

                # Apply the current pair mutations
                base[:, p] = ap[grid_i]
                base[:, q] = aq[grid_j]

                prev_p, prev_q = p, q
                t_gen += (time.perf_counter() - tg)

                # Score (m~361)
                ts = time.perf_counter()
                s = compute_scores_from_batch(base, model)
                t_score += (time.perf_counter() - ts)

                s = s.detach().float().cpu().numpy().astype(np.float32, copy=False).ravel()
                scores_out[done:done+m] = s
                done += m

                now = time.perf_counter()
                if verbose and ((now - last_report) > 10.0 or done == total):
                    rate = done / (now - t0)
                    eta_min = (total - done) / rate / 60 if rate > 0 else float("inf")
                    print(f"[INFO] {done:,}/{total:,} ({done/total:.1%}) | {rate:,.0f} var/s | ETA {eta_min:.1f} min")
                    print(f"       breakdown: gen={t_gen:.1f}s score={t_score:.1f}s")
                    last_report = now

    scores_out = scores_out[np.isfinite(scores_out)]
    if verbose:
        print(f"[INFO] Done all double mutants: N_valid={scores_out.size:,} total_time={time.perf_counter()-t0:.2f}s")

    return scores_out, meta

def generate_mutants(WT_sequence, num_variants, num_mutations, model, *, batch_size=512, device=None, seed=None):
    """
    Vectorized + batched mutant generation + scoring.

    - Mutates only WT positions where WT != '-'
    - Never mutates to '-'
    - Ensures mutated AA != WT AA at that position
    - Scores in batches via compute_scores_from_batch

    Returns
    -------
    np.ndarray float32, shape (num_variants,)
    """
    import numpy as np
    import torch

    # ---- RNG ----
    rng = np.random.default_rng(seed)

    # ---- device ----
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- inputs ----
    wt = np.frombuffer(WT_sequence.encode("ascii"), dtype=np.uint8)
    wt_chars = np.array(list(WT_sequence), dtype="<U1")
    L = wt_chars.size

    # IMPORTANT: uses WT_sequence (not global WT)
    non_gap_positions = np.where(wt_chars != "-")[0]
    if non_gap_positions.size == 0:
        raise ValueError("WT_sequence has no non-gap positions to mutate.")
    if num_mutations > non_gap_positions.size:
        raise ValueError(f"num_mutations={num_mutations} > non-gap positions={non_gap_positions.size}")

    # Allowed amino acids (exclude gap)
    # Uses your global AAs string if present; otherwise default.
    try:
        aa_list = [aa for aa in AAs if aa != "-"]
    except NameError:
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    aa_arr = np.array(aa_list, dtype="<U1")

    # Map AA -> index via aa2ind (your existing function)
    # We'll construct strings, then convert with aa2ind per-seq in a Python loop
    # (still fast because scoring dominates; string build is cheap).
    scores_out = np.empty((num_variants,), dtype=np.float32)

    # ---- main loop over batches ----
    model.eval()

    for start in range(0, num_variants, batch_size):
        end = min(start + batch_size, num_variants)
        b = end - start

        # Start from WT for all b sequences
        muts = np.tile(wt_chars, (b, 1))  # (b, L) array of single-char strings

        # Choose mutation positions: sample without replacement per sequence
        # We do this by shuffling and taking first k (vectorized trick)
        # indices in [0, non_gap_positions.size)
        perm = np.argsort(rng.random((b, non_gap_positions.size)), axis=1)
        chosen_idx = perm[:, :num_mutations]                           # (b, k) indices into non_gap_positions
        chosen_pos = non_gap_positions[chosen_idx]                     # (b, k) actual positions in sequence

        # For each mutation position, pick a new AA != WT[pos], != '-'
        # Vectorized: sample AA indices then fix collisions where sampled == WT
        sampled = aa_arr[rng.integers(0, aa_arr.size, size=(b, num_mutations))]  # (b, k)

        wt_at_pos = wt_chars[chosen_pos]  # (b, k) WT letters at those positions

        # If sampled equals WT at any spot, resample those spots until all differ.
        # Expected to converge quickly (20 AAs).
        same = (sampled == wt_at_pos)
        while np.any(same):
            sampled[same] = aa_arr[rng.integers(0, aa_arr.size, size=int(same.sum()))]
            same = (sampled == wt_at_pos)

        # Apply mutations
        # Advanced indexing: for each row i, set positions chosen_pos[i, j]
        row_idx = np.arange(b)[:, None]
        muts[row_idx, chosen_pos] = sampled

        # Convert to torch batch (b, L)
        # Build sequences -> aa2ind -> tensor
        seqs = ["".join(muts[i]) for i in range(b)]
        batch = torch.tensor([aa2ind(list(s)) for s in seqs], dtype=torch.long, device=device)

        # Score
        with torch.no_grad():
            s = compute_scores_from_batch(batch, model)  # (b,)
        s = s.detach().float().cpu().numpy().astype(np.float32)

        scores_out[start:end] = s

    return scores_out

def plot_msa_aa_position_heatmap(
    msa_seqs: Sequence[str],
    wt_seq: str,
    aa_order: str = "ACDEFGHIKLMNPQRSTVWY-",
    normalize: str = "count",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 4),
    show_wt_dots: bool = True,
    dot_size: float = 15.0,
    cmap_low: str = "#f7f7f7",
    cmap_high: str = "#40004b",
    show_direction_labels: bool = True,
    direction_top: str = "More Conserved",
    direction_bottom: str = "Less Conserved",
    xtick_every: int = 10,
    cbar_label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    if len(msa_seqs) == 0:
        raise ValueError("msa_seqs is empty.")

    L = len(msa_seqs[0])
    for i, s in enumerate(msa_seqs):
        if len(s) != L:
            raise ValueError(f"msa_seqs[{i}] length={len(s)} != {L}. Ensure alignment.")
    if len(wt_seq) != L:
        raise ValueError(f"wt_seq length={len(wt_seq)} != {L}. WT must be aligned to same length.")

    wt = wt_seq.upper()

    # indices of columns where WT != '-'
    keep_idx = np.array([i for i, ch in enumerate(wt) if ch != "-"], dtype=int)
    if keep_idx.size == 0:
        raise ValueError("WT contains only '-' gaps; nothing to plot.")

    # PKC positions (1..N_non_gap) for kept columns
    pkc_pos = np.arange(1, keep_idx.size + 1)

    aa_order = aa_order.upper()
    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
    A = len(aa_order)

    counts_full = np.zeros((A, L), dtype=np.int32)
    fallback = "-" if "-" in aa_to_idx else None

    for s in msa_seqs:
        s = s.upper()
        for pos, ch in enumerate(s):
            idx = aa_to_idx.get(ch, aa_to_idx.get(fallback, None))
            if idx is not None:
                counts_full[idx, pos] += 1

    counts = counts_full[:, keep_idx]

    if normalize not in {"count", "freq"}:
        raise ValueError("normalize must be 'count' or 'freq'.")

    data = counts.astype(np.float32)
    if normalize == "freq":
        data /= float(len(msa_seqs))

    custom_cmap = LinearSegmentedColormap.from_list(
        "msa_white_to_purple",
        [(0.0, cmap_low), (1.0, cmap_high)],
        N=256,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if cbar_label is None:
        cbar_label = "Count" if normalize == "count" else "Frequency"

    sns.heatmap(
        data,
        cmap=custom_cmap,
        ax=ax,
        cbar_kws={"label": cbar_label},
        linewidths=0.0,
        linecolor=None,
    )

    # WT dots (no fancy indexing on string)
    if show_wt_dots:
        wt_kept = [wt[i] for i in keep_idx]  # list[str] of chars
        xs = np.arange(len(wt_kept))
        ys = np.full(len(wt_kept), np.nan, dtype=np.float32)

        for x, ch in enumerate(wt_kept):
            y = aa_to_idx.get(ch, aa_to_idx.get(fallback, None))
            if y is not None:
                ys[x] = y

        valid = np.isfinite(ys)
        ax.scatter(xs[valid] + 0.5, ys[valid] + 0.5, color="black", s=dot_size, linewidths=0)

    ax.set_ylabel("Amino Acid", fontsize=14)
    ax.set_xlabel("PKC Amino Acid Position", fontsize=14)
    if title is not None:
        ax.set_title(title)

    ax.set_yticks(np.arange(A) + 0.5)
    ax.set_yticklabels(list(aa_order), rotation=0)

    Lk = keep_idx.size
    xticks_positions = np.arange(Lk)
    xticks_labels = [str(pkc_pos[i]) if (pkc_pos[i] % xtick_every == 0) else "" for i in xticks_positions]
    ax.set_xticks(xticks_positions + 0.5)
    ax.set_xticklabels(xticks_labels, rotation=0)

    if show_direction_labels:
        ax.text(Lk + 7, -2, direction_top, ha="center", va="center", fontsize=12, clip_on=False)
        ax.text(Lk + 7, A + 2, direction_bottom, ha="center", va="center", fontsize=12, clip_on=False)

    plt.tight_layout()
    return fig, ax, counts_full

def hamming_mutations_where_both_non_gap(
    seq: str,
    wt_seq: str,
    *,
    gap_char: str = "-",
    valid_aas: Optional[Set[str]] = None,  # e.g., set("ACDEFGHIKLMNPQRSTVWY")
    normalize: bool = False,
) -> float:
    """
    Count mutations only at positions where BOTH:
      - WT is not a gap
      - seq is not a gap
    Optionally require letters to be in valid_aas.
    """
    if len(seq) != len(wt_seq):
        raise ValueError(f"Length mismatch: len(seq)={len(seq)} vs len(wt_seq)={len(wt_seq)}")

    s = seq.upper()
    w = wt_seq.upper()

    denom = 0
    dist = 0
    for a, b in zip(s, w):
        # require both non-gap
        if b == gap_char or a == gap_char:
            continue

        # optional: require both are "real" amino acids
        if valid_aas is not None and (a not in valid_aas or b not in valid_aas):
            continue

        denom += 1
        if a != b:
            dist += 1

    if denom == 0:
        return 0.0
    return (dist / denom) if normalize else float(dist)


def plot_hamming_distance_from_wt(
    msa_seqs: Sequence[str],
    wt_seq: str,
    *,
    gap_char: str = "-",
    ignore_if_either_gap: bool = True,
    normalize: bool = False,
    bins: int = 50,
    kde: bool = True,
    figsize: Tuple[float, float] = (7, 4),
    color: str = "#40004b",
    title: Optional[str] = "Hamming distance from PKC (ignoring gaps)",
    ax: Optional[plt.Axes] = None,
):
    """
    Computes gap-ignoring Hamming distances from wt_seq for each sequence in msa_seqs
    and plots a histogram (+ optional KDE).

    Returns (fig, ax, distances).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(msa_seqs) == 0:
        raise ValueError("msa_seqs is empty.")

    # compute distances
    VALID = set("ACDEFGHIKLMNPQRSTVWY")
    dists = np.array(
        [
            hamming_mutations_where_both_non_gap(seq, wt_seq, valid_aas=VALID, normalize=False)
            for seq in msa_seqs
        ],
        dtype=float,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # histogram
    sns.histplot(dists, bins=bins, stat="count", kde=False, ax=ax, color=color)

    # optional KDE overlay (seaborn will auto-scale density; use a second axis if you want exact scaling)
    if kde and len(dists) > 1:
        sns.kdeplot(dists, ax=ax, color=color, lw=2)

    ax.set_xlabel("Hamming distance from PKC" + (" (normalized)" if normalize else ""))
    ax.set_ylabel("Count")
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax, dists




