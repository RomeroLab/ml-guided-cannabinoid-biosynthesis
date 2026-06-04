# Functions and models for simulated annealing

### Importing Modules
import numpy as np
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
import random
import pickle
import csv

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = '-------MAVKHLIVLKFKDEITEAQKEEFFKTYVNLVN--IIPAMKDVYW----GK-DVTQKNKEEGYTHIVEVTFESVETIQDYII-HPAHVGFGDVYRSFWEKLLIFDY-----TPRK-------'
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
    
# Define Computing Log Probability using VAE model
def compute_scores_from_batch(batch, model):
    """
    Computes the cross-entropy loss scores for a given batch of sequences using a ConvVAE.
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

def get_non_gap_indices(seq):
    """Get the indices of non-gap positions in the input MSA sequence"""
    return [i for i, aa in enumerate(seq) if aa != "-"]

def generate_all_point_mutants(seq, non_gap_indices, AA_options):
    """Generate all possible single point mutants of a sequence at non-gap positions
    Arguments:
    seq: starting seq - the original sequence to mutate
    non_gap_indices: list of indices corresponding to non-gap positions in the sequence
    AA_options: list of amino acid options at each position, if none defaults to all 20 AAs (default None)
    """
    all_mutants = []  # Initialize an empty list to store all the possible mutants
    for pos in non_gap_indices:  # Loop through each non-gap position in the input sequence
        for aa in AA_options[pos]:  # Loop through each amino acid at that position
            if seq[pos] != aa:  # If the current amino acid is not the same as the original one at that position
                mut = seq[pos] + str(pos) + aa  # Create a string to represent the mutation (e.g. G12A)
                all_mutants.append(mut)  # Add the mutation to the list of all mutants
                
    return all_mutants  # Return the list of all mutants

def mut2seq(seq, mutations):
    """Create mutations in form of A94T to seq
    Arguments:
    seq: starting seq - the original sequence to mutate
    mutations: list of mutations in form of ["A94T", "H99R"] or "A94T,H99R"
    """
    mutant_seq = seq  # Initialize the mutant sequence as the original sequence

    if type(mutations) is str:  # If mutations is a string, split it into a list of mutations
        mutations = mutations.split(',')
    for mut in mutations:  # Loop through each mutation in the list
        pos = int(mut[1:-1])  # Get the position of the mutation
        newAA = mut[-1]  # Get the new amino acid for the mutation
        if mut[0] != seq[pos]:  # If the wild-type amino acid at the mutation position does not match the original sequence, print a warning
            print('Warning: WT residue in mutation %s does not match WT sequence' % mut)
        mutant_seq = mutant_seq[:pos] + newAA + mutant_seq[pos + 1:]  # Apply the mutation to the mutant sequence

    return mutant_seq  # Return the mutant sequence

def find_top_n_mutations(VAE_fitness, all_mutants, WT, n=10):
    """
    Find the top n mutations with the highest fitness score from a list of all possible single point mutations.
    Arguments:
        VAE_fitness: function to calculate fitness score for a given sequence
        all_mutants: list of all possible single point mutants for the starting sequence
        WT: wild-type starting sequence
        n: number of top mutations to return (default 10)
    Returns:
        topn: list of n top mutations sorted by fitness score in descending order with the format 'A8C'
    """
    # evaluate fitness of all single mutants from WT
    single_mut_fitness = []
    for mut in all_mutants:
        pos = int(mut[1:-1])
        seq = WT[:pos] + mut[-1] + WT[pos+1:]
        fit = VAE_fitness(seq)
        single_mut_fitness.append((mut, fit))
    
    # find the best mutation per position
    best_mut_per_position = []
    for pos in range(len(WT)):
        # select the mutation with the highest fitness score for the current position
        position_mutants = [m for m in single_mut_fitness if int(m[0][1:-1]) == pos]
        if not position_mutants:
            continue
        best_mut_per_position.append(max(position_mutants, key=lambda x: x[1]))
    
    # take the top n mutations
    sorted_by_fitness = sorted(best_mut_per_position, key=lambda x: x[1], reverse=True)
    topn = [m[0] for m in sorted_by_fitness[:n]]

    # sort the top n mutations by position and format them as 'A8C'
    # topn_formatted = [WT[int(m[1:-1])] + str(int(m[1:-1])+1) + m[-1] for m in topn]
    
    # take the top n
    topn = tuple([n[1] for n in sorted([(int(m[1:-1]), m) for m in topn])])  # sort by position

    return topn_formatted

### This can mutate gaps that we do not want to mutate
def generate_random_mut(WT, AA_options, num_mut):
    # Create a list of all possible mutations for each position in the wild-type sequence
    AA_mut_options = []
    for WT_AA, AA_options_pos in zip(WT, AA_options):
        if WT_AA in AA_options_pos: # If the wild-type amino acid is an option at this position
            options = list(AA_options_pos).copy() # Create a copy of the list of possible AAs
            options.remove(WT_AA) # Remove the wild-type AA from the list of possible AAs
            AA_mut_options.append(options) # Add the list of possible mutations to the list of AA_mut_options
    
    # Create a list of random mutations
    mutations = []
    for n in range(num_mut):
        # Calculate the probability of each position mutating
        num_mut_pos = sum([len(row) for row in AA_mut_options]) # Count the number of positions that can mutate
        prob_each_pos = [len(row) / num_mut_pos for row in AA_mut_options] # Calculate the probability of each position mutating
        
        # Choose a position to mutate based on its probability
        rand_num = random.random() # Choose a random number between 0 and 1
        for i, prob_pos in enumerate(prob_each_pos):
            rand_num -= prob_pos
            if rand_num <= 0: # If the random number is less than or equal to the probability of this position mutating, choose this position
                # Choose a random mutation for this position
                mutations.append(WT[i] + str(i) + random.choice(AA_mut_options[i]))
                AA_mut_options.pop(i) # Remove this position from the list of AA_mut_options
                AA_mut_options.insert(i, []) # Add an empty list to the list of AA_mut_options to indicate that this position has already mutated
                break
    # Return the list of random mutations as a string
    return ','.join(mutations)


def generate_random_mut_non_gap_indices(WT, AA_options, num_mut, non_gap_indices, mutating_window_size):
    # Create a list of all possible mutations for each position in the wild-type sequence
    AA_mut_options = [[] for _ in range(len(WT))]  # Initialize a list with the length of WT

    if num_mut > mutating_window_size:
        raise ValueError('Number of mutations must be less than the length of WT being mutated (mutating_window_size)')
    
    # Fill only non-gap indices with mutation options
    for idx in non_gap_indices:
        WT_AA = WT[idx]
        AA_options_pos = AA_options[idx]
        if WT_AA in AA_options_pos:  # If the wild-type amino acid is an option at this position
            options = list(AA_options_pos).copy()  # Create a copy of the list of possible AAs
            options.remove(WT_AA)  # Remove the wild-type AA from the list of possible AAs
            AA_mut_options[idx] = options  # Set the list of possible mutations at the correct index
    
    # Create a list of random mutations
    mutations = []
    for _ in range(num_mut):
        # Calculate the probability of each position mutating
        num_mut_pos = sum([len(x) for x in AA_mut_options if x])  # Count the number of positions that can mutate
        if num_mut_pos == 0:
            break  # No more mutations possible
        
        prob_each_pos = [(len(x) / num_mut_pos if x else 0) for x in AA_mut_options]  # Probability for each position
        
        # Choose a position to mutate based on its probability
        rand_num = random.random()  # Choose a random number between 0 and 1
        cumulative_prob = 0
        
        for i, prob_pos in enumerate(prob_each_pos):
            cumulative_prob += prob_pos
            if rand_num <= cumulative_prob and AA_mut_options[i]:  # If the random number is less than or equal to the cumulative probability of this position mutating
                # Choose a random mutation for this position
                mutations.append(WT[i] + str(i) + random.choice(AA_mut_options[i]))
                AA_mut_options[i] = []  # Set this position to an empty list to indicate it cannot mutate again
                break

    # Return the list of random mutations as a string
    return ','.join(mutations)







