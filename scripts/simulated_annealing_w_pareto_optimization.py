#!/usr/bin/env python
# coding: utf-8

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
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.round_3.round_3_models import SeqFcnDataset, PTLModule
from models.VAE.ConvVAE import get_msa_from_fasta, ProtMSA, ConvVAE, compute_scores_from_batch
from SA_utils import get_non_gap_indices, generate_all_point_mutants, mut2seq, find_top_n_mutations, generate_random_mut_non_gap_indices

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = '-------MAVKHLIVLKFKDEITEAQKEEFFKTYVNLVN--IIPAMKDVYW----GK-DVTQKNKEEGYTHIVEVTFESVETIQDYII-HPAHVGFGDVYRSFWEKLLIFDY-----TPRK-------'
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap

# Load EnsMLPs
batch_size = 32 # typically powers of 2: 32, 64, 128, 256, ...
slen = len(WT) # length of protein
learning_rate = 5e-6 # important to optimize this
epochs = 10000 # rounds of training
num_models = 100 # 100 # number of models in ensemble
n_exps = 10
n_labels = n_exps*4
pattern = [1, 1, 1, 0.1]
full_repeats = n_labels // len(pattern)
additional_elements = n_labels % len(pattern)
weights = pattern * full_repeats + pattern[:additional_elements] # Weights for multi-task loss
num_models = 100

models = []
for i in range(num_models):
    model = PTLModule(slen, learning_rate, epochs, weights) # Instantiate the model previously defined
    checkpoint = torch.load(f'../models/round_3/model_ensemble/round_3_EnsMLP_{i}.ckpt')
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)

# Load VAE
batch_size = 64
ks = 4
nlatent = 40
epochs = 65
learning_rate=0.0000075
slen = len(WT)
VAE = ConvVAE(slen, ks, nlatent, learning_rate)
VAE.load_state_dict(torch.load('../models/VAE/VAE_weights.pt'))

# Simulated Annealing Parameters
type = 'C6' # Number of carbons in side chain of products
num_mut = 6

if num_mut == 3:
    VAE_Max = -210.17766 # (('L15A', 'K62E', 'W103L'), array([-210.17766], dtype=float32))
    start_mut = None # Use PKC1.0
    mut_rate = 2
    # num_trials = 150
    nsteps = 50000
    
    if type == 'C4':
        MTFCNN_Min = 0.17611008710227904 # C4, 3mut
        MTFCNN_Max = 0.6817577719688415   # C4, 3mut @ position 0 (('K26I', 'N38K', 'V72I'), 0.6817577719688415)
        VAE_Min = -222.05702   # C4, 3mut @ position 0 (('K26I', 'N38K', 'V72I'), 0.6817577719688415)
    else:
        MTFCNN_Min = 0.23489819448441268 # C6, 3mut
        MTFCNN_Max = 0.5841821901500225 # C6, 3mut @ position 0 (('K26I', 'Y33F', 'D47A'), 0.5841821901500225)
        VAE_Min = -222.7611 # C6, 3mut @ position 0 (('K26I', 'Y33F', 'D47A'), 0.5841821901500225)

if num_mut == 6:
    VAE_Max =  -204.41934 # (('L15A', 'V72F', 'G94E', 'Y99F', 'R100L', 'W103V'), array([-204.41934], dtype=float32))
    start_mut = None # Use PKC1.0
    mut_rate = 3
    # num_trials = 75
    nsteps = 100000
    
    if type == 'C4':
        MTFCNN_Min = 0.23449697555042803 # C4, 6mut
        MTFCNN_Max = 0.7144864574074745 # C4, 6mut @ position 11 (('K26I', 'K31V', 'V48F', 'K64E', 'V72I', 'G94W'), 0.7144864574074745)
        VAE_Min = -224.71594 # C4, 6mut @ position 11 (('K26I', 'K31V', 'V48F', 'K64E', 'V72I', 'G94W'), 0.7144864574074745)
    else:
        MTFCNN_Min = 0.1922477653250098 # C6, 6mut
        MTFCNN_Max = 0.6287983078509569 # C6, 6mut @ position 2 (('V14I', 'K26I', 'K31V', 'Y33F', 'D47A', 'V74S'), 0.6287983078509569)
        VAE_Min = -226.3465 # C6, 6mut @ position 2 (('V14I', 'K26I', 'K31V', 'Y33F', 'D47A', 'V74S'), 0.6287983078509569)

start_temp = -0.5
final_temp = -2.75
fixed_window_size = 19
WT = '-------MAVKHLIVLKFKDEITEAQKEEFFKTYVNLVN--IIPAMKDVYW----GK-DVTQKNKEEGYTHIVEVTFESVETIQDYII-HPAHVGFGDVYRSFWEKLLIFDY-----TPRK-------'
WT_no_gaps = 'MAVKHLIVLKFKDEITEAQKEEFFKTYVNLVNIIPAMKDVYWGKDVTQKNKEEGYTHIVEVTFESVETIQDYIIHPAHVGFGDVYRSFWEKLLIFDYTPRK' # PKC1.0 with no gaps
non_gap_indices = get_non_gap_indices(WT)
mutating_window_size = len(WT_no_gaps)-fixed_window_size # size of window where amino acids can be altered
with open('amino_acid_weight_lookup.pkl', 'rb') as f:
    aa_weights = pickle.load(f) # molecular weights for aas
residues_within_4A = [5, 7, 9, 23, 24, 27, 28, 30, 40, 49, 59, 72, 73, 78, 81, 82, 89, 92, 94, 96] # Define residues within 4 angstroms of docked OA and in active site
start_position = 6 # position window where amino acids can be altered begins
seed = random.randint(0, 100000) # Set random seeds for reproducibility
random.seed(seed)
np.random.seed(seed)
alpha_values = np.linspace(0, 1, num=50)

# Simulated Annealing Class
class SA_optimizer:
    def __init__(self, seq_fitness, WT, AA_options, num_mut, mutating_window_size, mut_rate, nsteps, cool_sched, non_gap_indices, start_temp, final_temp):
        
        """Initializes the SA_optimizer class with the following inputs:
        seq_fitness: a function that takes a sequence and returns its fitness
        WT: the wild-type sequence to be mutated
        AA_options: a list of possible amino acid substitutions for each position in the sequence
        num_mut: the number of mutations to make in each mutant sequence
        mut_rate: the rate at which mutations occur during simulated annealing
        nsteps: the number of steps in the cooling schedule for simulated annealing
        cool_sched: the cooling schedule to use for simulated annealing (either 'log' or 'lin')"""
        
        self.seq_fitness = seq_fitness
        self.WT = WT
        self.AA_options = AA_options
        self.num_mut = num_mut
        self.mutating_window_size = mutating_window_size
        self.mut_rate = mut_rate
        self.nsteps = nsteps
        self.cool_sched = cool_sched
        self.non_gap_indices = non_gap_indices
        self.start_temp = start_temp
        self.final_temp = final_temp
        self.close_sequences = []

    def optimize(self, start_mut=None):

        # If no starting mutation is provided, generate one randomly
        if start_mut is None:
            # start_mut = generate_random_mut(self.WT, self.AA_options, self.num_mut).split(',')
            start_mut = generate_random_mut_non_gap_indices(self.WT, self.AA_options, self.num_mut, self.non_gap_indices, self.mutating_window_size).split(',')
        # print(start_mut)
        
        # Generate a list of all possible point mutants for the wild-type sequence
        all_mutants = generate_all_point_mutants(self.WT, self.non_gap_indices, self.AA_options)

        # Ensure that the cooling schedule is either logarithmic or linear
        assert ((self.cool_sched == 'log') or (self.cool_sched == 'lin')), 'cool_sched must be \'log\' or \'lin\''

        # Set the temperature schedule based on the cooling schedule
        if self.cool_sched == 'log':
            temp = np.logspace(self.start_temp, self.final_temp, self.nsteps)
        if self.cool_sched == 'lin':
            temp = np.linspace(1000, 1e-9, self.nsteps)

        # Initialize variables to track progress and store results
        print('Simulated Annealing Progress: ')
        seq = mut2seq(self.WT, start_mut)
        fit, MTFCNN_score, VAE_score = self.seq_fitness(seq)
        current_seq = [start_mut, fit]  # Store the current sequence and its fitness
        self.best_seq = [start_mut, fit]  # Store the best sequence and its fitness found so far
        self.fitness_trajectory = [[fit, fit]]  # Store the trajectory of fitness values over time

        # for loop over decreasing temperatures
        for T in temp:
            # Create a mutant sequence based on the current sequence
            mutant = list(current_seq[0])

            # Choose the number of mutations to make to the current sequence
            n = np.random.poisson(self.mut_rate)
            n = min([self.num_mut - 1, max([1, n])])  # Bound the number of mutations within the range [1,num_mut-1]

            # Remove random mutations from the current sequence until it contains (num_mut-n) mutations
            while len(mutant) > (self.num_mut - n):
                mutant.pop(random.choice(range(len(mutant))))

            # Add back n random mutations to generate a new mutant sequence
            occupied = [m[1:-1] for m in mutant]  # Positions that already have a mutation
            mut_options = [m for m in all_mutants if m[1:-1] not in occupied]  # Mutations at unoccupied positions
            while len(mutant) < self.num_mut:
                mutant.append(random.choice(mut_options))
                occupied = [m[1:-1] for m in mutant]
                mut_options = [m for m in all_mutants if m[1:-1] not in occupied]

            # Sort mutations by position to clean up the format
            mutant = tuple([n[1] for n in sorted([(int(m[1:-1]), m) for m in mutant])])

            # Evaluate the fitness of the new mutant sequence
            fitness, MTFCNN_score, VAE_score = self.seq_fitness(mut2seq(self.WT, mutant))

            # Determine if the current sequence is close to the maximum fitness value
            if fitness > (0.55):
                # Add the current sequence and its fitness to the list
                self.close_sequences.append((mutant, fitness, MTFCNN_score, VAE_score))

            # If the mutant sequence is better than the best sequence found so far, update the best sequence
            if fitness > self.best_seq[1]:
                self.best_seq = [mutant, fitness, MTFCNN_score, VAE_score]

            # If mutant is worse than current seq, accept mutations with decreasing probability
            delta_F = fitness - current_seq[1]  # calculate the difference in fitness between the mutant sequence and the current sequence

            # ###############################################################################################
            # Printing first few accept probabilities. We want this to be 30-50%, preferrably 30-40%, but the overall SA curves appearance is more important
            accept_prob = np.exp(min([0, delta_F / (T)]))
            print(f"Acceptance probability: {accept_prob}")
            # ###############################################################################################
            
            if np.exp(min([0, delta_F / (T)])) > random.random():  # calculate the acceptance probability based on the temperature and delta_F
                current_seq = [mutant, fitness, MTFCNN_score, VAE_score]  # if the mutant is accepted, set the current sequence to the mutant sequence

            # store the current fitness in the fitness trajectory
            self.fitness_trajectory.append([self.best_seq[1], current_seq[1]])
            
        # Define your directory path and file name
        file_path = os.path.join(dir_path, f"close_sequences_{type}_{num_mut}mut_start_pos{start_position}_alpha{alpha}.pickle")
        # Serialize the list of close sequences to a pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(self.close_sequences, f)
        
        print('Simulated Annealing Progress: Done')

        return self.best_seq  # returns [best_mut, best_fit]

    def plot_trajectory(self, savefig_name=None):
        """
        Plots the fitness trajectory of the simulated annealing optimization algorithm.
        Args:
            savefig_name (str): optional file name to save the plot as an image file.
        """
        # Plot the fitness trajectory of the best and current mutants
        plt.plot(np.array(self.fitness_trajectory)[:, 0],'x', markersize=8, markeredgecolor='black', color='black')
        plt.plot(np.array(self.fitness_trajectory)[:, 1],'orange')
        
        # Add labels and legend
        plt.xlabel('Step')
        plt.ylabel('Fitness')
        plt.legend(['Best mut found', 'Current mut'])
        
        # Show or save the plot
        if savefig_name is None:
            plt.show()
        else:
            plt.savefig(savefig_name)
        
        # Close the plot window
        plt.close()

# MTFCNN for C4 experiments
class C4_seq2fitness_handler:
    def __init__(self, seq, models, VAE, VAE_Max, VAE_Min, MTFCNN_Min, MTFCNN_Max, alpha):
        self.seq = seq
        self.models = models
        self.VAE = VAE
        self.VAE_Min = VAE_Min
        self.VAE_Max = VAE_Max
        self.MTFCNN_Min = MTFCNN_Min
        self.MTFCNN_Max = MTFCNN_Max
        
    def seq2fitness(self, seq):
        # Score with EnsFCNN
        labels = []

        # Score Sequence for 100 models
        for model in self.models:
            pred_Y = model.predict(seq).astype(float) # Predict Label Scores
            labels.append(pred_Y) # Append label scores for each enzyme from all models

        ########################### EDIT for Round 2 ###########################
        # Calculate lower confidence bound for all 24 labels from all 100 models
        low_conf_bounds = np.quantile(labels, q=0.05, axis=0)
                
        # For C4 scoring
        MTFCNN_score = (low_conf_bounds[0][18] + low_conf_bounds[0][26])/2
        ########################### EDIT for Round 2 ###########################
        
        # Score with VAE
        torch_tensor = torch.tensor(aa2ind(list(seq))) # Convert to Torch Tensor
        torch_tensor_batch = torch_tensor[None,:] # Add Batch Dimension
        VAE_score = -compute_scores_from_batch(torch_tensor_batch, self.VAE).numpy()

        # Normalize
        VAE_norm = (VAE_score-self.VAE_Min)/(self.VAE_Max-self.VAE_Min)
        MTFCNN_norm = (MTFCNN_score-self.MTFCNN_Min)/(self.MTFCNN_Max-self.MTFCNN_Min)

        # # Calculate Euclidean distance from (1,1) and (VAE_norm, MTFCNN_norm)
        # Eu_dist = np.sqrt(np.square(1-VAE_norm) + np.square(1-MTFCNN_norm))
        # return -Eu_dist, MTFCNN_score, VAE_score
        Fitness_Score = alpha*VAE_norm + (1-alpha)*MTFCNN_norm
        return Fitness_Score, MTFCNN_score, VAE_score

# MTFCNN for C6 experiments
class C6_seq2fitness_handler:
    def __init__(self, seq, models, VAE, VAE_Max, VAE_Min, MTFCNN_Min, MTFCNN_Max, alpha):
        self.seq = seq
        self.models = models
        self.VAE = VAE
        self.VAE_Min = VAE_Min
        self.VAE_Max = VAE_Max
        self.MTFCNN_Min = MTFCNN_Min
        self.MTFCNN_Max = MTFCNN_Max
        
    def seq2fitness(self, seq):
        # Score with EnsFCNN
        labels = []

        # Score Sequence for 100 models
        for model in self.models:
            pred_Y = model.predict(seq).astype(float) # Predict Label Scores
            labels.append(pred_Y) # Append label scores for each enzyme from all models

        ########################### EDIT for Round 2 ###########################
        # Calculate lower confidence bound for all 24 labels from all 100 models
        low_conf_bounds = np.quantile(labels, q=0.05, axis=0)
                
        # For C6 scoring
        MTFCNN_score = (low_conf_bounds[0][22] + low_conf_bounds[0][30])/2
        ########################### EDIT for Round 2 ###########################
        
        # Score with VAE
        torch_tensor = torch.tensor(aa2ind(list(seq))) # Convert to Torch Tensor
        torch_tensor_batch = torch_tensor[None,:] # Add Batch Dimension
        VAE_score = -compute_scores_from_batch(torch_tensor_batch, self.VAE).numpy()

        # Normalize
        VAE_norm = (VAE_score-self.VAE_Min)/(self.VAE_Max-self.VAE_Min)
        MTFCNN_norm = (MTFCNN_score-self.MTFCNN_Min)/(self.MTFCNN_Max-self.MTFCNN_Min)

        # # Calculate Euclidean distance from (1,1) and (VAE_norm, MTFCNN_norm)
        # Eu_dist = np.sqrt(np.square(1-VAE_norm) + np.square(1-MTFCNN_norm))
        # return -Eu_dist, MTFCNN_score, VAE_score
        Fitness_Score = alpha*VAE_norm + (1-alpha)*MTFCNN_norm
        return Fitness_Score, MTFCNN_score, VAE_score

def mutating_window_rational_approach(non_gap_indices, WT_no_gaps, residues_within_4A, aa_weights, start_pos, mutating_window_size):
    """
    This function applies a sliding window approach to update the amino acid (AA)
    options for a sequence, starting from a given position. It sets the AA options
    to the corresponding AAs from `WT_no_gaps` within the specified window.
    
    Parameters:
    - WT_no_gaps (str): WT sequence without gaps.
    - start_pos (int): The starting position for the sliding window.
    - mutating_window (int): The length of the window where mutations are allowed.
    
    Returns:
    - list: A list of tuples, where each tuple contains the AA options for each position.
      Positions within the cloning window are set to the specific AA from `WT_no_gaps`,
      while other positions remain unchanged (all possible AAs).
    """
    AAs_options = 'ACDEFGHIKLMNPQRSTVWY'
    AA_options = [tuple([AA for AA in AAs_options]) for i in range(len(WT))]
    AA_options[non_gap_indices[0]] = WT_no_gaps[0] # Keep start codon

    window_end = start_pos + mutating_window_size
    for i in range(len(WT_no_gaps)):
        
        # Choose amino acids to keep frozen for cloning
        if i <= start_pos or i > window_end:
            AA_options[non_gap_indices[i]] = (WT_no_gaps[i])

        # Get amino acids that are either the same or have a smaller molecular weight
        elif i+1 in residues_within_4A:
            wt_aa_weight = aa_weights[WT_no_gaps[i]]
            smaller_or_same_AAs = [aa for aa in AAs_options if aa_weights[aa] <= wt_aa_weight]
            AA_options[non_gap_indices[i]] = smaller_or_same_AAs
    
    return AA_options

# create SA_trials folder if it doesn't exist
if not os.path.exists('SA_w_pareto_opt'):
    os.makedirs('SA_w_pareto_opt')

dir_path = f'SA_w_pareto_opt/{type}_{num_mut}mut_{nsteps}steps'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


for alpha in alpha_values:
    # Saving parameters
    params_str = f"""################################################
    Simulated Annealing Parameters
    ################################################
    non_gap_indices = {non_gap_indices}
    WT_no_gaps = '{WT_no_gaps}'
    start_mut = '{start_mut}'
    nsteps = {nsteps}
    num_mut = {num_mut}
    mut_rate = {mut_rate}
    start_temp = {start_temp}
    final_temp = {final_temp}
    type = '{type}'
    fixed_window_size = {fixed_window_size}
    mutating_window_size = {mutating_window_size}
    start_position = {start_position}
    seed = {seed}
    alpha = {alpha}
    ################################################
    Simulated Annealing Parameters
    ################################################
    """

    # Path for the parameters text file
    file_path = os.path.join(dir_path, f"parameters_alpha{alpha}.txt")

    # Write parameters to the file
    with open(file_path, "w") as file:
        file.write(params_str)

    print(f"Parameters saved to {file_path}")

    # Determine AA_options given sliding window
    AA_options = mutating_window_rational_approach(non_gap_indices, WT_no_gaps, residues_within_4A, aa_weights, start_position, mutating_window_size)

    # Running Simulated annealing
    # Set the file names with version numbers
    best_mutant_file = f"{dir_path}/best_{type}_{num_mut}mut_start_pos{start_position}_alpha{alpha}.pickle"
    trajectory_file = f"{dir_path}/traj_{type}_{num_mut}mut_start_pos{start_position}_alpha{alpha}.png"
    csv_filename = f"{dir_path}/fitness_trajectory_{type}_{num_mut}mut_start_pos{start_position}_alpha{alpha}.csv"

    # Create an instance of seq_fitness class for the current mutant
    seq_fitness = C6_seq2fitness_handler(WT, models, VAE, VAE_Max, VAE_Min, MTFCNN_Min, MTFCNN_Max, alpha)

    # Create an instance of SA_optimizer class for the current mutant
    sa_optimizer = SA_optimizer(seq_fitness.seq2fitness,
                                 WT,
                                 AA_options,
                                 num_mut=num_mut,
                                 mutating_window_size=mutating_window_size,
                                 mut_rate=mut_rate,
                                 nsteps=nsteps,
                                 cool_sched='log',
                                 non_gap_indices=non_gap_indices,
                                 start_temp=start_temp,
                                 final_temp=final_temp)

    # Optimize the mutant and store the best mutant and its fitness in a pickle file
    best_mut = sa_optimizer.optimize(start_mut)
    with open(best_mutant_file, 'wb') as f:
        pickle.dump((best_mut), f)

    # Save fitness trajectory in a CSV file
    with open(csv_filename, mode='w') as csv_file:
        fieldnames = ['Step', 'Fitness']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
        writer.writeheader()
        for step, (_, fitness) in enumerate(sa_optimizer.fitness_trajectory):
            writer.writerow({'Step': step, 'Fitness': float(fitness)})

    # Save Plotted Trajectory
    sa_optimizer.plot_trajectory(savefig_name=trajectory_file)

