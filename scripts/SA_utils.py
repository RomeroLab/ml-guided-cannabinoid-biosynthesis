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
# incorrect_WT = '-------MAVKHLIVLKFKDEITEAQKEEFFKTYVNLVN--IIPAMKDVYWGK-----DVTQKNKEEGYTHIVEVTFESVETIQDYII-HPAHVGFGDVYRSFWEKLLIFDYTPRK------------' # Incorrect alignment :(

aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap

# SeqFcnDataset is a data handling class.
class SeqFcnDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein sequence-function data"""

    def __init__(self, data_frame):
        self.data_df = data_frame

    def __getitem__(self, idx):
        # I convert amino acid sequences to torch tensors for model inputs
        # I convert all 24 labels, including -1 values, to torch tensors
        sequence = torch.tensor(aa2ind(list(self.data_df.Aligned_Sequence.iloc[idx]))) # Extract sequence at index idx
        labels = torch.tensor(self.data_df.iloc[idx, 3:-1].tolist()).float() # Extract labels for sequence at index idx and convert to a list
        return sequence, labels

    def __len__(self):
        return len(self.data_df)

# ProtDataModule splits the data into three different datasets.
class ProtDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data splitting"""

    def __init__(self, data_frame, batch_size, splits_path=None):
        # Call the __init__ method of the parent class
        super().__init__()

        # Store the batch size
        self.batch_size = batch_size
        self.data_df = data_frame
        
        if splits_path is not None:
            train_indices, val_indices, test_indices = self.load_splits(splits_path)
            # print(test_indices)
            
            # Shuffle the indices to ensure that the data from each cluster is mixed. Do I want this?
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
            
            # Store the indices for the training, validation, and test sets
            self.train_idx = train_indices
            self.val_idx = val_indices
            self.test_idx = test_indices
                
        else:
            # Initialize empty lists to hold the indices for the training, validation, and test sets
            train_indices = []
            val_indices = []
            test_indices = []

            for cluster in self.data_df['Cluster'].unique():
                # Get the indices of the rows in the DataFrame that belong to the current cluster
                cluster_indices = self.data_df[self.data_df['Cluster'] == cluster].index.tolist()
                # Check the size of the cluster
                if len(cluster_indices) == 1:
                    # Handle the case when the cluster has only one sample (e.g., add to training set)
                    train_indices.extend(cluster_indices)
                    continue
                
                # Split the cluster_indices into training and (validation + test) sets
                train, temp, _, _ = train_test_split(cluster_indices, cluster_indices, test_size=0.2, random_state=2)
                # Check the size of the cluster
                if len(temp) == 1:
                    # Handle the case when the cluster has only one sample (e.g., add to training set)
                    val_indices.extend(cluster_indices)
                    continue
                
                # Split the temporary set into validation and test sets
                val, test, _, _ = train_test_split(temp, temp, test_size=0.5, random_state=2)
                # Add the indices for the current cluster to the overall lists of indices
                train_indices.extend(train)
                val_indices.extend(val)
                test_indices.extend(test)
            
            # Shuffle the indices to ensure that the data from each cluster is mixed
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
            
            # Store the indices for the training, validation, and test sets
            self.train_idx = train_indices # Training data is used during model training
            self.val_idx = val_indices # Validation data is used to evaluate the model after epochs
            self.test_idx = test_indices # Testing data is used to evaluate the model after model training is complete

    # Prepare_data is called from a single GPU. Do not use it to assign state (self.x = y). Use this method to do
    # things that might write to disk or that need to be done only from a single process in distributed settings.
    def prepare_data(self):
        pass

    # Assigns train, validation and test datasets for use in dataloaders.
    def setup(self, stage=None):
              
        # Assign train/validation datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_data_frame = self.data_df.iloc[list(self.train_idx)]
            self.train_ds = SeqFcnDataset(train_data_frame)
            val_data_frame = self.data_df.iloc[list(self.val_idx)]
            self.val_ds = SeqFcnDataset(val_data_frame)
                    
        # Assigns test dataset for use in dataloader
        if stage == 'test' or stage is None:
            test_data_frame = self.data_df.iloc[list(self.test_idx)]
            self.test_ds = SeqFcnDataset(test_data_frame)
            
    #The DataLoader object is created using the train_ds/val_ds/test_ds objects with the batch size set during
    # initialization of the class and shuffle=True.
    def train_dataloader(self):
        return data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return data_utils.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)
    def test_dataloader(self):
        return data_utils.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True)
    
    def save_splits(self, path):
        """Save the data splits to a file at the given path"""
        with open(path, 'wb') as f:
            pickle.dump((self.train_idx, self.val_idx, self.test_idx), f)

    def load_splits(self, path):
        """Load the data splits from a file at the given path"""
        with open(path, 'rb') as f:
            self.train_idx, self.val_idx, self.test_idx = pickle.load(f)
            
            train_indices = self.train_idx
            val_indices = self.val_idx
            test_indices = self.test_idx
            
        return train_indices, val_indices, test_indices

# PTLModule is the actual neural network. Model architecture can be altered here.
class PTLModule(pl.LightningModule):
    """PyTorch Lightning Module that defines model and training"""
      
    # define network
    def __init__(self, slen, learning_rate, epochs, weights):
        super().__init__()
        
        # Creates an embedding layer in PyTorch and initializes it with the pretrained weights stored in aaindex
        AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
        self.embed = nn.Embedding(len(AAs), 16) # maps integer indices (a.a.'s)' to 16-dimensional vectors
        self.slen = slen # synthetic query sequence length (128 a.a.)
        self.ndim = self.embed.embedding_dim # dimensions of AA embedding

        # fully connected neural network
        ldims = [self.slen*self.ndim,65,40]
        self.dropout = nn.Dropout(p=0.2)
        self.linear_1 = nn.Linear(ldims[0], ldims[1])
        self.linear_2 = nn.Linear(ldims[1], ldims[2])
        
        # learning rate
        self.learning_rate = learning_rate
        self.save_hyperparameters() # log hyperparameters to file
        
        # # Define label weights as tensor
        # columns_w_labels = ['0_DVA/C4', '0_DVA/DVO','0_DVA/(DVA+DVO)', '0_(DVA+DVO)/VDAL',
        #                     '1_DVA/C4', '1_DVA/DVO','1_DVA/(DVA+DVO)', '1_(DVA+DVO)/VDAL',
        #                     '2_OA/C6', '2_OA/OL', '2_OA/(OA+OL)', '2_(OA+OL)/PDAL',
        #                     '3_OA/C6', '3_OA/OL', '3_OA/(OA+OL)', '3_(OA+OL)/PDAL',
        #                     '4_DVA/C4', '4_DVA/DVO', '4_DVA/(DVA+DVO)', '4_(DVA+DVO)/VDAL',
        #                     '5_OA/C6', '5_OA/OL', '5_OA/(OA+OL)', '5_(OA+OL)/PDAL',
        #                     '6_DVA/C4', '6_DVA/DVO', '6_DVA/(DVA+DVO)', '6_(DVA+DVO)/VDAL',
        #                     '7_OA/C6', '7_OA/OL', '7_OA/(OA+OL)', '7_(OA+OL)/PDAL',
        #                     '8_DVA/C4', '8_DVA/DVO', '8_DVA/(DVA+DVO)', '8_(DVA+DVO)/VDAL',
        #                     '9_OA/C6', '9_OA/OL', '9_OA/(OA+OL)', '9_(OA+OL)/PDAL']
        
        # Chosen weights, frozen
        self.label_weights = torch.tensor(weights, dtype=torch.float)
        
#         # Learnable weights
#         self.label_weights = nn.Parameter(torch.tensor(weights, dtype=torch.float))

    # FCN Model
    def forward(self, x):
        x = self.embed(x)
        x = x.view(-1,self.ndim*self.slen)
        x = self.linear_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear_2(x)
        return x
      
    def training_step(self, batch, batch_idx):
        sequence,scores = batch
        output = self(sequence)
        
        mask = (scores != -1).float() # create a mask tensor where 1 indicates a valid label and 0 indicates an invalid label 
        loss = nn.MSELoss(reduction='none')(output, scores) # Calculate MSE
        loss = torch.mul(loss, self.label_weights) # Use label weights tensor to weight the MSE loss
        regression_observations = torch.sum(mask) # Calculate Number of Observed Labels
        loss = torch.mul(loss, mask) # apply the mask to MSE to eliminate MSE's Calculated with missing labels
        loss = torch.div(torch.sum(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)), regression_observations) # computes the average loss over the observed labels, ignoring any invalid labels
        # The torch.nan_to_num() function replaces NaN, positive infinity, and negative infinity values in the loss tensor with 0.0
        # The torch.sum() function sums the resulting tensor
        # torch.div() divides the sum by the number of observed labels (regression_observations) to obtain the mean loss per label.
        
        if regression_observations == 0.0:
            # this might occur with small batch sizes, if there were no observations the loss is zero
            # No enzyme has no labels, but I included this in case I used this model in the future
            loss = 0.0

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step = False, on_epoch=True) # reports MSE loss to model
        return loss

    def validation_step(self, batch, batch_idx):
        sequence,scores = batch
        output = self(sequence)

        mask = (scores != -1).float() # create a mask tensor where 1 indicates a valid label and 0 indicates an invalid label (-1)
        loss = nn.MSELoss(reduction='none')(output, scores) # Calculate MSE
        loss = torch.mul(loss, self.label_weights) # Use label weights tensor to weight the MSE loss
        regression_observations = torch.sum(mask) # Calculate Number of Observed Labels
        loss = torch.mul(loss, mask) # apply the mask to MSE to eliminate Error's Calculated with missing labels
        loss = torch.div(torch.sum(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)), regression_observations) # computes the average loss over the observed labels, ignoring any invalid labels
        # The torch.nan_to_num() function replaces NaN, positive infinity, and negative infinity values in the loss tensor with 0.0
        # The torch.sum() function sums the resulting tensor
        # torch.div() divides the sum by the number of observed labels (regression_observations) to obtain the mean loss per label.
        
        if regression_observations == 0.0:
            # this might occur with small batch sizes, if there were no observations the loss is zero
            # No enzyme has no labels, but I included this in case I used this model in the future
            loss = 0.0
        
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step = False, on_epoch=True) # reports MSE loss to model
        return loss

    def test_step(self, batch):
        sequence,scores = batch
        output = self(sequence)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.001) # Weight Decay to penalize too large of weights
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) # No weight decay
        return optimizer
    
    def predict(self, sequence):
        ind = torch.tensor(aa2ind(list(sequence))) # Convert the amino acid sequence to a tensor of indices
        x = ind.view(1,-1) # Add a batch dimension to the tensor (put here instead of forward function)
        pred = self(x) # Apply the model to the tensor to get the prediction
        return pred.detach().numpy() # Detach the prediction from the computation graph and convert it to a NumPy array


# Load Data from MSA for ConvVAE format
def get_msa_from_fasta(filename):
    """
    get_msa_from_fasta is useful for getting sequences in MSA from fasta file
    """
    import Bio.SeqIO
    with open(filename, "rt") as fh: 
        return [r[1] for r in Bio.SeqIO.FastaIO.SimpleFastaParser(fh)]
    
class ProtMSA(torch.utils.data.Dataset):
    """
    ProtMSA is a helper class for loading and formating the data from MSA to a torch tensor
    """
    
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
    """
    A PyTorch Lightning Data Module to handle data splitting.
    """

    def __init__(self, MSA, batch_size, sample_weights, num_workers):
        super().__init__()
        self.MSA = MSA
        self.batch_size = batch_size
        train_val_test_split = [0.89314, 0.1, 0.00686] # 100 test sequences
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
    ConvVAE is a convolutional variational autoencoder that learns the probability of amino acids occurring
    at each position for 128 amino acids, learns from MSA sequences
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







