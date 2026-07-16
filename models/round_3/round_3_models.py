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

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MDAAKSQMAVKHLIVLKFKDEITEAQKEEFFKTYVNLVNKCIIPAMKDVYWLRSSGKLDVTQKNKEEGYTHIVEVTFESVETIQDYIIEHPAHVGFGDVYRSFWEKLLIFDYPSVLVTPRKIQLNSSY' # Synthetic Query Sequence
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

            if "Cluster" in self.data_df.columns:
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

            else:
                # No Cluster column: perform a standard random 80/10/10 split
                all_indices = self.data_df.index.tolist()

                train_indices, temp_indices = train_test_split(
                    all_indices,
                    test_size=0.2,
                    random_state=2,
                    shuffle=True,
                )

                val_indices, test_indices = train_test_split(
                    temp_indices,
                    test_size=0.5,
                    random_state=2,
                    shuffle=True,
                )
            
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
        self.embed = nn.Embedding(len(AAs), 16) # maps integer indices (a.a.'s)' to 16-dimensional vectors
        self.slen = slen # synthetic query sequence length (128 a.a.)
        self.ndim = self.embed.embedding_dim # dimensions of AA embedding

        # fully connected neural network
        ldims = [self.slen*self.ndim,65,48]
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
        #                     '9_OA/C6', '9_OA/OL', '9_OA/(OA+OL)', '9_(OA+OL)/PDAL',
        #                     '10_DVA/C4', '10_DVA/DVO', '10_DVA/(DVA+DVO)', '10_(DVA+DVO)/VDAL',
        #                     '11_OA/C6', '11_OA/OL', '11_OA/(OA+OL)','11_(OA+OL)/PDAL']
        
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

