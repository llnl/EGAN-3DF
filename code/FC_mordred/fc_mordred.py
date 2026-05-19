################################################################################
# Copyright 2024-2025 Lawrence Livermore National Security, LLC and other
# Heesung Shim (shim2@llnl.gov)
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# FC mordred model architecture used for training.
################################################################################

import torch
import torch.nn as nn

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]
   
class PotentialNetFullyConnected(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.2):
        super(PotentialNetFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_prob = dropout_prob

        nn1 = nn.Linear(in_channels, 100)
        nn.init.normal_(nn1.weight, mean=0.0, std=0.02)
        nn.init.constant_(nn1.bias, 1.0)
        self.fc_block1 = nn.Sequential(
            nn1, 
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(p=self.dropout_prob)
        )
        
        nn2 = nn.Linear(100, 50)
        nn.init.normal_(nn2.weight, mean=0.0, std=0.02)
        nn.init.constant_(nn2.bias, 1.0)
        self.fc_block2 = nn.Sequential(
            nn2,
            nn.ReLU(),
            #nn.Mish(), 
            nn.BatchNorm1d(50),
            nn.Dropout(p=self.dropout_prob)
        )

        self.fc_block3 = nn.Sequential(
            nn.Linear(50, out_channels)
        )
    
    def forward(self, mordred, return_hidden_feature=False):
        fc1= self.fc_block1(mordred)
        fc2= self.fc_block2(fc1)
        fc3= self.fc_block3(fc2) 
        return fc3, fc2
        


    
