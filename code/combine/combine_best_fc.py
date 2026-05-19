################################################################################
# Copyright 2024-2025 Lawrence Livermore National Security, LLC and other
# Heesung Shim (shim2@llnl.gov)
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fully connected architectire for the combined model
################################################################################


import torch
import torch.nn as nn

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

class PotentialNetFullyConnected2(torch.nn.Module):
    def __init__(self, in_set1_channels, in_set2_channels, out_channels, dropout_prob=0.2):
        super(PotentialNetFullyConnected2, self).__init__()
        self.in_set1_channels = in_set1_channels
        self.in_set2_channels = in_set2_channels
        self.out_channels = out_channels
        self.dropout_prob = dropout_prob
        print(f"channel1:{self.in_set1_channels}")
        print(f"channel2: {self.in_set2_channels}")

        nn1 = nn.Linear(in_set1_channels, 25)
        self.fc_block_m1 = nn.Sequential(
            nn1, 
            nn.ReLU(),
            nn.BatchNorm1d(25), 
            nn.Dropout(p=self.dropout_prob)
        )
        
        nn2 = nn.Linear(25, 10)
        self.fc_block_m2 = nn.Sequential(
            nn2,
            nn.ReLU(),
            nn.BatchNorm1d(10), 
            nn.Dropout(p=self.dropout_prob)
        )

        self.fc_block_s1 = nn.Sequential(
            nn.Linear(in_set2_channels, 25), 
            nn.ReLU(),
            nn.BatchNorm1d(25),
            nn.Dropout(p=self.dropout_prob)
        )

        self.fc_block_s2 = nn.Sequential(
            nn.Linear(25, 10), 
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Dropout(p=self.dropout_prob)
        )

        self.fc_block_all1 = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Dropout(p=self.dropout_prob) 
        )

        self.fc_block_all2 = nn.Sequential(
            nn.Linear(10, out_channels)
            
        )
    
    def forward(self, x1, x2):       
        m = self.fc_block_m1(x1)
        m1 = self.fc_block_m2(m)       
        s = self.fc_block_s1(x2)        
        s1 = self.fc_block_s2(s)  
        z = torch.cat((m1, s1), dim=1) 
        z = self.fc_block_all1(z)
        y = self.fc_block_all2(z)
        return y
        
    
    
