################################################################################
# Copyright 2024-2025 Lawrence Livermore National Security, LLC and other
# Heesung Shim (shim2@llnl.gov)
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Data preprocessing for the attention model
################################################################################

import os
import pandas as pd
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

h5_lig  = "ligand"
h5_coord = "coord"
h5_feat  = "feat"
h5_vdw  = "vdw"
h5_ligcoord = "lig_coord"
h5_ligfeat  = "lig_feat"
h5_aff = "affinity"

class Dataset_PDB(Dataset):
    def __init__(self, csv_filepaths, set1_npy_filepaths):
        print( f"Initializing Dataset_PDB with csv_fpath: {csv_filepaths}")
        print( f"set1_npy_filepaths: {set1_npy_filepaths}")
        super(Dataset_PDB, self).__init__()
        self.csv_filepaths = csv_filepaths
        self.set1_npy_filepaths = set1_npy_filepaths

        self.data_list = []  
         
        for csv_fpath,npy_filepath in zip(csv_filepaths,set1_npy_filepaths):
            print(f"Reading CSV file from path: {csv_fpath}")
            df = pd.read_csv(csv_fpath)
            grouped = df.groupby('pdbid')
            all_confid_values = np.load(npy_filepath)
            conf_ind = 0
            for pdbid, group in grouped:
                # Get all confid values for this pdbid (e.g., 10 confid values)
                pdbid_confid_indices = group['confid'].values
                affinity = group['affinity'].values[0]  # Only one affinity for each pdbid
                
                # For each confid, load the corresponding npy files
                confid_values_list = []
                for index in pdbid_confid_indices:
                    # Assuming the confid is an index or identifier for npy files
                    confid_values = all_confid_values[conf_ind+index-1, :]  # Assuming the naming format
                    confid_values_list.append(confid_values)
                confid_values_array = np.array(confid_values_list)        # Load the numpy file for the current confid
                       
                self.data_list.append([pdbid, pdbid_confid_indices, affinity, confid_values_array])
                conf_ind += len(pdbid_confid_indices)
                
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,ind):
        pdbid, confids, affinity, confid_values = self.data_list[ind]
         
        set1_tensor = torch.from_numpy(confid_values)
        label_aff = torch.tensor(np.expand_dims(affinity, axis=0)).float()

        return {
            "pdbid": pdbid,
            "confid_values": set1_tensor,
            "affinity": label_aff
            }