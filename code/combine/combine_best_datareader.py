################################################################################
# Copyright 2024-2025 Lawrence Livermore National Security, LLC and other
# Heesung Shim (shim2@llnl.gov)
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Data preprocessing for the combined model
################################################################################

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

h5_lig  = "ligand"
h5_coord = "coord"
h5_feat  = "feat"
h5_efeat = "efeat"
h5_vdw  = "vdw"
h5_ligcoord = "lig_coord"
h5_ligfeat  = "lig_feat"
h5_aff = "affinity"
h5_mordred = "mordred"

class Dataset_PDB(Dataset):
    def __init__(self, csv_filepaths, set1_npy_filepaths, set2_npy_filepaths):
        print( f"Initializing Dataset_PDB with csv_fpath: {csv_filepaths} ")
        print( f"set1_npy_filepaths: {set1_npy_filepaths} and set2_npy_filepaths: {set2_npy_filepaths}")
        super(Dataset_PDB, self).__init__()
        self.csv_filepaths = csv_filepaths
        self.set1_npy_filepaths = set1_npy_filepaths
        self.set2_npy_filepaths = set2_npy_filepaths       
        self.data_list = []  # store file_path, id, labels
        self.set1_array_list = []
        self.set2_array_list = []
        
        
        for csv_fpath in csv_filepaths:
            print(f"Reading CSV file from path: {csv_fpath}")
            df = pd.read_csv(csv_fpath)
            for row in df.itertuples():
                self.data_list.append([row.fdir, row.fn, str(row.pdbid), float(row.affinity)])

        for set1_npy_filepath in set1_npy_filepaths:
            print(f"Reading set1 file from path: {set1_npy_filepath}")
            set1 = np.load(set1_npy_filepath)
            for row in set1:
                self.set1_array_list.append(row)
                
        for set2_npy_filepath in set2_npy_filepaths:
            print(f"Reading set2 file from path: {set2_npy_filepath}")
            set2 = np.load(set2_npy_filepath)
            for row in set2:
                self.set2_array_list.append(row)


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,ind):
        fdir, fn, pdbid, affinity = self.data_list[ind]
        
        set1_row = self.set1_array_list[ind]
        set2_row = self.set2_array_list[ind]
        label_aff = torch.tensor(np.expand_dims(affinity, axis=0)).float()
        
        return {
            "pdbid": pdbid,
            "set1_value": set1_row,
            "set2_value": set2_row,
            "affinity": label_aff
            }



