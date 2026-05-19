################################################################################
# Copyright 2024-2025 Lawrence Livermore National Security, LLC and other
# Heesung Shim (shim2@llnl.gov)
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
#  Data preprocessing for the EGNN model
################################################################################


import os
import h5py
import pandas as pd
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from sklearn.metrics import pairwise_distances

h5_lig  = "ligand"
h5_coord = "coord"
h5_feat  = "feat"
h5_vdw  = "vdw"
h5_ligcoord = "lig_coord"
h5_ligfeat  = "lig_feat"
h5_aff = "affinity"



class Dataset_PDB(Dataset):
    def __init__(self, csv_filepaths, h5_filepaths, mlhdf_ver=2,label_type=1, h5_driver=None):
        print( f"Initializing Dataset_PDB with csv_fpath: {csv_filepaths} and h5_filepaths: {h5_filepaths}")
        super(Dataset_PDB, self).__init__()
        self.csv_filepaths = csv_filepaths
        self.h5_filepaths = h5_filepaths
        self.mlhdf_ver = mlhdf_ver
        self.label_type = label_type # 1: bindinf affinity, 2: pose classification
        self.h5_driver = h5_driver
        self.data_list = []  # store file_path, id, labels

        if csv_filepaths is not None and len(csv_filepaths) > 0:
            for csv_fpath in csv_filepaths:
                print(f"Reading CSV file from path: {csv_fpath}")
                df = pd.read_csv(csv_fpath)
                for row in df.itertuples():
                    self.data_list.append([row.fdir, row.fn, str(row.pdbid), float(row.affinity)])
        else:
            for fpath in self.h5_filepaths:
                fdir = os.path.dirname(fpath)
                print(f"fdir is {fdir}")
                fn = os.path.basename(fpath)
                print(f"fn is {fn}")
                with h5py.File(fpath, "r", driver=self.h5_driver) as h5:
                    for pdbid in list(h5):
                        affinity =  h5[pdbid].attrs.get('h5_aff', 0)
                        self.data_list.append([fdir, fn, str(pdbid),float(affinity)])
        self.__savecsv__(self.h5_filepaths[0][:-5]+"_all.csv")

    def __savecsv__(self, csv_filepath):
        df = pd.DataFrame(self.data_list, columns=["fdir", "fn", "pdbid", "affinity"])
        df.to_csv(csv_filepath, index=False)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,ind):
        fdir, fn, pdbid, affinity = self.data_list[ind]

        fpath = os.path.join(fdir, fn)
        with h5py.File(fpath, "r", driver=self.h5_driver) as h5:
            h5_data = h5[str(pdbid)]["ligand"]
        
            if h5_coord in h5_data and h5_feat in h5_data:
                coord = h5_data[h5_coord][:]
                feat = h5_data[h5_feat][:]

            else:
                raise KeyError(f"Missing keys in h5_data for {pdbid}")
            
            dist = pairwise_distances(coord, metric="euclidean")
            edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dist).float())
            data = Data(y=affinity)
            data.pos = torch.from_numpy(coord)
            data.x = torch.from_numpy(feat).float()
            data.edge_index = edge_index
            data.edge_attr = edge_attr.view(-1, 1)

        label_aff = torch.tensor(np.expand_dims(affinity, axis=0)).float()
        
        return {
            "fpath": fpath,
            "pdbid": pdbid,
            "data": data,
            "affinity": label_aff
            }
