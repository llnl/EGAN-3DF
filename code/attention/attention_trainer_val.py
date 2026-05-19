################################################################################
# Copyright 2024-2025 Lawrence Livermore National Security, LLC and other
# Heesung Shim (shim2@llnl.gov)
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Training, validation, and testing for the attention model
################################################################################

import os
import sys
sys.stdout.flush()
import numpy as np
import torch
import torch.nn as nn 
import math
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from attention_datareader import Dataset_PDB
from attention import AttentionNN

class Trainer:
    def __init__(self, configs, run_mode):
        self.configs = configs
        self.run_mode = run_mode

        use_cuda = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count()
        device = "cuda"
        if use_cuda:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
        print(use_cuda, cuda_count, self.device)
        self.init_model()

    def worker_init_fn(self, worker_id):
        np.random.seed(0)

    def valid_file(self, path):
        return os.path.isfile(path)

    def strip_prefix_if_present(self, state_dict, prefix):
        keys = sorted(state_dict.keys())
        if not all(key.startswith(prefix) for key in keys):
            return
        for key in keys:
            newkey = key[len(prefix):]
            state_dict[newkey] = state_dict.pop(key)

    def init_model(self):
        #get_feature_only = True if run_mode >=5 else False
        self.model = AttentionNN(input_dim=10, seq_len=10 , output_dim=self.configs["model"]["out_dim"], attention_mode=0, verbose=0)
                #get_feature_only=get_feature_only)
        self.model.to(self.device)

    def train(self):
        train_csvs = self.configs["dataset"]["train_csvs"]
        train_set1_npy = self.configs["dataset"]['train_set1_npy']
        val_csvs = self.configs["dataset"]["val_csvs"]
        val_set1_npy = self.configs["dataset"]['val_set1_npy']
        batch_size = self.configs["train"]["batch_size"]
        num_workers = self.configs["train"]["num_workers"]
        num_epochs = self.configs["train"]["epochs"]
        checkpoint_dir = self.configs["checkpoint"]["dir"]

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # Print file paths for debugging
        print(f"Train CSVs path: {train_csvs}")
        print(f"train_set1_npy: {train_set1_npy}")
        
        if torch.cuda.device_count() > 0:
            print("Training on multiple GPUs")
            # Your multi-GPU training code here
        else:
            print("Training on a CPU")
            # Your CPU training code here

        dataset = Dataset_PDB(train_csvs, train_set1_npy)
        val_dataset = Dataset_PDB(val_csvs, val_set1_npy)
        dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
        val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
        
        print("len of train:", len(dataloader))
        print("len of val:", len(val_dataloader))

        loss_fn = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.configs["train"]["lr"],weight_decay = self.configs["train"]["weight_decay"] )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        epoch_start = 0
        prefix = self.configs["checkpoint"]["prefix"]
        suffix = self.configs["checkpoint"]["suffix"]

        if isinstance(self.model, (DistributedDataParallel, DataParallel)):
            model_to_save = self.model.module
        else:
            model_to_save = self.model
        model_fn = os.path.join(self.configs["checkpoint"]["dir"], f"{prefix}_{suffix}.pth")
        if self.valid_file(model_fn):
            checkpoint = torch.load(model_fn)
            model_state_dict = checkpoint.pop("model_state_dict")
            self.strip_prefix_if_present(model_state_dict, "module.")
            model_to_save.load_state_dict(model_state_dict, strict=False)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            epoch_start = checkpoint["epoch"]
            step = checkpoint["step"]
            print(f"checkpoint loaded: {model_fn}")

        if not os.path.exists(os.path.dirname(self.configs["checkpoint"]["dir"])):
            os.makedirs(os.path.dirname(self.configs["checkpoint"]["dir"]))
        output_dir = os.path.dirname(self.configs["checkpoint"]["dir"])

        best_val_rmse = 1000.0
        step = 0
        epoch_losses = []
        for epoch_ind in range(epoch_start, num_epochs):
            self.model.train()
            batch_losses = []
            for batch_ind, batch in enumerate(dataloader):                
                input = batch["confid_values"].to(self.device)
                pdbid = batch["pdbid"].to(self.device)
                output_aff = batch["affinity"].to(self.device).squeeze(1)

                if not torch.cuda.is_available():
                    input = Batch.from_data_list(input)
                    
                output = self.model(input)
                pred = output["prediction"]

                loss_aff = loss_fn(pred[:,0].cpu().float(), output_aff.cpu().float())
                loss = loss_aff

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss = loss.cpu().data.item()
                batch_losses.append(batch_loss)
                print(f"[{epoch_ind+1}/{num_epochs}-{batch_ind+1}/{len(dataloader)}] training, loss: {batch_loss:.3f}, lr: {optimizer.param_groups[0]['lr']:.7f}")

            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
            scheduler.step()
            print("[%d/%d] training, epoch loss: %.3f" % (epoch_ind+1, num_epochs, epoch_loss))

 ################ validation #################
            ytrue_arr,ypred_arr,pdbid_arr,label_arr  = [],[],[],[]
            
            self.model.eval()
            with torch.no_grad():
                for bind, batch in enumerate(val_dataloader):
                    input = batch["confid_values"].to(self.device)
                    pdbid = batch["pdbid"].to(self.device)
                    output_aff = batch["affinity"].to(self.device)
                    output_aff = output_aff.squeeze(1)

                    if not torch.cuda.is_available():
                        input = Batch.from_data_list(input)

                    output = self.model(input)
                    pred = output["prediction"]

                    print("[%d/%d] evaluating " % (bind, len(val_dataloader)))

                    pdbid_arr.extend(pdbid)
                    label_arr.extend(output_aff.cpu().float().numpy())
                    ypred = pred[:,0].cpu().float().numpy()
                    ytrue = output_aff.squeeze().cpu().float().numpy()
                    ypred_arr.extend(ypred)
                    ytrue_arr.extend(ytrue)
                    

                rmse = math.sqrt(mean_squared_error(ytrue_arr, ypred_arr))
                mae = mean_absolute_error(ytrue_arr, ypred_arr)
                r2 = r2_score(ytrue_arr, ypred_arr)
                pearson, ppval = pearsonr(ytrue_arr, ypred_arr)
                spearman, spval = spearmanr(ytrue_arr, ypred_arr)
                print("RMSE: %.3f, MAE: %.3f, R^2 score: %.3f, Pearson: %.3f, Spearman: %.3f" % (rmse, mae, r2, pearson, spearman))
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    checkpoint_dict = {
                        "model_state_dict": model_to_save.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "step": step,
                        "epoch": epoch_ind
                    }
                    torch.save(checkpoint_dict, model_fn)
                    print(f"checkpoint saved: {model_fn}")
                    print("best validation rmse is : ", best_val_rmse)
                step += 1

################## Test ###########################
    def test(self):
        mlhdf_ver = self.configs["dataset"]["mlhdf_ver"]
        test_csvs = self.configs["dataset"]["test_csvs"]
        test_set1_npy = self.configs["dataset"]['test_set1_npy']
        batch_size = self.configs["test"]["batch_size"]
        num_workers = self.configs["test"]["num_workers"]
        pred_suffix = self.configs["test"]["pred_suffix"]
        ckpt_dir = self.configs["checkpoint"]["dir"]
        ckpt_prefix = self.configs["checkpoint"]["prefix"]
        ckpt_suffix = self.configs["checkpoint"]["suffix"]
        save_dir = self.configs["results_save_dir"]["dir"]
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Print file paths for debugging
        print(f"Test CSVs path: {test_csvs}")
        print(f"test_f_npy: {test_set1_npy}")
        # Implement the testing logic here
        test_dataset = Dataset_PDB(test_csvs, test_set1_npy)
        test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, 
                                             batch_size=batch_size, num_workers=num_workers, 
                                             pin_memory=True, drop_last=False)

        os.makedirs(self.configs["checkpoint"]["dir"], exist_ok=True)
        model_fn = os.path.join(self.configs["checkpoint"]["dir"], "%s_%s.pth" % (ckpt_prefix, ckpt_suffix))

        if os.path.isfile(model_fn) == False or os.path.getsize(model_fn) == 0:
            return 0, 0, 10000, 10000
        #print(self.model.state_dict().keys())

        try:
            checkpoint = torch.load(model_fn)
            model_state_dict = checkpoint.pop("model_state_dict")
            self.strip_prefix_if_present(model_state_dict, "module.")
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(model_state_dict, strict=False)
            else:
                self.model.load_state_dict(model_state_dict, strict=False)
            
            # others
            
            step = checkpoint["step"]
            
            print("step:", step)
            try:
                loss = checkpoint["loss"]
            except:
                loss = 10000
            try:
                val_loss = checkpoint["val_loss"]
            except:
                val_loss = 10000
            print("model checkpoint loaded: ", model_fn)

        except Exception as error:
            print("error occurred when model checkpoint loading! ", error) 
            epoch = 0
            step = 0
            loss = 10000
            val_loss = 10000


        ytrue_arr,ypred_arr,pdbid_arr,label_arr  = [],[],[],[]

        self.model.eval()
        with torch.no_grad():
            for ind, batch in enumerate(test_dataloader):
                input = batch["confid_values"].to(self.device)
                pdbid = batch["pdbid"].to(self.device)
                output_aff = batch["affinity"].to(self.device).squeeze(1)

                if not torch.cuda.is_available():
                    input = Batch.from_data_list(input)

                output = self.model(input)

                pred = output["prediction"]  
                attn_weights = output["attention"]  # (B, seq_len)               
                print("[%d/%d] evaluating " % (ind, len(test_dataloader)))
                pdbid_arr.extend(pdbid)
                label_arr.extend(output_aff.cpu().float().numpy())

                ypred = pred[:,0].cpu().float().numpy() 
                ytrue = output_aff.squeeze().cpu().float().numpy()
                ypred_arr.extend(ypred)
                ytrue_arr.extend(ytrue)


            rmse = math.sqrt(mean_squared_error(ytrue_arr, ypred_arr))
            mae = mean_absolute_error(ytrue_arr, ypred_arr)
            r2 = r2_score(ytrue_arr, ypred_arr)
            pearson, ppval = pearsonr(ytrue_arr, ypred_arr)
            spearman, spval = spearmanr(ytrue_arr, ypred_arr)
            print("RMSE: %.3f, MAE: %.3f, R^2 score: %.3f, Pearson: %.3f, Spearman: %.3f" % (rmse, mae, r2, pearson, spearman))

            
            np.save(f"{save_dir}/{ckpt_prefix}_{ckpt_suffix}_true.npy", ytrue_arr)
            np.save(f"{save_dir}/{ckpt_prefix}_{ckpt_suffix}_pred.npy", ypred_arr)
          
