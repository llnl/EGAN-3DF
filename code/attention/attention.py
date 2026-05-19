################################################################################
# Copyright 2024-2025 Lawrence Livermore National Security, LLC and other
# Heesung Shim (shim2@llnl.gov)
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Attention model architecture used for training.
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionNN(nn.Module):
    def __init__(self, input_dim=10, seq_len=10, output_dim=1, num_heads=5, attention_mode=0, verbose=0):
        super(AttentionNN, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.attention_mode = attention_mode
        self.verbose = verbose
        self.norm = nn.LayerNorm(input_dim)  # Layer normalization

        if attention_mode == 0:
            self.attention_layer = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        else:
            print("Transformer Encoder mode")
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
            self.attention_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.dropout = nn.Dropout(p=0.1)

        self.dense_block1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
        )
        self.dense_block2 = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
        )
        self.dense_block3 = nn.Linear(16, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.norm(x)

        if self.attention_mode == 0:
            # Multi-head self-attention
            out_att, attn_weights = self.attention_layer(x, x, x, need_weights=True, average_attn_weights=False)  # shape: (B, seq_len, input_dim), (B, num_heads, seq_len, seq_len)
            attn_weights = attn_weights.mean(dim=1)  # mean over heads → (B, N, N)
            attention_score = attn_weights.mean(dim=1)  # mean over sender → (B, N)
            attn_weights_to_return = attention_score               
            weights = attention_score.unsqueeze(-1)       # shape: (B, seq_len, 1)
            out_att_pooled = (x * weights).sum(dim=1)              # weighted sum
        else:
            out_att = self.attention_layer(x)                      # shape: (B, seq_len, input_dim)
            out_att_pooled = torch.mean(out_att, dim=1)            # simple average pooling

        if self.verbose:
            print("Pooled shape:", out_att_pooled.shape)

        out = self.dropout(out_att_pooled)
        out = self.dense_block1(out)
        out = self.dropout(out)
        out = self.dense_block2(out)
        out = self.dense_block3(out)

        return {
        "prediction": out,  # prediction value (B, output_dim)
        "attention": attn_weights_to_return  # attention weight (B, seq_len) or None
    }

