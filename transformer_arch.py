import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torch.nn.init as init
import os
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        # Multi-head self-attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)

        # Feedforward layer
        src2 = self.linear2(nn.functional.relu(self.linear1(src)))
        src = src + self.dropout(src2)

        # Layer normalization
        src = self.norm2(src)
        src = self.norm1(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.prediction_head = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)

        # Global average pooling across the sequence dimension
        x = x.mean(dim=1)

        # Prediction head
        output = self.prediction_head(x)
        return output


# Example usage:
input_size = 4  # Size of each token: [ck, xk, xk^2, 1]
d_model = 64  # Dimension of the model
nhead = 4  # Number of heads in multi-head attention
num_layers = 3  # Number of transformer layers
output_size = 3  # Size of the output: [ak, bk, dk]
batch_size = 1024  # Batch size

model = TransformerEncoder(input_size, d_model, nhead, num_layers, output_size)
input_data = torch.rand((batch_size, 10, input_size))  # Batch size of 1024, sequence length of 10
output = model(input_data)
print(output.shape)