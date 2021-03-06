"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv
import tqdm
import dgl


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


class SampleGAT(nn.Module):
    def __init__(self,
                num_layers,
                in_feat,
                num_hidden,
                num_classes,
                heads,
                activation,
                feat_drop,
                 attn_drop,
                negative_slope,
                residual):
        super(SampleGAT, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.heads = heads

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_feat, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, blocks, inputs):
        h = inputs

        for block, l in zip(blocks, range(self.num_layers)):
            torch.cuda.nvtx.range_push("layer_{}".format(l))
            h = self.gat_layers[l](block, h).flatten(1)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("layer_{}".format(self.num_layers))
        logits = self.gat_layers[-1](blocks[-1], h).mean(1)
        torch.cuda.nvtx.range_pop()
        return logits