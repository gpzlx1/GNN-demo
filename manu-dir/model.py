import torch as th
import torch.nn as nn
import torch.functional as F
import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm
import time

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        time_list = []
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            th.cuda.nvtx.range_push("layer_{}".format(l))

            th.cuda.synchronize()
            begin = time.time()
            h = layer(block, h)
            

            if l != len(self.layers) - 1:
                th.cuda.nvtx.range_push("active + drop")
                h = self.activation(h)
                h = self.dropout(h)
                th.cuda.nvtx.range_pop()
            th.cuda.nvtx.range_pop()

            th.cuda.synchronize()
            time_list.append((time.time() - begin) * 1000)
        return h, time_list
