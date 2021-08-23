"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import dataset
from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        torch.cuda.nvtx.range_push("dropout")
        h = self.dropout(inputs)
        torch.cuda.nvtx.range_pop()
        for l, layer in enumerate(self.layers):
            torch.cuda.nvtx.range_push("layer {}".format(l))
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                torch.cuda.nvtx.range_push("act")
                h = self.activation(h)
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("dropout")
                h = self.dropout(h)
                torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
        return h


def evaluate(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    if args.dataset == 'reddit':
        g, n_classes, train_nid = dataset.load_reddit()
    elif args.dataset == 'ogbn-products':
        g, n_classes, train_nid = dataset.load_ogbn_products()
    elif args.dataset == 'cit-patents':
        g, n_classes, train_nid = dataset.load_cit_patents()
    else:
        raise Exception('unknown dataset')
    
    torch.cuda.set_device(args.gpu)
    train_nfeat = torch.randn((g.num_nodes(), args.in_feat)).cuda()
    train_labels = torch.randint(0, n_classes, (g.num_nodes(),)).cuda()
    train_nid = train_nid.cuda()


    in_feats = train_nfeat.shape[1]
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d""" %
          (n_edges, n_classes,
           train_nid.shape[0]))


    # graph preprocess and calculate normalization factor
    g = g.int().to(args.gpu)

    # create GraphSAGE model
    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type)

    model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        
        # count
        torch.cuda.synchronize()
        t0 = time.time()

        # forward
        torch.cuda.nvtx.range_push("forward")
        logits = model(g, train_nfeat)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("loss")
        loss = F.cross_entropy(logits[train_nid], train_labels[train_nid])
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("backward")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.nvtx.range_pop()

        # count
        torch.cuda.synchronize()
        dur.append(time.time() - t0)

        #acc = evaluate(model, g, train_nfeat, train_labels, train_nid)
        acc = 0
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur[3:]), loss.item(),
                                            acc, n_edges / np.mean(dur[3:]) / 1000))

    print()
    #acc = evaluate(model, g, train_nfeat, train_labels, train_nid)
    #print("Test Accuracy {:.4f}".format(acc))
    print('{:.4f} {:.4f} {:.4f}'.format(np.mean(dur[3:]), 
        np.percentile(dur[3:], 97),
        np.percentile(dur[3:], 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=5,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument('--in-feat', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='reddit')
    args = parser.parse_args()
    print(args)

    main(args)
