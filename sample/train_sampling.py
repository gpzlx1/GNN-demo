import argparse
from networkx.classes.function import degree

from numpy.lib.function_base import percentile
from dataset import load_cit_patents
import time

import dataset
import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from dgl import backend
from sampler import sample_block


#### Entry point
def run(args, device, data):
    # Unpack data
    n_classes, train_g, train_nfeat, train_labels, train_nid = data
    in_feats = train_nfeat.shape[1]

    idx = torch.randperm(train_nid.nelement())
    train_nid = train_nid[idx]
    train_nid = train_nid.to(device)
    train_g = train_g.formats(['csc'])
    train_g = train_g.to("cuda")
    
    # Define model and optimizer

    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]


    srcnodes_layers = []
    dstnodes_layers = []
    edges_layers = [] 
    # Training loop
    for _ in range(len(fan_out)):
        srcnodes_layers.append([])
        dstnodes_layers.append([])
        edges_layers.append([])

    for epoch in range(args.num_epochs):
        for step in range((train_nid.nelement() + args.batch_size - 1) // args.batch_size):
            # Load the input features as well as output labels
            seeds = train_nid[step * args.batch_size : (step + 1) * args.batch_size]
            blocks = []
            nodes_all_types = [backend.to_dgl_nd(seeds)]
            for i, num_picks in enumerate(fan_out):
                block, nodes_all_types = sample_block(train_g, nodes_all_types, num_picks, False)
                srcnodes_layers[i].append(int(block.num_src_nodes()))
                dstnodes_layers[i].append(int(block.num_dst_nodes()))
                edges_layers[i].append(int(block.num_edges()))

    
    for i in range(len(fan_out)):
        print()
        print("srcnode: {:.0f} {:.0f} {:.0f}".format(np.mean(srcnodes_layers[i]), np.percentile(srcnodes_layers[i], 97), np.percentile(srcnodes_layers[i], 3)))
        print("dstnode: {:.0f} {:.0f} {:.0f}".format(np.mean(dstnodes_layers[i]), np.percentile(dstnodes_layers[i], 97), np.percentile(dstnodes_layers[i], 3)))
        print("edge: {:.0f} {:.0f} {:.0f}".format(np.mean(edges_layers[i]), np.percentile(edges_layers[i], 97), np.percentile(edges_layers[i], 3)))
        degree = []
        node = []
        for n1, n2, e in zip(srcnodes_layers[i], dstnodes_layers[i], edges_layers[i]):
            degree.append(float(e) / float(n1+n2))
            node.append(n1+n2)

        print("degree: {} {} {}".format(np.mean(degree), np.percentile(degree, 97), np.percentile(degree, 3)))
        print("node: {} {} {}".format(np.mean(node), np.percentile(node, 97), np.percentile(node, 3)))
    
 
        

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=1)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=50)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--in-feat', type=int, default=256)
    args = argparser.parse_args()

    device = torch.device('cuda:%d' % args.gpu)
    if args.dataset == 'reddit':
        g, n_classes, train_nid = dataset.load_reddit()
    elif args.dataset == 'ogbn-products':
        g, n_classes, train_nid = dataset.load_ogbn_products()
    elif args.dataset == 'cit-patents':
        g, n_classes, train_nid = dataset.load_cit_patents()
    else:
        raise Exception('unknown dataset')

    train_g =  g
    train_nfeat = torch.randn((g.num_nodes(), args.in_feat)).to(device)
    train_labels = torch.randint(0, n_classes, (g.num_nodes(),)).to(device)

    # Pack data
    data = n_classes, train_g, train_nfeat, train_labels, train_nid

    print(args)
    run(args, device, data)
