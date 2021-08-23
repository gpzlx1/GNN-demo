"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import dgl

import dataset

from gat import GAT


def main(args, device, data):
    # load and preprocess dataset
    n_classes, train_g, train_nfeat, train_labels, train_nid = data

    train_g = train_g.int().to(device)

    num_feats = train_nfeat.shape[1]
    n_edges = train_g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d""" %
          (n_edges, n_classes, train_nid.shape[0]))

    # add self loop
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(train_g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    print(model)
    model.cuda()

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        
        # count
        t0 = time.time()
        
        # forward
        logits = model(train_nfeat)
        loss = loss_fcn(logits[train_nid], train_labels[train_nid])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # count
        dur.append(time.time() - t0)


        acc = 0
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur[3:]) / 1000))

    print()
    print(dur)
    #acc = evaluate(model, g, train_nfeat, train_labels, train_nid)
    #print("Test Accuracy {:.4f}".format(acc))
    print('{:.4f} {:.4f} {:.4f}'.format(np.mean(dur[3:]), 
        np.percentile(dur[3:], 97),
        np.percentile(dur[3:], 3)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--in-feat', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='reddit')
    args = parser.parse_args()
    print(args)

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

    data = n_classes, train_g, train_nfeat, train_labels, train_nid
    main(args, device, data)
