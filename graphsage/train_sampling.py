import argparse

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

from model import SAGE


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

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
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        tic_step = time.time()
        for step in range((train_nid.nelement() + args.batch_size - 1) // args.batch_size):
            # Load the input features as well as output labels

            torch.cuda.nvtx.range_push("Iterations")

            torch.cuda.nvtx.range_push("sample")
            seeds = train_nid[step * args.batch_size : (step + 1) * args.batch_size]
            blocks = []
            nodes_all_types = [backend.to_dgl_nd(seeds)]
            for i, num_picks in enumerate(fan_out):
                torch.cuda.nvtx.range_push("{}-hops sample".format(i))
                block, nodes_all_types = sample_block(train_g, nodes_all_types, num_picks, False)
                blocks.insert(0, block)
                torch.cuda.nvtx.range_pop()
            input_nodes = blocks[0].srcdata[dgl.NID]
            torch.cuda.nvtx.range_pop()
            
            
            torch.cuda.nvtx.range_push("Fetch_Features")
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
            blocks = [block.to(device) for block in blocks]
            torch.cuda.nvtx.range_pop()

            # Compute loss and prediction
            torch.cuda.nvtx.range_push("Forward")
            batch_pred = model(blocks, batch_inputs)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Loss")
            loss = loss_fcn(batch_pred, batch_labels)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Backward")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.nvtx.range_pop()

            iter_tput.append(time.time() - tic_step)
            if step % args.log_every == 0:
                acc = 0
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc, len(seeds) / np.mean(iter_tput[-args.log_every:]), gpu_mem_alloc))
            tic_step = time.time()

            torch.cuda.nvtx.range_pop()

        toc = time.time()
        avg += toc - tic
        print('Epoch Time(s): {:.4f}'.format(toc - tic))

    print('Avg epoch time: {:.3f}'.format(avg / (epoch + 1)))
    print('{:.4f} {:.4f} {:.4f}'.format(np.mean(iter_tput[10:]) * 1000, 
        np.percentile(iter_tput[10:], 97) * 1000,
        np.percentile(iter_tput[10:], 3) * 1000))

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
