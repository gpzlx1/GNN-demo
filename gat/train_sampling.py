import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
from dgl import backend
from gat import SampleGAT

from dgl.data import RedditDataset
from sampler import sample_block

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def main(args):
    device = "cuda:0"
    data = RedditDataset(self_loop=True)
    _g = data[0]

    train_g = _g
    train_nfeat = _g.ndata.pop('feat')
    train_labels = _g.ndata.pop('label')

    n_classes = data.num_classes
    train_nid = torch.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    idx = torch.randperm(train_nid.nelement())
    train_nid = train_nid[idx]
    train_nid = train_nid.to(device)

    #g.ndata.clear()

    in_feats = train_nfeat.shape[1]
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]

    train_nfeat = train_nfeat.cuda()
    train_g = train_g.formats(['csc'])
    train_g = train_g.to(device)

    model = SampleGAT(args.num_layers,
                    in_feats,
                    args.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual)
    print(model)
    model = model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        tic_step = time.time()
        for step in range((train_nid.nelement() + args.batch_size - 1) // args.batch_size):

            seeds = train_nid[step * args.batch_size : (step + 1) * args.batch_size]
            blocks = []
            nodes_all_types = [backend.to_dgl_nd(seeds)]
            for num_picks in fan_out:
                block, nodes_all_types = sample_block(train_g, nodes_all_types, 10, False)
                blocks.insert(0, block)
            input_nodes = blocks[0].srcdata[dgl.NID]

            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
            blocks = [block.to(device) for block in blocks]

            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = 0
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc, np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic


    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    argparser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    argparser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    argparser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    argparser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    argparser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    argparser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    argparser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    args = argparser.parse_args()

    main(args)