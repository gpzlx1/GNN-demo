import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
from dgl import backend
from gat import SampleGAT

from sampler import sample_block
import dataset

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def run(args, device, data):

    n_classes, train_g, train_nfeat, train_labels, train_nid = data

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
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        
        avg += toc - tic


    print('Avg epoch time: {:.3f}'.format(avg / (epoch + 1)))
    print('{:.4f} {:.4f} {:.4f}'.format(np.mean(iter_tput[10:]) * 1000, 
        np.percentile(iter_tput[10:], 97) * 1000,
        np.percentile(iter_tput[10:], 3) * 1000))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=1)
    argparser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    argparser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    argparser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=50)
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
    run(args, device, data)