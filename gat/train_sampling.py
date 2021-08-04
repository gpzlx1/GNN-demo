import argparse
from types import new_class
import numpy as np
import networkx as nx
import time
import torch
from torch._C import set_flush_denormal
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args

from gat import SampleGAT
from utils import EarlyStopping

from dgl.data import RedditDataset

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def evaluate(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    num_workers = 4
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, nfeat, device, args.batch_size, num_workers)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def main(args):
    device = "cuda:0"
    data = RedditDataset(self_loop=True)
    _g = data[0]

    train_g = val_g = test_g = _g
    train_nfeat = val_nfeat = test_nfeat = _g.ndata.pop('feat')
    train_labels = val_labels = test_labels = _g.ndata.pop('label')

    n_classes = data.num_classes
    train_nid = torch.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
    #g.ndata.clear()

    in_feats = train_nfeat.shape[1]
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]

    train_nfeat = train_nfeat.cuda()
    val_nfeat = test_nfeat = train_nfeat

    dataloader_device = torch.device('cpu')
    if args.sample_gpu:
        # copy only the csc to the GPU
        train_g = train_g.formats(['csc'])
        train_nid = train_nid.to(device)
        train_g = train_g.to(device)
        dataloader_device = device

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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])

    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        device=dataloader_device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
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
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device)
            print('Eval Acc {:.4f}'.format(eval_acc))
            test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device)
            print('Test Acc: {:.4f}'.format(test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
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
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    argparser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    argparser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    argparser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    argparser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    args = argparser.parse_args()

    main(args)