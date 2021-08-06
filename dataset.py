import os
from types import new_class
import dgl
from networkx.readwrite.graph6 import n_to_data
from numpy.lib.utils import source
import torch as th
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset

def load_reddit():
    data = RedditDataset(self_loop=True)
    g = data[0]
    n_classes = data.num_classes
    train_nid = th.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    g.ndata.clear()
    return g, n_classes, train_nid

def load_ogbn_products():
    data = DglNodePropPredDataset(name="ogbn-products")
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    labels = labels[:, 0]
    #n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    n_classes = 41
    train_nid = splitted_idx['train']
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, n_classes, train_nid

def load_cit_patents():
    cache_path = "/home/ubuntu/demo/datasets/cit-Patents/data.bin"
    if os.path.exists(cache_path):
        g = dgl.data.utils.load_graphs(cache_path)[0][0]
        train_nid = th.arange(0, g.number_of_nodes(), dtype=th.int64)
        n_classes = 41
        return g, n_classes, train_nid

    source_path = "/home/ubuntu/demo/datasets/cit-Patents/cit-Patents.txt"
    out = None
    with open(source_path, 'r') as f:
        out = f.readlines()

    map_dir = {}
    count = int(0)
    for edge in out:
        if edge.startswith("#"):
            continue
        for n in edge.strip().split("\t"):
            if n not in map_dir:
                map_dir[n] = int(count)
                count = count + 1

    src = []
    dst = []
    for edge in out:
        if edge.startswith("#"):
            continue
       
        edge = edge.strip().split("\t")
        if edge:
            src.append(int(map_dir[edge[0]]))
            dst.append(int(map_dir[edge[1]]))

    g = dgl.graph((src, dst), num_nodes=3774768)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    dgl.save_graphs(cache_path, [g])

    train_nid = th.arange(0, g.number_of_nodes(), dtype=th.int64)
    n_classes = 41
    return g, n_classes, train_nid
    

if __name__ == "__main__":
    import time
    functions = [load_reddit, load_ogbn_products, load_cit_patents]
    for f in functions:
        begin = time.time()
        g, n_classes, train_nid = f()
        print(g)
        print(n_classes)
        print(train_nid)
        end = time.time()
        print("Time Cost for {} : {}".format(f, end - begin))