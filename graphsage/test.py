import torch
import dgl
from dgl.data import RedditDataset

sampler_not = dgl.dataloading.MultiLayerNeighborSampler([25])
sampler_replace = dgl.dataloading.MultiLayerNeighborSampler([25], replace=True)

data = RedditDataset(self_loop=True)
g = data[0]

train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
idx = torch.randperm(train_nid.nelement())
train_nid = train_nid[idx]
seeds = train_nid[3000:3100]
print(seeds)

for sampler, string in zip([sampler_not, sampler_replace], ["replace=False", "replace=True"]): 
    print(string)

    seeds_cpu = seeds
    g_cpu = g
    print(g_cpu.device, seeds_cpu.device)
    ret_cpu = sampler.sample_blocks(g_cpu, seeds_cpu)[0]
    print(ret_cpu.srcdata[dgl.NID], ret_cpu.srcdata[dgl.NID].shape)

    seeds_cuda = seeds.to("cuda")
    g_cuda = g.formats(['csc'])
    g_cuda = g_cuda.to("cuda")
    print(g_cuda.device, seeds_cuda.device)
    ret_cuda = sampler.sample_blocks(g_cuda, seeds_cuda)[0]
    print(ret_cuda.srcdata[dgl.NID], ret_cuda.srcdata[dgl.NID].shape)

    print()

