import torch as th

import dgl.ndarray as nd
from dgl import utils
from dgl import backend
from dgl.transform import _CAPI_DGLToBlock
from dgl.heterograph import DGLHeteroGraph, DGLBlock
from dgl.sampling.neighbor import _CAPI_DGLSampleNeighbors


def sample_block(graph, nodes_all_types, fanout, replace=False):
    #th.cuda.nvtx.range_push("Sample Neighbors")
    subgidx = _CAPI_DGLSampleNeighbors(
            graph._graph, 
            nodes_all_types,
            backend.to_dgl_nd(backend.tensor([fanout], dtype=backend.int64)),
            'in',  
            [nd.array([], ctx=nd.cpu())] * len(graph.etypes),
            replace
        )
    subgraph = DGLHeteroGraph(subgidx.graph, graph.ntypes, graph.etypes)
    #th.cuda.nvtx.range_pop()

    #th.cuda.nvtx.range_push("To Block")
    new_graph_index, src_nodes_nd, induced_edges_nd = _CAPI_DGLToBlock(
        subgraph._graph, nodes_all_types, True
    )
    block = DGLBlock(new_graph_index, (subgraph.ntypes, subgraph.ntypes), subgraph.etypes)

    src_node_ids = [backend.from_dgl_nd(src) for src in src_nodes_nd]
    dst_node_ids = [backend.from_dgl_nd(dst) for dst in nodes_all_types]

    node_frames = utils.extract_node_subframes_for_block(subgraph, src_node_ids, dst_node_ids)
    block._node_frames = node_frames
    #th.cuda.nvtx.range_pop()
    
    return block, src_nodes_nd