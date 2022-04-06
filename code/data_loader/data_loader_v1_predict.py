import glob
import json
import os
from collections import defaultdict
from typing import Dict, Any, Union, Callable
import dgl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from general_util.logger import get_child_logger
from data_loader.data_utils import EmbeddingMatrix
logger = get_child_logger('Dataset')

class SubgraphDataset(Dataset):
    def __init__(self, triple_file: str, split: str, test_subgraph_dir):
        logger.info(f'Loading data file from {triple_file}.')
        self.split = split
        self.triples = json.load(open(triple_file, 'r'))
        if split == 'test':
            self.dir = test_subgraph_dir

    def __getitem__(self, index) -> T_co:

        triple = self.triples[index]

        all_nodes = set()
        all_dgl_graph, all_src, all_dst, all_node2re_id, all_re_id2node = [], [], [], [], []

        tri_graph = torch.load(os.path.join(self.dir, '{}.pt'.format(triple["user"])))

        UO_OI_IA = self._load_subgraph(tri_graph)
        for each in UO_OI_IA:
            all_dgl_graph.append(each[0])
            all_src.append(each[1])
            all_dst.append(each[2])
            all_node2re_id.append(each[3])
            all_re_id2node.append(each[4])
            all_nodes.update(each[5])

        return all_dgl_graph, all_node2re_id, all_re_id2node, list(all_nodes), triple, int(triple["user"])

    def __len__(self):
        return len(self.triples)

    def _load_subgraph(self, tri_graph: dict):

        UO_nodes, UO_orig_edges = tri_graph['UO_nodes'], tri_graph['UO_graph']
        OI_nodes, OI_orig_edges = tri_graph['OI_nodes'], tri_graph['OI_graph']
        IA_nodes, IA_orig_edges = tri_graph['IA_nodes'], tri_graph['IA_graph']

        uo_node2re_id = {}
        uo_re_id2node = {}
        for i, node in enumerate(UO_nodes):
            uo_node2re_id[node] = i
            uo_re_id2node[i] = node

        uo_mapped_src = []
        uo_mapped_dst = []
        for e in UO_orig_edges:
            uo_mapped_src.append(uo_node2re_id[e[0]])
            uo_mapped_dst.append(uo_node2re_id[e[1]])

        oi_node2re_id = {}
        oi_re_id2node = {}
        for i, node in enumerate(OI_nodes):
            oi_node2re_id[node] = i
            oi_re_id2node[i] = node

        oi_mapped_src = []
        oi_mapped_dst = []
        for e in OI_orig_edges:
            oi_mapped_src.append(oi_node2re_id[e[0]])
            oi_mapped_dst.append(oi_node2re_id[e[1]])

        ia_node2re_id = {}
        ia_re_id2node = {}
        for i, node in enumerate(IA_nodes):
            ia_node2re_id[node] = i
            ia_re_id2node[i] = node

        ia_mapped_src = []
        ia_mapped_dst = []
        for e in IA_orig_edges:
            ia_mapped_src.append(ia_node2re_id[e[0]])
            ia_mapped_dst.append(ia_node2re_id[e[1]])

        uo_dgl_graph = dgl.graph((torch.tensor(uo_mapped_src + uo_mapped_dst), torch.tensor(uo_mapped_dst + uo_mapped_src)))
        oi_dgl_graph = dgl.graph((torch.tensor(oi_mapped_src + oi_mapped_dst), torch.tensor(oi_mapped_dst + oi_mapped_src)))
        ia_dgl_graph = dgl.graph((torch.tensor(ia_mapped_src + ia_mapped_dst), torch.tensor(ia_mapped_dst + ia_mapped_src)))

        UO = [uo_dgl_graph, uo_mapped_src, uo_mapped_dst, uo_node2re_id, uo_re_id2node, UO_nodes]
        OI = [oi_dgl_graph, oi_mapped_src, oi_mapped_dst, oi_node2re_id, oi_re_id2node, OI_nodes]
        IA = [ia_dgl_graph, ia_mapped_src, ia_mapped_dst, ia_node2re_id, ia_re_id2node, IA_nodes]
        return UO, OI, IA