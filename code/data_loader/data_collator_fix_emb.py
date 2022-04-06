import json

import dgl, os
import torch
from PIL import Image
from general_util.logger import get_child_logger
from data_loader.data_utils import EmbeddingMatrix
from torchvision import transforms
logger = get_child_logger('SubgraphCollatorVocab')

def get_s(src_emb_index, dst_emb_index, src_id2vocab, dst_id2vocab, edge, k):
    SD = torch.zeros((len(src_emb_index), len(dst_emb_index)))
    for i, src in enumerate(src_emb_index.tolist()):
        for j, dst in enumerate(dst_emb_index.tolist()):
            if src_id2vocab[src] in edge.keys():
                if dst_id2vocab[dst] in edge[src_id2vocab[src]]:
                    SD[i][j] = 1
    SS = torch.matmul(SD, SD.t())
    SS_one = torch.ones_like(SS)
    SS_zero = torch.zeros_like(SS)
    batch_ss = torch.where(SS >= k, SS_one, SS_zero)
    return batch_ss

class SubgraphCollatorVocab:
    def __init__(self,
                 user_vocab: str,
                 outfit_vocab: str,
                 attr_vocab: str,
                 item_vocab: str,
                 node_vocab: str,
                 img2path: str,
                 uo_edge: str,
                 oi_edge: str,
                 ia_edge: str,
                 embedding: EmbeddingMatrix):

        self.user_vocab = json.load(open(user_vocab, 'r'))
        self.outfit_vocab = json.load(open(outfit_vocab, 'r'))
        self.attr_vocab = json.load(open(attr_vocab, 'r'))
        self.item_vocab = json.load(open(item_vocab, 'r'))
        self.node_vocab = torch.load(node_vocab)
        self.item2imgpath = json.load(open(img2path))
        self.item2imgpath_kset = set(self.item2imgpath.keys())

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.ia = json.load(open(uo_edge))
        self.oi = json.load(open(oi_edge))
        self.uo = json.load(open(ia_edge))
        self.item_id2vocab = {}
        for item, id in self.item_vocab.items():
            self.item_id2vocab[id] = item
        self.attr_id2vocab = {}
        for attr, id in self.attr_vocab.items():
            self.attr_id2vocab[id] = attr
        self.outfit_id2vocab = {}
        for outfit, id in self.outfit_vocab.items():
            self.outfit_id2vocab[id] = outfit
        self.user_id2vocab = {}
        for user, id in self.user_vocab.items():
            self.user_id2vocab[id] = user

        self.k = 1

        self.node2type = {}
        for k, v_ls in self.node_vocab.items():
            for v in v_ls:
                if v not in self.node2type:
                    self.node2type[v] = k
                else:
                    # assert self.node2type[v] == k, (self.node2type[v], k)  # Check repetition.
                    ...
        assert 'u' in self.node_vocab
        assert 'o' in self.node_vocab
        assert 'i' in self.node_vocab
        assert 'a' in self.node_vocab
        self.embedding = embedding

    def __call__(self, batch):

        all_dgl_graph, all_node2re_id, all_re_id2node, all_nodes, all_triples = zip(*batch)
        # print(all_nodes)
        batch_size = len(all_dgl_graph)
        _nodes = set()
        _node2emb = {}

        max_subgraph_num = len(all_dgl_graph[0])

        for b in range(batch_size):
            _nodes.update(all_nodes[b])

        users = []
        user_emb_index = []
        outfits = []
        outfit_emb_index = []
        items = []
        item_emb_index = []
        attributes = []
        attr_emb_index = []

        for _node in _nodes:
            _node_type = self.node2type[_node]
            if _node_type == 'i':
                items.append(_node)
                item_emb_index.append(self.item_vocab[_node])
            elif _node_type == 'a':
                attributes.append(_node)
                attr_emb_index.append(self.attr_vocab[_node])
            elif _node_type == 'u':
                users.append(_node)
                user_emb_index.append(self.user_vocab[_node])
            elif _node_type == 'o':
                outfits.append(_node)
                outfit_emb_index.append(self.outfit_vocab[_node])
            else:
                raise RuntimeError(f"Unrecognized node type: {_node_type}.")

        item_emb_index = torch.tensor(item_emb_index, dtype=torch.long)
        attr_emb_index = torch.tensor(attr_emb_index, dtype=torch.long)
        outfit_emb_index = torch.tensor(outfit_emb_index, dtype=torch.long)
        user_emb_index = torch.tensor(user_emb_index, dtype=torch.long)

        # S
        batch_s_ii = get_s(item_emb_index, attr_emb_index, self.item_id2vocab, self.attr_id2vocab, self.ia, self.k)
        batch_s_oo = get_s(outfit_emb_index, item_emb_index, self.outfit_id2vocab, self.item_id2vocab, self.oi, self.k)
        batch_s_uu = get_s(user_emb_index, outfit_emb_index, self.user_id2vocab, self.outfit_id2vocab, self.uo, self.k)

        node2emb_index = {}
        for i, _node in enumerate(users + outfits + items + attributes):
            node2emb_index[_node] = i

        img_list = []
        for item in items:
            if item in self.item2imgpath_kset:
                path = self.item2imgpath[item]
                with open(path, "rb") as f:
                    img = Image.open(f).convert("RGB")
                img_list.append(self.transform(img))
            else:
                img_list.append(torch.zeros(3, 150, 150))
        img_list = torch.stack(img_list, dim=0)

        user_index, pos_outfit_index, neg_outfit_index = [], [], []
        for b in range(batch_size):
            triple = all_triples[b]
            user, o_pos, o_neg = triple[0], triple[1], triple[2]
            user_index.append(node2emb_index[user])
            pos_outfit_index.append(node2emb_index[o_pos])
            neg_outfit_index.append(node2emb_index[o_neg])
        user_index = torch.tensor(user_index)
        pos_outfit_index = torch.tensor(pos_outfit_index)
        neg_outfit_index = torch.tensor(neg_outfit_index)

        tri_graph, tri_input_emb_index = [], []
        for l in range(max_subgraph_num):
            all_graphs, all_input_emb_index = [], []
            for b in range(batch_size):
                all_graphs.append(all_dgl_graph[b][l])
                sg_node_num = len(all_re_id2node[b][l])
                sg_input_emb_index = list(map(lambda x: node2emb_index[all_re_id2node[b][l][x]], range(sg_node_num)))
                all_input_emb_index.extend(sg_input_emb_index)
            graph = dgl.batch(all_graphs)
            input_emb_index = torch.tensor(all_input_emb_index, dtype=torch.long)
            tri_graph.append(graph)
            tri_input_emb_index.append(input_emb_index)

        return {
            "UO_graph": tri_graph[0],
            "OI_graph": tri_graph[1],
            "IA_graph": tri_graph[2],
            "UO_input_emb_index": tri_input_emb_index[0],
            "OI_input_emb_index": tri_input_emb_index[1],
            "IA_input_emb_index": tri_input_emb_index[2],
            "user_index": user_index,
            "pos_outfit_index": pos_outfit_index,
            "neg_outfit_index": neg_outfit_index,
            "item_text": torch.index_select(self.embedding.item_text, dim=0, index=item_emb_index),
            "attr_text": torch.index_select(self.embedding.attr_text, dim=0, index=attr_emb_index),
            "item_image_ori": img_list,
            "outfit_emb_index": outfit_emb_index,
            "user_emb_index": user_emb_index,
            "batch_s_ii": batch_s_ii,
            "batch_s_oo": batch_s_oo,
            "batch_s_uu": batch_s_uu
        }
