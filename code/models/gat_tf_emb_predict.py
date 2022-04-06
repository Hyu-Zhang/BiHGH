import json, sys

import dgl
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import Module
import os
from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from models.modeling_utils import init_weights, initialize_vision_backbone
from general_util.metrics import ROC, NDCG, MRR_NDCG_score
logger = get_child_logger('GATTransformerVocab')
from general_util.mrr import get_mrr
import shutil
from torchvision.models import resnet18


def sign(x):
    """Return hash code of x."""

    return x.detach().sign()

class GATTransformer(Module, LogMixin):
    def __init__(self,
                 user_embedding: str,
                 outfit_embedding: str,
                 user_vocab: str,
                 outfit_vocab: str,
                 freeze_user_emb: bool = False,
                 freeze_outfit_emb: bool = False,
                 vision_model: str = 'resnet18',
                 text_hidden_size: int = 768,
                 img_hidden_size: int = 2048,
                 hidden_size: int = 768,
                 regulation_rate: float = 0.1,
                 hash_hidden_size : int = 256,
                 margin: float = 0.1,
                 tanh_k: float = 0.5,
                 gnn: Module = None):
        super().__init__()
        self.hidden_size = hidden_size
        # User embedding
        user_vocab = json.load(open(user_vocab, 'r'))
        user_embedding = torch.load(user_embedding, map_location='cpu')
        self.user_embedding_layer = nn.Embedding(len(user_vocab),
                                                 text_hidden_size + img_hidden_size).from_pretrained(user_embedding, freeze=freeze_user_emb)
        self.user_proj = nn.Linear(text_hidden_size + img_hidden_size, hidden_size)

        # Outfit embedding
        outfit_vocab = json.load(open(outfit_vocab, 'r'))
        outfit_embedding = torch.load(outfit_embedding, map_location='cpu')
        self.outfit_embedding_layer = nn.Embedding(len(outfit_vocab),
                                                 text_hidden_size + img_hidden_size).from_pretrained(outfit_embedding, freeze=freeze_outfit_emb)

        self.outfit_proj = nn.Linear(text_hidden_size + img_hidden_size, hidden_size)

        # Item image encoding
        self.resnet = initialize_vision_backbone(vision_model)
        # Item text embedding

        self.item_proj = nn.Linear(hidden_size * 2, hidden_size)
        # Attribute text embedding
        self.attr_proj = nn.Linear(text_hidden_size, hidden_size)

        self.gat = gnn
        self.txt_mlp = nn.Linear(text_hidden_size, hidden_size)
        self.hash_hidden = hash_hidden_size
        self.hash_mlp = nn.Linear(hidden_size, self.hash_hidden)

        self.scale = tanh_k
        self.margin = margin
        self.regulation_rate = regulation_rate
        self.gat.apply(init_weights)
        self.init_metric("auc", "mrr", "ndcg")

    def forward(self,
                UO_graph,
                OI_graph,
                IA_graph,
                UO_input_emb_index,
                OI_input_emb_index,
                IA_input_emb_index,
                user_index: Tensor,
                pos_outfit_index: Tensor,
                neg_outfit_index: Tensor,
                pos_mask: Tensor,
                neg_mask: Tensor,
                item_text: Tensor,
                attr_text: Tensor,
                item_image_ori: Tensor,
                outfit_emb_index: Tensor,
                user_emb_index: Tensor,
                batch_s_ii: Tensor,
                batch_s_oo: Tensor,
                batch_s_uu: Tensor,
                all_names: Tensor,
                epoch : int
                ):

        num_images = item_image_ori.size(0)
        image_emb = self.resnet(item_image_ori)
        image_emb = image_emb.reshape(num_images, -1) # N, d
        text_emb = self.txt_mlp(item_text) # N, d

        # [num_layer, seq_len, h] -> [seq_len, num_layer * h]
        item_emb = self.item_proj(torch.cat([image_emb, text_emb], dim=-1))

        attr_emb = self.attr_proj(attr_text[:, 0, :])
        outfit_emb = self.outfit_proj(self.outfit_embedding_layer(outfit_emb_index))

        user_emb = self.user_proj(self.user_embedding_layer(user_emb_index))
        node_emb = torch.cat([user_emb, outfit_emb, item_emb, attr_emb], dim=0)
        # [N, h] N: node num

        node_feat = None
        tri_graph = [UO_graph, OI_graph, IA_graph]
        tri_input_emb_index = [UO_input_emb_index, OI_input_emb_index, IA_input_emb_index]
        ex_tri_input_emb_index = [
            t.unsqueeze(-1).expand(-1, node_emb.size(-1)) for t in tri_input_emb_index
        ]
        iter_ids = [2, 1, 0, 1, 2]
        node_emb = node_emb.to(dtype=torch.float)
        for iter_id in iter_ids:
            # IA OI UO OI IA
            node_feat = torch.index_select(node_emb, dim=0, index=tri_input_emb_index[iter_id])
            # print(node_feat.size(0))
            node_feat = node_feat.to(dtype=torch.float)
            with autocast(enabled=False):
                node_feat = self.gat(tri_graph[iter_id], node_feat)
            node_emb = torch.scatter(node_emb, dim=0, index=ex_tri_input_emb_index[iter_id], src=node_feat)

        node_emb = self.hash_mlp(node_emb)
        node_emb = torch.tanh(torch.mul(node_emb, self.scale))

        u_h = torch.index_select(node_emb, 0, index=user_index) # [batch, h]
        batch_size = u_h.size()[0]
        node_emb = node_emb.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size*node_emb.size(0), -1)
        p_o_h = torch.index_select(node_emb, 0, index=pos_outfit_index.reshape(batch_size*pos_outfit_index.size(-1))) # [batch*n1, h]
        n_o_h = torch.index_select(node_emb, 0, index=neg_outfit_index.reshape(batch_size*neg_outfit_index.size(-1))) # [batch*n2, h]

        temp_pos_u = u_h.unsqueeze(1).expand(-1, pos_outfit_index.size(-1), -1).reshape(batch_size*pos_outfit_index.size(-1), -1)
        pos_score = torch.mean(p_o_h * temp_pos_u, dim=-1).reshape(batch_size, pos_outfit_index.size(-1))
        temp_neg_u = u_h.unsqueeze(1).expand(-1, neg_outfit_index.size(-1), -1).reshape(batch_size*neg_outfit_index.size(-1), -1)
        neg_score = torch.mean(n_o_h * temp_neg_u, dim=-1).reshape(batch_size, neg_outfit_index.size(-1))

        b_u_h, b_p_o_h, b_n_o_h = sign(u_h), sign(p_o_h), sign(n_o_h)

        b_temp_pos_u = b_u_h.unsqueeze(1).expand(-1, pos_outfit_index.size(-1), -1).reshape(batch_size*pos_outfit_index.size(-1), -1)
        b_pos_score = torch.mean(b_p_o_h * b_temp_pos_u, dim=-1).reshape(batch_size, pos_outfit_index.size(-1))
        b_temp_neg_u = b_u_h.unsqueeze(1).expand(-1, neg_outfit_index.size(-1), -1).reshape(batch_size*neg_outfit_index.size(-1), -1)
        b_neg_score = torch.mean(b_n_o_h * b_temp_neg_u, dim=-1).reshape(batch_size, neg_outfit_index.size(-1))

        all_ndcg = 0
        all_auc = 0
        all_mrr = 0
        all_my_mrr = 0
        all_mean_ndcg = 0

        all_b_ndcg = 0
        all_b_auc = 0
        all_b_mrr = 0
        all_b_my_mrr = 0
        all_b_mean_ndcg = 0

        for i in range(batch_size):

            pos = pos_score[i].tolist()[:pos_mask[i].item()]
            neg = neg_score[i].tolist()[:neg_mask[i].item()]

            score_list = []
            for j, p in enumerate(pos):
                score_list.append([p, neg[j*10:j*10+9]])# 0-9 10-19
            my_mrr = get_mrr(score_list)
            all_my_mrr += my_mrr

            ndcg, mrr = MRR_NDCG_score([pos], [neg])
            mean_ndcg = NDCG([pos], [neg])
            aucs, mean_auc = ROC([pos], [neg])

            b_pos = b_pos_score[i].tolist()[:pos_mask[i].item()]
            b_neg = b_neg_score[i].tolist()[:neg_mask[i].item()]

            b_score_list = []
            for j, p in enumerate(b_pos):
                b_score_list.append([p, b_neg[j*10:j*10+9]])# 0-9 10-19
            b_my_mrr = get_mrr(b_score_list)
            all_b_my_mrr += b_my_mrr

            b_ndcg, b_mrr = MRR_NDCG_score([b_pos], [b_neg])
            b_mean_ndcg = NDCG([b_pos], [b_neg])
            b_aucs, b_mean_auc = ROC([b_pos], [b_neg])

            all_ndcg += ndcg
            all_mean_ndcg += mean_ndcg.mean()
            all_auc += mean_auc
            all_mrr += mrr

            all_b_ndcg += b_ndcg
            all_b_mean_ndcg += b_mean_ndcg.mean()
            all_b_auc += b_mean_auc
            all_b_mrr += b_mrr

        all_ndcg /= batch_size
        all_auc /= batch_size
        all_mrr /= batch_size
        all_my_mrr /= batch_size
        all_mean_ndcg /= batch_size

        all_b_ndcg /= batch_size
        all_b_auc /= batch_size
        all_b_mrr /= batch_size
        all_b_my_mrr /= batch_size
        all_b_mean_ndcg /= batch_size

        return {
            "ndcg": all_ndcg,
            "auc": all_auc,
            "mrr": all_mrr,
            "mymrr": all_my_mrr,
            "meanndcg": all_mean_ndcg,
            "b_ndcg": all_b_ndcg,
            "b_auc": all_b_auc,
            "b_mrr": all_b_mrr,
            "b_mymrr": all_b_my_mrr,
            "b_meanndcg": all_b_mean_ndcg
        }
