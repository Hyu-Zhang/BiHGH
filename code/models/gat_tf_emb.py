import json, sys
import dgl
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import Module
# from torchvision import models
from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from models.modeling_utils import init_weights, initialize_vision_backbone
from general_util.metrics import ROC, NDCG, MRR_NDCG_score
logger = get_child_logger('GATTransformerVocab')
from general_util.mrr import get_mrr
from torchvision.models import resnet18, resnet34


def contrastive_loss(margin, im, s):
    size, dim = im.shape
    scores = im.matmul(s.t()) / dim
    diag = scores.diag()
    zeros = torch.zeros_like(scores)
    # shape #item x #item
    # sum along the row to get the VSE loss from each image
    cost_im = torch.max(zeros, margin - diag.view(-1, 1) + scores)
    # sum along the column to get the VSE loss from each sentence
    cost_s = torch.max(zeros, margin - diag.view(1, -1) + scores)
    # to fit parallel, only compute the average for each item
    vse_loss = cost_im.sum(dim=1) + cost_s.sum(dim=0) - 2 * margin
    # for data parallel, reshape to (size, 1)
    return vse_loss / (size - 1)

def sign(x):
    """Return hash code of x."""

    return x.detach().sign()

def cal_similarity_loss(batch_s, emb):

    fi = torch.cosine_similarity(emb.unsqueeze(0), emb.unsqueeze(1), dim=-1) # len IXI
    similarity_loss = -torch.sum(batch_s * fi - torch.log(torch.ones_like(fi) + torch.exp(fi)))
    similarity_loss /= len(emb)
    return similarity_loss

class GATTransformer(Module, LogMixin):
    def __init__(self,
                 user_embedding: str,
                 outfit_embedding: str,
                 user_vocab: str,
                 outfit_vocab: str,
                 freeze_user_emb: bool = False,
                 freeze_outfit_emb: bool = False,
                 vision_model: str = 'resnet34',
                 text_hidden_size: int = 768,
                 img_hidden_size: int = 2048,
                 hidden_size: int = 768,
                 regulation_rate: float = 0.1,
                 hash_hidden_size : int = 256,
                 margin: float = 0.1,
                 tanh_k: float = 0.5,
                 gnn: Module = None):
        super().__init__()
        self.regulation_rate = regulation_rate
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
        self.init_metric("loss", "auc", "mrr", "ndcg")

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
                item_text: Tensor,
                attr_text: Tensor,
                item_image_ori: Tensor,
                outfit_emb_index: Tensor,
                user_emb_index: Tensor,
                batch_s_oo: Tensor,
                batch_s_uu: Tensor,
                epoch: int
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

        # Map 512 to lower dimensional space
        node_emb = self.hash_mlp(node_emb)
        node_emb = torch.tanh(torch.mul(node_emb, self.scale))

        # Get consecutive representations of user and outfit
        u_h = torch.index_select(node_emb, 0, index=user_index) # [batch, h]
        batch_size = u_h.size()[0]
        p_o_h = torch.index_select(node_emb, 0, index=pos_outfit_index)  # [batch, h]
        n_o_h = torch.index_select(node_emb, 0, index=neg_outfit_index)  # [batch, h]
        # Calculate the score of user and positive and negative cases
        pos_logits = torch.sum(u_h * p_o_h, dim=-1, keepdim=True).reshape(batch_size, 1)
        neg_logits = torch.sum(u_h * n_o_h, dim=-1, keepdim=True).reshape(batch_size, 1)

        # bpr loss
        x = pos_logits - neg_logits
        bprloss = -torch.mean(torch.log(torch.sigmoid(x)))
        # vse loss
        vse_loss = contrastive_loss(self.margin, image_emb, text_emb).mean()
        # similarity loss
        hash_user_emb = node_emb[:len(user_emb)]
        hash_outfit_emb = node_emb[len(user_emb):len(user_emb) + len(outfit_emb)]
        similarity_outfit_loss = cal_similarity_loss(batch_s_oo, hash_outfit_emb)/batch_size
        similarity_user_loss = cal_similarity_loss(batch_s_uu, hash_user_emb)/batch_size
        # total loss
        all_loss = bprloss + vse_loss + self.regulation_rate*similarity_outfit_loss + similarity_user_loss

        # hash code
        b_u_h, b_p_o_h, b_n_o_h = sign(u_h), sign(p_o_h), sign(n_o_h)
        b_pos_logits = torch.sum(b_u_h * b_p_o_h, dim=-1, keepdim=True).reshape(batch_size, 1)
        b_neg_logits = torch.sum(b_u_h * b_n_o_h, dim=-1, keepdim=True).reshape(batch_size, 1)

        logits = torch.stack([neg_logits, pos_logits], dim=-1)
        ndcg, mrr = MRR_NDCG_score(pos_logits.tolist(), neg_logits.tolist())
        mean_ndcg = NDCG(pos_logits.tolist(), neg_logits.tolist())
        aucs, mean_auc = ROC(pos_logits.tolist(), neg_logits.tolist())

        b_ndcg, b_mrr = MRR_NDCG_score(b_pos_logits.tolist(), b_neg_logits.tolist())
        b_mean_ndcg = NDCG(b_pos_logits.tolist(), b_neg_logits.tolist())
        b_aucs, b_mean_auc = ROC(b_pos_logits.tolist(), b_neg_logits.tolist())

        # validation
        # if not self.training:
        #     # pos 1 / neg 0
        #     # true_num = pos_score.size(-1)
        #     mean_ndcg, avg_ndcg, mrr_li = NDCG(pos_logits.tolist(), neg_logits.tolist())
        #     aucs, mean_auc = ROC(pos_logits.tolist(), neg_logits.tolist())
        #     self.eval_metrics.update("loss", val=all_loss.item(), n=batch_size)
        #     self.eval_metrics.update("auc", val=mean_auc, n=batch_size)
        #     self.eval_metrics.update("ndcg", val=mean_ndcg.mean(), n=batch_size)
        #     self.eval_metrics.update("mrr", val=mrr, n=batch_size)

        return {
            "loss": all_loss,
            "bprloss": bprloss,
            "vseloss": vse_loss,
            "similarity_outfit_loss": similarity_outfit_loss,
            "similarity_user_loss": similarity_user_loss,
            "logits": logits,
            "mean ndcg": mean_ndcg.mean(),
            "auc": mean_auc,
            "mrr@10": mrr,
            "ndcg@10": ndcg,
            "binary mean ndcg": b_mean_ndcg.mean(),
            "binary auc": b_mean_auc,
            "binary mrr@10": b_mrr,
            "binary ndcg@10": b_ndcg,
        }
