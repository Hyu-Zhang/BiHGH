import json
import os

import torch
from torch import nn
from torchvision.models import resnet18
from tqdm import tqdm
import argparse

batch_size = 256
text = 'item_text_emb_weight.cls.pt'
text = torch.load(text)
img = 'item_img.pt'
img = torch.load(img)
item_vocab = 'item_vocab.json'
vocab = json.load(open(item_vocab))

def initializer(_oi_edges):
    global oi_edges
    oi_edges = _oi_edges


def load_item_embedding(item):
    if item in vocab.keys():
        id = vocab[item]
        item_text = text[id]
        item_img = img[id]
    else:
        print("error")
        item_text = torch.zeros(768)
        item_img = torch.zeros(512)

    return item_text, item_img


def get_outfit_embedding(_outfit):
    with torch.no_grad():
        text_emb_ls = []
        img_ls = []
        for item in oi_edges[_outfit]:
            text, img = load_item_embedding(item)
            if text is None and img is None:
                continue
            text_emb_ls.append(text)
            img_ls.append(img)
        text_emb = torch.stack(text_emb_ls, dim=0)
        img_emb = torch.stack(img_ls, dim=0)
        assert img_emb.size(0) == text_emb.size(0)
    # return torch.stack(emb_ls, dim=0).mean(dim=0)
    return torch.cat([text_emb, img_emb], dim=-1).mean(dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_vocab', default='vocab.pt', type=str)
    parser.add_argument('--oi_edge_file', default='OI_550.json', type=str)
    parser.add_argument('--output_dir', default='/', type=str)

    args = parser.parse_args()

    node_vocab = torch.load(args.node_vocab)

    oi = json.load(open(args.oi_edge_file, 'r'))

    # Process user embedding
    initializer(oi)
    outfit_embedding = {}
    for o in tqdm(node_vocab['o'], total=len(node_vocab['o'])):
        outfit_embedding[o] = get_outfit_embedding(o)

    torch.save(outfit_embedding, os.path.join(args.output_dir, 'outfit_embedding.pt'))

    # Process user vocabulary
    outfit_emb_weight = []
    outfit_vocab = {}
    for i, (outfit, outfit_emb) in enumerate(outfit_embedding.items()):
        # print(outfit_emb)
        outfit_emb_weight.append(outfit_emb)
        outfit_vocab[outfit] = i
    outfit_emb_weight = torch.stack(outfit_emb_weight, dim=0)

    torch.save(outfit_emb_weight, os.path.join(args.output_dir, 'outfit_emb_weight.pt'))
    json.dump(outfit_vocab, open(os.path.join(args.output_dir, 'outfit_vocab.json'), 'w'))

    print("Done.")
