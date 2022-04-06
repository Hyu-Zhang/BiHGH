import json
import os
import torch
from torch import nn
from torchvision.models import resnet18
from tqdm import tqdm
import argparse


outfit2id = json.load(open('outfit_vocab.json'))
outfit_emb = torch.load('outfit_emb_weight.pt')

def initializer(_uo_edges):
    global uo_edges
    uo_edges = _uo_edges


def load_outfit_embedding(outfit):

    if outfit in outfit2id.keys():
        id =  outfit2id[outfit]
        embed = outfit_emb[id]
    else:
        embed = torch.zeros(1280)

    return embed


def get_user_embedding(_user):

    with torch.no_grad():
        user_embed = []
        for outfit in uo_edges[_user]:
            embed = load_outfit_embedding(outfit)
            user_embed.append(embed)

        user_embed = torch.stack(user_embed, dim=0)

    return user_embed.mean(dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_vocab', default='vocab.pt', type=str)
    parser.add_argument('--uo_edge_file', default='UO_build.json', type=str)
    parser.add_argument('--output_dir', default='/', type=str)

    args = parser.parse_args()

    node_vocab = torch.load(args.node_vocab)
    uo = json.load(open(args.uo_edge_file, 'r'))

    # Process user embedding
    initializer(uo)

    user_embedding = {}
    for u in tqdm(node_vocab['u'], total=len(node_vocab['u'])):
        user_embedding[u] = get_user_embedding(u)
    torch.save(user_embedding, os.path.join(args.output_dir, 'user_embedding.pt'))

    user_emb_weight = []
    user_vocab = {}
    for i, (user, user_emb) in enumerate(user_embedding.items()):
        user_emb_weight.append(user_emb)
        user_vocab[user] = i
    user_emb_weight = torch.stack(user_emb_weight, dim=0)

    torch.save(user_emb_weight, os.path.join(args.output_dir, 'user_emb_weight.pt'))
    json.dump(user_vocab, open(os.path.join(args.output_dir, 'user_vocab.json'), 'w'))

    print("Done.")
