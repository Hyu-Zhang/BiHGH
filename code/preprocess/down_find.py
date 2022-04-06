
from functools import partial
import argparse
from collections import defaultdict
from typing import Dict, Set
import json, os, torch
import sys
from collections import defaultdict
from tqdm import tqdm
from multiprocess import Pool

def _initializer(uo, oi, ia):
    global uo_graph, oi_graph, ia_graph
    uo_graph = uo
    oi_graph = oi
    ia_graph = ia

def pair(data):
    # train val
    # user, o_pos, o_neg = data[0], data[1], data[2] # user
    #
    # all_outfit = []
    # all_outfit.append(o_pos)
    # all_outfit.append(o_neg)
    # test
    user, o_pos, o_neg = data['user'], data['pos'], data['neg']
    all_outfit = []
    all_outfit.extend(o_pos)
    all_outfit.extend(o_neg)


    # all_user = []
    # all_user.append(user)
    uo_edge = []
    if user in uo_graph:
        for outfit in uo_graph[user]:
            uo_edge.append((user, outfit))
            all_outfit.append(outfit)

    uo_edge = list(set(uo_edge))
    uo_nodes = []
    for item in uo_edge:
        uo_nodes.append(item[0])
        uo_nodes.append(item[1])
    uo_nodes = list(set(uo_nodes))

    oi_edge = []
    all_item = []
    for outfit in all_outfit:
        if outfit in oi_graph:
            for item in oi_graph[outfit]:
                oi_edge.append((outfit, item))
                all_item.append(item)
    oi_edge = list(set(oi_edge))
    oi_nodes = []
    for item in oi_edge:
        oi_nodes.append(item[0])
        oi_nodes.append(item[1])
    oi_nodes = list(set(oi_nodes))

    ia_edge = []
    for item in all_item:
        if item in ia_graph:
            for attr in ia_graph[item]:
                ia_edge.append((item, attr))
    ia_edge = list(set(ia_edge))
    ia_nodes = []
    for item in ia_edge:
        ia_nodes.append(item[0])
        ia_nodes.append(item[1])
    ia_nodes = list(set(ia_nodes))

    dict = {}
    dict['UO_nodes'] = uo_nodes
    dict['UO_graph'] = uo_edge
    dict['OI_nodes'] = oi_nodes
    dict['OI_graph'] = oi_edge
    dict['IA_nodes'] = ia_nodes
    dict['IA_graph'] = ia_edge
    # O 》 U 》 O
    # torch.save(dict, 'train_down_find/{}.pt'.format("_".join(data)))
    torch.save(dict, 'test_down_find/{}.pt'.format(user))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/', type=str)
    parser.add_argument('--num_workers', type=int, default=24)
    args = parser.parse_args()

    ia = json.load(open(os.path.join(args.data_dir, 'IA_550.json'), 'r'))
    # ai = json.load(open(os.path.join(args.data_dir, 'AI_550.json'), 'r'))
    oi = json.load(open(os.path.join(args.data_dir, 'OI_550.json'), 'r'))
    # io = json.load(open(os.path.join(args.data_dir, 'IO_550.json'), 'r'))
    # ou = json.load(open(os.path.join(args.data_dir, 'OU_build.json'), 'r'))
    uo = json.load(open(os.path.join(args.data_dir, 'UO_build.json'), 'r'))

    path = '/test.json'
    data = json.load(open(path))
    _initializer(uo, oi, ia)
    with Pool(32) as p:
        _results = list(tqdm(
            p.imap(pair, data),
            total=len(data)
        ))

