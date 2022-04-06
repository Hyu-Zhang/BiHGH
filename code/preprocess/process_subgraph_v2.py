import torch
import argparse
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
import os
from typing import Dict, Set
import json
from tqdm import tqdm

_edges: Dict[str, Dict[str, Set]]


def _initializer(e: Dict[str, Dict[str, Dict[str, Set]]]):
    global _edges
    _edges = e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/', type=str)
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--output_dir', default='/', type=str)
    args = parser.parse_args()

    ia = json.load(open(os.path.join(args.data_dir, 'IA_550.json'), 'r'))
    ai = json.load(open(os.path.join(args.data_dir, 'AI_550.json'), 'r'))
    oi = json.load(open(os.path.join(args.data_dir, 'OI_550.json'), 'r'))
    io = json.load(open(os.path.join(args.data_dir, 'IO_550.json'), 'r'))
    ou = json.load(open(os.path.join(args.data_dir, 'OU_build.json'), 'r'))
    uo = json.load(open(os.path.join(args.data_dir, 'UO_build.json'), 'r'))

    edges = defaultdict(dict)

    edges['i']['a'] = ia
    edges['a']['i'] = ai
    edges['o']['u'] = ou
    edges['u']['o'] = uo
    edges['o']['i'] = oi
    edges['i']['o'] = io

    print(f"Length:\nIA: {len(ia)}\tAI: {len(ai)}\tOU: {len(ou)}\tUO: {len(uo)}\tIO:{len(io)}\tOI:{len(oi)}")

    user_ids = set()
    outfit_ids = set()
    item_ids = set()
    attr_ids = set()

    for i_id, a_ls in ia.items():
        item_ids.add(i_id)
        attr_ids.update(a_ls)
    for a_id, i_ls in ai.items():
        attr_ids.add(a_id)
        item_ids.update(i_ls)

    for i_id, o_ls in io.items():
        item_ids.add(i_id)
        outfit_ids.update(o_ls)
    for o_id, i_ls in oi.items():
        outfit_ids.add(o_id)
        item_ids.update(i_ls)

    for o_id, u_ls in ou.items():
        outfit_ids.add(o_id)
        user_ids.update(u_ls)
    for u_id, o_ls in uo.items():
        user_ids.add(u_id)
        outfit_ids.update(o_ls)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_nodes = {
        'u': list(user_ids),
        'o': list(outfit_ids),
        'i': list(item_ids),
        'a': list(attr_ids)
    }
    print(f"Node num:\nAttribute: {len(attr_ids)}\tUser: {len(user_ids)}\tOutfit: {len(outfit_ids)}\tItem: {len(item_ids)}")