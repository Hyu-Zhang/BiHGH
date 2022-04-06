import argparse
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.data.dataset import T_co
import torch, os, json
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
# from multiprocess import Pool
from tqdm import tqdm
import shutil
from transformers import AutoTokenizer, AutoModelForMaskedLM
# device = torch.device("cuda:3")
batch_size = 1024

tokenizer = AutoTokenizer.from_pretrained("bert-base-Japanese-char")

model = AutoModelForMaskedLM.from_pretrained("bert-base-Japanese-char").cuda()
model.eval()

text_exp_dir = 'item_txt_expression'

def JP_Initialization(text):
    with torch.no_grad():
        # text = text[:500]
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

        attention_mask = inputs['attention_mask']
        for key in inputs:
            inputs[key] = inputs[key].cuda()

        outputs = model(**inputs, output_hidden_states=True)
        hidden_state = torch.stack([temp.mean(1) for temp in outputs[1]], dim=0)  # (13,targets_num,768)

        hidden_state = hidden_state.mean(0)  # (targets_num,768)
        return hidden_state


def load_item_embedding(item):
    path = os.path.join(text_exp_dir, "{}.pt".format(item))
    if os.path.exists(path):
        text = torch.load(path)
        if text:
            feat = JP_Initialization(text)
        else:
            feat = torch.zeros(512, 768)
    else:
        feat = torch.zeros(512, 768)

    return feat

class myDataset(Dataset):
    def __init__(self):
        node_vocab = torch.load('vocab.pt')
        self.items = list(node_vocab['i'])
        # print(self.items[0])

    def __getitem__(self, index) -> T_co:
        # print(load_item_embedding(self.items[index]))
        return self.items[index], load_item_embedding(self.items[index])

    def __len__(self):
        return len(self.items)

if __name__ == '__main__':

    vocab = json.load(open('item_vocab.json'))
    id2vocab = {}
    for key,value in vocab.items():
        id2vocab[value] = key
    mydataset = myDataset()
    loader = DataLoader(dataset=mydataset, batch_size=batch_size,  num_workers=20, shuffle=False)

    name_list = []
    tensor_list = []
    with torch.no_grad():
        for i, item in enumerate(tqdm(loader)):
            name, tensor = item
            name_list.extend(name)
            tensor_list.extend(tensor.tolist())

    name_dict = {}
    for i, name in enumerate(name_list):
        name_dict[name] = i
    txt_tensor = []
    for i in range(len(vocab)):
        txt_tensor.append(torch.tensor(tensor_list[name_dict[id2vocab[i]]]))

    txt_tensor = torch.stack(txt_tensor, dim=0)
    torch.save(txt_tensor, "item_text_emb_weight.cls.pt")

    print("Done.")
