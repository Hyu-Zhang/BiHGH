
from torchvision.models import resnet18, resnet34
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.data.dataset import T_co
import torch, json
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda:1")
batch_size = 1024

resnet = nn.Sequential(*list(resnet34(pretrained=True).children())[:-1]).to(device=device)
resnet.eval()

imgname2path = 'img2path_gutai.json'
imgname2path = json.load(open(imgname2path))
def extractor(img_path):
    transform = transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))]
    )

    transform_one = transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))]
    )
    try:
        img = Image.open(img_path)
        try:
            img = transform(img)
            return img
        except Exception as e:

            img = transform_one(img)
            return img
    except Exception as e:
        img = torch.zeros(3, 150, 150)
        # print('error============',img_path)
        return img

def load_item_embedding(item):
    # print(item)
    if item not in imgname2path.keys():
        image = torch.zeros(3, 150, 150)
        # print("not find")
    else:
        image = imgname2path[item]
        # image = os.path.join(img_dir, f"{item}_m.jpg")
        image = extractor(image)
    # print("============1===================")
    return image

class myDataset(Dataset):
    def __init__(self):
        node_vocab = torch.load('vocab.pt')
        self.items = list(node_vocab['i'])
        # print(self.items[0])

    def __getitem__(self, index) -> T_co:
        # print(load_item_embedding(self.items[index]))
        return self.items[index], load_item_embedding(self.items[index])
        # # return self.items[index]
        # print("=============2==================")
        # return load_item_embedding(self.items[index])

    def __len__(self):
        return len(self.items)

if __name__ == '__main__':
    vocab = json.load(open('item_vocab.json'))
    id2vocab = {}
    for key,value in vocab.items():
        id2vocab[value] = key
    mydataset = myDataset()
    loader = DataLoader(dataset=mydataset, batch_size=batch_size,  num_workers=8, shuffle=False)
    name_list = []
    tensor_list = []
    with torch.no_grad():
        for i, item in enumerate(tqdm(loader)):
            name, tensor = item
            batch_tensor = resnet(tensor.to(device)).reshape(tensor.size(0), -1).cpu()
            name_list.extend(name)
            tensor_list.extend(batch_tensor.tolist())

    name_dict = {}
    for i, name in enumerate(name_list):
        name_dict[name] = i
    img_tensor = []
    for i in range(len(vocab)):
        img_tensor.append(torch.tensor(tensor_list[name_dict[id2vocab[i]]]))

    img_tensor = torch.stack(img_tensor, dim=0)
    torch.save(img_tensor, "item_img.pt")

    print("Done.")
