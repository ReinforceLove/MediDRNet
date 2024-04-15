import numpy as np
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.functional import normalize
from PIL import Image
import torch
import pandas as pd
import numpy as np


def data_info(path):
    num = np.zeros(5)
    list_path = path
    img_list = []
    with open(list_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            img_list.append(line)
    weights = np.zeros(len(img_list))
    for i in img_list:
        t = int(i.split('_')[-1])
        num[t] += 1
        # print(t)
    index_w = sum(num) / num
    # print(index_w)
    for id, i in enumerate(img_list):
        t = int(i.split('_')[-1])
        weights[id] = index_w[t]

    return weights


class DRD_dataset(Dataset):
    def __init__(self, base_dir, list_dir, purpose):
        self.datadir = base_dir
        self.purpose = purpose
        self.img_list = self._read_list(list_dir)
        self.contrast_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((-90, 90), expand=False, )], p=0.5),

            # transforms.RandomApply([
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),

            transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5028999, 0.5019743, 0.501623],
                                             std=[0.05806068, 0.062229224, 0.04611738]),
        ])
        self.train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((-90, 90), expand=False, )], p=0.5),

            transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5028999, 0.5019743, 0.501623],
                                             std=[0.05806068, 0.062229224, 0.04611738]),
        ])
        self.test_transform = transforms.Compose([transforms.ToTensor(),  ])

    def _read_list(self, list_path):
        list_data = []
        with open(list_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                list_data.append(line)
        return list_data

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idex):
        item = self.img_list[idex]
        # print(item)
        img = Image.open(self.datadir + '/' + item + '.jpeg')
        label = int(item.split('_')[-1])
        sample = {}
        if self.purpose == 'contrast':
            data = self.contrast_transform(img)
            sample = {'img': data, 'label': label}
            return sample
        elif self.purpose == 'classification':
            img = self.train_transform(img)
            sample = {'img': img, 'label': label}
            return sample
        else:
            img = self.test_transform(img)
            sample = {'img': img, 'label': label}
            return sample


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    weights = data_info('../data/train1.list')

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True)
    dataset = DRD_dataset('../data/DRD/train', '../data/train1.list', 'contrast')
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=32, num_workers=0)

    dataset1 = DRD_dataset('../data/DRD610/train', '../data/train610.list', 'test')#[0.5028953, 0.50196934, 0.50161314] [0.057567358, 0.064187214, 0.04282246]
    dataloader1 = torch.utils.data.DataLoader(dataset1, shuffle=False, batch_size=1, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X in dataloader1:
        for d in range(3):
            mean[d] += X['img'][:, d, :, :].mean()
            std[d] += X['img'][:, d, :, :].std()
    mean.div_(len(dataset1))
    std.div_(len(dataset1))
    print(list(mean.numpy()), list(std.numpy()))
    # for i in range(10):
    # print('------------------------------------------------------', i)
    # for id, i in enumerate(dataloader1):
    # img1 = i['img1'][0, :]
    # img2 = i['img2'][0, :]
    # print(id, len(dataloader1))
    # print(i['label'])
    # print(j['label'])
    # t = torchvision.transforms.ToPILImage()
    # data1 = t(img1)
    # data2 = t(img2)
    # data1.show()
    # data2.show()
    # print(img1.shape, img2.shape)
    # cv2.imshow('aaa', img1)
    # key = 0  # 创建并初始化key变量为0
    # while key != 27:
    #     key = cv2.waitKey()
    #     cv2.destroyAllWindows()

    # print(i['img'].cuda().shape)
