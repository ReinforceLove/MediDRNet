import pandas as pd
import numpy as np


def read_list(list_path):
    list_data = []
    with open(list_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            list_data.append(line)
    return list_data


def data_info():
    num = np.zeros(5)
    list_path = '../data/train.list'
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
    index_w = sum(num)/num
    # print(index_w)
    for id,i in enumerate(img_list):
        t = int(i.split('_')[-1])
        weights[id] = index_w[t]
    # print(weights)
    # label_file_path = 'trainLabels.csv'
    # df = pd.read_csv(label_file_path)

    # for i in range(5):
    #     num[i] = df.loc[df['level'] == i].shape[0]
    # weights = sum(num)/num
    # return weights, sum(num)


data_info()
