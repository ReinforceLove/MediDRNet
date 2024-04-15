import cv2
import csv
import glob
import numpy as np
import pandas as pd


def scaleRadius(img, scale):
    x = img[img.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2 + 1e-6
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


scale = 300
target_size = (610, 610)
list = glob.glob("train/*.jpeg")
label_file_path = 'trainLabels.csv'
target_path = 'DRD610/train/'
df = pd.read_csv(label_file_path)
print(df.head())
df.set_index('image', inplace=True)

sum = len(list)
for i, f in enumerate(list):  # + glob.glob("test/*.jpeg"):
    try:
        t = f[6:-5]
        a = cv2.imread(f, 1)

        # 缩放图像以达到指定的半径
        a = scaleRadius(a, scale)
        # 减去局部均值颜色
        a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
        # 移除外部10%
        b = np.zeros(a.shape)
        cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
        a = a * b + 128 * (1 - b)
        a = cv2.resize(a, target_size)
        level_of_image = df.loc[t, 'level']
        name = f'{t}_{level_of_image}.jpeg'
        print(f'[{i}/{sum}]', f, t, level_of_image, name,a.shape)
        cv2.imwrite(target_path + name, a)
    except:
        print("处理出错:", f)
    # cv2.imwrite(target_path + name, a)

