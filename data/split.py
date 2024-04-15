import glob
import random


def save(data, path):
    file = open(path, 'w')
    for line in data:
        file.write(line + '\n')
    file.close()


# 312
list = glob.glob("DRD610/train/*.jpeg")
random.shuffle(list)
num_train = []
num_test = []
sum2 = [0] * 5
n = [2132, 2132, 2132, 561, 397]

lens = len(list)
for i, f in enumerate(list):
    a = f[13:-5]
    t = int(a.split('_')[-1])
    print(a, t)
    if sum2[t] < 312:
        sum2[t] += 1
        num_test.append(a)
    # elif sum2[t] < n[t]:
    else:
        sum2[t] += 1
        num_train.append(a)
print(sum2)

save(num_train, 'train610.list')
save(num_test, 'test610.list')
# save(num_val,'val.list')
