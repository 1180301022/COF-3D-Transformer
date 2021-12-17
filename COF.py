import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

max_len = 2500  # 单分子最大原子数，截断操作

class COF(Dataset):
    def __init__(self, path="./convert", mode="train"):
        self.file_names = os.listdir(path)  # 训练集+测试集 文件名
        self.mode = mode
        self.path = path
        self.labels = pd.read_csv("./properties.csv")[[" name", " surface area [m^2/g]"]]
        self.labels.set_index(" name", inplace=True)

    def __len__(self):
        if self.mode == "train":
            return 55218  # 90%
        else:
            return 6135  # 10%

    def __getitem__(self, item):
        name = self.file_names[item] if self.mode == "train" else self.file_names[55218 + item]
        atom, pos3d = process_single_item(name, self.path)
        label = self.labels.loc[" " + name[: -4]].values[0]
        return (atom, pos3d), label


class COFWithDistanceMatrix(COF):
    def __init__(self, path="./convert", mode="train"):
        super().__init__(path, mode)

    def __getitem__(self, item):
        name = self.file_names[item] if self.mode == "train" else self.file_names[55218 + item]
        atom, pos3d = process_single_item(name, self.path)
        matrix = EuclideanDistances(pos3d, pos3d)
        l = self.labels.loc[" " + name[: -4]].values[0]
        return (atom, pos3d, matrix), l


class COFWithVirtualNode(COF):
    def __init__(self, path="./convert", mode="train"):
        super(COFWithVirtualNode, self).__init__(path, mode)

    def __getitem__(self, item):
        data, label = super().__getitem__(item)
        atom, pos3d = data
        atom = np.insert(atom, 0, 0)
        pos3d = np.insert(pos3d, 0, [.0, .0, .0], axis=0)
        return (atom, pos3d), label

periodic_dict = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11,
                 "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21,
                 "Ti": 22}

def process_single_item(file_name, path="./convert"):
    # 读取x
    f = open(file=(path + "/" + file_name), mode="r")
    atom = []
    pos3d = []
    f.readline()
    f.readline()  # 跳过前两行
    for i, line in enumerate(f.readlines()):
        if i == 2500:
            break
        line = line.split()
        atom.append(periodic_dict[line[0]])
        pos3d.append([float(line[1]), float(line[2]), float(line[3])])
    f.close()

    # padding
    n = len(atom)
    atom.extend([0 for _ in range(max_len - n)])
    pos3d.extend([[0, 0, 0] for _ in range(max_len - n)])

    atom = np.array(atom).reshape([max_len])
    pos3d = np.array(pos3d).reshape([max_len, 3])

    return atom, pos3d

def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = np.dot(A,BT)
    SqA =  A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return np.array(ED)

# d = COFWithVirtualNode()
# s = d.__getitem__(1)
# b = 1