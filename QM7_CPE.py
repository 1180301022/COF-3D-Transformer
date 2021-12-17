import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat

"""QM7数据集的准备"""


class QM7(Dataset):
    def __init__(self, mode, path='./qm7.mat'):
        file = loadmat(path)
        if mode == "train":
            index_list = [1, 2, 3, 4]
        else:
            index_list = [0]
        P = file['P'][index_list].flatten()
        self.mode = mode
        self.label = file['T'][0, P]  # label 原子化能量
        self.position = file['R'][P]  # 三维坐标
        self.atom = file['Z'][P]  # 原子种类

    def __len__(self):
        if self.mode == "train":
            return 5732  # 80%
        else:
            return 1433  # 20%

    def __getitem__(self, item):
        dist = get_distance_matrix(self.position[item])
        return (self.atom[item], dist), self.label[item]


def get_distance_matrix(pos):
    """
    pos: 23 x 3
    return: 23 x 23
    """
    res = [[0 for _ in range(23)] for _ in range(23)]
    for i in range(23):
        for j in range(23):
            if abs(np.sum(pos[i])) < 1e-7 or abs(np.sum(pos[j])) < 1e-7:
                continue
            # 对角线
            elif i == j:
                res[i][j] = 0
                continue
            else:
                atom1 = pos[i]
                atom2 = pos[j]
                minus = atom1 - atom2
                square = np.square(minus)
                res[i][j] = np.sum(square)
    return np.array(res)
