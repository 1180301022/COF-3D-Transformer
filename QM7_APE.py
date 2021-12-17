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
            return 5732
        else:
            return 1433

    def __getitem__(self, item):
        label = self.label[item]  # 标签
        data = (self.atom[item], self.position[item])  # 分子类型，坐标
        return data, label
