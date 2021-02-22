import csv
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset
import torch


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class read_dataset(Dataset):

    def __init__(self, opt):
        self.path = opt.data_path + opt.data_name
        self.input = {}
        self.input['A'] = []
        self.input['B'] = []
        f = open(self.path, 'r')
        self.reader_num = csv.reader(f)
        self.sizes = 0
        for i in self.reader_num:
            self.sizes += 1
        self.split_ratio = opt.split_ratio
        self.isTrain = opt.isTrain
        self.cut = int(self.sizes * self.split_ratio)
        self.data_loading()

    def data_loading(self):
        n = 0
        f1 = open(self.path, 'r')
        self.reader = csv.reader(f1)
        for i in self.reader:
            if self.isTrain:
                if 0 < n < int(self.split_ratio * self.sizes):
                    self.input['A'].append(self.oneHot(i[0], dim='A'))
                    self.input['B'].append(self.oneHot(i[1], dim='B'))
            else:
                if n >= int(self.split_ratio * self.sizes) and n > 0:
                    self.input['A'].append(self.oneHot(i[0], dim='A'))
                    self.input['B'].append(self.oneHot(i[1], dim='B'))
            n = n + 1

    def oneHot(self, sequence, dim='A'):
        if dim == 'A':
            oh = np.zeros([4, len(sequence)])
            oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
            for i in range(len(sequence)):
                if sequence[i] == 'M':
                    oh[:, i] = np.random.rand(4)
                else:
                    oh[oh_dict[sequence[i]], i] = 1
            return oh
        elif dim == 'B':
            oh = np.zeros([4, len(sequence)])
            for i in range(len(sequence)):
                oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
                oh[oh_dict[sequence[i]], i] = 1
            return oh

    def __getitem__(self, item):
        A = self.input['A'][item]
        B = self.input['B'][item]
        A = transforms.ToTensor()(A)
        A = torch.squeeze(A)
        A = A.float()
        B = transforms.ToTensor()(B)
        B = torch.squeeze(B)
        B = B.float()
        return {'A': A, 'B': B}

    def __len__(self):
        return len(self.input['A'])