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


class yeast_dataset(Dataset):

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
                    self.input['A'].append(i[0])
                    self.input['B'].append(i[1])
            else:
                if n >= int(self.split_ratio * self.sizes):
                    self.input['A'].append(i[0])
                    self.input['B'].append(i[1])
            n = n + 1

    def oneHot(self, sequence, dim=8):
        if dim == 8:
            oh = np.zeros([8, len(sequence)])
            for i in range(len(sequence)):
                oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'M': 4, 'H': 5, 'N': 6, 'Z': 7}
                oh[oh_dict[sequence[i]], i] = 1
            return oh
        elif dim == 4:
            oh = np.zeros([4, len(sequence)])
            for i in range(len(sequence)):
                oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
                oh[oh_dict[sequence[i]], i] = 1
            return oh

    def __getitem__(self, item):
        A = self.input['A'][item]
        B = self.input['B'][item]
        A = self.oneHot(A, dim=8)
        B = self.oneHot(B, dim=4)
        A = transforms.ToTensor()(A)
        A = torch.squeeze(A)
        A = A.float()
        B = transforms.ToTensor()(B)
        B = torch.squeeze(B)
        B = B.float()
        return {'A': A, 'B': B}

    def __len__(self):
        return len(self.input['A'])