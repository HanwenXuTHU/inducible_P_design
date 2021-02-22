import xlrd
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset
import torch
import random


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class SeqDataset(Dataset):

    def __init__(self, path='others/ecoli_predictdata.xlsx', isTrain=True, isGpu=True, isZero=True):
        self.path = path
        book1 = xlrd.open_workbook(self.path)
        sheet = book1.sheet_by_name('ecoli_predictdata')
        nrows = sheet.nrows
        random.seed(0)
        index = list(np.arange(nrows))
        index = index[1 : ]
        random.shuffle(index)
        self.pSeq = []
        self.expr = []
        self.isReal = []
        self.isTrain = isTrain
        self.split_r = 0.9
        self.isGpu = isGpu
        if self.isTrain:
            start, end = 1, int(nrows*self.split_r)
        else:
            start, end = int(nrows*self.split_r), nrows - 1
        for i in range(start, end):
            if isZero == True:
                self.pSeq.append(self.oneHot(sheet.cell(index[i], 0).value))
                self.isReal.append(sheet.cell(index[i], 1).value)
                self.expr.append(sheet.cell(index[i], 2).value)
            else:
                if sheet.cell(index[i], 2).value > 0:
                    self.pSeq.append(self.oneHot(sheet.cell(index[i], 0).value))
                    self.isReal.append(sheet.cell(index[i], 1).value)
                    self.expr.append(sheet.cell(index[i], 2).value)

    def oneHot(self, sequence):
        oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        oh = np.zeros([4, len(sequence)])
        for i in range(len(sequence)):
            oh[oh_dict[sequence[i]], i] = 1
        return oh

    def __getitem__(self, item):
        X = self.pSeq[item][:, :]
        Y = self.isReal[item]
        Z = self.expr[item]
        X = transforms.ToTensor()(X)
        X = torch.squeeze(X)
        X = X.float()
        Y = transforms.ToTensor()(np.asarray([[Y]]))
        Y = torch.squeeze(Y)
        Y = Y.long()
        Z = transforms.ToTensor()(np.asarray([[np.log2(Z)]]))
        Z = torch.squeeze(Z)
        Z = Z.float()
        if self.isGpu:
            X, Y, Z = X.cuda(), Y.cuda(), Z.cuda()
        return {'x': X,'y': Y,'z':Z}

    def __len__(self):
        return len(self.isReal)


