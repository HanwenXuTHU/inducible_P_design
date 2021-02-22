import xlrd
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset
import torch
import random
from torch.utils.data import DataLoader
import collections
import pandas as pd

class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class predictDataset(Dataset):

    def __init__(self, path='results/tet-100_2021-02-18-18-34-35_results.xlsx', isTrain=True, isGpu=True, isZero=True):
        self.path = path
        book1 = xlrd.open_workbook(self.path)
        sheet = book1.sheet_by_name('tet-100_2021-02-18-18-34-35_res')
        nrows = sheet.nrows
        random.seed(0)
        index = list(np.arange(nrows))
        self.pSeq = []
        self.split_r = 0.9
        self.isGpu = isGpu
        start, end = 0, int(nrows) - 1
        for i in range(start, end):
            self.pSeq.append(self.oneHot(sheet.cell(index[i + 1], 1).value))

    def oneHot(self, sequence):
        oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        oh = np.zeros([4, len(sequence)])
        for i in range(len(sequence)):
            oh[oh_dict[sequence[i]], i] = 1
        return oh

    def __getitem__(self, item):
        X = self.pSeq[item][:, :]
        X = transforms.ToTensor()(X)
        X = torch.squeeze(X)
        X = X.float()
        if self.isGpu:
            X = X.cuda()
        return X

    def __len__(self):
        return len(self.pSeq)


def decode_oneHot(seq):
    keys = ['A', 'T', 'C', 'G', 'M', 'N', 'H', 'Z']
    dSeq = ''
    for i in range(np.size(seq, 1)):
        pos = np.argmax(seq[:, i])
        dSeq += keys[pos]
    return dSeq


def main():
    model_real_path = 'others/predict_real.pth'
    model_expr_path = 'others/predict_expr.pth'
    dataset_input = DataLoader(dataset=predictDataset(isTrain=False, isGpu=True), batch_size=1,
                              shuffle=False)
    model_real = torch.load(model_real_path)
    model_expr = torch.load(model_expr_path)
    pSeqList = []
    isRealList = []
    exprList = []
    for k, inputLoader in enumerate(dataset_input):
        isReal = model_real(inputLoader)
        _, isReal = isReal.max(1)
        expr = model_expr(inputLoader)
        isReal = isReal.detach()
        isReal = isReal.cpu().float().numpy()
        seqs = inputLoader.detach()
        seqs = seqs.cpu().float().numpy()
        expr = expr.detach()
        expr = expr.cpu().float().numpy()
        for i in range(np.size(isReal)):
            isRealList.append(isReal[i])
            tempSeq = seqs[i, :, :]
            pSeqList.append(decode_oneHot(tempSeq))
            exprList.append(2**(expr[i, 0]))
    predictResults = collections.OrderedDict()
    predictResults['seq'] = pSeqList
    predictResults['isReal'] = isRealList
    predictResults['expr'] = exprList
    predictResults = pd.DataFrame(predictResults)
    predictResults.to_csv('others/predict_results.csv', index=False)
    debug = 0


if __name__ == '__main__':
    main()