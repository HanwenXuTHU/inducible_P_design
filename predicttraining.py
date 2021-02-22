import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from predictdataset import SeqDataset
from predictmodel import expr_predict, real_predict
from matplotlib import pyplot as plt
import numpy as np
import seqlogo


class MotifAnalysis:
    def __init__(self):
        self.batch_size = 2048
        self.lr = 0.001
        self.lr_expr = 0.0005
        self.gpu = True
        trainD = SeqDataset(isTrain=True, isGpu=self.gpu)
        self.train_num = len(trainD.pSeq)
        testD = SeqDataset(isTrain=False, isGpu=self.gpu)
        self.test_num = len(testD.pSeq)
        self.dataset_train = DataLoader(dataset=SeqDataset(isTrain=True, isGpu=self.gpu), batch_size=self.batch_size, shuffle=True)
        self.dataset_train_nz = DataLoader(dataset=SeqDataset(isTrain=True, isGpu=self.gpu, isZero=False), batch_size=self.batch_size,
                                        shuffle=True)
        self.dataset_test = DataLoader(dataset=SeqDataset(isTrain=False, isGpu=self.gpu), batch_size=self.batch_size, shuffle=False)
        self.epoch = 100
        self.model_real = real_predict(input_nc=4, output_nc=4)
        self.model_expr = expr_predict(input_nc=4, output_nc=4)
        self.save_path = 'others/'
        if self.gpu:
            self.model_expr=self.model_expr.cuda()
            self.model_real = self.model_real.cuda()
        self.loss_y = torch.nn.CrossEntropyLoss()
        self.loss_z = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model_real.parameters(), lr=self.lr)
        self.optimizer_expr = torch.optim.Adam(self.model_expr.parameters(), lr=self.lr_expr)

    def training(self):
        for ei in range(self.epoch):
            train_loss_y = 0
            train_loss_z = 0
            train_num_y = 0
            train_num_z = 0
            test_loss_y = 0
            test_loss_z = 0
            test_num = 0
            train_num_correct = 0
            output_after_conv1 = 0
            input_data = 0
            for trainLoader in self.dataset_train:
                train_data, train_y, train_z = trainLoader['x'], trainLoader['y'], trainLoader['z']
                predict = self.model_real(train_data)
                predict_y = predict
                loss_y = self.loss_y(predict_y, train_y)
                self.optimizer.zero_grad()
                loss_y.backward()
                self.optimizer.step()
                train_loss_y += loss_y
                train_num_y = train_num_y + 1
                _, pred = predict_y.max(1)
                train_num_correct += (pred == train_y).sum().item()
            for trainLoader in self.dataset_train_nz:
                train_data, train_y, train_z = trainLoader['x'], trainLoader['y'], trainLoader['z']
                predict = self.model_expr(train_data)
                predict_z = torch.squeeze(predict)
                loss_z = self.loss_z(predict_z, train_z)
                self.optimizer_expr.zero_grad()
                loss_z.backward()
                self.optimizer_expr.step()
                train_loss_z += loss_z
                train_num_z = train_num_z + 1
            num_correct = 0
            test_predict_expr = []
            test_real_expr = []
            for testLoader in self.dataset_test:
                test_data, test_y, test_z = testLoader['x'], testLoader['y'], testLoader['z']
                predict_y = self.model_real(test_data)
                predict_z = self.model_expr(test_data)
                predict_y = predict_y.detach()
                predict_z = predict_z.detach()
                test_loss_y += self.loss_y(predict_y, test_y)
                predict_z1 = predict_z.cpu().float().numpy()
                real_z1 = test_z.cpu().float().numpy()
                for i in range(np.size(real_z1)):
                    if real_z1[i] > 0:
                        test_real_expr.append(real_z1[i])
                        test_predict_expr.append(predict_z1[i, 0])
                test_loss_z += self.loss_z(predict_z, test_z)
                test_num = test_num + 1
                _, pred = predict_y.max(1)
                num_correct += (pred == test_y).sum().item()
            coefs = np.corrcoef(test_real_expr, test_predict_expr)
            coefs = coefs[0, 1]
            print('epoch:{} train correct:{} evaluation correct:{} train_loss z:{} test_loss z:{} test_coefs:{}'.format(ei, train_num_correct/self.train_num, num_correct / self.test_num, train_loss_z/train_num_z, test_loss_z/test_num, coefs))
        predict_expr = []
        real_expr = []
        for testLoader in self.dataset_test:
            test_data, test_y, test_z = testLoader['x'], testLoader['y'], testLoader['z']
            predict_z = self.model_expr(test_data)
            predict_z = predict_z.detach()
            predict_z = predict_z.cpu().float().numpy()
            real_z = test_z.cpu().float().numpy()
            for i in range(np.size(real_z)):
                if real_z[i] > 0:
                    real_expr.append(real_z[i])
                    predict_expr.append(predict_z[i, 0])
        ## scatter
        real_expr = np.asarray(real_expr)
        predict_expr = np.asarray(predict_expr)
        #figure = plt.figure()
        plt.scatter(real_expr, predict_expr)
        coefs = np.corrcoef(real_expr, predict_expr)
        coefs = coefs[0, 1]
        plt.title('pearson coefficient:{}'.format(coefs))
        plt.savefig('others/predict_scatter.png')
        torch.save(self.model_expr, self.save_path + '/predict_expr.pth')
        torch.save(self.model_real, self.save_path + '/predict_real.pth')





    def motif_detect(self, detect_ratio = 0.1, conv_size = 7, conv_num = 32):
        input_conv = np.load('input_conv.npy')
        output_conv = np.load('output_conv.npy')
        data_size = np.size(output_conv, 0)
        in_len = np.size(input_conv, 2)
        op_len = np.size(output_conv, 2)
        diff = int((in_len - op_len) / 2)
        self.motifs = np.zeros([conv_num, 4, conv_size])
        for k in range(data_size):
            for i in range(op_len):
                for j in range(conv_num):
                    stOut = np.sort(-output_conv[k, j, :])
                    T = -stOut[int(detect_ratio*op_len)]
                    if output_conv[k, j, i] > T:
                        self.motifs[j, :, :] += input_conv[k, :, i : i + 2 * diff + 1] * output_conv[k, j, i]
        for i in range(conv_size):
            for j in range(conv_num):
                self.motifs[j, :, i] /= np.sum(self.motifs[j, :, i])
        for j in range(conv_num):
            pwm = self.motifs[j, :, :]
            np.savetxt('motif_conv' + str(j) + '.txt', pwm)


def main():
    analysis = MotifAnalysis()
    analysis.training()
    #analysis.motif_detect()

if __name__ == '__main__':
    main()






