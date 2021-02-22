from motif2sequence import P2P_model
from read_dataset import read_dataset
from options import train_opt, test_opt
from torch.utils.data import DataLoader
from tensor2seq import save_sequence, tensor2seq
from tansfer_fasta import csv2fasta
import numpy as np


def log_info(dir, epoch, train_loss):
    with open(dir, 'a') as f:
        print('epoch: {} training loss G: {}\n'.format(epoch, train_loss))
        f.write('epoch: {} training loss G: {}\n'.format(epoch, train_loss))


def main():
    opt_train = train_opt()
    opt_test = test_opt()
    data_train = read_dataset(opt_train)
    data_test = read_dataset(opt_test)
    sequence_length = np.size(data_train.input['A'][0], 1)
    model = P2P_model(opt_train, sequence_length=sequence_length)
    dataset_train = DataLoader(dataset=data_train, batch_size=3000, shuffle=True)
    dataset_test = DataLoader(dataset=data_test, batch_size=1, shuffle=False)
    model.setup(opt_train)
    for epoch in range(opt_train.epoch_count, opt_train.n_epochs + opt_train.n_epochs_decay + 1):
        model.update_learning_rate()
        num = 0
        train_loss = 0
        for i, data in enumerate(dataset_train):
            model.set_input(data)
            model.optimize_parameters()
            train_loss += model.loss_G
            num += 1
        train_loss /= num
        log_info(opt_train.log_dir, epoch, train_loss)
    tensorSeq = []
    for i, data in enumerate(dataset_test):
        model.set_input(data)
        model.test()
        tensorSeq.append(model.get_current_visuals())
    csv_name = save_sequence(tensorSeq, opt_train.project_name + '_')
    csv2fasta(csv_name, opt_train.results_path, opt_train.data_name)
    model.save_networks(epoch)


if __name__ == '__main__':
    main()