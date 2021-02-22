import basemodel
from motif2sequence import P2P_model
from options import input_opt
from read_dataset import read_dataset
from torch.utils.data import DataLoader
from tensor2seq import save_sequence, tensor2seq
from tansfer_fasta import csv2fasta
#from utils.qc.seq_criteria import seq_criteria
import numpy as np


def main():
    opt_gen = input_opt()
    data_test = read_dataset(opt_gen)
    sequence_length = np.size(data_test.input['A'][0], 1)
    model = P2P_model(opt_gen, sequence_length=sequence_length)
    model.setup(opt_gen)
    dataset_test = DataLoader(dataset=data_test, batch_size=1, shuffle=False)
    tensorSeq = []
    for i, data in enumerate(dataset_test):
        model.set_input(data)
        model.test()
        tensorSeq.append(model.get_current_visuals())
    csv_name = save_sequence(tensorSeq, opt_gen.project_name + '_')
    csv2fasta(csv_name, opt_gen.results_path, opt_gen.data_name)
    #seq_criteria('results/tet-100_2021-02-18-18-34-35_results.csv')

if __name__ == '__main__':
    main()

