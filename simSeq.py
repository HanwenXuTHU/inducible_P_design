import pandas as pd
import numpy as np


class simSeq():

    def __init__(self, ref_path='ecoli_100.csv'):
        ref = pd.read_csv(ref_path, index_col=None)
        self.seqs = list(ref['output'])
        self.bg_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self.bg_order_3 = self.get_background()
        self.seq_length = len(self.seqs[0])

    def get_background(self):
        bg_order_3 = np.zeros([4, 4, 4, 4])
        for i in range(len(self.seqs)):
            seq_t = self.seqs[i]
            for j in range(len(seq_t) - 4):
                bg_order_3[self.bg_dict[seq_t[j]], self.bg_dict[seq_t[j + 2]], self.bg_dict[seq_t[j + 3]], self.bg_dict[seq_t[j + 4]]] += 1
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    bg_order_3[i, j, k, :] = bg_order_3[i, j, k, :]/np.sum(bg_order_3[i, j, k, :])
        return bg_order_3

    def simulate_seq(self):
        bg_list = ['A', 'T', 'C', 'G']
        sim_s = ''
        start_seq = np.random.randint(0, 4, 3)
        sim_s += bg_list[start_seq[0]]
        sim_s += bg_list[start_seq[1]]
        sim_s += bg_list[start_seq[2]]
        for i in range(self.seq_length - 3):
            sim_i = np.random.choice(range(4), p=[self.bg_order_3[self.bg_dict[sim_s[i]], self.bg_dict[sim_s[i+1]], self.bg_dict[sim_s[i+2]], 0],
                                                  self.bg_order_3[self.bg_dict[sim_s[i]], self.bg_dict[sim_s[i+1]], self.bg_dict[sim_s[i+2]], 1],
                                                  self.bg_order_3[self.bg_dict[sim_s[i]], self.bg_dict[sim_s[i+1]], self.bg_dict[sim_s[i+2]], 2],
                                                  self.bg_order_3[self.bg_dict[sim_s[i]], self.bg_dict[sim_s[i+1]], self.bg_dict[sim_s[i+2]], 3]])
            sim_s += bg_list[sim_i]
        return sim_s