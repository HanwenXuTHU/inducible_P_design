import pandas as pd
import numpy as np
import collections


def replace_char(string, char, index):
    string = list(string)
    string[index] = char
    return ''.join(string)


def generate_input(motif = 'ttgtgagcggataacaa', gen_num = 100, length = 100):
    motif = motif.upper()
    bg = ['M']
    motif_length = len(motif)
    gen_data = collections.OrderedDict()
    gen_data['input'] = []
    gen_data['output'] = []
    oh_dict = ['A', 'T', 'C', 'G']
    for i in range(length - motif_length + 1):
        motif_pos = i
        for j in range(gen_num):
            line_n = ''
            line_op = ''
            for k in range(length):
                line_n += bg[0]
                line_op += oh_dict[np.random.randint(0, 4)]
            for k in range(motif_length):
                line_n = replace_char(line_n, motif[k], motif_pos + k)
            gen_data['input'].append(line_n)
            gen_data['output'].append(line_op)
    gen_data = pd.DataFrame(gen_data)
    gen_data.to_csv('others/tet.csv', index=False)

generate_input()




