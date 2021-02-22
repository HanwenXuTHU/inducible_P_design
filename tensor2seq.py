"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import collections
import pandas as pd
import time


def tensor2seq(input_sequence, label):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_sequence, np.ndarray):
        if isinstance(input_sequence, torch.Tensor):  # get the data from a variable
            sequence_tensor = input_sequence.data
        else:
            return input_sequence
        sequence_numpy = sequence_tensor[0].cpu().float().numpy()  # convert it into a numpy array
    else:  # if it is a numpy array, do nothing
        sequence_numpy = input_sequence
    return decode_oneHot(sequence_numpy, label)


def save_sequence(tensorSeq, name=''):
    i = 0
    results =collections.OrderedDict()
    results['realA'] = []
    results['fakeB'] = []
    results['realB'] = []
    for seqT in tensorSeq:
        for label, seq in seqT.items():
            seq = tensor2seq(seq, label)
            results[label].append(seq)
            i = i + 1
    results = pd.DataFrame(results)
    save_name = 'results/' + name + time.strftime('%Y-%m-%d-%H-%M-%S_', time.localtime(time.time())) + 'results.csv'
    results.to_csv(save_name, index=False)
    return save_name


def decode_oneHot(seq, label):
    keys = ['A', 'T', 'C', 'G']
    dSeq = ''
    for i in range(np.size(seq, 1)):
        if label == 'realA':
            if np.max(seq[:, i]) != 1:
                dSeq += 'M'
            else:
                pos = np.argmax(seq[:, i])
                dSeq += keys[pos]
        else:
            pos = np.argmax(seq[:, i])
            dSeq += keys[pos]
    return dSeq