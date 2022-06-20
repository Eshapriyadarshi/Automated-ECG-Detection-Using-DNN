import numpy as np 
import pandas as pd 
import os
import pickle
import matplotlib.pyplot as plt
import csv
from scipy import stats

def detect_peaks(signal, threshold=0.6, qrs_filter=None):
    '''
    Peak detection algorithm using cross corrrelation and threshold 
    '''
    if qrs_filter is None:
        # create default qrs filter, which is just a part of the sine function
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)
    
    # normalize data
    signal = (signal - signal.mean()) / signal.std()

    # calculate cross correlation
    similarity = np.correlate(signal, qrs_filter, mode="same")
    similarity = similarity / np.max(similarity)

    # return peaks (values in ms) using threshold
    return signal[similarity > threshold].index, similarity

if __name__== '__main__':
    print(pickle.__version__)

