import numpy as np 
import pandas as pd 
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


def record_file_preparation(record_file):
    signals = []
    window_size = 180
    df = pd.read_csv(record_file) 
    X = list()
    with open(record_file, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|') # read CSV file\
        row_index = -1
        for row in spamreader:
            if(row_index >= 0):
                signals.insert(row_index, int(row[1]))
            row_index += 1

        signals = stats.zscore(signals)
       
    segment = list()    
    beat = list()
    win = 235
    i=0
    peaks, _ = detect_peaks(df.iloc[:,1])
    #print(peaks)
    df1 = pd.DataFrame(peaks)
    df1.rename(columns={0:"'sample #'"},inplace=True)
    joindf = pd.merge(df,df1,on="'sample #'",how='inner')
    joindf.drop(joindf.columns[2],axis = 1,inplace=True)

    for j in range (joindf.shape[0]):
        beat.append(joindf["'sample #'"][(joindf["'sample #'"]>i) & (joindf["'sample #'"]<win)].max())
        i += 235
        win += 235
    r_peaks = [x for x in beat if str(x) != 'nan']
    
    for i in (r_peaks):    
        
        if(window_size <= i and i < (len(signals) - window_size)):

            segment = signals[i-window_size:i+window_size]
            X.append(segment)
    return X


def predict_result(file_path, model):
    X = record_file_preparation(file_path)
    X = pd.DataFrame(X)
    test_x = X.to_numpy()
    test_x = test_x.reshape(len(test_x), test_x.shape[1],1)
    predictions = model.predict(test_x)
    y_pred = (predictions > 0.5)
    res = pd.DataFrame(y_pred)
    return res



