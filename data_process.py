import numpy as np
np.random.seed(seed=123)
from sklearn.utils import shuffle
import pandas as pd
#helper class for SVM data processing 
class DataPreprocess:
    #object constructor
    def __init__(self, test_data_path, sensor_cols, label_col, interval):
       
        self.test_data_path = test_data_path
        self.sensor_cols = sensor_cols
        self.label_col = label_col
        self.interval =  interval
        self.test_data, self.test_labels = self.load_data()
        self.test_data, self.test_labels = self.data_process(self.test_data, self.test_labels, self.interval)
        
    def load_data(self):
        
        test_data = pd.read_csv(self.test_data_path)
        test_label = test_data[self.label_col].to_numpy()
        test_data = test_data[self.sensor_cols].to_numpy()
        return test_data, test_label
    
    #prepare data and return labels     
    def data_process(self, data_set, label_set, interval):
        data=[]
        labels = []
        for i in range(0,len(data_set)-interval):
            vector=data_set[i].tolist()+data_set[i+interval].tolist()
            label = 1 if (label_set[i] == 'Normal') and (label_set[i+interval] == 'Normal') else -1
            data.append(vector)
            labels.append(label)
        return np.asarray(data), np.asarray(labels) 
