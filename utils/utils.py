#!/usr/bin/env python3
### Burak Less Data Experiment Utils

### GENERIC
import copy
import datetime
import io
import os
from os import listdir
from os.path import isfile, join, isdir
import sys
from functools import partial

### DATA PROCESS
import pandas as pd
import numpy as np
import ast 
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
import re

### PLOTTING & LOGS
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pylab import rcParams
rcParams['figure.figsize'] = 8, 6

### DATA STORING
import h5py
import pickle
import json

### RANDOM
import random
import time
#from numpy.random import seed

import multiprocessing
from multiprocessing import Pool
#print("CPU COUNT:", multiprocessing.cpu_count())
from fast_features import generate_features
from scipy.stats import ks_2samp



###PREDICT UTILS###
def plot_cm(labels, predictions, name):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('{}'.format(name))    
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    
def majority_filter_traditional(seq, width):
    offset = width // 2
    seq = [0] * offset + seq
    result = []
    for i in range(len(seq) - offset):
        a = seq[i:i+width]
        result.append(max(set(a), key=a.count))
    return result        
    
def consecutive_filter(seq,width):
    
    result = []
    for index in range(len(seq)):
        tmp_set_list = list(set(seq[index:index+width]))
        if len(tmp_set_list) == 1 and tmp_set_list[0] == seq[index]:
            result.append(seq[index])
        else:
            result.append(0) #Assumes healthy label is 0

    return result

def calculate_miss_rates(true_label,pred_label):
    alarm_dict = {}

    normal_true_idx = np.where(true_label==0)[0]
    anom_true_idx = np.where(true_label!=0)[0]

    #Find number of normal samples labeled as anomalous
    fp_deploy = pred_label[normal_true_idx][pred_label[normal_true_idx] != 0]

    false_alarm_rate = len(fp_deploy) / len(normal_true_idx)
    logging.info("Total normal runs classified as anomaly: %s, Total normal runs %s ",str(len(fp_deploy)),str(len(normal_true_idx)))
    logging.info(false_alarm_rate)

    #Find number of anomalous samples labeled as normal
    fn_deploy = pred_label[anom_true_idx][pred_label[anom_true_idx] == 0]

    anom_miss_rate = len(fn_deploy) / len(anom_true_idx)
    logging.info("Total anom runs classified as normal: %s, Total anom runs %s ",str(len(fn_deploy)),str(len(anom_true_idx)))
    logging.info(anom_miss_rate) 
    
    alarm_dict['false_alarm_rate'] = false_alarm_rate
    alarm_dict['anom_miss_rate'] = anom_miss_rate
    
    return alarm_dict    
    
def false_anom_rate_calc(true_label,pred_label,conf,cv_index,name,save):
    """
        Calculates false alarm rate and anomaly miss rate
        Assumes 0 is normal label and other labels are anomalies
        
        Args:
            true_label: Array composed of integer labels, e.g., [0,0,4,2]
            pred_label: Array composed of integer labels, e.g., [0,0,4,2]
    """    
    # • False alarm rate: The percentage of the healthy windows that are identified as anomalous (any anomaly type).
    # • Anomaly miss rate: The percentage of the anomalous windows that are identified as healthy
    alarm_dict = {}
        
    normal_true_idx = np.where(true_label==0)[0]
    anom_true_idx = np.where(true_label!=0)[0]
        
    #Find number of normal samples labeled as anomalous
    fp_deploy = pred_label[normal_true_idx][pred_label[normal_true_idx] != 0]

    false_alarm_rate = len(fp_deploy) / len(normal_true_idx)
    logging.info("Total normal runs classified as anomaly: %s, Total normal runs %s ",str(len(fp_deploy)),str(len(normal_true_idx)))
    logging.info(false_alarm_rate)
    
    #Find number of anomalous samples labeled as normal
    fn_deploy = pred_label[anom_true_idx][pred_label[anom_true_idx] == 0]

    anom_miss_rate = len(fn_deploy) / len(anom_true_idx)
    logging.info("Total anom runs classified as normal: %s, Total anom runs %s ",str(len(fn_deploy)),str(len(anom_true_idx)))
    logging.info(anom_miss_rate)    
    
    alarm_dict['false_alarm_rate'] = false_alarm_rate
    alarm_dict['anom_miss_rate'] = anom_miss_rate
    
    if save:
        json_dump = json.dumps(alarm_dict)
        f_json = open(conf['results_dir'] / ("{}_alert_dict.json".format(name)),"w")
        f_json.write(json_dump)
        f_json.close()    

def analysis_wrapper_multiclass(true_labels, pred_labels,conf,cv_index,name,name_cm='Deployment Data',save=True,plot=True):
    """
        true_labels: it should be in the format of an array [0,2,1,3,...]
        pred_labels: it should be in the format of an array [0,1,1,4,...]        
    """
    from sklearn.metrics import classification_report
    logging.info("####################################")

    logging.info("%s\n%s",name_cm,classification_report(y_true=true_labels, y_pred =pred_labels))
    logging.info("#############")
    deploy_report = classification_report(y_true=true_labels, y_pred =pred_labels,output_dict=True)

    if save:
        logging.info("Saving results")
        cv_path = conf['results_dir']
        json_dump = json.dumps(deploy_report)
        f_json = open(cv_path / ("{}_report_dict.json".format(name)),"w")
        f_json.write(json_dump)
        f_json.close() 
        
    if plot:
        plot_cm(true_labels, pred_labels,name=name_cm)    
        
    false_anom_rate_calc(true_labels,pred_labels,conf,cv_index,name,save)    
    
    
class WindowShopper: 

    def __init__(self, data, labels, window_size = 64, trim=30, silent=False):
        '''Init'''
        self.data = data
        self.labels = labels
        if self.labels is not None:
            self.label_count = len(labels['anom'].unique()) #Automatically assuming anomaly classification
        self.trim = trim
        self.silent = silent

        #Windowed data and labels
        self.windowed_data = []
        self.windowed_label = []
        
        #Output shape
        self.window_size = window_size
        self.metric_count = len(data.columns)
        self.output_shape = (self.window_size, self.metric_count)    

        #Prepare windows
        self._get_windowed_dataset()
    
    #Not calling this but it is good to have
    def _process_sample_count(self):
        self.per_label_count = {x: 0 for x in self.labels[self.labels.columns[0]].unique()}
        self.sample_count = 0
        for node_id in self.data.index.get_level_values('node_id').unique():
            counter = 0
            cur_array = self.data.loc[node_id, :, :]
            for i in range(self.trim, len(cur_array) - self.window_size - self.trim):
                counter += 1
            self.sample_count += counter
            self.per_label_count[self.labels.loc[node_id, self.labels.columns[0]]] += counter

    def _get_windowed_dataset(self):

        if self.labels is not None:
            #Iterate unique node_ids
            for node_id in self.labels.index.unique():
              # print(node_id)
                cur_array = self.data.loc[node_id,:,:]

                temp_data = []
                temp_label = []
                #Iterate over application runtime
                for i in range(self.trim, len(cur_array) - self.window_size - self.trim):

                    self.windowed_data.append(cur_array.iloc[i:i+self.window_size].to_numpy(
                      dtype=np.float32).reshape(self.output_shape))
                    self.windowed_label.append(self.labels.loc[node_id])

            self.windowed_data = np.dstack(self.windowed_data)
            self.windowed_data = np.rollaxis(self.windowed_data,2)
            if not self.silent:
                logging.info("Windowed data shape: %s",self.windowed_data.shape)
            #FIXME: column names might be in reverse order for HPAS data, Used app, anom for Cori data but it was anom,app

            self.windowed_label = pd.DataFrame(np.asarray(self.windowed_label).reshape(len(self.windowed_label),2),columns=['app','anom'])

            if not self.silent:
                logging.info("Windowed label shape: %s",self.windowed_label.shape)
        else:
            logging.info("Deployment selection - no label provided")
            
            cur_array = self.data

            temp_data = []
            temp_label = []
            #Iterate over application runtime
            for i in range(self.trim, len(cur_array) - self.window_size - self.trim):

                self.windowed_data.append(cur_array.iloc[i:i+self.window_size].to_numpy(
                  dtype=np.float32).reshape(self.output_shape))

            self.windowed_data = np.dstack(self.windowed_data)
            self.windowed_data = np.rollaxis(self.windowed_data,2)                
                
            self.windowed_label = None

    def return_windowed_dataset(self):

        return self.windowed_data, self.windowed_label
    
def granularityAdjust(data,granularity=60):
    
    result = pd.DataFrame()
    for nid in data.index.get_level_values('node_id').unique():
        temp_data = data[data.index.get_level_values('node_id') == nid]
        temp_data = temp_data.iloc[ \
            (temp_data.index.get_level_values('timestamp').astype(int) -
             int(temp_data.index.get_level_values('timestamp')[0])) \
            % granularity == 0]
        result = pd.concat([result,temp_data])
                
    return result    

class MyEncoder:
    def fit_transform(self, labels,dataset=None):
        self.dataset = dataset
        self.fit_anom(labels)
        self.fit_appname(labels)
        return self.transform(labels)

    def fit_anom(self, labels):
        self.anoms = labels['anom'].unique()
        self.anom_dict = {}
        for idx, i in enumerate(self.anoms):
            self.anom_dict[i] = idx
            
    def fit_appname(self,labels):
        self.apps = labels['app'].unique()
        self.app_dict = {}
        for idx, i in enumerate(self.apps):
            self.app_dict[i] = idx
            
    def transform(self, labels):
#         if self.dataset == 'tpds':
#             labels['anom'] = labels['anom'].apply(self.anom_dict.get)
#             labels['app'] = labels['app'].apply(self.app_dict.get)
#         elif self.dataset == 'hpas':
        labels['anom'] = labels['anom'].apply(self.anom_dict.get)
        labels['app'] = labels['app'].apply(self.app_dict.get)            

#         elif self.dataset == 'cori':
#             raise NotImplemented

        #labels.rename(columns={'anomaly':"anom",'appname':"app"},inplace=True)
        
        return labels    
    
    
#TODO: Make the second reader parallel
_TIMESERIES = None

def _get_features(node_id, features=None, **kwargs):
    global _TIMESERIES
    assert (
        features == ['max', 'min', 'mean', 'std', 'skew', 'kurt',
                     'perc05', 'perc25', 'perc50', 'perc75', 'perc95']
    )
#    print("Kwargs Trial",kwargs['trim']);

    if isinstance(_TIMESERIES, pd.DataFrame):
        df = pd.DataFrame(
            generate_features(
                np.asarray(_TIMESERIES.loc[node_id, :, :].values.astype('float'), order='C'),
                trim=kwargs['trim']
            ).reshape((1, len(_TIMESERIES.columns) * 11)),
            index=[node_id],
            columns=[feature + '_' + metric
                     for metric in _TIMESERIES.columns
                     for feature in features])
        return df
    else:
        # numpy array format compatible with Burak's notebooks
        return generate_features(
                np.asarray(_TIMESERIES[node_id].astype(float), order='C'),
                trim=kwargs['trim']

            ).reshape((1, _TIMESERIES.shape[2] * 11))

class _FeatureExtractor:
    def __init__(self, features=None, window_size=None, trim=None):
        self.features = features
        self.window_size = window_size
        self.trim = trim

    def __call__(self, node_id):
        return _get_features(
            node_id, features=self.features,
            window_size=self.window_size, trim=self.trim)

class TSFeatureGenerator:
    """Wrapper class for time series feature generation"""

    def __init__(self, trim=60, threads=multiprocessing.cpu_count(),
                 features=['max', 'min', 'mean', 'std', 'skew', 'kurt',
                           'perc05', 'perc25', 'perc50', 'perc75', 'perc95']):
        self.features = features
        self.trim = trim
        self.threads = threads

    def fit(self, x, y=None):
        """Extracts features
            x = training data represented as a Pandas DataFrame
            y = training labels (not used in this class)
        """
        return self

    def transform(self, x, y=None):
        """Extracts features
            x = testing data/data to compare with training data
            y = training labels (not used in this class)
        """
        global _TIMESERIES
        _TIMESERIES = x
        if isinstance(x, pd.DataFrame):
            with Pool(processes=self.threads) as pool:
                result = pool.map(
                    _FeatureExtractor(features=self.features,
                                      window_size=0, trim=self.trim),
                    x.index.get_level_values('node_id').unique())
                pool.close()
                pool.join()
                return pd.concat(result)
        else:
            # numpy array format compatible with Burak's notebooks
            result = [
                      _FeatureExtractor(features=self.features,
                                  window_size=0, trim=self.trim)(i) for i in range(len(x))]
            return np.concatenate(result, axis=0)
        
    def transform_window(self, x, y=None,window=45):
        """Extracts features
            x = testing data/data to compare with training data
            y = training labels (not used in this class)
        """
        global _TIMESERIES
        _TIMESERIES = x
        if isinstance(x, pd.DataFrame):
            with Pool(processes=self.threads) as pool:
                result = pool.map(
                    _FeatureExtractor(features=self.features,
                                      window_size=window, trim=self.trim),
                    x.index.get_level_values('node_id').unique())
                pool.close()
                pool.join()
                return pd.concat(result)
        else:
            # numpy array format compatible with Burak's notebooks
            result = [
                      _FeatureExtractor(features=self.features,
                                  window_size=window, trim=self.trim)(i) for i in range(len(x))]
            return np.concatenate(result, axis=0)        

def generate_rolling_features(time_series, features=None, window_size=0, trim=60, skip=None):
    
    assert(features is not None)
    if trim != 0:
        time_series = time_series[trim:- trim]
    if window_size > len(time_series) or window_size < 1:
        window_size = len(time_series)
    df_rolling = time_series.rolling(window_size)
    columns = time_series.columns
    df_features = []
    col_map = {}

    def add_feature(f, name):
        nonlocal df_features
        nonlocal df_rolling
        col_map = {}
        for c in columns:
            col_map[c] = feature + '_' + c
        df_features.append(f(df_rolling)[window_size - 1::skip].rename(index=str, columns=col_map))

    percentile_regex = re.compile(r'perc([0-9]+)')
    for feature in features:
        percentile_match = percentile_regex.fullmatch(feature)
        if feature == 'max':
            add_feature(lambda x: x.max(), feature)
        elif feature == 'min':
            add_feature(lambda x: x.min(), feature)
        elif feature == 'mean':
            add_feature(lambda x: x.mean(), feature)
        elif feature == 'std':
            add_feature(lambda x: x.var(), feature)
        elif feature == 'skew':
            add_feature(lambda x: x.skew().fillna(0), feature)
        elif feature == 'kurt':
            add_feature(lambda x: x.kurt().fillna(-3), feature)
        elif percentile_match is not None:
            quantile = float(percentile_match.group(1)) / 100
            add_feature(lambda x: x.quantile(quantile), feature)
        else:
            raise ValueError("Feature '{}' could not be parsed".format(feature))

    df = pd.concat(df_features, axis=1)
    return df        
        
        
def get_nids_apps(metadata,appname):

    nids = metadata[metadata['app'] == appname]['node_ids']
    nids = nids.apply(ast.literal_eval)
    nids_list = []
    for temp_list in nids:
        nids_list = nids_list + temp_list

    return nids_list

def smart_select(label_df, case, anom_type=None, app_type=None):

    anom_dict = dict(label_df['anom'].value_counts())
    logging.info("Anomaly distribution %s", anom_dict)
    app_dict = dict(label_df['app'].value_counts())
    logging.info("App distribution %s",app_dict)


    #Select only one anomaly
    if case == 1:
        logging.info("Selected ANOMALY type: %s",anom_type)
        return pd.DataFrame(label_df[label_df['anom'] == anom_type])
    #Select only one app
    elif case == 2:
        logging.info("Selected APP type: %s",app_type)
        return pd.DataFrame(label_df[label_df['app'] == app_type])
    #Select multiple anoms
    elif case == 3:
        logging.info("Selected ANOMALY types: %s",anom_type)
        return pd.DataFrame(label_df[label_df['anom'].isin(anom_type)])
    #Select multiple apps
    elif case == 4:
        logging.info("Selected APP types: %s",app_type)
        return pd.DataFrame(label_df[label_df['app'].isin(app_type)])
    #Select multiple apps and anoms
    elif case ==5:
        logging.info("Selected APP type, %s", app_type)
        logging.info("Selected ANOM type, %s",anom_type)

        try:
            if(len(label_df[label_df['anom'].isin(anom_type) & label_df['app'].isin(app_type)]) == 0):
                raise Exception
            else:
                return label_df[label_df['anom'].isin(anom_type) & label_df['app'].isin(app_type)]        
        except:
            logging.info("Provided combination does NOT exist!")
            return

        
    else:
        logging.info("Invalid case selection")        
        return

def read_h5file(READ_PATH, filename):

    logging.info("Reading h5file!")

    if isdir(READ_PATH):
        tempFilename = str(filename) + ".h5"
        tempPath = join(READ_PATH,str(tempFilename))
        hf_read = h5py.File(tempPath, 'r')
        tempData = np.array(hf_read.get(filename))

        return tempData

    else:
        logging.info("Error in PATH!")

#Reads the h5 file and csv file names windowed_test_data and windowed_test_label
def read_windowed_test_data(READ_PATH):

    windowed_test_data = read_h5file(READ_PATH,'windowed_test_data')
    windowed_test_label = pd.read_csv(join(READ_PATH,"windowed_test_label.csv"))
    
    logging.info("Windowed test data shape: %s", windowed_test_data.shape)
    logging.info("Windowed test label shape: %s", windowed_test_label.shape)

    return windowed_test_data, windowed_test_label

#Reads the h5 file and csv file names windowed_train_data and windowed_train_label
def read_windowed_train_data(READ_PATH):

    windowed_train_data = read_h5file(READ_PATH,'windowed_train_data')
    windowed_train_label = pd.read_csv(join(READ_PATH,"windowed_train_label.csv"))
    
    logging.info("Windowed train data shape: %s", windowed_train_data.shape)
    logging.info("Windowed train label shape: %s", windowed_train_label.shape)

    return windowed_train_data, windowed_train_label


#Used for visualization experiments
def save_data(input_data, input_label, data_name, label_name):

    output_directory = "/content/drive/My Drive/PhD/Colab Notebooks/Monitoring_Colab/visualization/"
    data_folder = "data"

    if not os.path.isdir(output_directory + data_folder):
        os.mkdir(output_directory + data_folder)

    hf = h5py.File(output_directory+data_folder+"/"+data_name+'.h5', 'w')
    hf.create_dataset(data_name, data=input_data)
    hf.close()
    logging.info("Saved %s", data_name)

    input_label.to_csv(output_directory+data_folder+"/"+label_name+'.csv',index=False)
    logging.info("Saved %s", label_name)                     

#Used for visualization experiments
def load_data(data_name, label_name):

    output_directory = "/content/drive/My Drive/PhD/Colab Notebooks/Monitoring_Colab/visualization/"
    data_folder = "data"

    READ_PATH = output_directory + data_folder

    data = read_h5file(READ_PATH,data_name)
    label = pd.read_csv(join(READ_PATH,label_name+".csv"))

    return data,label

def prediction_wrapper(model,test_data,test_label,title=None,plot_name=None,class_names=None):
    preds = model.predict(test_data)
    print(classification_report(test_label, preds, target_names=class_names))
    disp = plot_confusion_matrix(model, test_data, test_label,
                                display_labels=class_names,
                                cmap=plt.cm.Oranges,
                                normalize='true')
    if title is None:
        disp.ax_.set_title("Test - Normalized Confusion Matrix")
    else:
        disp.ax_.set_title(title)
    
    if plot_name is not None:
        plt.savefig(output_directory+folder_name+"/"+plot_name)

### Feature Selection
def get_p_values_per_data(target_anomalous_features,target_healthy_features):
    #target_anomalous_features, _ = data_object.train_data(anomalous_features)
    #target_healthy_features, _ = data_object.train_data(healthy_features)
    if len(target_anomalous_features) == 0 or \
            len(target_healthy_features) == 0:
        logging.warn('Make sure that the excluded item is an application')
        return pd.Series([1] * len(healthy_features.columns),
                         healthy_features.columns, name='feature')
    
    p_values = [None] * len(target_healthy_features.columns)
    for f_idx, feature in enumerate(target_healthy_features.columns):
        p_values[f_idx] = ks_2samp(target_anomalous_features[feature],
                                   target_healthy_features[feature])[1]
    p_values_series = pd.Series(p_values, target_healthy_features.columns,
                                name='feature')
    return p_values_series

def benjamini_hochberg(p_values_df, apps, anomalies, fdr_level):
    n_features = len(p_values_df)
    selected_features = set()
    for app in apps:
        for anomaly in anomalies:
            col_name = '{}_{}'.format(app, anomaly)
            target_col = p_values_df[col_name].sort_values()
            K = list(range(1, n_features + 1))
            # Calculate the weight vector C
            weights = [sum([1 / i for i in range(1, k + 1)]) for k in K]
            # Calculate the vector T to compare to the p_value
            T = [fdr_level * k / n_features * 1 / w
                 for k, w in zip(K, weights)]
            # select
            selected_features |= set(target_col[target_col <= T].index)
    return selected_features
        
        
### DEPRECATED
#For old comparison experiments
def read_train_data(READ_PATH,folderName):

    if isdir(join(READ_PATH,folderName)):
        READ_PATH = join(READ_PATH,folderName)
        windowed_train_data = read_h5file(READ_PATH,'windowed_train_data')
        windowed_train_label = pd.read_csv(join(READ_PATH,"windowed_train_label.csv"))
        
        logging.info("Windowed train data shape: %s", windowed_train_data.shape)
        logging.info("Windowed train label shape: %s", windowed_train_label.shape)

        return windowed_train_data, windowed_train_label
    
# class WindowShopper: 

#     def __init__(self, data, labels, window_size = 64, trim=30, silent=False, selectedFeatures=None):
#         #Init
#         self.data = data
#         self.labels = labels
#         self.label_count = len(labels['anom'].unique()) #Automatically assuming anomaly classification
#         self.trim = trim
#         self.silent = silent
        
#         #Output shape
#         self.window_size = window_size
#         self.metric_count = len(data.columns)
#         self.metrics = data.columns
#         self.output_shape = (self.window_size, self.metric_count)    
        
#         self.selectedFeatures = selectedFeatures
        
#         if self.selectedFeatures is None:
#             logging.info("No feature selection, will generate an array without feature extraction")            
#             #Windowed data and labels
#             self.windowed_data = []
#             self.windowed_label = []
#             self._get_windowed_dataset()
#         else:     
#             logging.info("Feature selection, will generate a dataframe with feature extraction")            
#             #Windowed data and labels            
#             self.windowed_data = pd.DataFrame()
#             self.windowed_label = []            
#             self.feature_generator = TSFeatureGenerator(trim=self.trim)
#             self._get_windowed_dataset_fs()
#         #Prepare windows

    
#     #Not calling this but it is good to have
#     def _process_sample_count(self):
#         self.per_label_count = {x: 0 for x in self.labels[self.labels.columns[0]].unique()}
#         self.sample_count = 0
#         for node_id in self.data.index.get_level_values('node_id').unique():
#             counter = 0
#             cur_array = self.data.loc[node_id, :, :]
#             for i in range(self.trim, len(cur_array) - self.window_size - self.trim):
#                 counter += 1
#             self.sample_count += counter
#             self.per_label_count[self.labels.loc[node_id, self.labels.columns[0]]] += counter
            
#     def _get_windowed_dataset_fs(self):
        
#         feature_generator = TSFeatureGenerator(trim=0)        
        
#         #Iterate unique node_ids
#         for node_id in self.labels.index.unique():
#           # print(node_id)
#             cur_array = self.data[self.data.index.get_level_values('node_id') == node_id]
#             logging.info("Node id: %s",node_id)
#             temp_data = []
#             temp_label = []
#             #Iterate over application runtime
#             for i in range(self.trim, len(cur_array) - self.window_size - self.trim):
                
#                 extracted_node_data = feature_generator.transform(cur_array.iloc[i:i+self.window_size])
#                 self.windowed_data = pd.concat([self.windowed_data,extracted_node_data[self.selectedFeatures]])                
#                 self.windowed_label.append(self.labels.loc[node_id])
#                 if i % 100 == 0:
#                     logging.info("Windowed data shape: %s",self.windowed_data.shape)
#                     logging.info("Windowed label shape: %s",len(self.windowed_label))
                
#         if not self.silent:
#             logging.info("Windowed data shape: %s",self.windowed_data.shape)
#         #FIXME: column names might be in reverse order for HPAS data, Used app, anom for Cori data but it was anom,app
#         self.windowed_label = pd.DataFrame(np.asarray(self.windowed_label).reshape(len(self.windowed_label),2),columns=['app','anom'])
            
#         if not self.silent:
#             logging.info("Windowed label shape: %s",self.windowed_label.shape)
            

#     def _get_windowed_dataset(self):

#         #Iterate unique node_ids
#         for node_id in self.labels.index.unique():
#           # print(node_id)
#             cur_array = self.data.loc[node_id,:,:]        
          
#             temp_data = []
#             temp_label = []
#             #Iterate over application runtime
#             for i in range(self.trim, len(cur_array) - self.window_size - self.trim):

#                 self.windowed_data.append(cur_array.iloc[i:i+self.window_size].to_numpy(
#                   dtype=np.float32).reshape(self.output_shape))
#                 self.windowed_label.append(self.labels.loc[node_id])

#         self.windowed_data = np.dstack(self.windowed_data)
#         self.windowed_data = np.rollaxis(self.windowed_data,2)
#         if not self.silent:
#             logging.info("Windowed data shape: %s",self.windowed_data.shape)
#         #FIXME: column names might be in reverse order for HPAS data, Used app, anom for Cori data but it was anom,app
#         self.windowed_label = pd.DataFrame(np.asarray(self.windowed_label).reshape(len(self.windowed_label),2),columns=['app','anom'])
            
#         if not self.silent:
#             logging.info("Windowed label shape: %s",self.windowed_label.shape)

#     def return_windowed_dataset(self):

#         return self.windowed_data, self.windowed_label    
    
    
    
