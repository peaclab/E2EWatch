#!/usr/bin/env python
# coding: utf-8


from abc import ABC, abstractmethod
from pathlib import Path
import os

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split


import numpy as np

import json 
import logging, sys
logging.basicConfig(format='%(asctime)s %(levelname)-7s %(message)s',
                    stream=sys.stderr, level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

#Python files
from config import Configuration
from utils import *

from tqdm import tqdm
import pickle 
import gc


class BaseDataset(ABC):
    """Anomaly detection dataset base class."""
    
    def __init__(self):
        logging.info("BaseDataset Class Initialization")        
        
        super().__init__()

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.anom_classes = None  # tuple with original class labels that define the outlier class

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    @abstractmethod
    def load_dataset(self):
        pass
        
    def __repr__(self):
        return self.__class__.__name__        

        
class HPCDeploymentDataset(BaseDataset):
        
    def __init__(self, conf):
        super().__init__()      
        logging.info("HPCDataset Class Initialization")
        self.conf = conf
        
    def read_label(self,TRAIN_DATA=True):
        """Read train or test label"""

        if TRAIN_DATA:
            raw_labels = pd.read_csv(self.conf['hdf_data_path'] / 'normal_labels.csv',index_col = ['node_id'])
        else:
            raw_labels = pd.read_csv(self.conf['hdf_data_path'] / 'anomaly_labels.csv',index_col = ['node_id'])
            
###INVESTIGATE: 
#         if self.conf['system'] == 'volta':
#             raw_labels = raw_labels[raw_labels['anom'] != 'linkclog']

        if self.conf['system'] == 'eclipse':
            raw_labels = raw_labels.rename(columns={'appname':'app','anomaly':'anom'})
            raw_labels = raw_labels[raw_labels['anom'] != 'iometadata']    

        return raw_labels   
    
    def granularity_adjust(self,data,granularity=60):

        result = pd.DataFrame()
        for nid in data.index.get_level_values('node_id').unique():
            temp_data = data[data.index.get_level_values('node_id') == nid]
            temp_data = temp_data.iloc[ \
                (temp_data.index.get_level_values('timestamp').astype(int) -
                 int(temp_data.index.get_level_values('timestamp')[0])) \
                % granularity == 0]
            result = pd.concat([result,temp_data])

        return result    
   
    
    def prepare_labels(self):
        
        """Prepares labels for each CV set"""        
        #Common encoder for train and test labels 
        encoder = MyEncoder()

        normal_labels = self.read_label(TRAIN_DATA=True)   
        anomaly_labels = self.read_label(TRAIN_DATA=False)
        
        logging.info("Anomaly label shape %s",anomaly_labels.shape)       
        logging.info("Anomaly label dist: \n%s",anomaly_labels['anom'].value_counts())

        logging.info("Normal shape %s",normal_labels.shape)       
        logging.info("Normal label dist: \n%s",normal_labels['anom'].value_counts())        
        
        if self.conf['system'] == 'eclipse':
            
            normal_labels.drop(normal_labels[normal_labels['app'] == 'miniAMR'].index,inplace=True)        
            anomaly_labels.drop(anomaly_labels[anomaly_labels['app'] == 'miniAMR'].index,inplace=True)         
            #After dropping the miniAMR, update the labels
            normal_labels.to_csv(self.conf['hdf_data_path'] / 'normal_labels.csv')
            anomaly_labels.to_csv(self.conf['hdf_data_path'] / 'anomaly_labels.csv')
            
#         elif self.conf['system'] == 'volta':
            
#             normal_labels.drop(normal_labels[normal_labels['app'] == 'miniAMR'].index,inplace=True)        
#             anomaly_labels.drop(anomaly_labels[anomaly_labels['app'] == 'miniAMR'].index,inplace=True)         
#             #After dropping the miniAMR, update the labels
#             normal_labels.to_csv(self.conf['hdf_data_path'] / 'normal_labels.csv')
#             anomaly_labels.to_csv(self.conf['hdf_data_path'] / 'anomaly_labels.csv')
                                    
        
        all_labels = pd.concat([normal_labels, anomaly_labels])
        all_labels = encoder.fit_transform(all_labels)
        
#         normal_labels = encoder.transform(normal_labels)         
#         anomaly_labels = encoder.transform(anomaly_labels) 
        
        anom_dict = encoder.anom_dict
        app_dict = encoder.app_dict        
                
        if not (self.conf['experiment_dir'] / ('anom_dict.json')).exists():    

            json_dump = json.dumps(anom_dict)
            f_json = open(self.conf['experiment_dir'] / "anom_dict.json","w")
            f_json.write(json_dump)
            f_json.close()         

            json_dump = json.dumps(app_dict)
            f_json = open(self.conf['experiment_dir'] / "app_dict.json","w")
            f_json.write(json_dump)
            f_json.close()
        else:
            logging.info("Anom and app dict already exists")         
                                  
        #CHANGE SPLIT PARAMETER, if you want more than 5 folds 
        skf = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)

        n_nodeids = all_labels.shape[0]
        cv_index = 0
        logging.info("Generating labels for CV folders")
        for train_index, test_index in skf.split(np.zeros(n_nodeids),all_labels['anom']):
            logging.info("CV fold %s",cv_index)
            cv_path = self.conf['experiment_dir'] / ("CV_" + str(cv_index))                        
            #cv_path = conf['output_training_dir'] / ("CV_" + str(cv_index))
            if not cv_path.exists():
                cv_path.mkdir(parents=True)
                
            train_label = all_labels.iloc[train_index]
            test_label = all_labels.iloc[test_index]
                                    
            logging.info("Train data class dist \n%s\n",train_label['anom'].value_counts())    
            logging.info("Train data app dist \n%s\n",train_label['app'].value_counts())                
            logging.info("Test data class dist \n%s\n",test_label['anom'].value_counts())                 
            logging.info("Test data app dist \n%s\n",test_label['app'].value_counts())                 

            train_label.to_csv(cv_path / 'train_label.csv')
            test_label.to_csv(cv_path / 'test_label.csv')            
            
            cv_index += 1                    
                              
    @abstractmethod
    def prepare_data(self):
        pass    
        
class EclipseDeploymentDataset(HPCDeploymentDataset):
    """
        EclipseDeploymentDataset class for datasets for Eclipse HPC monitoring data and designed for deployment to Eclipse
        #TODO: Update
    """
    
    def __init__(self, conf):
        super().__init__(conf)
        logging.info("EclipseDeploymentDataset Class Initialization")        
        
        self.normal_class = [0]
        self.anom_classes = [1,2,3,4]
        self.classes = [0,1,2,3,4]
                           
    def _extract_features_and_windowize(self,all_data,selected_label):
        """
            Feature extraction and windowing combination
            It returns timestamp as an index and concats all the timestamp values below each other
        """
        features = ['max', 'min', 'mean', 'std', 'skew', 'kurt','perc05', 'perc25', 'perc50', 'perc75', 'perc95'] 
        
        label = pd.DataFrame(columns=['app','anom'])
        label.index.name = 'node_id'
        node_job_len = {}
        data = []
        
        counter = 0        
        for node_id in selected_label.index:
            
            #Minute granularity
            if self.conf['granularity'] != 0:    
                try:
                    temp_extracted_data = generate_rolling_features(all_data.loc[node_id],features=features,window_size=self.conf['window_size'],trim=0,skip=None)
                except:
                    logging.info("Key doesn't exist %s",node_id)
            #Seconds granularity                    
            else:
                try:
                    temp_extracted_data = generate_rolling_features(all_data.loc[node_id],features=features,window_size=self.conf['window_size'],trim=60,skip=15)                
                except:
                    logging.info("Key doesn't exist %s",node_id)
                    
            timestamps = temp_extracted_data.index.get_level_values('timestamp').values            
            multiindex = list(zip(np.repeat(node_id,len(timestamps)),timestamps))
            new_index = pd.MultiIndex.from_tuples(multiindex, names=['node_id', 'timestamp'])
            temp_extracted_data.index = new_index
            data.append(temp_extracted_data)                     

#             new_index = str(node_id) + '_' + temp_extracted_data.index        
#             temp_extracted_data.index = new_index
#             temp_extracted_data.index.name = 'node_id'                                     
#             data.append(temp_extracted_data)
            
            node_job_len[node_id] = temp_extracted_data.shape[0]             

            if (counter % 300) == 0:
                logging.info(counter)
                
            counter += 1
            
        #Repeat the rows number of specified times
        selected_label['times'] = node_job_len.values()
        label = selected_label.loc[selected_label.index.repeat(selected_label.times)]
        label.drop(columns=['times'],inplace=True)        
        
        data = pd.concat(data, axis=0)

        logging.info("Feature extracted shape %s",data.shape)
        logging.info("Feature extracted label shape %s",label.shape)
        
        return data, label
   
                            
    def prepare_data(self):
        
        """Prepares data according to the labels"""
        
        CV_NUM_STR = ("CV_" + str(self.conf['cv_fold']))
        logging.info("Preparing data for CV Fold %s",self.conf['cv_fold'])
        
        #Load the labels first for a quick check
        anomaly_labels = pd.read_csv(self.conf['hdf_data_path'] / 'anomaly_labels.csv',index_col=['node_id'])
        anomaly_labels = anomaly_labels.rename(columns={'appname':'app','anomaly':'anom'})
        
        normal_labels = pd.read_csv(self.conf['hdf_data_path'] / 'normal_labels.csv',index_col=['node_id'])
        normal_labels = normal_labels.rename(columns={'appname':'app','anomaly':'anom'})
        
        
        train_label = pd.read_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'train_label.csv',index_col=['node_id'])
        
        logging.info("Train label shape %s",train_label.shape)       
        logging.info("Train data label dist: \n%s",train_label['anom'].value_counts())
        
        test_label = pd.read_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'test_label.csv',index_col=['node_id'])        
        
        logging.info("Test label shape %s",test_label.shape)   
        logging.info("Test data label dist: \n%s",test_label['anom'].value_counts()) 
                                        
        assert len(anomaly_labels) + len(normal_labels) == len(train_label) + len(test_label)            
        logging.info("Label assertion passed")                
        
        #These two code blocks read previously saved train and test data - do NOT confuse
        anomaly_data = pd.read_hdf(self.conf['hdf_data_path'] / 'anomaly_data.hdf','anomaly_data')
        anomaly_data = anomaly_data[[x for x in anomaly_data.columns if 'per_core' not in x]]
                    
        #FIXME: Doing that because there are some miniAMR runs that we are dropping from the labels
        #They are coming from 4 node application runs                
##Alternative fix instead of loc, however didn't work with multiindex        
#         anomaly_data = anomaly_data.reindex(anomaly_labels.index)        
        anomaly_data = anomaly_data.loc[anomaly_labels.index]             
        anomaly_data.dropna(inplace=True)        
        logging.info("Anomaly data shape: %s",anomaly_data.shape)

        normal_data = pd.read_hdf(self.conf['hdf_data_path'] / 'normal_data.hdf','normal_data')
        normal_data = normal_data[[x for x in normal_data.columns if 'per_core' not in x]]
                    
        #FIXME: Doing that because there are some miniAMR runs that we are dropping from the labels
        #They are coming from 4 node application runs
##Alternative fix instead of loc, however didn't work with multiindex        
#         normal_data = normal_data.reindex(anomaly_labels.index)        
        normal_data = normal_data.loc[normal_labels.index]                      
        normal_data.dropna(inplace=True)         
        logging.info("Normal data shape: %s",normal_data.shape)

        all_data = pd.concat([normal_data,anomaly_data])
        #all_data = all_data.dropna()
        logging.info("Is NaN: %s",np.any(np.isnan(all_data)))
                        
        all_labels = pd.concat([train_label,test_label])
        
        all_data_unique_nids = list(all_data.index.get_level_values('node_id').unique())
        all_labels = all_labels.loc[all_data_unique_nids]
        all_label_unique_nids = list(all_labels.index)

        assert all_data_unique_nids == all_label_unique_nids

        logging.info("Data shape: %s",all_data.shape)
        logging.info("Unique NodeIds in Data: %s",len(all_data_unique_nids))
        logging.info("Unique NodeIds in Label: %s",len(all_label_unique_nids))    
        
        #Update the train and test labels according to the dropped labels
        train_label = train_label[train_label.index.isin(all_labels.index)]
        train_label.to_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'train_label.csv')

        test_label = test_label[test_label.index.isin(all_labels.index)]
        test_label.to_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'test_label.csv')        
                            
        if self.conf['feature_select']:
            cache_path = self.conf['experiment_dir'] / '{}_feature_p_values.hdf'.format(self.conf['system'])
                        
            apps = set(all_labels['app'].unique())
            anomalies = self.anom_classes
            
            if cache_path.exists():
                logging.info('Retrieving feature p-values')
                p_values_df = pd.read_hdf(cache_path)
            else:    
                
                logging.info('Calculating feature p-values')
                all_columns = all_data.columns                
                                
                p_values_df = pd.DataFrame()
                pbar = tqdm(total=len(apps)*len(anomalies))

                for app in apps:
                    n_anomalous_runs = len(all_labels[all_labels['app'] == app][all_labels['anom'] != self.normal_class[0]])
                    
                    #Select labels with "normal" label
                    selected_labels = all_labels.loc[all_labels['app'] == app].loc[all_labels['anom'] == self.normal_class[0]]
                    healthy_features, _ = self._extract_features_and_windowize(all_data,selected_labels)
    
                    for anomaly in anomalies:
                        col_name = '{}_{}'.format(app, anomaly)
                        
                        selected_labels = all_labels.loc[all_labels['app'] == app].loc[all_labels['anom'] == anomaly]
                        anomalous_features, _ = self._extract_features_and_windowize(all_data,selected_labels)
                        p_values_df[col_name] = get_p_values_per_data(anomalous_features,healthy_features)

                        pbar.update(1)   

                p_values_df.to_hdf(cache_path,key='key')
            fdr_level = 0.01
            self.selected_features = benjamini_hochberg(p_values_df, apps, anomalies, fdr_level)
            pd.DataFrame(self.selected_features).to_csv(self.conf['experiment_dir'] / 'selected_features.csv')
            logging.info('Selected %d features', len(self.selected_features))
        else:
            logging.info("No feature selection")

                        
        logging.info("---- TRAIN DATA ----")                        
        #logging.info("Generating windows")

        processed_train_data, processed_train_label = self._extract_features_and_windowize(all_data,train_label)

        ### Save data as hdf
        logging.info("Saving training data")
        processed_train_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR / 'train_data.hdf',key='train_data',complevel=9)     
        processed_train_label.to_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'train_label.csv')

        logging.info("Train data shape %s",processed_train_data.shape)
        logging.info("Train label shape %s",processed_train_label.shape)     

        del processed_train_data
        gc.collect() 

        logging.info("---- TEST DATA ----")
        logging.info("Generating windows")

        processed_test_data, processed_test_label = self._extract_features_and_windowize(all_data,test_label)

        logging.info("Saving test data")
        processed_test_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR / 'test_data.hdf',key='test_data',complevel=9)
        processed_test_label.to_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'test_label.csv')      

        logging.info("Test data shape %s",processed_test_data.shape)    
        logging.info("Test label shape %s",processed_test_label.shape)          

        logging.info("Saved data and labels\n")

        del processed_test_data
        gc.collect()         
                             
                                                                                               
    def load_dataset(self,scaler='MinMax',scalerSave=True): 
        from pickle import dump
        
        CV_NUM_STR = ("CV_" + str(self.conf['cv_fold'])) 
        DATA_PATH = self.conf['experiment_dir'] / CV_NUM_STR
        
        selected_features = pd.read_csv(self.conf['experiment_dir'] / 'selected_features.csv')

        X_train = pd.read_hdf(DATA_PATH / 'train_data.hdf',key='train_data')
        X_train.reset_index(drop=True)
        
        if self.conf['feature_select']: 
            X_train = X_train[selected_features['0'].values]
            
        self.y_train = pd.read_csv(DATA_PATH / 'train_label.csv',index_col=['node_id'])
        #self.y_train = self.y_train.loc[X_train.index]
        self.y_train.index.name = 'node_id'        


        X_test = pd.read_hdf(DATA_PATH / 'test_data.hdf',key='test_data')
        X_test.reset_index(drop=True)       
        
        if self.conf['feature_select']: 
            X_test = X_test[selected_features['0'].values]
        
        self.y_test = pd.read_csv(DATA_PATH / 'test_label.csv',index_col=['node_id'])
        #self.y_test = self.y_test.loc[X_test.index]
        self.y_test.index.name = 'node_id'         

        logging.info("Train data shape %s",X_train.shape)
        logging.info("Train label shape %s",self.y_train.shape)

        logging.info("Test data shape %s",X_test.shape)
        logging.info("Test label shape %s",self.y_test.shape)        

        with open(self.conf['experiment_dir'] / ('anom_dict.json')) as f:
            ANOM_DICT = json.load(f)        

        if scaler == 'Standard':

            # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)        
            scaler = StandardScaler().fit(X_train)
            self.X_train = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns,index=X_train.index)
            self.X_test = pd.DataFrame(scaler.transform(X_test),columns=X_train.columns,index=X_test.index)
            
        elif scaler == 'MinMax':
            # Scale to range [0,1]
#             all_data = pd.concat([X_train,X_test])
#             minmax_scaler = MinMaxScaler().fit(all_data)            
            minmax_scaler = MinMaxScaler().fit(X_train)
    
            if scalerSave:
                dump(minmax_scaler, open(self.conf['model_dir'] / 'scaler.pkl','wb'))
                logging.info("Scaler is saved in the model directory")    
            self.X_train = pd.DataFrame(minmax_scaler.transform(X_train),columns=X_train.columns,index=X_train.index)
            self.X_test = pd.DataFrame(minmax_scaler.transform(X_test),columns=X_test.columns,index=X_test.index)
            
#             self.X_train = all_data.loc[self.y_train.index]
#             self.X_test = all_data.loc[self.y_test.index]     

        return self.X_train, self.y_train, self.X_test, self.y_test        


class VoltaDeploymentDataset(HPCDeploymentDataset):
    """
        VoltaDeploymentDataset class for datasets for Volta HPC monitoring data
        This class is designed to experiment with equal number of instances in each anomaly and separate normal labels
        Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
        to also return the semi-supervised target as well as the index of a data sample.
    """
    
    def __init__(self, conf):
        super().__init__(conf)
        logging.info("VoltaSampledDataset Class Initialization")        
        
        self.normal_class = [0]
        self.anom_classes = [1,2,3,4]
        self.classes = [0,1,2,3,4]
                                                     
    def prepare_data(self):
        
        """Prepares data according to the labels"""
        
        CV_NUM_STR = ("CV_" + str(self.conf['cv_fold']))
        logging.info("Preparing data for CV Fold %s",self.conf['cv_fold'])
        
                
        #These two code blocks read previously saved train and test data - do NOT confuse
        anomaly_data = pd.read_hdf(self.conf['hdf_data_path'] / 'anomaly_data.hdf','anomaly_data')
        anomaly_data = anomaly_data[[x for x in anomaly_data.columns if 'per_core' not in x]]
        logging.info("Anomaly data shape: %s",anomaly_data.shape)

        normal_data = pd.read_hdf(self.conf['hdf_data_path'] / 'normal_data.hdf','normal_data')
        normal_data = normal_data[[x for x in normal_data.columns if 'per_core' not in x]]
        logging.info("Normal data shape: %s",normal_data.shape)

        all_data = pd.concat([normal_data,anomaly_data])
        logging.info("Full data shape: %s",all_data.shape)

        all_data = all_data.dropna()
        logging.info("Is NaN: %s",np.any(np.isnan(all_data)))
        logging.info("Data shape: %s",all_data.shape)

        CV_NUM_STR = ("CV_" + str(self.conf['cv_fold']))
        
        train_label = pd.read_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'train_label.csv',index_col=['node_id'])
        train_data = all_data[all_data.index.get_level_values('node_id').isin(train_label.index)]
        logging.info("Train data shape %s",train_data.shape)  
        logging.info("Train label shape %s",train_label.shape)   
        
        test_label = pd.read_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'test_label.csv',index_col=['node_id'])
        test_data = all_data[all_data.index.get_level_values('node_id').isin(test_label.index)]
        logging.info("Test data shape %s",test_data.shape)    
        logging.info("Test label shape %s",test_label.shape)  
        
        logging.info("Train data label dist: \n%s",train_label['anom'].value_counts())
        logging.info("Test data label dist: \n%s",test_label['anom'].value_counts())            
                        
        if self.conf['feature_select']:
            cache_path = self.conf['experiment_dir'] / '{}_feature_p_values.hdf'.format(self.conf['system'])
            all_labels = pd.concat([train_label,test_label])            
            apps = set(all_labels['app'].unique())
            anomalies = self.anom_classes
            
            if cache_path.exists():
                logging.info('Retrieving feature p-values')
                p_values_df = pd.read_hdf(cache_path)
            else:    
                
                logging.info('Calculating feature p-values')
                all_columns = train_data.columns
                all_labels = pd.concat([train_label,test_label])
                                
                p_values_df = pd.DataFrame()
                pbar = tqdm(total=len(apps)*len(anomalies))

                for app in apps:
                    n_anomalous_runs = len(all_labels[all_labels['app'] == app][all_labels['anom'] != self.normal_class[0]])

                    healthy_node_ids = set(list(all_labels[all_labels['app'] == app][all_labels['anom'] == self.normal_class[0]].index))
                    temp_node_data = all_data[all_data.index.get_level_values('node_id').isin(healthy_node_ids)]

                    
                    feature_generator = TSFeatureGenerator(trim=30)
                    healthy_features = feature_generator.transform(temp_node_data)

                    for anomaly in anomalies:
                        col_name = '{}_{}'.format(app, anomaly)
                        anomalous_node_ids = set(list(all_labels[all_labels['app'] == app][all_labels['anom'] == anomaly].index))
                        temp_node_data = all_data[all_data.index.get_level_values('node_id').isin(anomalous_node_ids)]

                        anomalous_features = feature_generator.transform(temp_node_data)

                        p_values_df[col_name] = get_p_values_per_data(anomalous_features,healthy_features)

                        pbar.update(1)   

                p_values_df.to_hdf(cache_path,key='key')
            fdr_level = 0.01
            selected_features = benjamini_hochberg(p_values_df, apps, anomalies, fdr_level)
            pd.DataFrame(selected_features).to_csv(self.conf['experiment_dir'] / 'selected_features.csv')
            logging.info('Selected %d features', len(selected_features))
            
            logging.info('Selected %d features', len(selected_features))
        else:
            logging.info("No feature selection")
                                
        
        
        if self.conf['feature_extract']:
            #FIXME: It might need an update for TPDS data 
            logging.info("Generating features")    
            feature_generator = TSFeatureGenerator(trim=0) #Don't change the trim
            
            train_data = feature_generator.transform(train_data)
            test_data = feature_generator.transform(test_data)
            
        ### Save data as hdf
        logging.info("Saving training data")
        train_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR / 'train_data.hdf',key='train_data',complevel=9)
        
        train_label = train_label.loc[train_data.index]
        train_label.index.name = 'node_id'        
        train_label.to_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'train_label.csv')


        logging.info("Saving test data")
        test_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR / 'test_data.hdf',key='test_data',complevel=9)
        
        test_label = test_label.loc[test_data.index]
        test_label.index.name = 'node_id'      
        test_label.to_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'test_label.csv')                 
        
        logging.info("Train data shape %s",train_data.shape)
        logging.info("Train label shape %s",train_label.shape)        
        logging.info("Test data shape %s",test_data.shape)    
        logging.info("Test label shape %s",test_label.shape)          
           
        logging.info("Saved data and labels\n")
        logging.info("Train data label dist: \n%s",train_label['anom'].value_counts())
        logging.info("Test data label dist: \n%s",test_label['anom'].value_counts())              
            
        
        
    def load_dataset(self,scaler='MinMax',time=False,borghesi=False): 
           
        if not time:
            
            CV_NUM_STR = ("CV_" + str(self.conf['cv_fold'])) 
            DATA_PATH = self.conf['experiment_dir'] / CV_NUM_STR

            ###Read training data
            if borghesi:
                X_train = pd.read_hdf(DATA_PATH / 'train_data_borghesi.hdf',key='train_data_borghesi')
                self.y_train = pd.read_csv(DATA_PATH / 'train_label_borghesi.csv',index_col=['node_id'])
                self.y_train = self.y_train.loc[X_train.index]
                self.y_train.index.name = 'node_id'        
                
            else:
                X_train = pd.read_hdf(DATA_PATH / 'train_data.hdf',key='train_data')
                self.y_train = pd.read_csv(DATA_PATH / 'train_label.csv',index_col=['node_id'])
                self.y_train = self.y_train.loc[X_train.index]
                self.y_train.index.name = 'node_id'        

            ###Read test data     
            if borghesi:
                X_train = pd.read_hdf(DATA_PATH / 'test_data_borghesi.hdf',key='test_data_borghesi')
                self.y_train = pd.read_csv(DATA_PATH / 'test_label_borghesi.csv',index_col=['node_id'])
                self.y_train = self.y_train.loc[X_train.index]
                self.y_train.index.name = 'node_id'        
                        
            else:
                X_test = pd.read_hdf(DATA_PATH / 'test_data.hdf',key='test_data')
                self.y_test = pd.read_csv(DATA_PATH / 'test_label.csv',index_col=['node_id'])
                self.y_test = self.y_test.loc[X_test.index]
                self.y_test.index.name = 'node_id'        


            logging.info("Train data shape %s",X_train.shape)
            logging.info("Train label shape %s",self.y_train.shape)

            logging.info("Test data shape %s",X_test.shape)
            logging.info("Test label shape %s",self.y_test.shape)        

            with open(self.conf['experiment_dir'] / ('anom_dict.json')) as f:
                ANOM_DICT = json.load(f)        

            if scaler == 'Standard':

                # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)        
                scaler = StandardScaler().fit(X_train)
                self.X_train = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns,index=X_train.index)
                self.X_test = pd.DataFrame(scaler.transform(X_test),columns=X_train.columns,index=X_test.index)

            elif scaler == 'MinMax':
                # Scale to range [0,1]
                minmax_scaler = MinMaxScaler().fit(X_train)
                self.X_train = pd.DataFrame(minmax_scaler.transform(X_train),columns=X_train.columns,index=X_train.index)
                self.X_test = pd.DataFrame(minmax_scaler.transform(X_test),columns=X_test.columns,index=X_test.index)


            return self.X_train, self.y_train, self.X_test, self.y_test        

        else:
            
            CV_NUM_STR = ("CV_" + str(self.conf['cv_fold'])) 
            DATA_PATH = self.conf['experiment_dir'] / CV_NUM_STR

            ###Read training data
            self.X_train = pd.read_hdf(DATA_PATH / 'train_data.hdf',key='train_data')

            self.y_train = pd.read_csv(DATA_PATH / 'train_label.csv',index_col=['node_id'])            
            
            
            ###Read test data            
            self.X_test = pd.read_hdf(DATA_PATH / 'test_data.hdf',key='test_data')

            self.y_test = pd.read_csv(DATA_PATH / 'test_label.csv',index_col=['node_id'])     
            
            
            logging.info("Train data shape %s",self.X_train.shape)
            logging.info("Train label shape %s",self.y_train.shape)

            logging.info("Test data shape %s",self.X_test.shape)
            logging.info("Test label shape %s",self.y_test.shape)               
            
            
            return self.X_train, self.y_train, self.X_test, self.y_test             
        
               
    
