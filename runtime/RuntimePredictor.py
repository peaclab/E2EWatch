#!/usr/bin/env python
# coding: utf-8

import datetime
import os
from os import listdir
from os.path import isfile, join
import re
import sys
import logging
from pathlib import Path
import argparse

from functools import reduce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from csv import writer

#ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

#UTILS
import sys
sys.path.insert(1,'/usr3/graduate/baksar/projectx/E2EWatch/utils/')
from utils import *


logging.basicConfig(format='%(asctime)s %(levelname)-7s %(message)s',
                    stream=sys.stderr, level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

class RuntimePredictor():
    
    
    def __init__(self, pickle_dir, 
                 pickle_name="eclipse_rf.pickle", 
                 feature_select = True, 
                 window_size = 60, 
                 granularity = 0,
                 DEBUG=True):
        
        """Init method
        
        - **parameters**, **types**, **return** and **return types**::
        
            :param pickle_dir: Directory that has the pickle model
            :param pickle_name: Pickle model name
            :param granularity: If the collected data is not collected per second granularity, specify this                
            :param window_size: If granularity is 0, it represents seconds, if granularity is 60, it represents minutes
        
        """
        
        self.pickle_dir = pickle_dir
        self.pickle_name = pickle_name
        self.window_size = window_size
        self.granularity = granularity
        self._load_model()
        self._load_scaler()
        self._load_anom_dict()
        self.feature_select = feature_select
        if self.feature_select:
            self._select_features()
        self.DEBUG = DEBUG
                
    def _select_features(self):
        
        #self.selected_features = open(self.pickle_dir / "selected_features.txt").read().splitlines()
        self.selected_features = pd.read_csv(self.pickle_dir / 'selected_features.csv')
        self.selected_features = self.selected_features['0'].values

        trial = []
        #When you use generate rolling features it doesn't add ::vmstat
        if self.feature_select:
            for feature in self.selected_features:
                trial.append(feature.split('::')[0])
            self.selected_features = trial
            
        logging.info('Loaded selected %d features', len(self.selected_features)) 
        
    def _load_model(self):
        '''Read the pickled model'''
        
        try:
            with open(self.pickle_dir / self.pickle_name, 'rb') as file:  
                self.model = pickle.load(file) 
                logging.info("Model loaded")
        except FileNotFoundError:
            logging.info("Model pickle doesn't exist")
            raise

    def _load_scaler(self):
        '''Read the pickled scaler'''
        
        try:
            with open(self.pickle_dir / 'scaler.pkl', 'rb') as file:  
                self.scaler = pickle.load(file) 
                logging.info("Scaler loaded")                
        except FileNotFoundError:
            logging.info("Scaler pickle doesn't exist")
            raise
            
            
    def _load_anom_dict(self):
        '''Read anom_dict to reverse encoding'''
        try:
            with open(self.pickle_dir / 'anom_dict.json') as f:
                self.anom_dict = json.load(f)              
        except FileNotFoundError:
            logging.info("Anomaly encoding dictionary doesn't exist")
            
    def _granularityAdjust(self,data,granularity=60):

        result = pd.DataFrame()
        for nid in data.index.get_level_values('node_id').unique():
            temp_data = data[data.index.get_level_values('node_id') == nid]
            temp_data = temp_data.iloc[ \
                (temp_data.index.get_level_values('timestamp').astype(int) -
                 int(temp_data.index.get_level_values('timestamp')[0])) \
                % granularity == 0]
            result = pd.concat([result,temp_data])

        return result    
            
    
    def predict_from_DF(self,runtime_data):
        
        """Process runtime monitoring data and make predictions with the existing model 

        Args:
            runtime_data: Dataframe that contains runtime HPC monitoring data

        Returns:
            Node by node runtime prediction results along with classifier confidence         
        """ 
        if not isinstance(runtime_data, pd.DataFrame):
            raise ValueError("should provide a pandas dataframe")
        
        #Drop NaN
        runtime_data.dropna(inplace=True)

        runtime_data['component_id'] = runtime_data['component_id'].astype(int)
        runtime_data = runtime_data.rename(columns={'component_id':'node_id'})
    
        round_factor = 1000 #Currently runtime data is collected every 60 seconds
        runtime_data['timestamp'] = round(runtime_data['timestamp'].astype(int) / round_factor)
        runtime_data['timestamp'] = runtime_data['timestamp'].astype(int) 
        runtime_data = runtime_data.set_index(['node_id','timestamp'])        
                
        #Per minute granularity data    
        if self.granularity != 0:
            runtime_data = self._granularityAdjust(runtime_data,granularity=60)
                
        ###Results will be stored in here
        node_results = pd.DataFrame()
        temp_feature_data = pd.DataFrame()
        
        logging.info("Preparing results for each node")
        for nid in runtime_data.index.get_level_values('node_id').unique():

            node_data = runtime_data.loc[nid,:,:]
                                    
            features = ['max', 'min', 'mean', 'std', 'skew', 'kurt','perc05', 'perc25', 'perc50', 'perc75', 'perc95'] 
            feature_train_data = pd.DataFrame()

            if self.granularity != 0:    
                feature_data = generate_rolling_features(node_data,features=features,window_size=self.window_size,trim=0)
            else:
                feature_data = generate_rolling_features(node_data,features=features,window_size=self.window_size,trim=60,skip=15)
#                 feature_data = generate_rolling_features(node_data,features=features,window_size=3,trim=0)                

            if self.feature_select:
                feature_data = feature_data[self.selected_features]                
                
            feature_data = pd.DataFrame(self.scaler.transform(feature_data),columns=feature_data.columns,index=feature_data.index)
                    
            #Testing pipeline
            preds_encoded = self.model.predict(feature_data)
            preds_prob = self.model.predict_proba(feature_data)

            preds = []
            for pred in preds_encoded:
                for key,value in self.anom_dict.items():
                    if value == pred:
                        preds.append(key)
                    
            node_data = feature_data

            timestamps = feature_data.index.get_level_values('timestamp').values            
            multiindex = list(zip(np.repeat(nid,len(timestamps)),timestamps))
            index = pd.MultiIndex.from_tuples(multiindex, names=['node_id', 'timestamp'])
            temp_results = pd.DataFrame(index=index)

            temp_results['preds'] = preds
            temp_results['prob'] = np.max(preds_prob,axis=1)
            node_results = pd.concat([node_results,temp_results])    

        return node_results                                                    
                               
    def TEST_predict_from_DF(self,runtime_data):
        
        """This is a TEST function for offline testing of the job data. 
        This function can be used to test job data inside the training/test set
        Format is not as same as runtime data returns by the RuntimeFramework so do NOT
        use for runtime results

        Args:
            runtime_data: Dataframe that contains runtime HPC monitoring data
            Dataframe can contain either one node data or multiple node data

        Returns:
            Node by node runtime prediction results along with classifier confidence         
        """ 
        if not isinstance(runtime_data, pd.DataFrame):
            raise ValueError("should provide a pandas dataframe")
                    
        new_columns = [column.split("::")[0] for column in runtime_data.columns]
        runtime_data.columns = new_columns
        
        if self.granularity != 0:                    
            runtime_data = self._granularityAdjust(runtime_data,granularity=60)
        node_results = pd.DataFrame()
        temp_feature_data = pd.DataFrame()
        
        logging.info("Preparing results for each node")
        for nid in runtime_data.index.get_level_values('node_id').unique():

            node_data = runtime_data.loc[nid,:,:]
                                    
            features = ['max', 'min', 'mean', 'std', 'skew', 'kurt','perc05', 'perc25', 'perc50', 'perc75', 'perc95'] 
            feature_train_data = pd.DataFrame()

            if self.granularity != 0:    
                feature_data = generate_rolling_features(node_data,features=features,window_size=self.window_size,trim=0)
#                 feature_data = generate_rolling_features(node_data,features=features,window_size=3,trim=0)                
            else:
                feature_data = generate_rolling_features(node_data,features=features,window_size=self.window_size,trim=60,skip=15)

            if self.feature_select:
                feature_data = feature_data[self.selected_features]                
                
            feature_data = pd.DataFrame(self.scaler.transform(feature_data),columns=feature_data.columns,index=feature_data.index)
                    
            #Testing pipeline
            preds_encoded = self.model.predict(feature_data)
            preds_prob = self.model.predict_proba(feature_data)

            preds = []
            for pred in preds_encoded:
                for key,value in self.anom_dict.items():
                    if value == pred:
                        preds.append(key)
            node_data = feature_data

            timestamps = feature_data.index.get_level_values('timestamp').values            
            
            multiindex = list(zip(np.repeat(nid,len(timestamps)),timestamps))
            index = pd.MultiIndex.from_tuples(multiindex, names=['node_id', 'timestamp'])
            temp_results = pd.DataFrame(index=index)

            temp_results['preds'] = preds
            temp_results['prob'] = np.max(preds_prob,axis=1)
            node_results = pd.concat([node_results,temp_results])    

        return node_results                                                    
