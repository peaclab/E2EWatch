#!/usr/bin/env python
# coding: utf-8


import logging
import sys,os
from pathlib import Path
sys.path.insert(1,'/usr3/graduate/baksar/projectx/E2EWatch/utils/')
from config import Configuration
from utils import *
from datasets import EclipseDeploymentDataset
import seaborn as sns


import argparse
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier 

from time import time
from collections import defaultdict

import xgboost as xgb
import lightgbm as lgbm
import pickle 

def main():
        
    logging.basicConfig(format='%(asctime)s %(levelname)-7s %(message)s',
                        stream=sys.stderr, level=logging.DEBUG)
    logging.info("Running main function, master!")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--windowSize")    
    parser.add_argument("--featureSelect")   
    parser.add_argument("--system")   
    parser.add_argument("--modelName")    

    args = parser.parse_args()
                
    WINDOW_SIZE = args.windowSize
    FS = int(args.featureSelect)
    SYSTEM =  str(args.system)
    MODEL_NAME = str(args.modelName)
    MODELS = ['lgbm','xgboost','rf']
    
    if not MODEL_NAME in MODELS:
        logging.info("Model does not supported %s", MODEL_NAME)    
        raise
            
    logging.info("Predict function for Window Size %s, feature_select %s, Model Name %s", WINDOW_SIZE,FS,MODEL_NAME)    

    #CV_FOLDS = [0,1,2,3,4]
    CV_FOLDS = [0]
        
    for CV_INDEX in CV_FOLDS:
               
        
        conf = Configuration(ipython=True,
                             overrides={
                                 'system' : SYSTEM,
                                 'operation':'label_generate', 
                                 'exp_name':'final_window_{}sec'.format(WINDOW_SIZE),
                                 #'hdf_data_path': Path('/projectnb/peaclab-mon/aksar/datasets/eclipse_final_minute_hdfs'),     
                                 'model_config': 'random_forest',                                 
                                 #Label Generation
                                 'num_split': 5,
                                 #Data Generation                          
                                 'cv_fold':int(CV_INDEX), #Required only for data_generate and pipeline options
                                 'granularity': 0,
                                 'windowing': True,                         
                                 'window_size' : int(WINDOW_SIZE),                                  
                                 'feature_select': True if FS else False,
                                 'feature_extract': True,
                             })

        eclipseDataset = EclipseDeploymentDataset(conf)
        
        logging.info("Experiment name: %s", str(conf['experiment_dir']).split("/")[-1])           
        logging.info("CV INDEX: %s", str(CV_INDEX))                
        logging.info("HDF PATH %s",conf['hdf_data_path'])
        
        with open(conf['experiment_dir'] / ('anom_dict.json')) as f:
            ANOM_DICT = json.load(f) 

        with open(conf['experiment_dir'] / ('app_dict.json')) as f:
            APP_DICT = json.load(f)          
        
                       
        X_train, y_train, X_test, y_test = eclipseDataset.load_dataset() 
                
        y_train.reset_index(inplace=True)
        y_test.reset_index(inplace=True)     
                
        for APP_ENCODED in [0,1,2,3,4,5]:
            
            #Refresh this everytime
            MODEL_NAME = str(args.modelName)

            UNKNOWN = set([APP_ENCODED])
            UNKNOWN_APPNAME = str([appname for appname, encoded in APP_DICT.items() if encoded == APP_ENCODED][0])

            logging.info('Unknown app name %s',UNKNOWN_APPNAME)

            ALL_APPS = set(y_train['app'].unique())
            TRAIN_APPS = list(ALL_APPS - UNKNOWN)
            TEST_APPS = list(UNKNOWN)


            logging.info('Training apps %s',TRAIN_APPS)
            logging.info('Testing apps %s',TEST_APPS)


            temp_y_train = y_train[y_train['app'].isin(TRAIN_APPS)]
            temp_X_train = X_train.iloc[temp_y_train.index]
            #temp_X_train = X_train.loc[temp_y_train.index]
            print("Apps:",temp_y_train['app'].unique())
            print("Anoms:",temp_y_train['anom'].unique())    


            temp_y_test = y_test[y_test['app'].isin(TEST_APPS)]    
            temp_X_test = X_test.iloc[temp_y_test.index]
            #temp_X_test = X_test.loc[temp_y_test.index]
            print("Apps:",temp_y_test['app'].unique())
            print("Anoms:",temp_y_test['anom'].unique())    


            if conf['feature_select']:
                MODEL_NAME = MODEL_NAME + '-fs'

            logging.info('Model name: %s',MODEL_NAME)
            
            with open(conf['tuning_dir'] / (MODEL_NAME + "_best_params.json"), 'r') as json_file:
                best_params = json.load(json_file)            


            if 'rf' in MODEL_NAME:
                
                tuned_model=RandomForestClassifier(n_estimators= 250,#int(best_params['n_estimators']),
                                                     max_depth=10,#int(best['max_depth']),
                                                     min_samples_leaf=int(best_params['min_samples_leaf']),
                                                     min_samples_split=int(best_params['min_samples_split']),
                                                     n_jobs=mp.cpu_count()
                                                  )      
                
            elif 'lgbm' in MODEL_NAME:

                tuned_model=lgbm.LGBMClassifier(n_estimators=250,                           
                                                 max_depth=int(best_params['max_depth']),
                                                 num_leaves=int(best_params['num_leaves']),
                                                 colsample_bytree=float(best_params['colsample_bytree']),                                  
                                                 #learning_rate=float(best_params['learning_rate']),
                                                 n_jobs=mp.cpu_count()
                                               )      
                
                
            elif 'xgboost' in MODEL_NAME:

                tuned_model=xgb.XGBClassifier(n_estimators=250,    
                                              max_depth=int(best_params['max_depth']),
                                              num_leaves=int(best_params['num_leaves']),
                                              colsample_bytree=float(best_params['colsample_bytree']),                                  
                                              #learning_rate=float(best_params['learning_rate']),                                  
                                              n_jobs=mp.cpu_count()
                                             )                      

            logging.info("Fitting!") 
            tuned_model.fit(temp_X_train.values,temp_y_train['anom'].astype('int'))                          
            preds = tuned_model.predict(temp_X_test.values)
            logging.info("Done with training and testing {}".format(MODEL_NAME))
            
            logging.info("Generating report!")
            ### Saves classification report where all apps are combined
            logging.info("########################### BEST ###########################") 
            report_dict = classification_report(y_true=temp_y_test['anom'].astype('int'), y_pred=preds, labels=temp_y_test['anom'].astype('int'))
            print(report_dict)
            logging.info("########################### BEST ###########################") 
            report_dict = classification_report(y_true=temp_y_test['anom'].astype('int'), y_pred=preds, labels=temp_y_test['anom'].astype('int'),output_dict=True)                

            
            RESULT_NAME = 'UnknownApp_' + UNKNOWN_APPNAME + '_' + MODEL_NAME
            logging.info(RESULT_NAME)        

            ### Plot confusion matrix and save the diagnosing results and FAR & ADR
            if WINDOW_SIZE != 0:
                if conf['granularity'] != 0:
                    analysis_wrapper_multiclass(temp_y_test['anom'].astype('int'),preds,
                                                conf,CV_INDEX,RESULT_NAME + '_test',
                                               name_cm='{} min Windowing - {}'.format(WINDOW_SIZE,SYSTEM))        
                else:
                    analysis_wrapper_multiclass(temp_y_test['anom'].astype('int'),preds,
                                                conf,CV_INDEX,RESULT_NAME + '_test',
                                               name_cm='{} sec Windowing - {}'.format(WINDOW_SIZE,SYSTEM))

            else: 
                analysis_wrapper_multiclass(temp_y_test['anom'].astype('int'),preds,
                                        conf,CV_INDEX,RESULT_NAME + '_test',
                                       name_cm='No Windowing - {}'.format(SYSTEM))            
        
        logging.info("########################### DONE ###########################")
if __name__ == '__main__':
    main()




