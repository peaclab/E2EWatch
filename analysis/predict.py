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
import pickle 

import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier 

from time import time
from collections import defaultdict
import pickle 

#Hyperparameter Tuning
from hyperopt import tpe,hp,Trials
from hyperopt.fmin import fmin
from sklearn.model_selection import cross_val_score, StratifiedKFold

import xgboost as xgb
import lightgbm as lgbm

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
        
    #CV_FOLDS = [0,1,2,3,4]
    CV_FOLDS = [0]
     # rf will become -> rf_test_alert_dict.json while saving
    
    for CV_INDEX in CV_FOLDS:
        
        
        logging.info("Predict function for Window Size %s, Feature Selection %s", WINDOW_SIZE, FS)
        
        conf = Configuration(ipython=True,
                             overrides={
                                 'system' : SYSTEM,
                                 'operation':'label_generate', 
                                 'exp_name':'final_window_{}sec'.format(WINDOW_SIZE),
                                 #'hdf_data_path': Path('/projectnb/peaclab-mon/aksar/datasets/eclipse_final_hdfs'),     
                                 'model_config': 'random_forest',                                 
                                 #Label Generation
                                 'num_split': 5,
                                 #Data Generation                          
                                 'cv_fold':int(CV_INDEX), #Required only for data_generate and pipeline options
                                 #'granularity': 60,
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
        
        
        if conf['feature_select']:
            MODEL_NAME = MODEL_NAME + '-fs'      
            
        logging.info('Model name: %s',MODEL_NAME)            
        
        #Read the saved params 
        with open(conf['tuning_dir'] / (MODEL_NAME + "_best_params.json"), 'r') as json_file:
            best_params = json.load(json_file)

        time_dict = defaultdict()

        logging.info("Setting up the classifier!")
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
            
            
        time_dict[MODEL_NAME] = {}

        logging.info("Fitting!") 
        start_time = time()
        tuned_model.fit(X_train.values,y_train['anom'].astype('int'))  
        elapsed_time = time() - start_time
        logging.info("Training time %s",np.round(elapsed_time,3))
        time_dict[MODEL_NAME]['training_time'] = elapsed_time        

        start_time = time()   
        preds = tuned_model.predict(X_test.values)
        elapsed_time = time() - start_time
        logging.info("Inference time %s seconds",np.round(elapsed_time,3))
        time_dict[MODEL_NAME]['inference_time'] = elapsed_time

        

        logging.info("Generating report!")
        ### Saves classification report where all apps are combined
        logging.info("########################### BEST ###########################") 
        report_dict = classification_report(y_true=y_test['anom'].astype('int'), y_pred=preds, labels=y_test['anom'].unique())
        print(report_dict)
        logging.info("########################### BEST ###########################") 
        report_dict = classification_report(y_true=y_test['anom'].astype('int'), y_pred=preds, labels=y_test['anom'].unique(),output_dict=True)       
        
        
        logging.info("Saving the timer!")
        with open(conf['results_dir'] / '{}_time_dict.json'.format(MODEL_NAME), 'w') as file:  
            json.dump(time_dict, file)

        logging.info("Saving the model!")
        with open(conf['model_dir'] / 'eclipse_{}.pickle'.format(MODEL_NAME), 'wb') as file:  
            pickle.dump(tuned_model, file)            
        
        
        ### Plot confusion matrix and save the diagnosing results and FAR & ADR
        if WINDOW_SIZE != 0:
            if conf['granularity'] != 0:
                analysis_wrapper_multiclass(y_test['anom'].astype('int'),preds,
                                            conf,CV_INDEX,MODEL_NAME + '_test',
                                           name_cm='{} min Windowing - {}'.format(WINDOW_SIZE,SYSTEM))        
            else:
                analysis_wrapper_multiclass(y_test['anom'].astype('int'),preds,
                                            conf,CV_INDEX,MODEL_NAME + '_test',
                                           name_cm='{} sec Windowing - {}'.format(WINDOW_SIZE,SYSTEM))

        else: 
            analysis_wrapper_multiclass(y_test['anom'].astype('int'),preds,
                                    conf,CV_INDEX,MODEL_NAME + '_test',
                                   name_cm='No Windowing - {}'.format(SYSTEM))         
            
        
        ###Save prediction probabilities
        pred_proba = tuned_model.predict_proba(X_test.values)       
        
        prediction_df = pd.DataFrame(columns=['true_label','pred_label','Prediction Confidence','Anomaly'])
        prediction_df['pred_label'] = preds
        prediction_df['true_label'] = y_test['anom'].astype('int').values
        prediction_df['Prediction Confidence'] = np.max(pred_proba,axis=1) 

        correct_pred_df = prediction_df[prediction_df['true_label'] == prediction_df['pred_label']]

        #Modification
        for key,value in ANOM_DICT.items():
                correct_pred_df.loc[correct_pred_df[correct_pred_df['true_label'] == value].index,'Anomaly'] = key

        correct_pred_df.to_pickle(conf['results_dir'] / '{}_pred_confidence.pickle'.format(MODEL_NAME))      
        
        
        ### Saves wrong predictions for possible examination 
        wrong_predictions = y_test.copy()
        wrong_predictions['pred'] = preds

        wrong_predictions = wrong_predictions[wrong_predictions['anom'] != wrong_predictions['pred']]

        wrong_predictions.to_csv(conf['model_dir'] / '{}_wrong_predictions.csv'.format(MODEL_NAME))            
                                                                            
        logging.info("########################### DONE ###########################")
        
if __name__ == '__main__':
    main()




