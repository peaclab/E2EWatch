#!/usr/bin/env python
# coding: utf-8



def pipeline(conf):
    
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier    

    READ_DIR = conf['output_training_dir'] / ('CV_' + str(conf['cv_fold']))

#     if not conf['windowing']:    

#         if conf['feature_select']:
#             data = pd.read_hdf(READ_DIR / 'nowindow_train_data_fs.hdf', key='nowindow_train_data_fs')
#             label = pd.read_csv(READ_DIR / 'nowindow_train_label_fs.csv',index_col=['node_id'])
#         else:
#             data = pd.read_hdf(READ_DIR / 'nowindow_train_data.hdf', key='nowindow_train_data')        
#             label = pd.read_csv(READ_DIR / 'nowindow_train_label.csv',index_col=['node_id'])

#     else:

#         if conf['feature_select']:
#             data = pd.read_hdf(READ_DIR / 'window_train_data_fs.hdf', key='window_train_data_fs')        
#             label = pd.read_csv(READ_DIR / 'window_train_label_fs.csv',index_col=['node_id'])

#         else:
#             data = pd.read_hdf(READ_DIR / 'window_train_data.hdf', key='window_train_data')                
#             label = pd.read_csv(READ_DIR / 'window_train_label.csv',index_col=['node_id'])

    logging.info("Reading Train Data & Label Completed")
    logging.info("Data shape: %s",data.shape)
    logging.info("Label shape: %s",label.shape)        


    pipeline_rf = Pipeline([
        ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced'))
    ])      

    logging.info("Fitting pipeline!")
    pipeline_rf.fit(data.values,label['anom'].astype('int'))


    ###Save the model 
    logging.info("Saving the model!")

    import pickle 

    with open(READ_DIR / 'eclipse_rf.pickle', 'wb') as file:  
        pickle.dump(pipeline_rf, file)

    with open(READ_DIR / 'eclipse_rf_py2.pickle', 'wb') as file:  
        pickle.dump(pipeline_rf, file, protocol=2)    



#     if not conf['windowing']:

#         if conf['feature_select']:        

#             data = pd.read_hdf(READ_DIR / 'nowindow_test_data_fs.hdf', key='nowindow_test_data_fs')                        
#             label = pd.read_csv(READ_DIR / 'nowindow_test_label_fs.csv',index_col=['node_id'])
#         else:
#             data = pd.read_hdf(READ_DIR / 'nowindow_test_data.hdf', key='nowindow_test_data')                        
#             label = pd.read_csv(READ_DIR / 'nowindow_test_label.csv',index_col=['node_id'])        

#     else:
#         if conf['feature_select']:      
#             data = pd.read_hdf(READ_DIR / 'window_test_data_fs.hdf', key='window_test_data_fs')                        
#             label = pd.read_csv(READ_DIR / 'window_test_label_fs.csv',index_col=['node_id'])        
#         else:
#             data = pd.read_hdf(READ_DIR / 'window_test_data.hdf', key='window_test_data')                        
#             label = pd.read_csv(READ_DIR / 'window_test_label.csv',index_col=['node_id'])

    logging.info("Reading Test Data & Label Completed")    
    logging.info("Data shape: %s",data.shape)
    logging.info("Label shape: %s",label.shape)            


    logging.info("Testing pipeline!")
    preds = pipeline_rf.predict(data)

    logging.info("Generating report!")
    ### Saves classification report where all apps are combined
    report_dict = classification_report(y_true=label['anom'].astype('int'), y_pred=preds, labels=label['anom'].unique())
    print(report_dict)
    report_dict = classification_report(y_true=label['anom'].astype('int'), y_pred=preds, labels=label['anom'].unique(),output_dict=True)


    ### Save the predictions
    if not conf['windowing']:
        if conf['feature_select']:  
            hf = h5py.File(conf['output_training_dir'] / ('CV_' + str(conf['cv_fold'])) / 'nowindow_preds_fs.h5', 'w')
            hf.create_dataset('nowindow_preds_fs', data=preds, compression='gzip',compression_opts=9)
            hf.close()  

        else:
            hf = h5py.File(conf['output_training_dir'] / ('CV_' + str(conf['cv_fold'])) / 'nowindow_preds.h5', 'w')
            hf.create_dataset('nowindow_preds', data=preds, compression='gzip',compression_opts=9)
            hf.close()  
    else:
        if conf['feature_select']:
            hf = h5py.File(conf['output_training_dir'] / ('CV_' + str(conf['cv_fold'])) / 'window_preds_fs.h5', 'w')
            hf.create_dataset('window_preds_fs', data=preds, compression='gzip',compression_opts=9)
            hf.close()        
        else:
            hf = h5py.File(conf['output_training_dir'] / ('CV_' + str(conf['cv_fold'])) / 'window_preds.h5', 'w')
            hf.create_dataset('window_preds', data=preds, compression='gzip',compression_opts=9)
            hf.close()

    ### Save the prediction results        
    if not conf['windowing']:

        if conf['feature_select']:
            json_dump = json.dumps(report_dict)
            f_json = open(READ_DIR / "report_dict_nowindow_fs.json","w")
            f_json.write(json_dump)
            f_json.close()    

        else:
            json_dump = json.dumps(report_dict)
            f_json = open(READ_DIR / "report_dict_nowindow.json","w")
            f_json.write(json_dump)
            f_json.close()    

    else:

        if conf['feature_select']:
            json_dump = json.dumps(report_dict)
            f_json = open(READ_DIR / "report_dict_window_fs.json","w")
            f_json.write(json_dump)
            f_json.close()    

        else:
            json_dump = json.dumps(report_dict)
            f_json = open(READ_DIR / "report_dict_window.json","w")
            f_json.write(json_dump)
            f_json.close()   