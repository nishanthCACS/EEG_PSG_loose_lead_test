#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:15:35 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

this script contains the functions related to events to txt format
"""
import logging
from sleep_EEG_loose_lead_detect.preprocessing.find_bad_epochs import writeFile, formatDataTable


logger = logging.getLogger("events_logger")
while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
# Set logging level to the logger
# logger.setLevel(logging.DEBUG) # <-- THIS!
logger.setLevel(logging.INFO)
logger.propagate = False

yasa_logger = logging.getLogger("yasa")
# to avoid pring the not detecting any spindles
yasa_logger.disabled = True

def preprocess_events_to_txt(prerocess_event_txt_file_name, saving_dir, epoch_status,  preprocess_txt_only_detected_bad=True,comma_sep=False, save_csv=False):
    # --------------------------------------------------------------------------
    # this function cretaes the events to txt function after preprocessing
    # 
    #  return the epoch in index start from 1
    # 
    # preprocess_txt_only_detected_bad: this only select the epoches effected in the events annotation
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # this holds the rows of events
    # --------------------------------------------------------------------------
    preprocess_events_txt=[]
    for e in range(0,len(epoch_status)):
        if preprocess_txt_only_detected_bad:               
            if epoch_status[e]!='normal':
                preprocess_events_txt.append([e+1, str(epoch_status[e])])
        else:
            preprocess_events_txt.append([e+1, str(epoch_status[e])])
    
    # --------------------------------------------------------------------------
    # assign txt format to save as txt file
    # --------------------------------------------------------------------------
    if prerocess_event_txt_file_name.split('.')[-1]!='txt':
        if save_csv and  prerocess_event_txt_file_name.split('.')[-1]!='csv':
            prerocess_event_txt_file_name = prerocess_event_txt_file_name+'.csv'
        else:
            prerocess_event_txt_file_name = prerocess_event_txt_file_name+'.txt'

    # --------------------------------------------------------------------------
    # whether txt file in comma seprated format
    # --------------------------------------------------------------------------
    if comma_sep:
        col_sep=','
    else:
        col_sep='\t'
    
    writeFile(saving_dir+prerocess_event_txt_file_name,formatDataTable(preprocess_events_txt,col_sep=col_sep))
    logger.info("saved in "+saving_dir+prerocess_event_txt_file_name)    


def only_sleep_epoches_events_to_txt(prerocess_event_txt_file_name, saving_dir, epoch_status, comma_sep=False, save_csv=False):
    # --------------------------------------------------------------------------
    # this function cretaes the events to txt function
    # 
    #  return the epoch in index start from 1
    # 
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # this holds the rows of events
    # --------------------------------------------------------------------------
    preprocess_events_txt=[]
    for e in range(0,len(epoch_status)):
        preprocess_events_txt.append([e+1, str(epoch_status[e])])
    
    # --------------------------------------------------------------------------
    # assign txt format to save as txt file
    # --------------------------------------------------------------------------
    if prerocess_event_txt_file_name.split('.')[-1]!='txt':
        if save_csv and  prerocess_event_txt_file_name.split('.')[-1]!='csv':
            prerocess_event_txt_file_name = prerocess_event_txt_file_name+'.csv'
        else:
            prerocess_event_txt_file_name = prerocess_event_txt_file_name+'.txt'
    
    # --------------------------------------------------------------------------
    # whether txt file in comma seprated format
    # --------------------------------------------------------------------------
    if comma_sep:
        col_sep=','
    else:
        col_sep='\t'
    
    writeFile(saving_dir+prerocess_event_txt_file_name,formatDataTable(preprocess_events_txt,col_sep=col_sep))
    logger.info("saved in "+saving_dir+prerocess_event_txt_file_name)    

