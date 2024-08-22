#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 21:46:46 2021

@author: yeungds
@modified by: yuanyuan.li@nih.gov
@modified by: amlan.talukder@nih.gov
@modified by: anandanadarajah.nishanth@nih.gov inter channels coherence in the selected way 
modified on Mon Jan 16 12:06:35 2023 to add the signal_root to to calculate teh correlation vector
modified on Fri Jan 27 14:37:34 2023 to only load the root data
modified on Mon Jul 24 16:23:31 2023 to trace-back the events origin from loose-lead

modified on Wed Jan 17 20:20:15 2024 to manually extract the sleep-related startids and skip those events

"""
import numpy as np
import mne
import logging

from copy import deepcopy
from sleep_EEG_loose_lead_detect.GUI_interface.percentage_bar_vis  import percent_complete

#%% logger intialisation
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("load_EDFS_with_events")
while logger.handlers:
      logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

''' 
to get the channels and events information presented in the .edf files
please use the  functions
   > get_channel_event_infor
      -get_root_channels
      -get_event_infor
'''

def get_root_channels(in_edf):
    #----------------------------------------------------------------------------------
    # Load channel information while loading the edf
    #----------------------------------------------------------------------------------
    edf = mne.io.read_raw_edf(in_edf, preload=False, verbose=False, stim_channel=None)
    
    return edf.ch_names


def get_event_infor(in_edf):

    #----------------------------------------------------------------------------------
    # Load edf file with event information
    #----------------------------------------------------------------------------------
    edf = mne.io.read_raw_edf(in_edf, preload=False, verbose=False, stim_channel=None)
    _, event_ids = mne.events_from_annotations(edf)

    events_all =[]
    for e_name in event_ids:
        events_all.append(str(e_name.lower()))
    return events_all

def get_channel_event_infor(in_edf):

    #----------------------------------------------------------------------------------
    # Load edf file with event information
    #----------------------------------------------------------------------------------
    edf = mne.io.read_raw_edf(in_edf, preload=False, verbose=False, stim_channel=None)
    _, event_ids = mne.events_from_annotations(edf)

    events_all =[]
    try:
        for e_name in event_ids:
            events_all.append(str(e_name.lower()))
    except:
        logger.warning(in_edf+ " event details issue")

    try:
        ch_names=edf.ch_names
    except:
        logger.warning(in_edf+ " channel names details issue")
        ch_names=[]

    return ch_names, events_all


#----------------------------------------------------------------------------------
def get_assigned_channels(ch_names =['F3', 'F4', 'C3', 'C4', 'O1', 'O2']):
    '''
    Assign the interstted sleep-relevant channels
    even after the re-referencing this will be there
    
    '''
    return ch_names

#----------------------------------------------------------------------------------
def load_root_dataset(in_edf, epoch_sec, micro_volt_scaling=1e6,
                      re_reference_mastoid=True,verbose=False,GUI_percentile=True):

    '''
    I/P: 
        in_edf: edf location and edf name
        epoch_sec: 30 sec

    re_reference_mastoid:    giving option to chooose the channels without 
        re-refencing via the re_reference_mastoid
        
        re-referencing is always prefered due to aviod the channel miss placement issue
    '''
    ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'M1', 'M2']
    #----------------------------------------------------------------------------------
    # Load edf file
    #----------------------------------------------------------------------------------
    if verbose:
        logger.info('loading %s iniated',in_edf)
    if GUI_percentile:
        percent_complete(1, 100, bar_width=60, title="Loading-edf", print_perc=True)

    # try:
    edf = mne.io.read_raw_edf(in_edf, preload=False, verbose=False, stim_channel=None)
    # except:
    #     edf = mne.io.read_raw_edf(in_edf, preload=False, verbose=False, stim_channel=None, encoding='latin1')

    sampling_freq = edf.info['sfreq']
    signals = edf.get_data(picks=ch_names)

    # mne automatically converts to V, convert back to uV
    signals *= micro_volt_scaling
    
    if verbose:
        logger.info('annotation extaraction %s iniated',in_edf)
        logger.info('sampling frquency '+str(sampling_freq))

    if GUI_percentile:
        percent_complete(20, 100, bar_width=60, title="Loading-edf", print_perc=True)


    annotations = edf._annotations
    whole_annotations = np.concatenate([annotations.onset[:, None].astype(object),
                                                annotations.duration[:, None].astype(
                                                    object),
                                                annotations.description[:, None].astype(object)], axis=1)
    # re-reference the channels 
    if re_reference_mastoid:
            
        signals_root = np.array([
            signals[ch_names.index('F3')] - signals[ch_names.index('M2')],
            signals[ch_names.index('F4')] - signals[ch_names.index('M1')],
            signals[ch_names.index('C3')] - signals[ch_names.index('M2')],
            signals[ch_names.index('C4')] - signals[ch_names.index('M1')],
            signals[ch_names.index('O1')] - signals[ch_names.index('M2')],
            signals[ch_names.index('O2')] - signals[ch_names.index('M1')],
            ])
    else:
        signals_root = deepcopy(np.array([
        signals[ch_names.index('F3')],
        signals[ch_names.index('F4')],
        signals[ch_names.index('C3')],
        signals[ch_names.index('C4')],
        signals[ch_names.index('O2')]]))
        signals[ch_names.index('O1')],
    

 
    ch_names  = get_assigned_channels()

    #----------------------------------------------------------------------------------
    # Get sleep stage annotations
    #----------------------------------------------------------------------------------

    # These are all possible sleep stage events from the known data   
    # made changes in the n4 stage is especially annotated by the 0
    # and the unknows are annotated by 6
    sleep_stage_event_to_id_mapping = {'sleep stage w': 5,
                                        'sleep stage r': 4, 
                                        'sleep stage 1': 3, 
                                        'sleep stage 2': 2, 
                                        'sleep stage 3': 1,
                                        'sleep stage 4': 0, 
                                        'sleep stage n1': 3, 
                                        'sleep stage n2': 2, 
                                        'sleep stage n3': 1,
                                        'sleep stage n4': 0,
                                        'sleep stage n': 6,
                                        'sleep stage ?': 6}


    ev_or, event_ids = mne.events_from_annotations(edf,verbose=False)

    relevant_sleep_events = {}
    for e_name in event_ids:
        if e_name.lower() in sleep_stage_event_to_id_mapping:
            relevant_sleep_events[e_name] = sleep_stage_event_to_id_mapping[e_name.lower()]
    
    events, _ = mne.events_from_annotations(edf, relevant_sleep_events, chunk_duration=epoch_sec,verbose=False)
    
    if verbose:
        logger.info('events are assigned')
    if GUI_percentile:
        percent_complete(50, 100, bar_width=60, title="Loading-edf", print_perc=True)
        
    window_size = int(epoch_sec*sampling_freq)

    '''
    we can directly use the events information 
        to Calculate start position ids for each epoch starting from labeled sleep stages

    '''
    # start_ids=events[:,0]
    len_signals = signals_root.shape[1] # (Channels x Data)
    start_ids = np.arange(events[0][0], len_signals-window_size+1, window_size)

    '''
    to trace the orgin of the edf events
    
    '''
    sel_index_sl_st, flag_other_start_id_extraction =  obtain_sel_index_sl_st(ev_or, events, start_ids)
    
    '''
    if any break events occurs in the middle this is flagged and manusally extract the events
    '''
    if flag_other_start_id_extraction:
        logger.warn(in_edf+" Selected events start ids are not continious, seems some breaks may happened in the recording")
        logger.warning('Due to this missalignment due to time stamp missing; the down strean analysis is performed based on assumption of alignment')
        logger.warning('Assumption:  health check performed by precheck tool with the EEG and the obtained aligned edf is used for the analysis')
    
        sel_index_sl_st, start_ids = re_intiate_manually_obtain_sel_index_sl_st(whole_annotations,events,ev_or,
                                                                    sleep_stage_event_to_id_mapping,
                                                                        window_size, epoch_sec=epoch_sec)
        logger.warning('Since the edf file missing the time stamp with the signal we cannot fully gurantee the outcomes')

          
    if verbose:
        logger.info('assigning the selected sleep-ids')
    if GUI_percentile:
        percent_complete(80, 100, bar_width=60, title="Loading-edf", print_perc=True)
        
    sleep_events = np.zeros(signals.shape[1]) + np.nan
    for start, _, stage in events:
        sleep_events[start:start+window_size] = stage

    '''
    # Write all sleep related events to file
    fname = out_loc + in_name + "_edfstage.txt"
    evt_df.to_csv(fname, sep='\t', header=True, index=True)
    '''
    if verbose:
        logger.info('loading edf completed')
    if GUI_percentile:
        percent_complete(100, 100, bar_width=60, title="Loading-edf", print_perc=True)
    logger.info('')
    logger.info('')

    return signals_root, sleep_events, ch_names, sampling_freq, start_ids, sel_index_sl_st, whole_annotations, ev_or

#----------------------------------------------------------------------------------
def obtain_sel_index_sl_st(ev_or, events, start_ids):
    '''
    since the algorithm 
    based on the sleep events the edf file must have the
    annoated events with the intended sleep-events in the same duration
    go through the start_ids and select only the selected events
    
    need to hold this if we are going to provide the detected annotations map back to the edf
    
    '''
    flag_other_start_id_extraction=False

    sel_index_sl_st=np.zeros((len(start_ids)))
    p=0
    for s in start_ids:
        try:
            sel_index_sl_st[p]=np.where(ev_or[:,0]==s)[0][0]
        except:
            flag_other_start_id_extraction=True
            break
        p+=1
    
    # since the indexes are integer this whould be
    sel_index_sl_st = sel_index_sl_st.astype(int)
    if not flag_other_start_id_extraction: 
        if  not np.array_equal(ev_or[list(sel_index_sl_st),0],events[:,0]):
            logger.error("Events orgin not match with the selected events, please report with the edf file to track the issue")
            flag_other_start_id_extraction=True


    return sel_index_sl_st, flag_other_start_id_extraction



def re_intiate_manually_obtain_sel_index_sl_st(whole_annotations,events,ev_or,
                                               sleep_stage_event_to_id_mapping,
                                               # start_id_manual_or,
                                               window_size,
                                               epoch_sec=30):
    '''
    recheck the sleep stage events for robust edf handling
    if any breaks happens in the middle lets reintiae and calculate
    '''
    # start_id_manual=deepcopy(start_id_manual_or)
    start_id_manual=[]
    sel_index_sl_st_manual=[]
    # sleep_stages_epoch_wise=[]
    i=-1
    for t, d, desc in whole_annotations:
        i+=1
        epoch_id = int(t//epoch_sec)
         
        # Check if this is a sleep stage annotation
        if desc.lower() not in sleep_stage_event_to_id_mapping:
    
            continue
        label = sleep_stage_event_to_id_mapping[desc.lower()]
        n_epochs = int(d//epoch_sec)
         
        for j in range(n_epochs):
            # sleep_stages_epoch_wise.append(label)
            index_raw = int((epoch_id + j) * window_size)
            
            #this will make sure to place the right startindex for sleep-stages
            # start_id_manual[i]=index_raw
            start_id_manual.append(index_raw)
        sel_index_sl_st_manual.append(i)
        
    if not np.array_equal(ev_or[sel_index_sl_st_manual,0],events[:,0]):
          logger.error("Events orgin not match with the selected events, please report with the edf file to track the issue")
    return np.array(sel_index_sl_st_manual),start_id_manual


def isEDFAligned(eeg_data_raw, annotations, sampling_freq, epoch_size):
    '''
    function credit goes to Dr. Amlan Talukdar
    This function just help to check the provided annotation aligined with the given edf file

    As from the analysis some times the end criteria may not full-fill (use the function with your own-risk)
    
    '''
    whole_annotations = np.concatenate([annotations.onset[:, None].astype(object),
                                        annotations.duration[:, None].astype(object),
                                        annotations.description[:, None].astype(object)], axis=1)
 
    # --------------------------------------------------------------------------
    # Check if the last annotation time matches the last EEG epoch
    # --------------------------------------------------------------------------
    last_annot_time = whole_annotations[-1][0] + whole_annotations[-1][1]
    last_annot_epoch = int(last_annot_time//epoch_size)
    last_signal_epoch = int(eeg_data_raw.shape[1]//int(epoch_size * sampling_freq))
 
    return last_annot_epoch == last_signal_epoch