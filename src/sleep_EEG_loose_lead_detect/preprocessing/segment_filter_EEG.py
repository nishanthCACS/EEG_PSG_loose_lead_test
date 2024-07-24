''#!/usr/bin/env python3
# -*- coding: utf-8 -*-'''
"""***********************************
Created on Mon Sep 13 21:46:46 2021

@author: yeungds
@modified by: yuanyuan.li@nih.gov
@modified by: amlan.talukder@nih.gov

@modified by:anandanadarajn2@nih.gov on 29-Oct-2022 to accomadate the YASA-spindle detection (earlier the power_noise_handler is included)
modified on Mon Jul 3 08:36:19 2023 to accomadate the amplitude-threshold based on age (this can be done in plenty of wasys)
modified on Wed Aug  9 09:59:08 2023 discription and verbose is added

**************************"""
import logging
import numpy as np
from scipy.signal import detrend
from scipy.stats import mode
from mne.filter import filter_data, notch_filter
# from mne.time_frequency import psd_array_multitaper
from sleep_EEG_loose_lead_detect.preprocessing.power_noise_handler import powerline_noice_magnificant_checker
from sleep_EEG_loose_lead_detect.GUI_interface.percentage_bar_vis  import percent_complete



#%% logger intialisation
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("preprocess_time")
while logger.handlers:
      logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


epoch_status_explanation = [
    'normal',
    'NaN in sleep stage',
    'NaN in EEG',
    'overly high/low amplitude',
    'flat signal',
    'bad events']


def segment_EEG(EEG, labels, window_time, Fs, start_ids, notch_freq=None, bandpass_freq=None, 
                start_end_remove_window_num=0, amplitude_thres=2000, to_remove_mean=False, bad_epochs=None,
                notch_freq_essential_checker=False,
                channel_specific_preprocess=False, ch_names=[],
                return_filtered_EEG=False,
                quantile_normalise_time=False, verbose=False,GUI_percentile=True ):

    '''
    Parameters
    ----------
    EEG : numpy array of signal
        DESCRIPTION: EEG signal that in array format
    labels : list of ints
        DESCRIPTION: sleep_stages related to the EEG array
    window_time : int
        DESCRIPTION. length of the epoch 

    Fs : int
        DESCRIPTION. Sampling rate
    hypno_start_time_idx : int
        DESCRIPTION. index belong to the starting point of the EEG belongs to sleep-stages (or relevant to sleep-study)
    notch_freq : float, optional
        DESCRIPTION. To remove the power line noise this filter is applied commonly
        (60Hz or 50Hz) like in US 60Hz, and Europe 50Hz, based on the data colloected region this should be applied
    bandpass_freq : [float, float], optional
        DESCRIPTION.range of frequency to band pass for sleep [0.5 Hz- 32.5 Hz]
    amplitude_thres : float, optional
        DESCRIPTION. The value assign anly amplitude value > amplitude_thres is marked as high amplitude
        The default is 2000.
    to_remove_mean : Bool, optional
        DESCRIPTION. this iwll remove the mean of the signal
    bad_epochs : list of indexes, optional
        DESCRIPTION. list of indexes marked as bad epohces like batroom break, etc.
        The default is None.
    notch_freq_essential_checker : Bool, optional
        DESCRIPTION. chek one epoch of the selected (first) channels' power near the line noise, and the average power in the lowest power of interest
        5Hz-25Hz since in the sleep-EEG this will have the low-power, slow-wave(delta) region has higher power
        whether a The default is False.
        
        
    channel_specific_preprocess: Bool, optional
        to return the preprocessed events with the channel information that has changed from the normal to other status with the channel information
        
        Even this condition is true this doen't gurantee all the other channels are good for this specific epoch 
        That need to be varified seperately 
        
        
        if not channel_specific_preprocess just return the epoch status without channel specific information
        like nan value, high/lower amplitude etc.
        
 
    
    quantile_normalise_time : Bool, optional
        DESCRIPTION. Normalise the time domin signal with the quantile normalisation or not
        The default is False.

    verbose: bool
        DESCRIPTION. printing the intermediate stages
        The default is False.

    Returns
    -------
    None.

    '''

    # --------------------------------------------------------------------------
    # step_time : int
    # DESCRIPTION. used to check the flat signal while sliding window (with window time) with the sliding step step-time apart
    
    # identify the flat signal and remove them
    # this is intiated by cheking the standard deviation of the temporal signal 
    
    #if the signal is flat more than the flat second period then the epoch is marked as 
    # --------------------------------------------------------------------------
    std_thres = 0.2
    std_thres2 = 1.
    flat_seconds = 5
    padding = 0
     
    if to_remove_mean:
        EEG = EEG - np.nanmean(EEG,axis=1, keepdims=True)
    window_size = int(round(window_time*Fs))

    # step_size = int(round(step_time*Fs))
    step_size = window_size
    flat_length = int(round(flat_seconds*Fs))
    
    # --------------------------------------------------------------------------
    # since the start_ids are used to trace back the events the window_size and step_size are maintained with the same value
    # --------------------------------------------------------------------------
    # start_ids = np.arange(hypno_start_time_idx, EEG.shape[1]-window_size+1, step_size)

    # --------------------------------------------------------------------------
    # other preprocessing steps like remove the infinity and nan values etc.
    # --------------------------------------------------------------------------
    if verbose:
        logger.info('Nan and infinity check intiated')

    labels_ = []
    for si in start_ids:
        labels2 = labels[si:si+window_size]
        labels2[np.isinf(labels2)] = np.nan
        labels2[np.isnan(labels2)] = -1
        
        # --------------------------------------------------------------------------
        # to keep the same kind of syntax and compbility
        # --------------------------------------------------------------------------
        # label_2 = mode(labels2).mode[0]
        label__ = mode(labels2,keepdims=True).mode[0]
        # label__ = mode(labels2,keepdims=False).mode

        # if not np.array_equal(label__,label_2):
        #     print('label__: ',label__)
        #     print('label_2: ',label_2)

        if label__==-1:
            labels_.append(np.nan)
        else:
            labels_.append(label__)
    labels = np.array(labels_)
    
    if verbose:
        logger.info('Nan and infinity check assigned')
    if GUI_percentile:
        percent_complete(10, 100, bar_width=60, title="Preprocess", print_perc=True)

    # --------------------------------------------------------------------------
    # first assign normal to all epoch status
    # --------------------------------------------------------------------------
    epoch_status = [epoch_status_explanation[0]]*len(start_ids)
    
    # --------------------------------------------------------------------------
    # check nan sleep stage
    # --------------------------------------------------------------------------
    if np.any(np.isnan(labels)):
        ids = np.where(np.isnan(labels))[0]
        for i in ids:
            epoch_status[i] = epoch_status_explanation[1]
            # epoch_status[i] = epoch_status_explanation[2]+ ' channels '+','.join([ch_names[x] for x in list(np.where(nan2d[i,:])[0])])

    # --------------------------------------------------------------------------
    # just a constant to check the notch filter status    
    # --------------------------------------------------------------------------
    notch_filter_skipped=True
    if verbose:
        logger.info('Notch filter check intiated')

    # --------------------------------------------------------------------------
    # just check the essential ity of the notch filter applying need
    # --------------------------------------------------------------------------
    if Fs/2>notch_freq and bandpass_freq is not None and np.max(bandpass_freq)>=notch_freq:
        # --------------------------------------------------------------------------
        # notch_freq_essential_checker check the  signal's power line noise presence
        # if that powerline noise is already supressed by the recorder we can skip the power -line noise notch filter
        # this is performed by only checking by one of the EEG channel's one epoch by applying FFT
        # --------------------------------------------------------------------------
        if notch_freq_essential_checker:
            ch_sel=1#for F4 channel in default input
            ids_normal = np.where(~np.isnan(labels))[0]
            EEG_segs_sel = EEG[ch_sel,list(map(lambda x:np.arange(x-padding,x+window_size+padding), start_ids[[ids_normal[0],ids_normal[len(ids_normal)//2]]]))]
            if powerline_noice_magnificant_checker(EEG_segs_sel,Fs):
                EEG = notch_filter(EEG, Fs, notch_freq, fir_design="firwin", verbose=False) 
                notch_filter_skipped=False
            else:
                if verbose:
                    logger.info("power line noise is not much just left without Notch filter")
                if GUI_percentile:
                    percent_complete(30, 100, bar_width=60, title="Preprocess", print_perc=True)
        else:
            EEG = notch_filter(EEG, Fs, notch_freq, fir_design="firwin", verbose=False) 
            notch_filter_skipped=False
            if verbose:
                logger.info('Notch filteration done ')
            if GUI_percentile:
                percent_complete(30, 100, bar_width=60, title="Preprocess", print_perc=True)
    # --------------------------------------------------------------------------
    #  Apllying the band pass filter
    # --------------------------------------------------------------------------

    if bandpass_freq is None:
        fmin = None
        fmax = None
    else:
        fmin = bandpass_freq[0]
        fmax = bandpass_freq[1]
        if fmax>=Fs/2:
            fmax = None


    if bandpass_freq is not None:
        if verbose:
            logger.info('Band pass filteration  begin with %.2f Hz, %.2f Hz',fmin,fmax)
      
        EEG = filter_data(EEG, Fs, fmin, fmax, fir_design="firwin", verbose=False)
    
    # --------------------------------------------------------------------------
    #  Assigning minimal value and get quantiles
    # --------------------------------------------------------------------------
    EEG2 = np.array(EEG)
    EEG2[np.abs(EEG2)<1e-5] = np.nan
    q1,q2,q3 = np.nanpercentile(EEG2, (25,50,75), axis=1)
    
    # --------------------------------------------------------------------------
    # Segment into epochs
    # --------------------------------------------------------------------------
    EEG_segs = EEG[:,list(map(lambda x:np.arange(x-padding,x+window_size+padding), start_ids))].transpose(1,0,2) 
        
    if channel_specific_preprocess:
        if  len(ch_names)!=np.shape(EEG_segs)[1]:
            raise Exception("The ch_names should be the length of channels present in the EEG")
    
    
    if verbose:
        logger.info('segmentaion done')
    if GUI_percentile:
        percent_complete(60, 100, bar_width=60, title="Preprocess", print_perc=True)

    nan2d = np.any(np.isnan(EEG_segs), axis=2)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    for i in nan1d:
        # epoch_status[i] = epoch_status_explanation[2]
        epoch_status = _channel_specific_preprocess_events_handler(epoch_status,i,
                                                    2,nan2d,
                                                    channel_specific_preprocess,epoch_status_explanation, ch_names)
    
    amplitude_large2d = np.any(np.abs(EEG_segs)>amplitude_thres, axis=2)
    amplitude_large1d = np.where(np.any(amplitude_large2d, axis=1))[0] 
    for i in amplitude_large1d:
        # epoch_status[i] = epoch_status_explanation[3]
        epoch_status = _channel_specific_preprocess_events_handler(epoch_status,i,
                                                    3,amplitude_large2d,
                                                    channel_specific_preprocess,epoch_status_explanation, ch_names)
    # --------------------------------------------------------------------------
    # if there is any flat signal with flat_length
    # --------------------------------------------------------------------------
    short_segs = EEG_segs.reshape(EEG_segs.shape[0], EEG_segs.shape[1], EEG_segs.shape[2]//flat_length, flat_length)
    flat2d = np.any(detrend(short_segs, axis=3).std(axis=3)<=std_thres, axis=2)
    flat2d = np.logical_or(flat2d, np.std(EEG_segs,axis=2)<=std_thres2)
    flat1d = np.where(np.any(flat2d, axis=1))[0]
    for i in flat1d:
        # epoch_status[i] = epoch_status_explanation[4]
        epoch_status = _channel_specific_preprocess_events_handler(epoch_status,i,
                                                    4,flat2d,
                                                    channel_specific_preprocess,epoch_status_explanation, ch_names)
    # --------------------------------------------------------------------------
    # Mark epochs with bad events
    # --------------------------------------------------------------------------
    epoch_status = np.array(epoch_status)
    if not (bad_epochs is None):
        indx = np.searchsorted(start_ids, bad_epochs)
        # --------------------------------------------------------------------------
        #  only annotate the channels with teh epoch-status not normal with channel specific infor for 
        #  preprocess step detections
        # --------------------------------------------------------------------------
        if len(bad_epochs)>0:
            for i in indx:
                if not channel_specific_preprocess and epoch_status[i]==epoch_status_explanation[0]:           
                    epoch_status[i] = epoch_status_explanation[5]
                else:
                    epoch_status[i] = epoch_status[i]+'; '+epoch_status_explanation[5]


    # --------------------------------------------------------------------------
    # normalize signal
    # --------------------------------------------------------------------------    
    if quantile_normalise_time:

        nch = EEG_segs.shape[1]
        EEG_segs = (EEG_segs - q2.reshape(1,nch,1)) / (q3.reshape(1,nch,1)-q1.reshape(1,nch,1))
    if verbose:
        logger.info('Preprocess done')
    if GUI_percentile:
        percent_complete(100, 100, bar_width=60, title="Preprocess", print_perc=True)

    # --------------------------------------------------------------------------
    # 
    # to return the bandpassed notch filtered EEG 
    # --------------------------------------------------------------------------

    if return_filtered_EEG:
        return EEG_segs, EEG, labels, start_ids, epoch_status, q1,q2,q3, notch_filter_skipped
    else:
        return EEG_segs, labels, start_ids, epoch_status, q1,q2,q3, notch_filter_skipped


def _channel_specific_preprocess_events_handler(epoch_status,i,
                                                ep_ex_indx,arr_2d,
                                                channel_specific_preprocess,epoch_status_explanation, ch_names):
    # --------------------------------------------------------------------------
    # Not channel specific such all the events are annotated without the channel specific info
    #  for an example flat signal not mention which channel is flat etc.
    # --------------------------------------------------------------------------    
    if not channel_specific_preprocess:           
        epoch_status[i] = epoch_status_explanation[ep_ex_indx]
    else:
        # --------------------------------------------------------------------------
        # Due to channel specific such all the previous  events
        # and  channel specific preprocess info combined
        # 
        # just cheeck the normal event annotaion 
        # --------------------------------------------------------------------------    
        if  epoch_status[i]!=epoch_status_explanation[0]:
            epoch_status[i] =  epoch_status[i]+'; '+epoch_status_explanation[ep_ex_indx]+ ' channels '+','.join([ch_names[x] for x in list(np.where(arr_2d[i,:])[0])])
        else:
            epoch_status[i] = epoch_status_explanation[ep_ex_indx]+ ' channels '+','.join([ch_names[x] for x in list(np.where(arr_2d[i,:])[0])])
    return epoch_status