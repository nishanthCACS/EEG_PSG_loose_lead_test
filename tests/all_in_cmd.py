#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 06:58:03 2024

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved
"""

import sys
import os
import logging

import numpy as np
import pandas as pd
import pickle


from copy import deepcopy
from collections import Counter
from multiprocessing import Pool#, TimeoutError


# --------------------------------------------------------------------------
# load package functions
# assign the working directory of the package
# example sys.path.append(".../GitHub_packaging/EEG_PSG_loose_lead_test/src/")

# --------------------------------------------------------------------------
sys.path.append("src")
# sys.path.append("/Users/anandanadarajn2/Documents/a_Reports_EEG_Latex_full/EEG_arousal_detection_loose_channel_detection/ground_truth/codes/packaging/GitHub_packaging/EEG_PSG_loose_lead_test/src/")


from sleep_EEG_loose_lead_detect.optional_parameters import parameter_assignment
#checking only one get_root_channels function whether the loading suceeds
# # here non-normlaised EEG is obtained in the preproceeing step in time domain
from sleep_EEG_loose_lead_detect.preprocessing.load_EDF_dataset_events import load_root_dataset

from sleep_EEG_loose_lead_detect.preprocessing.find_bad_epochs import markBadEpochs
from sleep_EEG_loose_lead_detect.preprocessing.segment_filter_EEG import segment_EEG 


# # here MT-spectrum and correlation in one step to avoid saving the spectrum 
from sleep_EEG_loose_lead_detect.channel_correlation_outlier.correlation_functions import MT_based_correltion_calc_in_continious_segs
# # from channel_correlation_outlier.correlation_functions import time_correlation_coefficient_retriever_for_cont_seg_main



# this is belong to approach-2 or using the mean of the correlation
from sleep_EEG_loose_lead_detect.channel_correlation_outlier.poolers_Z_standardization_funcs import obtain_mean_corr
# this functions are common to both approaches
from sleep_EEG_loose_lead_detect.outlier_common.out_lier_based_fun import find_loose_leads_based_mean, find_loose_leads_based_mean_seperately
from sleep_EEG_loose_lead_detect.outlier_common.MT_variance_for_structural_aritifact_funcs import find_vertical_spikes
from sleep_EEG_loose_lead_detect.preprocessing.cont_segs_of_whole_EEG import save_cont_segs, continious_seg_to_np

# --------------------------------------------------------------------------
# spindle retaleed function based on YASA
# --------------------------------------------------------------------------
from sleep_EEG_loose_lead_detect.preprocessing.spindle_fun import spindle_detect_main
from sleep_EEG_loose_lead_detect.preprocessing.detected_spindle_one_hot import convert_to_full_one_hot_mapping_based_cont_seg_sp_sw_all_channels

# --------------------------------------------------------------------------
# pin point the loose-lead
# --------------------------------------------------------------------------
'''
not checked yet
'''
from sleep_EEG_loose_lead_detect.loose_lead_events.events_to_txt import preprocess_events_to_txt,only_sleep_epoches_events_to_txt
from sleep_EEG_loose_lead_detect.loose_lead_events.loose_lead_to_events_origin_funcs import epoch_status_to_events_annot
from sleep_EEG_loose_lead_detect.loose_lead_events.loose_lead_to_events_origin_funcs import main_loose_lead_to_origin_events_dic, event_annotations_origin_for_edf#, event_origin_for_edf
from sleep_EEG_loose_lead_detect.loose_lead_events.pin_point_funcs import pin_point_loose_lead



# --------------------------------------------------------------------------
#logger intialisation
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("sleep_loose_lead")
while logger.handlers:
      logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
logger.propagate = False



def pre_proprint(epoch_status):
    sm = Counter(epoch_status)
    for k, v in sm.items():
        logger.info(f'{k}: {v}/{len(epoch_status)}+ {v*100./len(epoch_status)}%')

 


# --------------------------------------------------------------------------
#logger intialisation
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("sleep_loose_lead")
while logger.handlers:
      logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
logger.propagate = False



def pre_proprint(epoch_status):
    sm = Counter(epoch_status)
    for k, v in sm.items():
        logger.info(f'{k}: {v}/{len(epoch_status)}+ {v*100./len(epoch_status)}%')


def EEG_sleep_loose_lead(in_name_temp, opt_paramters):
    '''
    Parameters
    ----------
    in_name_temp : str
        the edf file name going to be extracted.
    loading_dir_pre : object
        directory information for loading and saving etc.
    pred_slow_waves : Bool, optional
        DESCRIPTION. Predict the slow-waves using the YASA tool
    pred_spindles :  Bool, optional
        DESCRIPTION. Predict the spindles using the YASA tool
    T : int, optional
        DESCRIPTION. the window duration for multi-taper spectral estimation in sec. The default is 4sec.
        Don't choose the window length arbitarily this highly impact the spectral estimation. 
        The 4sec is good choise for sampling frequency Fs= 256
    amplitude_high_same_all_age : Bool, optional
        DESCRIPTION. defining the higher threhold value differently for each subjects age 
        
        The default is False.
        two categories provided based on the age_therh: 5 means 
            the suject's age > 5 years are marked for higher when the value > amplitude_thres_higer_ages
            the suject's age < 5 years are marked for higher when the value > amplitude_thres_lower_ages
            Normally the smaller the age the the brain power is higher
    
    verbose: Bool, optional
        DESCRIPTION: print all the information of wlkthrough in the pipe-line
        
    Returns
    -------
    None.
 

    '''
    
    # loading_dir_pre = opt_paramters.assign_user_inputs_for_directories(docker_image=False)
    loading_dir_pre = opt_paramters.loading_dir_pre
    GUI_percentile = opt_paramters.GUI_percentile    

    
    pred_slow_waves = opt_paramters.pred_slow_waves
    pred_spindles = opt_paramters.pred_spindles 
    
    T = opt_paramters.T
    amplitude_high_same_all_age = opt_paramters.amplitude_high_same_all_age
    avoid_spindle_loc = opt_paramters.avoid_spindle_loc
    verbose  = opt_paramters.verbose
    
    
    # --------------------------------------------------------------------------
    # assigning the directories
    # --------------------------------------------------------------------------
    in_loc = loading_dir_pre.in_loc
    in_bad_events = loading_dir_pre.bad_events_file_path    
    
    pred_slow_waves = opt_paramters.pred_slow_waves
    pred_spindles = opt_paramters.pred_spindles 
    
    T = opt_paramters.T
    amplitude_high_same_all_age = opt_paramters.amplitude_high_same_all_age
    avoid_spindle_loc = opt_paramters.avoid_spindle_loc
    verbose  = opt_paramters.verbose
    
    if avoid_spindle_loc:
        save_spindle_loc = loading_dir_pre.save_spindle_loc
        
    
    
    # --------------------------------------------------------------------------
    #  if the file not ends with edf add edf extention
    # --------------------------------------------------------------------------
    
    if in_name_temp.split('.')[-1]!='edf':
        in_edf = in_loc + in_name_temp + ".edf"
    else:
        in_edf = in_loc + in_name_temp 
        in_name_temp = '.'.join(in_name_temp.split('.')[0:-1])
    logger.warning('*** filename: %s', in_edf)
    
    if len(opt_paramters.tag)>0:
        in_name_temp= in_name_temp+opt_paramters.tag
    # --------------------------------------------------------------------------
    # Assigning prprocess paramter values
    # --------------------------------------------------------------------------
    epoch_length, line_freq, bandpass_freq, normal_only, notch_freq_essential_checker, amplitude_thres = opt_paramters.preprocess_par()
    
    # --------------------------------------------------------------------------
    # developments optional to user
    # --------------------------------------------------------------------------
    save_events_origin = opt_paramters.save_events_origin
    sleep_stage_preprocess_origin = opt_paramters.sleep_stage_preprocess_origin
    
    temp={}
    temp['age']=float(in_name_temp.split('_')[2])
    temp['sex']=in_name_temp.split('_')[1]
    
    # --------------------------------------------------------------------------
    # Load the EEG raw data and events
    # --------------------------------------------------------------------------
    EEG_root, sleep_stages_or, EEG_channels, Fs, start_time_idx,  sel_index_sl_st, whole_annotations, ev_or = load_root_dataset(in_edf, epoch_length,GUI_percentile=GUI_percentile)
    
    # --------------------------------------------------------------------------
    # save the events for checking purpose
    # since we are running bulk of EEGs to avoid the excess space this is saved in a temporary directory
    # and hold in a temporary list to delete if no loose-lead detected
    # --------------------------------------------------------------------------
    if save_events_origin:
        np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_sel_index_sl_st",sel_index_sl_st)
        np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_ev_or",ev_or)    
    
    
    temp['Fs']=Fs
    
    # --------------------------------------------------------------------------
    #  make deep copy to extract the spindle location
    # --------------------------------------------------------------------------
    EEG_root_copy=deepcopy(EEG_root)
    if avoid_spindle_loc:
        sleep_stages_origin =  deepcopy(sleep_stages_or)
    
    
    sleep_stages=  deepcopy(sleep_stages_or)
    
    # --------------------------------------------------------------------------
    #   mark the bad epoches; some times edfs have rich information other than the sleep-stages
    #   like patients out for bathroom break, restroom, etc. The epoches fell into the bathroom break is just noise
    #   so these kind of noise present data supposed to be removed from the sleep-stages annotation
    #
    # However each technician use their own terminolgy. 
    # we haven't remove all the bad-events here, just some known events that is surely known as bad-events
    # we have provided the bad events in the /docs/bad_events_bathroom.txt files that can be updatable
    #
    # some potential events are purposefully leaved like movenments, arousal etc. Since these information can be later 
    # used to find the subjects disease based studies etc
    # --------------------------------------------------------------------------
    bad_epochs = markBadEpochs(in_edf, in_bad_events, epoch_sec=epoch_length)
    if loading_dir_pre.keep_signature_dic['bad_epochs']:
        np.save(loading_dir_pre.bad_epochs_folder+ in_name_temp + "_bad_epochs",bad_epochs)

    # --------------------------------------------------------------------------
    # Segment EEG into 30sec epochs, apply notch & band filters, mark bad epochs and normalization
    # this is preprocessing the EEG signal in time domain and return the preprocessed signal in time domain
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # this will force the events with the channels information
    #   channel_specific_preprocess=True 
    # if not channel_specific_preprocess just return the epoch status without channel specific information
    # like nan value, high/lower amplitude etc.
    # --------------------------------------------------------------------------
    channel_specific_preprocess = opt_paramters.channel_specific_preprocess

    # --------------------------------------------------------------------------
    # EEG_channels extracted from the load_root_dataset
    # such that filaly the default channels will be endup in
    # ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
    # --------------------------------------------------------------------------
    ch_names = EEG_channels

    
    # --------------------------------------------------------------------------
    # in the fuiltertaion process the 
    # EEG_root_copy have higher chance of filetisation
    #  so keep them in a seperate deep copy is prefered
    # 
    #  to feed the filterd EEG through the MT-make sure assign this as True
    # basically the epoches points to the EEG_root_copies location
    # --------------------------------------------------------------------------
    filtered_EEG_MT=True
    epochs, EEG_root_copy, sleep_stages, epoch_start_idx, epoch_status, q1,q2,q3, notch_filter_skipped = segment_EEG(EEG_root_copy, 
                                                                                    sleep_stages, 
                                                                                    epoch_length, 
                                                                                    Fs, start_time_idx, 
                                                                                    notch_freq=line_freq, 
                                                                                    bandpass_freq=bandpass_freq, 
                                                                                    amplitude_thres=amplitude_thres, 
                                                                                    channel_specific_preprocess=channel_specific_preprocess,
                                                                                    ch_names=ch_names,
                                                                                    bad_epochs=bad_epochs,notch_freq_essential_checker=notch_freq_essential_checker,
                                                                                    return_filtered_EEG=True,
                                                                                    verbose=verbose,
                                                                                    GUI_percentile=GUI_percentile)
    
    
    preprocess_events_to_txt(in_name_temp+'_pre_process.txt',loading_dir_pre.out_loc_txt, epoch_status,  preprocess_txt_only_detected_bad=True)
    
    
    
    
    temp['q1']=np.array(q1)
    temp['q2']=np.array(q2)
    temp['q3']=np.array(q3)
    
    temp['notch_filter_skipped']=notch_filter_skipped
    
    temp['empty_EEG_segs']=False
    if epochs.shape[0] <= 0:
        temp['empty_EEG_segs']=True
        raise ValueError('Empty EEG segments')
    
    if normal_only:
        good_ids = np.where(epoch_status=='normal')[0]
        sel_id_name='good_ids'
        if len(good_ids)<=300:
            pre_proprint(epoch_status)
            temp['less_300']=True
        else:
            temp['less_300']=False
    else:
        raise Exception("need to modify the epoch status to selecting status, sin cethe outlier detection is build based only on normal epoch in mind")
    
    # --------------------------------------------------------------------------
    #for the time being only the good ids we can separatly chechk the higher/ lower cropped portions
    # --------------------------------------------------------------------------
    sel_ids = good_ids
    
    # --------------------------------------------------------------------------
    #  When we are avoiding the spindle occuring location
    #   while creating the spindle location the connious segmentation is done
    # 
    # --------------------------------------------------------------------------
    if avoid_spindle_loc:
        # --------------------------------------------------------------------------
        #  for spindle detect ion we feed the unpreprocessed (not filter applied) the EEG data
        # --------------------------------------------------------------------------
        data=deepcopy(EEG_root)
    
    
        
        # spindle detection intiated 
        sw_columns_ordered =  ['Start', 'End', 'MidCrossing','Duration', 'NegPeak', 'PosPeak',  'ValNegPeak', 'ValPosPeak', 
                    'PTP', 'Slope', 'Frequency',  'IdxChannel']
    
        sp_columns_ordered =  ['Start', 'End', 'Duration', 'Peak', 'Amplitude', 'RMS', 'AbsPower',
                'RelPower', 'Frequency', 'Oscillations', 'Symmetry', 'IdxChannel']
        
        pred_slow_waves=True
        pred_spindles =True
        
    
        cont_EEG_segments,start_keys,cont_EEG_segments_np, sp_sw_dic_format =  spindle_detect_main(in_name_temp, sel_ids, start_time_idx,
                                           data, sleep_stages_origin, save_spindle_loc, sel_id_name=sel_id_name,
                Fs=Fs, epoch_length=epoch_length, window_time=epoch_length, s_loc=0, e_loc=1,
                sw_columns_ordered=sw_columns_ordered,sp_columns_ordered=sp_columns_ordered,
                pred_slow_waves=pred_slow_waves, pred_spindles =pred_spindles)
    
        
       
      
    else:
        # --------------------------------------------------------------------------
        # to obtain the continious segmentations for the loose lead detection
        # 
        # whether save the continious segments or not
        # 
        # --------------------------------------------------------------------------
        save_cont_seg=False
        cont_EEG_segments_use =  save_cont_segs(in_name_temp, loading_dir_pre.out_loc_NREM_REM, sel_ids, start_time_idx, Fs,
                                                sel_id_name='good_ids', epoch_length=epoch_length,
                                                save_cont_seg=save_cont_seg,GUI_percentile=GUI_percentile)
    
        start_keys = list(cont_EEG_segments_use.keys())
        cont_EEG_segments_np = continious_seg_to_np(cont_EEG_segments_use, start_keys)
    
    if normal_only:
        good_ids = np.where(epoch_status=='normal')[0]
        #to make sure the index will be passed correctly later
        good_ids = good_ids.astype(int)
    
        sleep_stages_orgin = deepcopy(sleep_stages)
        sleep_stages = sleep_stages[good_ids]
    
        epochs = epochs[good_ids]
        #earlier direct passing worked now need to feed like this
        epoch_start_idx = np.array([epoch_start_idx[i] for i in good_ids])
        epoch_start_idx = epoch_start_idx.astype(int)
    
    # --------------------------------------------------------------------------
    # saving the preprocessed sleep-stages
    # --------------------------------------------------------------------------
    if sleep_stage_preprocess_origin:
        np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + '_sleep_stages_orgin',sleep_stages_orgin)
    
    # --------------------------------------------------------------------------
    # overall preprocessing steps in time domain is completed here 
    # Here onwards MT-estoimation and correlation based loose-lead detection begins
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    #this assign values are especillay for correlation based loose lead detection
    # adjust the boundry values and mapped by tanh
    # z_transform=True
    # --------------------------------------------------------------------------
    
    b_val=opt_paramters.b_val
    
    intersted_sleep_stages_REM=opt_paramters.intersted_sleep_stages_REM# R sleep stage
    intersted_sleep_stages_NREM=opt_paramters.intersted_sleep_stages_NREM#N1,N2,N3,N4
    intersted_sleep_stages_NREM_REM_comb=opt_paramters.intersted_sleep_stages_NREM_REM_comb#N1,N2,N3,N4 and R
    
    cross_correlation_ref_dic= opt_paramters.cross_correlation_ref_dic
    
    # --------------------------------------------------------------------------
    # db_scale = False means the correlation values are calculated from the basic amplitude
    # as this functions is limited to 1 sec sliding through the MT-spectrum
    # as the corrrelation value of the temporal information is obtained in 1 sec resolution can bve later down sampled 
    # like sliding 2 sec by removing the one sample in the middle (kind of downsampling)
    # 
    # The correltion is calulated bween the given frquency band f_min_interst-f_max_interst
    # Here we used the bandpassed range to calculate the correltaion
    #
    # save_MT_spectrum: will onkygive the spectrum output which can be later used to save
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # either filterd EEG can be fed through the MT-estimation 
    # --------------------------------------------------------------------------
    if filtered_EEG_MT:
        EEG_MT_feed = EEG_root_copy
    else:
        EEG_MT_feed =   EEG_root
        
    
    # --------------------------------------------------------------------------
    # to calculate correlation in the db scale or given scale
    # --------------------------------------------------------------------------
    db_scale=False
    correlation_flatten_MT_not_opt, sleep_stages_annot_flatten, MT_spec_not_db= MT_based_correltion_calc_in_continious_segs(EEG_MT_feed, sleep_stages, 
                                                                                                        cont_EEG_segments_np, 
                                            Fs,ch_names, start_time_idx=start_time_idx, T=T, db_scale=db_scale,
                                            f_min_interst=bandpass_freq[0], f_max_interst=bandpass_freq[1], 
                                            window_time=epoch_length, padding=0, sleep_stage_anoot_ext=True,
                                            interested_channels_index=[],root_data=True,
                                            b_val=b_val,inter_mediate_transform=False, 
                                            optim_bandwidth=False,
                                            save_MT_spectrum=True,
                                            GUI_percentile=GUI_percentile)
    
    '''
    spindle and MT-coorelatiion one hot encoding
    '''
    # --------------------------------------------------------------------------
    # spindle_interest_channel= like if we are only targetting the central channels for the spindle calculation
    # --------------------------------------------------------------------------
    if avoid_spindle_loc:
        # sp_sw_dic_format_copy =deepcopy(sp_sw_dic_format)
        spindle_possible_channels =[]#probabily on C3 like   
    
        spindle_enc, spindle_info_hold =  convert_to_full_one_hot_mapping_based_cont_seg_sp_sw_all_channels(sp_sw_dic_format,
                                                                          sleep_stages_or,cont_EEG_segments_np,Fs,
                                                                            MT_spec_not_db, start_time_idx, ep_length=30,
                                                                            o_p_adjuster=3,            boundry_cond_sec_frac=0.5,combine_intersted_channels_to_all=False,
                                                                            ch_names = ch_names, spindle_interest_channel=spindle_possible_channels)
        
    
    
        
    else:
        spindle_enc=[]
    
    if len(spindle_enc)>0:
        in_name_temp = in_name_temp+'_avoid_spindle'
    '''
      detect the outliers based only on using the mean-correlation values
      
          since the mtholdolgy proposed here has two approaches
      using the 
          approach-2: mean correlation 
                          or
          approach-1: independendtly use the cmbintaion
    '''
    # --------------------------------------------------------------------------
    #  break the contuniious grops while consideritin gthe given spindle or 
    #  already predicted i/p given
    # --------------------------------------------------------------------------
    break_spindle_flatten, break_flat_MT_flatten, z_transform,  inter_mediate_transform,  Fisher_based, flat_MT_consider =opt_paramters.intial_parameters_outlier_vertical_spikes()
    
    # --------------------------------------------------------------------------
    #  since the  flat_MT need to be donebased on the region of standard deviation
    #  that need to be selected whether region after preprcess or full 
    #  and the std_thres value need to be selected with caution
    # --------------------------------------------------------------------------
    if flat_MT_consider:
        # --------------------------------------------------------------------------
        #to consider the detection as continious block or not
        #  while the varience based loose-lead detection performed
        # --------------------------------------------------------------------------
        cont_seg_wise=False
        # only_flat_MT=False
        # --------------------------------------------------------------------------
        # we can assign the strict threhold like this or we can assign the moving window based approach
        # --------------------------------------------------------------------------
        std_thres=opt_paramters.std#this will only make the spikes this vlaue can be even lower like 3
        if not db_scale:
            #converting the MT_spec_raw to MT_spec_db   
            MT_spec_db = 10*np.log10(np.concatenate(MT_spec_not_db,axis=2))
        else:
            MT_spec_db = np.concatenate(MT_spec_not_db,axis=2)
        # cont_seg_wise=True
        flat_MT = find_vertical_spikes(MT_spec_db,MT_spec_not_db,std_thres=std_thres,cont_seg_wise=cont_seg_wise)
    else:
        flat_MT=[]
    
    # --------------------------------------------------------------------------
    # this is like using methology based on the distribution 
    # in this script we are not checking the distribution or moving window
    #  Here we used moving window based approach
    # --------------------------------------------------------------------------
    tail_check_bin, GMM_based,    factor_check, threh_prob, outlier_basic_con,\
            moving_window_based, moving_window_size, th_fact,\
            sep, global_check, only_good_points,\
            sorted_median_cont_grp_comp_sort_median_cond, sorted_median_cont_grp_comp_sort_max_cond, sorted_median_cont_grp_comp_sort_quan_cond,\
            cont_seg_wise, cont_threh, threh_prob_artf_cont_seg, \
            with_conv,thresh_min_conv,thresh_in_sec,outlier_presene_con_lenth_th, \
            with_fill_period=opt_paramters.methodology_related_paramaters_for_outlier_detection()
    # --------------------------------------------------------------------------
    #finalise the loose lead detection the following parameters are just annotate the whole lead as loose-lead based on the conditions
    # --------------------------------------------------------------------------
    loose_lead_period_min,\
        percentage_on_given_period_while_sliding,\
        overall_percent_check,\
        apply_conv_window,num_occurance, percent_check, loose_conv_wind,stride_size, conv_type = opt_paramters.assign_par_finalise_lead_loose_due_to_amoun_of_presented_artifacts()
    
    len_period_tol_min=loose_lead_period_min

            
    '''
    to get the preprocessed mapped information
    '''
    # --------------------------------------------------------------------------
    #  using the dictionary format that 
    #  this is due to easy to represent the events in loose-lead detection as well
    # 
    #  The following function map the preprocess events to the events-orgin index
    #  the mapped preprocessed event origin is later used for .edf creation
    # --------------------------------------------------------------------------

        
    # --------------------------------------------------------------------------
    # if we have plan to save the edf files 
    #  we need to keep this events files
    # --------------------------------------------------------------------------
    
    ep_loose_lead_pre_or, ep_loose_lead_dur_pre_or, even_id_unique_or =epoch_status_to_events_annot(epoch_status, sel_index_sl_st,  epoch_sec=epoch_length,
                                                                                ep_loose_lead={},    ep_loose_lead_dur={},    even_id_unique=[])
        
    ep_loose_lead_pre=deepcopy(ep_loose_lead_pre_or)
    ep_loose_lead_dur_pre=deepcopy(ep_loose_lead_dur_pre_or)
    even_id_unique=deepcopy(even_id_unique_or)
    '''
        to run the NREM and REM seperately 
    '''
    
    # --------------------------------------------------------------------------
    # to track the correlation-coefficients are based on the final loose-leads present
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # first time the mean are calcualted without knowing the mean
    # --------------------------------------------------------------------------
    loose_lead_channels_old=[]
    loose_lead_channels=[]
    
    # --------------------------------------------------------------------------
    # the number of itertaion are subject specific such that while loop is used 
    #  Here the "n" is number of channels O(n)
    #
    # the condition-1
    #   to force to run first time
    # --------------------------------------------------------------------------
    re_iter_count = 0
    
    # --------------------------------------------------------------------------
    # the condition-2 and 3 check the re-iteration of mean calculation essentiality
    # 
    # the condition-1
    #       (len(loose_lead_channels)< (len(ch_names)-2) checks
    # atleast one good- channel (not supected as loose lead till now)
    # excist for outlier calculation 
    # --------------------------------------------------------------------------
    # the condition-2
    #   len(loose_lead_channels_old)!=len(loose_lead_channels)
    #   make sure the re_iteration found any new loose-lead such that no-need for the re-iteration
    # --------------------------------------------------------------------------
    
    
    while (re_iter_count==0) or (len(loose_lead_channels)< (len(ch_names)-2) and len(loose_lead_channels_old)!=len(loose_lead_channels)):
        # --------------------------------------------------------------------------
        #   since we are calculated loose-leads based on the same condition
        # --------------------------------------------------------------------------
        print("re_iter_count: ",re_iter_count)
        if sep:
            # --------------------------------------------------------------------------
            #   this will make the outlier detection run on NREM and REM seperately
            #  make sure to mainatin a seperate copy of raw-calucltion to avoid recalculation of correlation
            # --------------------------------------------------------------------------
            corr_raw =deepcopy(correlation_flatten_MT_not_opt)
            
            # --------------------------------------------------------------------------
            # correlation_flatten_MT_not_opt_re is only obtained for saving the correlation for later analysis
            # --------------------------------------------------------------------------
    
            correlation_flatten_MT_not_opt_re = obtain_mean_corr(corr_raw, ch_names = ch_names,loose_lead_channels=loose_lead_channels)
        
            corr_raw =deepcopy(correlation_flatten_MT_not_opt)
            arousal_annot_NREM_REM_re = find_loose_leads_based_mean_seperately(corr_raw, sleep_stages_annot_flatten, 
                    ch_names,cross_correlation_ref_dic,
                    intersted_sleep_stages_NREM, intersted_sleep_stages_REM,
                    outlier_basic_con=outlier_basic_con, 
                    b_val=b_val,inter_mediate_transform=inter_mediate_transform,z_transform=z_transform,Fisher_based=Fisher_based,
                    thresh_min=5, #cont_threh=cont_threh,
                    flat_MT_consider=flat_MT_consider,
                    flat_MT=flat_MT,
                    avoid_spindle_loc=avoid_spindle_loc, spindle_enc=spindle_enc,
                    break_spindle_flatten=break_spindle_flatten, break_flat_MT_flatten=break_flat_MT_flatten,
                    tail_check_bin= tail_check_bin, factor_check=factor_check,                                         
                    GMM_based=GMM_based, threh_prob =threh_prob,          
                    cont_EEG_segments_np=cont_EEG_segments_np,
                    threh_prob_artf_cont_seg=threh_prob_artf_cont_seg,  cont_threh=cont_threh,# cont_seg_wise=cont_seg_wise,
                    moving_window_based=moving_window_based,
                    moving_window_size=moving_window_size, th_fact=th_fact,
            global_check = global_check, only_good_points=only_good_points,
            sorted_median_cont_grp_comp_sort_median_cond=sorted_median_cont_grp_comp_sort_median_cond,
        sorted_median_cont_grp_comp_sort_max_cond=sorted_median_cont_grp_comp_sort_max_cond,
        sorted_median_cont_grp_comp_sort_quan_cond=sorted_median_cont_grp_comp_sort_quan_cond,loose_lead_channels=loose_lead_channels)
        
        
        else:
            '''
                to run the NREM and REM combinely 
            '''
            corr_raw =deepcopy(correlation_flatten_MT_not_opt)
            # --------------------------------------------------------------------------
            # correlation_flatten_MT_not_opt_re is only obtained for saving the correlation for later analysis
            # --------------------------------------------------------------------------
            correlation_flatten_MT_not_opt_re = obtain_mean_corr(corr_raw, ch_names = ch_names,loose_lead_channels=loose_lead_channels)
        
            arousal_annot_NREM_REM_re = find_loose_leads_based_mean(corr_raw, sleep_stages_annot_flatten, 
                        ch_names,cross_correlation_ref_dic,
                        intersted_sleep_stages_NREM_REM_comb,
                        outlier_basic_con=outlier_basic_con, 
                        b_val=b_val,inter_mediate_transform=inter_mediate_transform,z_transform=z_transform,Fisher_based=Fisher_based,
                        flat_MT_consider=flat_MT_consider,
                        flat_MT=flat_MT,
                        avoid_spindle_loc=avoid_spindle_loc, spindle_enc=spindle_enc,
                        break_spindle_flatten=break_spindle_flatten, break_flat_MT_flatten=break_flat_MT_flatten,
                        tail_check_bin= tail_check_bin, factor_check=factor_check,                                         
                        GMM_based=GMM_based, threh_prob =threh_prob,          
                        cont_EEG_segments_np=cont_EEG_segments_np,
                        threh_prob_artf_cont_seg=threh_prob_artf_cont_seg,  cont_threh=cont_threh,#cont_seg_wise =cont_seg_wise,
                        moving_window_based=moving_window_based,
                        moving_window_size=moving_window_size, th_fact=th_fact,
                global_check = global_check, only_good_points=only_good_points,
                sorted_median_cont_grp_comp_sort_median_cond=sorted_median_cont_grp_comp_sort_median_cond,
        sorted_median_cont_grp_comp_sort_max_cond=sorted_median_cont_grp_comp_sort_max_cond,
        sorted_median_cont_grp_comp_sort_quan_cond=sorted_median_cont_grp_comp_sort_quan_cond,loose_lead_channels=loose_lead_channels,
        GUI_percentile=GUI_percentile)
        
    
        loose_lead_channels_old=deepcopy(loose_lead_channels)
        # break
        '''
            Earlier this was implemented using the hard threhold now this can be accept the parameter
             like pinpoint parameters while avoidng the conflict seperately accepting parameters to unify without effect
             like 
             with_fill_period, etc.
        '''
        loose_lead_once, loose_channel_pin_point, loose_lead_channels =  pin_point_loose_lead(arousal_annot_NREM_REM_re,
        apply_conv_window=apply_conv_window, thresh_min_conv=thresh_min_conv, thresh_in_sec=thresh_in_sec, outlier_presene_con_lenth_th=outlier_presene_con_lenth_th,
                loose_check_with_fill_period=with_fill_period,  len_period_tol_min=len_period_tol_min,
                loose_lead_period_min =loose_lead_period_min,overall_percent_check=overall_percent_check,
                percentage_on_given_period_while_sliding=percentage_on_given_period_while_sliding,
                num_occurance=num_occurance, percent_check= percent_check,
                loose_conv_wind=loose_conv_wind,     stride_size=stride_size, conv_type=conv_type,
                ch_names = ch_names)
        
        # --------------------------------------------------------------------------
        # to save the results in first iteration
        # --------------------------------------------------------------------------
        if re_iter_count==0:
            temp['loose_lead_channels']=[]
            if sep:
                np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_NREM_REM_sep_outlier_annot",arousal_annot_NREM_REM_re)
    
            else:
                np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_NREM_REM_com_outlier_annot",arousal_annot_NREM_REM_re)
       
        # --------------------------------------------------------------------------
        # to avoid the re-iterations
        # --------------------------------------------------------------------------
        if loose_lead_once:
            re_iter_count +=1
        else:
            break
    
        # loose_lead_channels = list(set(loose_lead_channels_old+loose_lead_channels))
        # print("loose lead chanenls: ",loose_lead_channels)
    
    if re_iter_count>0:
        temp['loose_lead_channels']=deepcopy(loose_lead_channels)
    
        if loading_dir_pre.keep_signature_dic['annota_NREM_REM']:
        # --------------------------------------------------------------------------
        #  if the user don't need this no-need save the pickles
        # --------------------------------------------------------------------------
    
            if sep:
                np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_NREM_REM_sep_outlier_annot_re",arousal_annot_NREM_REM_re)
                np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_correlation_flatten_MT_not_opt_re_sep_",correlation_flatten_MT_not_opt_re)
            else:
                np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_NREM_REM_com_outlier_annot_re",arousal_annot_NREM_REM_re)
                np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_correlation_flatten_MT_not_opt_re_com_",correlation_flatten_MT_not_opt_re)
    # if re_iter_count==0:
    #     if sep:
    #         np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_correlation_flatten_MT_not_opt_re_sep_",correlation_flatten_MT_not_opt_re)
    #     else:
    #         np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_correlation_flatten_MT_not_opt_re_com_",correlation_flatten_MT_not_opt_re)
    
    '''
    atleast one outlier need to be detected to annotate the events
    '''
    
    if np.sum(arousal_annot_NREM_REM_re)>0:
        # --------------------------------------------------------------------------
        #  for creating tghe edf files we need this
        # --------------------------------------------------------------------------
    
        ep_loose_lead = deepcopy(ep_loose_lead_pre)
        ep_loose_lead_dur = deepcopy(ep_loose_lead_dur_pre)
    
        ep_loose_lead, ep_loose_lead_dur,even_id_unique = main_loose_lead_to_origin_events_dic(arousal_annot_NREM_REM_re, sel_index_sl_st, cont_EEG_segments_np,   
                                                                                ep_loose_lead=ep_loose_lead,    ep_loose_lead_dur=ep_loose_lead_dur, even_id_unique=even_id_unique,
                                              epoch_sec=30, T=4, sliding_size=1, ch_names=   ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'],
                                              assign_comment= 'loose lead', with_conv=True,
                                          outlier_presene_con_lenth_th=outlier_presene_con_lenth_th, thresh_min_conv=thresh_min_conv, thresh_in_sec= True, conv_type = "same",
                                          with_fill_period=with_fill_period,  len_period_tol_min=len_period_tol_min, show_single_outliers_before_combine_tol=True)
                   
        whole_annot_cp = event_annotations_origin_for_edf(whole_annotations, ep_loose_lead, ep_loose_lead_dur, in_place_event=False,  along_with_original_event_id=False)
        if sep:
            with open(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_epoch_sep_status.pickle", 'wb') as handle:
                pickle.dump(whole_annot_cp, handle, protocol=pickle.HIGHEST_PROTOCOL)
            del handle
        #    np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_epoch_sep_status",whole_annot_cp)
        else:
            with open(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_epoch_com_status.pickle", 'wb') as handle:
                pickle.dump(whole_annot_cp, handle, protocol=pickle.HIGHEST_PROTOCOL)
            del handle
        #    np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_epoch_com_status",whole_annot_cp)
        '''
        to save the events inplace
        '''    
        ep_loose_lead = deepcopy(ep_loose_lead_pre)
        ep_loose_lead_dur = deepcopy(ep_loose_lead_dur_pre)
    
        ep_loose_lead, ep_loose_lead_dur,even_id_unique = main_loose_lead_to_origin_events_dic(arousal_annot_NREM_REM_re, sel_index_sl_st, cont_EEG_segments_np,   
                                                                                ep_loose_lead=ep_loose_lead,    ep_loose_lead_dur=ep_loose_lead_dur, even_id_unique=even_id_unique,
                                              epoch_sec=30, T=4, sliding_size=1, ch_names=   ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'],
                                              assign_comment= 'loose lead', with_conv=with_conv,
                                          outlier_presene_con_lenth_th=outlier_presene_con_lenth_th, thresh_min_conv=thresh_min_conv, thresh_in_sec= True, conv_type = "same",
                                          with_fill_period=with_fill_period,  len_period_tol_min=len_period_tol_min, show_single_outliers_before_combine_tol=True)
        # --------------------------------------------------------------------------
        #  to save teh events inplce
        # --------------------------------------------------------------------------
    
        whole_annot_cp = event_annotations_origin_for_edf(whole_annotations, ep_loose_lead, ep_loose_lead_dur, in_place_event=True,  along_with_original_event_id=True)
        
        sleep_events_with_final_marking_epoches = whole_annot_cp[sel_index_sl_st][:,2]
        # --------------------------------------------------------------------------
        # this is only to save the loose-leads, with inplace since the epoch number is obtained from the inplcae location 
        # --------------------------------------------------------------------------
        only_sleep_epoches_events_to_txt(in_name_temp+'_only_loose_lead.txt', loading_dir_pre.out_loc_txt, sleep_events_with_final_marking_epoches, comma_sep=False, save_csv=False)
        if sep:
            np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_epoch_sep_status_inplace",whole_annot_cp)
        else:
            np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_epoch_com_status_inplace",whole_annot_cp)
    if loading_dir_pre.keep_signature_dic['annota_NREM_REM']:
        # --------------------------------------------------------------------------
        #  if the user don't need this no-need save the pickles
        # --------------------------------------------------------------------------
        np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_cont_EEG_segments_np",cont_EEG_segments_np)
        
        np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_epoch_start_idx",epoch_start_idx)
        #the epoch status and the start_time_idx both are same just placed here for cheking purpose
        np.save(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_start_time_idx",start_time_idx)
    
    
    '''
    saving modiifed EDF 
    
    '''
    #  we need to call this when using MNE higher verstion since this package build in MNE version 0.2.3.4
    #  MNE version 0.2.4 suppots
    # edf_obj = EDFInfo(path_edf_file=in_edf, path_annotations=path_annotations)
    # edf_obj.createEDF()
    
    with open(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_epoch_status.pickle", 'wb') as handle:
        pickle.dump(epoch_status, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle
    
    if loading_dir_pre.keep_signature_dic['annota_NREM_REM']:
        # --------------------------------------------------------------------------
        #  if the user don't need this no-need save the pickles
        # --------------------------------------------------------------------------
        with open(loading_dir_pre.out_loc_NREM_REM+ in_name_temp + "_corr_MT_not_opt.pickle", 'wb') as handle:
            pickle.dump(correlation_flatten_MT_not_opt, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del handle
    
    if loading_dir_pre.keep_signature_dic['sleep_anot']:
        # --------------------------------------------------------------------------
        #  if the user don't need this no-need save the pickles
        # --------------------------------------------------------------------------
        np.save(loading_dir_pre.out_loc_outlier_sleep_anot +in_name_temp + '_sleep_annot_flatten',sleep_stages_annot_flatten)
    
    if loading_dir_pre.keep_signature_dic['MT_spec']:
        # --------------------------------------------------------------------------
        #  if the user don't need this no-need save the pickles
        # --------------------------------------------------------------------------
        with open(loading_dir_pre.out_loc_outlier_MT_spec+in_name_temp+'_MT.pickle', 'wb') as handle:
            pickle.dump(MT_spec_not_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del handle
    
    if loading_dir_pre.keep_signature_dic['dic']:
        dic_name = loading_dir_pre.out_loc_dic + in_name_temp + "_dic.txt"
        temp_dict = [deepcopy(temp)]
        temp_df = pd.DataFrame.from_dict(temp_dict) 
        temp_df.to_csv(dic_name, sep=',', header=True, index=True)


if __name__ == '__main__':
    '''
    getting user input and call loose-lead detection
    '''
    opt_paramters = parameter_assignment()
    loading_dir_pre = opt_paramters.assign_user_inputs_for_directories()
    # logger.debug(loading_dir_pre.edf_files)
    for f in loading_dir_pre.edf_files:
        EEG_sleep_loose_lead(f,opt_paramters)
    