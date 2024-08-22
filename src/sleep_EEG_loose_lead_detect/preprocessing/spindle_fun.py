#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:20:19 2022

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved


The spindle detection is fully based on the YASA

to avoid confustions final the indexes of epoch_segemnets are relative to the intial-given-segments such those start ids are directly usable
 (not relative to the sel_ids e.g: good_ids)

 modified on Dec-01-2022 to include sw_sat, sp_sat returm
"""
import numpy as np
import os
import logging

from copy import deepcopy
from yasa import sw_detect,spindles_detect

from sleep_EEG_loose_lead_detect.preprocessing.cont_segs_of_whole_EEG import find_cont_EEG_segments, continious_seg_to_np
from sleep_EEG_loose_lead_detect.GUI_interface.percentage_bar_vis  import percent_complete

logger = logging.getLogger("spindle")
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


sw_columns_ordered =  ['Start', 'End', 'MidCrossing','Duration', 'NegPeak', 'PosPeak',  'ValNegPeak', 'ValPosPeak', 
                       'PTP', 'Slope', 'Frequency',  'IdxChannel']
    
sp_columns_ordered =  ['Start', 'End', 'Duration', 'Peak', 'Amplitude', 'RMS', 'AbsPower',
        'RelPower', 'Frequency', 'Oscillations', 'Symmetry', 'IdxChannel']




def int_por_in_the_chunk(ev_np,chunk_start_exact_time_pos,Fs,s_loc=0,e_loc=2):
    '''
    This function gives the exact event_location in overall EEG group
    s_loc= Start coloumn index
    e_loc = End
    '''
    ev_np[:,[s_loc,e_loc]] = (np.copy(ev_np[:,[s_loc,e_loc]]) *Fs)+chunk_start_exact_time_pos
    return ev_np

def find_epoch_position_relevant_to_spindle_or_slow_wave(exact_pos_int,start_time_idx,window_size):
    '''
    this function finds the segmented epoche"s location relevant to spindle
    
    '''
    eph_split_chk_ind = -1
    while exact_pos_int < start_time_idx[eph_split_chk_ind]:
        eph_split_chk_ind=eph_split_chk_ind-1
        
    eph_split_chk_ind = len(start_time_idx)+eph_split_chk_ind
    if not exact_pos_int<start_time_idx[eph_split_chk_ind]+window_size:
        raise("some thing wrong in epoch position sel for spindle")
        
    return eph_split_chk_ind

def sw_or_spindle_objest_postion_info_ret(ob,sel_cols,group_of_interest_por, eeg_seg_interest_pos, unique_all, sleep_stages, chunk_start_exact_time_pos, start_time_idx, window_size, Fs=256,s_loc=0,e_loc=1):
    '''
    This is the main function retrieve the information of epoches
    
    '''
    # incase no-spindle found in the algorithm
    events =  ob.summary()
    ev_np = events[sel_cols].to_numpy()

    ev_np = int_por_in_the_chunk(ev_np,chunk_start_exact_time_pos,Fs,s_loc=s_loc,e_loc=e_loc)

    masks_all = deepcopy(ob.get_mask())
    group_of_interest_por.append(deepcopy(ev_np))
    
    #first two indexes point the epoch of the segments last two the hypnos sleep
    eeg_seg_interest_pos_np=np.zeros((np.shape(ev_np)[0],4))
    for ev in range(np.shape(ev_np)[0]):
        # to find the EEG_seg portion's spindle/ sw location
        eeg_seg_interest_pos_np[ev,0] = find_epoch_position_relevant_to_spindle_or_slow_wave(ev_np[ev,s_loc],start_time_idx,window_size)
        eeg_seg_interest_pos_np[ev,1] = find_epoch_position_relevant_to_spindle_or_slow_wave(ev_np[ev,e_loc],start_time_idx,window_size)
     
        eeg_seg_interest_pos_np[ev,2]=sleep_stages[int(ev_np[ev,s_loc])]
        eeg_seg_interest_pos_np[ev,3]=sleep_stages[int(ev_np[ev,e_loc])]

        if len(np.unique(eeg_seg_interest_pos_np[ev,[0,1]]))!=1 and  len(np.unique(eeg_seg_interest_pos_np[ev,[2,3]]))!=1:
            unique_all=False
    eeg_seg_interest_pos.append(deepcopy(eeg_seg_interest_pos_np))
    return group_of_interest_por, eeg_seg_interest_pos, masks_all, unique_all





def spindle_detect_main(in_name_temp, sel_ids, start_time_idx, data, sleep_stages, save_spindle_loc, sel_id_name='good_ids',
                        Fs=256, epoch_length=30, window_time=30, s_loc=0, e_loc=1,
                        sw_columns_ordered=sw_columns_ordered,sp_columns_ordered=sp_columns_ordered,
                        pred_slow_waves=True, pred_spindles =True,
                        save_pred_spindle=False,save_pred_slow_waves=False,
                        save_cont_seg=False,
                        verbose=False,GUI_percentile=True,
                        yasa_verbose=False):
    '''
    this function first break into continious segments (to make sure to avoid the intereputions not effect)
    '''
    if not yasa_verbose:
        yasa_logger = logging.getLogger("yasa")
        # to avoid pring the not detecting any spindles
        yasa_logger.disabled = True
        
        # --------------------------------------------------------------------------
        # to turnoff the MNE logger used by the YASA
        # --------------------------------------------------------------------------
        mne_logger = logging.getLogger("mne")  # one selection here used across mne-python
        mne_logger.disabled = True
        logger.info("MNE verbose deactivated")

        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    if verbose:
        if pred_spindles:
            logger.info("Running spindle detection  %s" , (in_name_temp))
        if pred_slow_waves:
            logger.info("Running slow-wave detection  %s" , (in_name_temp))


    window_size = int(round(window_time*Fs))


    # --------------------------------------------------------------------------
    # 
    # same as save continiuous segement part
    # --------------------------------------------------------------------------

    epoch_groups_sample_size=Fs*epoch_length
    
    cont_EEG_segments = find_cont_EEG_segments(sel_ids,start_time_idx,epoch_groups_sample_size)

    start_keys = list(cont_EEG_segments.keys())
    cont_EEG_segments_np = continious_seg_to_np(cont_EEG_segments, start_keys)
    
    """
     spindle detection
        
        sw_columns = ['Start', 'NegPeak', 'MidCrossing', 'PosPeak', 'End', 'Duration',
               'ValNegPeak', 'ValPosPeak', 'PTP', 'Slope', 'Frequency', 'Channel',
               'IdxChannel']
            sp_coloumns= ['Start', 'Peak', 'End', 'Duration', 'Amplitude', 'RMS', 'AbsPower',
                'RelPower', 'Frequency', 'Oscillations', 'Symmetry', 'Channel',
                'IdxChannel']
    """     
    logger.info("Running on continoius EEG segments of = %i", len(start_keys))
    
    sw_unique_all=True
    sp_unique_all=True
    
    sw_sat =False
    sp_sat=False

    sw_group_of_interest_por=[]
    sp_group_of_interest_por=[]

    
    sw_eeg_seg_interest_pos = []
    sp_eeg_seg_interest_pos = []
    sw_co_occured=[]
    for sel_chk in range(0,len(start_keys)):
        '''
        retrieve the data for applying spindle detection algorithm
        select the continious EEG portion (chunk) 
        '''
        chunk_start_exact_time_pos= start_time_idx[start_keys[sel_chk]]
        if cont_EEG_segments[start_keys[sel_chk]]==len(start_time_idx):
            chunk_end_exact_time_pos=int(start_time_idx[-1]+epoch_groups_sample_size)
        else:
            chunk_end_exact_time_pos=start_time_idx[cont_EEG_segments[start_keys[sel_chk]]]
        sel_EEG_chunk =  data[:,chunk_start_exact_time_pos:chunk_end_exact_time_pos]
        # corres_hyno_chunk = sleep_stages[chunk_start_exact_time_pos:chunk_end_exact_time_pos]
            
        '''
        need to get the hypnogram information 
        to with the selected segments as it is
        '''
        
        
        '''
        here apply selected spindle detection algorithm/ slow-wave detction algorithms
        based on the default par in 0.6.1 YASA
        '''
        if pred_spindles:
            sp = spindles_detect(sel_EEG_chunk, Fs,  include=(1, 2, 3), 
                                 freq_sp=(12, 15),  freq_broad=(1, 30), duration=(0.5, 2), min_distance=500, 
                                 thresh={"rel_pow": 0.2, "corr": 0.65, "rms": 1.5},
                                 multi_only=False,    remove_outliers=False,     verbose=False)
            if not isinstance(sp, type(None)):
                sp_sat=True
                sp_group_of_interest_por, sp_eeg_seg_interest_pos, sp_masks, sp_unique_all = sw_or_spindle_objest_postion_info_ret(sp,sp_columns_ordered,sp_group_of_interest_por, sp_eeg_seg_interest_pos, sp_unique_all, sleep_stages, chunk_start_exact_time_pos, start_time_idx, window_size, Fs=Fs, s_loc=s_loc,e_loc=e_loc)
              
                sp_group_of_interest_por_com = np.vstack(sp_group_of_interest_por)
                sp_eeg_seg_interest_pos_com = np.vstack(sp_eeg_seg_interest_pos)
            
        '''
        slow wave detection
        based on the default par in 0.6.1 YASA
        '''
        if pred_slow_waves:
            sw = sw_detect(sel_EEG_chunk, Fs, include=(2, 3), freq_sw=(0.3, 1.5),
                        dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_neg=(40, 200),
                        amp_pos=(10, 150), amp_ptp=(75, 350), coupling=False,
                        remove_outliers=False, verbose=False)
            '''
            
            choose the slow_wave coloumns and spindle coloumns to be appear 
            '''    
            if not isinstance(sw, type(None)):
                sw_sat =True
                sw_group_of_interest_por, sw_eeg_seg_interest_pos, sw_masks, sw_unique_all = sw_or_spindle_objest_postion_info_ret(sw,sw_columns_ordered,sw_group_of_interest_por, sw_eeg_seg_interest_pos, sw_unique_all, sleep_stages, chunk_start_exact_time_pos, start_time_idx, window_size, Fs=Fs, s_loc=s_loc,e_loc=e_loc)

                sw_group_of_interest_por_com = np.vstack(sw_group_of_interest_por)
                sw_eeg_seg_interest_pos_com = np.vstack(sw_eeg_seg_interest_pos)
    
        
        '''
        include the spindle and SW overlapped portions to find any relevant info
        '''
    
        if (sw_sat and sp_sat):
            try:
                sw_co_occured = sw.find_cooccurring_spindles(sp.summary(), lookaround=1.2)
                if not isinstance(sw_co_occured, type(None)):
                    sw_co_occured.append([deepcopy(sw_co_occured),deepcopy(sw)])
            except:
                pass
        # logger.info("Remaining continoius EEG segments of = %i", len(start_keys)-sel_chk)
        if verbose:
            if sel_chk%10==0:
                logger.info("Remaining %i continoius EEG segments  %s" % (len(start_keys)-sel_chk,in_name_temp))
                
        # --------------------------------------------------------------------------
        if GUI_percentile:
            percent_complete((sel_chk/len(start_keys)*100), 100, bar_width=60, title="Spindle detectiion", print_perc=True)
            
    if verbose:
        logger.info("Done spindle detection  %s" , (in_name_temp))
    if GUI_percentile:
        percent_complete(100, 100, bar_width=60, title="Spindle detectiion", print_perc=True)
        
    if save_cont_seg or save_pred_slow_waves or save_pred_spindle:
        os.chdir('/')
        os.chdir(save_spindle_loc)
                
    if save_cont_seg:
        np.save(in_name_temp+'_cont_EEG_seg_info_'+sel_id_name,cont_EEG_segments)


    if save_pred_slow_waves and pred_slow_waves and sw_sat:
        np.save(in_name_temp+'_sw_fea_'+sel_id_name,sw_group_of_interest_por_com)
        np.save(in_name_temp+'_sw_sl_st_info_'+sel_id_name,sw_eeg_seg_interest_pos_com.astype(int))
    elif pred_slow_waves and not sw_sat:
        logger.warning('No Slow wave is detected for %s',in_name_temp)
        sw_group_of_interest_por_com=[] 
        sw_eeg_seg_interest_pos_com=[]

    if save_pred_spindle and pred_spindles and sp_sat:

        np.save(in_name_temp+'_sp_fea_'+sel_id_name,sp_group_of_interest_por_com)
        np.save(in_name_temp+'_sp_sl_st_info_'+sel_id_name,sp_eeg_seg_interest_pos_com.astype(int))
    elif pred_spindles and not sp_sat:
        logger.warning('No spindle detected for %s',in_name_temp)
        sp_group_of_interest_por_com=[]
        sp_eeg_seg_interest_pos_com=[]
   
    if not pred_spindles:
        sp_group_of_interest_por_com=[]
        sp_eeg_seg_interest_pos_com=[]
    
    if not pred_slow_waves:
        sw_group_of_interest_por_com=[] 
        sw_eeg_seg_interest_pos_com=[]
        

    # --------------------------------------------------------------------------
    #  this will hold all the values of spindles in one dictionary format  
    # 
    # --------------------------------------------------------------------------

    sp_sw_dic_format={}
    sp_sw_dic_format['sw_sat']=sw_sat
    sp_sw_dic_format['sp_sat']=sp_sat
    sp_sw_dic_format['sp_eeg_seg_interest_pos_com']=sp_eeg_seg_interest_pos_com
    sp_sw_dic_format['sp_group_of_interest_por_com']=sp_group_of_interest_por_com
    sp_sw_dic_format['sw_eeg_seg_interest_pos_com']=sw_eeg_seg_interest_pos_com
    sp_sw_dic_format['sw_group_of_interest_por_com']=sw_group_of_interest_por_com
   
    # --------------------------------------------------------------------------
    # to turn-ON the MNE logger
    # --------------------------------------------------------------------------
    if not yasa_verbose:
        mne_logger = logging.getLogger("mne")  # one selection here used across mne-python
        mne_logger.disabled = False    
    return cont_EEG_segments,start_keys,cont_EEG_segments_np, sp_sw_dic_format


