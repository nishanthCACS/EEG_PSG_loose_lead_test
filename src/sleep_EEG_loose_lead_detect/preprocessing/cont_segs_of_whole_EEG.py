#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:22:52 2022

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

once the preprocess intial steps are done thhen there will be the continious chaukis of the EEG data
these functions are intended to seperate the continious segmentation from whole EEGs
"""
import os
import logging

import numpy as np

from copy import deepcopy
from sleep_EEG_loose_lead_detect.GUI_interface.percentage_bar_vis  import percent_complete

logger = logging.getLogger("cont_EEG_segs")
while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def chk_next_continious(sel_ids,start_pos_sel_id_indx,start_time_idx,epoch_groups_sample_size):
    '''
    This function checks the next epoch's starting position iscontinous  for the selected_pos_idexes 
    
    '''
    return start_time_idx[sel_ids[start_pos_sel_id_indx]]+epoch_groups_sample_size == start_time_idx[sel_ids[start_pos_sel_id_indx+1]]


def find_cont_EEG_segments(sel_ids,start_time_idx,epoch_groups_sample_size=0,Fs=256,epoch_length=30,GUI_percentile=True):

    '''
    This function uses start_time_idx to obtain the continious segments
    
         to do this go-through the goodids and starindex to find teh continious EEG groups
    
    start_time_idx: strating positions of the sel_ids correspondiong epoches
    Fs= sampling freq 
    
    return the 
        cont_EEG_segments: sel_ids as starting with correspoding ending locations (on the already splited epoch groups)
    '''
    if GUI_percentile:
        percent_complete(0, 100, bar_width=60, title="contniuous-grouping", print_perc=True)

    if epoch_groups_sample_size==0:
        epoch_groups_sample_size = Fs*epoch_length
    
    sel_ids = np.sort(sel_ids)
    
    start_pos_sel_id_indx =0
    start_chunk_key_idex=0
    
    cont_EEG_segments={}
    while start_pos_sel_id_indx<len(sel_ids)-1:
        # --------------------------------------------------------------------------
        # based on the start_pos_sel_id_indx on the sel_ids, 
        #  to decide the continuity
        # check the distance between the next epoch
        # --------------------------------------------------------------------------
        if chk_next_continious(sel_ids, start_pos_sel_id_indx,start_time_idx,epoch_groups_sample_size):
            # --------------------------------------------------------------------------
            #  if the next epoch is continious just goahead and check the next epoch
            # --------------------------------------------------------------------------
            start_pos_sel_id_indx=start_pos_sel_id_indx+1
        else:
            # --------------------------------------------------------------------------
            # Else just assign the lastly enocunterd sel_ids[start_pos_sel_id_indx] 
            # next index as ending of continious epoch
            #  since the python last index is not included in the counting
            # --------------------------------------------------------------------------

            cont_EEG_segments[sel_ids[start_chunk_key_idex]]= deepcopy(sel_ids[start_pos_sel_id_indx]+1)
            start_pos_sel_id_indx=start_pos_sel_id_indx+1
            start_chunk_key_idex = deepcopy(start_pos_sel_id_indx)
        
        # --------------------------------------------------------------------------
        # shows the approximate percentage of the continious segmentation done
        # this part not guranttes thos much percentage exist for calculation
        # --------------------------------------------------------------------------
        if GUI_percentile:
            percent_complete(start_pos_sel_id_indx, len(sel_ids), bar_width=60, title="contniuous-grouping", print_perc=True)
    if GUI_percentile:
        percent_complete(100, 100, bar_width=60, title="contniuous-grouping", print_perc=True)
    # --------------------------------------------------------------------------
    # finalise the border crteria
    # --------------------------------------------------------------------------
    if sel_ids[start_chunk_key_idex] not in list(cont_EEG_segments.keys()):
        cont_EEG_segments[sel_ids[start_chunk_key_idex]] = sel_ids[-1]+1

    return cont_EEG_segments

def save_cont_segs(in_name_temp, save_spindle_loc, sel_ids, start_time_idx, Fs, sel_id_name='good_ids', epoch_length=30,
                   save_cont_seg=True,GUI_percentile=True):
    '''
        this function returns only the continious segments
        for the given epoch_groups_sample_size in secs (default=30sec)
    '''
    
    epoch_groups_sample_size=Fs*epoch_length
    cont_EEG_segments = find_cont_EEG_segments(sel_ids,start_time_idx,epoch_groups_sample_size,GUI_percentile=GUI_percentile)
    if save_cont_seg:
        os.chdir('/')
        os.chdir(save_spindle_loc)
        np.save(in_name_temp+'_cont_EEG_seg_info_'+sel_id_name,cont_EEG_segments)
    else:
        return cont_EEG_segments
    

def continious_seg_to_np(cont_EEG_segments, start_keys):
    '''
    This function converts the cont_EEG_segments to numpy format with start_epoch_index and end_epoch_index
    '''
    cont_EEG_segments_np = np.zeros((len(start_keys),2))
    for s_k_i in range(0,len(start_keys)):
        cont_EEG_segments_np[s_k_i,0]=start_keys[s_k_i]
        cont_EEG_segments_np[s_k_i,1]=cont_EEG_segments[start_keys[s_k_i]]
    
    return cont_EEG_segments_np.astype(int)

def function_change_rel_idx_of_cont_segs(cont_EEG_segments):
    '''
    
    cont_EEG_segments : in the given EEG files segments

    '''
    # --------------------------------------------------------------------------
    # convert the continious segments to relative index
    # --------------------------------------------------------------------------
    cont_segmnets_rel_index =np.zeros((len(cont_EEG_segments),2))
    start_idx=0
    for i in range(0,len(cont_EEG_segments)):
        cont_segmnets_rel_index[i,0]=int(start_idx)
        start_idx = start_idx+cont_EEG_segments[i,1]-cont_EEG_segments[i,0]
        cont_segmnets_rel_index[i,1]=int(start_idx)
    
    return cont_segmnets_rel_index.astype(int)
