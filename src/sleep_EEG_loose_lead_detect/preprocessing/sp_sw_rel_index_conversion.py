#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 00:13:49 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved
"""


import numpy as np
import logging

from copy import deepcopy


logger = logging.getLogger("sp_sw_rel_index_convertion")
while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def convert_the_indexes_rel_to_sel_ids(sp_sw_dic_format,
                                       cont_EEG_segments,sel_ids, start_time_idx,
         find_sw_an_spinldes_out_in_same=True,
         star_end_cond=True, based_on_ep=True, based_on_sl_st=False,
         save_rel_indexs=False,save_converted_index_loc='',sel_id_name='good_ids'):
        
    '''
    obtain the statistics of the sleep stages with epoches
    
    Find both spindles and Slow-waves together
    find_sw_an_spinldes_out_in_same=True
    
    
    star_end_cond=True
    based_on_ep=True
    based_on_sl_st=False
    
    sw_sat: to check  atleast one slow-wave detected
    sp_sat: to check atleast one spindle detected

    this function will convert the given indexes relative to th eselectd ids and for further processing with MT-sec wise usage
    '''

    
    sw_sat =  sp_sw_dic_format['sw_sat']
    sp_sat =  sp_sw_dic_format['sp_sat']
    sp_eeg_seg_interest_pos_com = sp_sw_dic_format['sp_eeg_seg_interest_pos_com']
    sp_group_of_interest_por_com = sp_sw_dic_format['sp_group_of_interest_por_com']
    sw_eeg_seg_interest_pos_com = sp_sw_dic_format['sw_eeg_seg_interest_pos_com']
    sw_group_of_interest_por_com = sp_sw_dic_format['sw_group_of_interest_por_com']


    
    # --------------------------------------------------------------------------
    # first convert the spindle/ slowave occuring poistion relative to the epoch's starting position as
    # --------------------------------------------------------------------------
    if sw_sat:
        sw_group_of_interest_por_com = find_sp_se_relative_postion_in_epoch(sw_group_of_interest_por_com,sw_eeg_seg_interest_pos_com,start_time_idx)

    if sp_sat:
        sp_group_of_interest_por_com = find_sp_se_relative_postion_in_epoch(sp_group_of_interest_por_com,sp_eeg_seg_interest_pos_com,start_time_idx)
        
    if sw_sat:
        sw_cont_spindle_contain_segs_np = find_continious_seg_for_the_given_main(sw_eeg_seg_interest_pos_com, cont_EEG_segments, 
                                               star_end_cond=star_end_cond,based_on_ep=based_on_ep,based_on_sl_st=based_on_sl_st)
    if sp_sat:
        sp_cont_spindle_contain_segs_np = find_continious_seg_for_the_given_main(sp_eeg_seg_interest_pos_com, cont_EEG_segments, 
                                               star_end_cond=star_end_cond,based_on_ep=based_on_ep,based_on_sl_st=based_on_sl_st)
    
    # --------------------------------------------------------------------------
    # then greb the starting ending position and the channel poition as seprate blocks
    # this canbe easily done by grebbing first two indexes and last-channel index

    # convert the index relative to the sel_ids
    # --------------------------------------------------------------------------
    if sp_sat:
        sp_cont_segs_np_in_sel_ids_index = convert_the_indexes_to_sel_idexes(sp_cont_spindle_contain_segs_np,sel_ids)
    if sw_sat:
        sw_cont_segs_np_in_sel_ids_index = convert_the_indexes_to_sel_idexes(sw_cont_spindle_contain_segs_np,sel_ids)     
  
    logger.info("loading sp_eeg_seg_interest_pos_com ending position can be neglected %s",in_name_temp)

    if sp_sat:
        sp_eeg_seg_interest_pos_com[:,[0,1]] = convert_the_indexes_to_sel_idexes_of_interest_pos(sp_eeg_seg_interest_pos_com[:,[0,1]], sel_ids)
    if sw_sat:
        sw_eeg_seg_interest_pos_com[:,[0,1]] = convert_the_indexes_to_sel_idexes_of_interest_pos(sw_eeg_seg_interest_pos_com[:,[0,1]], sel_ids)
   
    
    sp_eeg_seg_interest_pos_com_rel ={}
    
    if save_rel_indexs:
        sel_id_name = 'rel_ind_'+  sel_id_name

        os.chdir('/')
        os.chdir(save_converted_index_loc)
        
    if save_rel_indexs and sp_sat:
        np.save(in_name_temp+'_sp_fea_'+sel_id_name,sp_group_of_interest_por_com)
        np.save(in_name_temp+'_sp_cont_EEG_seg_info__'+sel_id_name,sp_cont_segs_np_in_sel_ids_index)
        np.save(in_name_temp+'_sp_sl_st_info_'+sel_id_name,sp_eeg_seg_interest_pos_com.astype(int))
    if sp_sat:
        sp_eeg_seg_interest_pos_com_rel['sp_eeg_seg_interest_pos_com']=sp_eeg_seg_interest_pos_com.astype(int)
        sp_eeg_seg_interest_pos_com_rel['sp_group_of_interest_por_com']=sp_group_of_interest_por_com
        sp_eeg_seg_interest_pos_com_rel['sp_cont_segs_np_in_sel_ids_index']=sp_cont_segs_np_in_sel_ids_index

        
    if save_rel_indexs and sw_sat:
        np.save(in_name_temp+'_sw_fea_'+sel_id_name,sw_group_of_interest_por_com)
        np.save(in_name_temp+'_sw_cont_EEG_seg_info__'+sel_id_name,sw_cont_segs_np_in_sel_ids_index)
        np.save(in_name_temp+'_sw_sl_st_info_'+sel_id_name,sw_eeg_seg_interest_pos_com.astype(int))
    
    if sw_sat:
        sp_eeg_seg_interest_pos_com_rel['sw_eeg_seg_interest_pos_com']=sw_eeg_seg_interest_pos_com.astype(int)
        sp_eeg_seg_interest_pos_com_rel['sw_group_of_interest_por_com']=sw_group_of_interest_por_com
        sp_eeg_seg_interest_pos_com_rel['sw_cont_segs_np_in_sel_ids_index']=sw_cont_segs_np_in_sel_ids_index

        
    sp_eeg_seg_interest_pos_com_rel['sw_sat']=sw_sat
    sp_eeg_seg_interest_pos_com_rel['sp_sat']=sp_sat

    
    return sp_eeg_seg_interest_pos_com_rel



def find_sp_se_relative_postion_in_epoch(given_group_of_int,eeg_seg_interest_pos_com,start_time_idx):
    # --------------------------------------------------------------------------
    # helps to retrieve the given_groups interested portion sleep-spindles or slow-waves starting postion relave to the given epoch
    # --------------------------------------------------------------------------
    start_end_pos_in_sel_ind = given_group_of_int[:,[0,1]]
    sel_epoch_ind_t = eeg_seg_interest_pos_com[:,0].astype(int)
    sel_epoch_ind = [sel_id for sel_id in sel_epoch_ind_t]
    
    start_end_pos_in_sel_ind_rel_epoch = np.zeros(np.shape(start_end_pos_in_sel_ind))
    # using the starting time index to find the relative position of the spindle/ slow-wave in the correspodning epoches
    start_end_pos_in_sel_ind_rel_epoch[:,0] = start_end_pos_in_sel_ind[:,0] - start_time_idx[sel_epoch_ind] 
    start_end_pos_in_sel_ind_rel_epoch[:,1] = start_end_pos_in_sel_ind[:,1] - start_time_idx[sel_epoch_ind] 
    given_group_of_int[:,[0,1]]=deepcopy(start_end_pos_in_sel_ind_rel_epoch)
    return given_group_of_int



def convert_the_indexes_to_sel_idexes(map_index_needed,sel_ids):
    # --------------------------------------------------------------------------
    # upto now the segments are present relative to the given segments
    # Now convert the indexes of segmets relative to the sel_ids (good_ids) 
    
    # In case if we already acquired we can use them
    # --------------------------------------------------------------------------
    map_index_needed = map_index_needed.astype(int)
    
    
    #first obtain the mapped index
    rel_index_map = rel_index_map_fun(sel_ids)
    

    mapped_index = np.zeros(np.shape(map_index_needed))
    for m_i in range(0,np.shape(map_index_needed)[0]):
        # for c_i in range(0,np.shape(sw_cont_spindle_contain_segs_np)[1]):
        # Time being mots of these are in strat, and ending poistion (as index 1)
        for c_i in [0,1]:
            # since the ending condition in continious segment doesn't fell in the sel_ids
            if c_i==0:
                mapped_index[m_i,c_i] = rel_index_map[map_index_needed[m_i,c_i]]
                logger.debug('chkecinh map_index_needed[m_i,0] %i', map_index_needed[m_i,c_i])
            else:
                logger.debug('chkecinh map_index_needed[m_i,1] %i', map_index_needed[m_i,c_i])
                mapped_index[m_i,c_i] = rel_index_map[map_index_needed[m_i,c_i]-1]+1
                
     
    return mapped_index

def convert_the_indexes_to_sel_idexes_of_interest_pos(map_index_needed,sel_ids):
    '''
    unpto now the segments are present relative to the given segments
    Now convert the indexes of segmets relative to the sel_ids (good_ids) 
    
    In case if we already acquired we can use them
    
    especillay this function is designed for spindle/ sleepwave ocuured index
    such that starting and ending position of the atcula epoch such that both indexes are pointing the poistion of the interest occurs
        (ending poistion is not increated by one index)
    '''
    map_index_needed = map_index_needed.astype(int)
    
   
    #first obtain the mapped index
    rel_index_map = rel_index_map_fun(sel_ids)
    
    mapped_index = np.zeros(np.shape(map_index_needed))
    for m_i in range(0,np.shape(map_index_needed)[0]):
        # for c_i in range(0,np.shape(sw_cont_spindle_contain_segs_np)[1]):
        # Time being mots of these are in strat, and ending poistion (as index 1)
        for c_i in [0,1]:
            mapped_index[m_i,c_i] = rel_index_map[map_index_needed[m_i,c_i]]
    return mapped_index


def rel_index_map_fun(sel_ids):  
    # --------------------------------------------------------------------------
    #  function maps the selected ids to relative index 
    #  such that after the selection is done
    #  rel_index_map gives the actual location of the sel_ids
    # --------------------------------------------------------------------------
    rel_index_map = {}
    for r_i in range(0,len(sel_ids)):
        rel_index_map[sel_ids[r_i]] = r_i
    return rel_index_map


def find_continious_seg_for_the_given_helper(epoch_indexes, epoch_indexes_of_eeg_seg_position, cont_EEG_segments):
    """
        helper function for function: find_continious_seg_for_the_given_main
    """
    
    #first find the unique interested epoches
    unique_interested_epoches= np.unique(epoch_indexes)
    # then find the intersted continious-segment that has spindles as first one and last one (avoid the boundry issues)
    # maybe we can include some boundry-conditions later
    cont_spindle_contain_segs = {}
    
    ch_ep_ind=0
    sel_seg_ind=0
    checked_intial_seg_has_spindle=False

    while sel_seg_ind<np.shape(cont_EEG_segments)[0] and ch_ep_ind<len(unique_interested_epoches):
        
        ch_ep = unique_interested_epoches[ch_ep_ind]

        cont_spindle_contain_segs[cont_EEG_segments[sel_seg_ind,0]]=[ch_ep,0]
            
        while cont_EEG_segments[sel_seg_ind,0]<=ch_ep<cont_EEG_segments[sel_seg_ind,1]:
            cont_spindle_contain_segs[cont_EEG_segments[sel_seg_ind,0]][1]=ch_ep+1
            ch_ep_ind=ch_ep_ind+1
            if ch_ep_ind==len(unique_interested_epoches):
                break
            ch_ep = unique_interested_epoches[ch_ep_ind]
            #to check the first index not there in the continoius segement 
            checked_intial_seg_has_spindle=True


        while cont_EEG_segments[sel_seg_ind,0] > ch_ep or ch_ep>cont_EEG_segments[sel_seg_ind,1]:
            sel_seg_ind=sel_seg_ind+1
                
        #to remove the first contious segment not has sp/sw
        if not checked_intial_seg_has_spindle:
            del cont_spindle_contain_segs[cont_EEG_segments[0,0]]
            checked_intial_seg_has_spindle=True # to make sure to avoid the re-occurance

    cont_spindle_contain_segs_keys = list(cont_spindle_contain_segs.keys())

    cont_spindle_contain_segs_np=np.zeros((len(cont_spindle_contain_segs_keys),2))
    for c_p in range(0,len(cont_spindle_contain_segs_keys)):
        cont_spindle_contain_segs_np[c_p,:]=deepcopy(cont_spindle_contain_segs[cont_spindle_contain_segs_keys[c_p]])
    return cont_spindle_contain_segs_np


def find_continious_seg_for_the_given_main(eeg_seg_interest_pos_com, cont_EEG_segments, star_end_cond=True,based_on_ep=True,based_on_sl_st=True):
    '''
    This function uses the already obtained 
                                    cont_EEG_segments
    
    This returns the first index of the continious epoch starts and one-epoch after the continious epoch ends 
    to give the right boundry condition
    
    epoch_indexes_of_eeg_seg_position: selected indexs of eeg_seg_interest_pos_np(this is for spindle, slowwave, etc.;
        in this script it would be
        sw_eeg_seg_interest_pos_com or sp_eeg_seg_interest_pos_com)
    
    
    For an ex: epoch_indexes_of_eeg_seg_position may holds the indexes of uniques start and end position spindle contained epoches. 
    
    star_end_cond: are we consider the strat end position (if not all the indexes are choosen for the time being)
    if we consider star_end_cond, then are we choosing based_on_ep 

    '''
    if star_end_cond:
        unique_epoch, unique_sl_st =  obtain_unique_start_end_sp_sw_index_mask(eeg_seg_interest_pos_com)
        if based_on_ep:
            #choose index based on the epoches which contain the spindle/ slow wave within the epoch it self
            epoch_indexes_of_eeg_seg_position = np.where(unique_epoch==1)[0]
        elif based_on_sl_st:
            #choose index based on the contain the spindle/ slow wave within the same- sleep stage annotated
            epoch_indexes_of_eeg_seg_position = np.where(unique_sl_st==1)[0]
        # print('Overall: ' ,np.shape(eeg_seg_interest_pos_com)[0])
        # print('unique_epoch: ',np.sum(unique_epoch))
        # print('unique_sl_st: ',np.sum(unique_sl_st))
        epoch_indexes = eeg_seg_interest_pos_com[epoch_indexes_of_eeg_seg_position,0]
    else:
        epoch_indexes = eeg_seg_interest_pos_com[:,0]
    
    
    cont_spindle_contain_segs_np = find_continious_seg_for_the_given_helper(epoch_indexes, epoch_indexes_of_eeg_seg_position, cont_EEG_segments)
    return cont_spindle_contain_segs_np

def obtain_unique_start_end_sp_sw_index_mask(eeg_seg_interest_pos_np):
    '''
    obtain the statistics of the sleep stages with epoches
    '''
   
    unique_epoch=np.zeros((np.shape(eeg_seg_interest_pos_np)[0]))
    unique_sl_st=np.zeros((np.shape(eeg_seg_interest_pos_np)[0]))
    for ev in range(0,np.shape(eeg_seg_interest_pos_np)[0]):
        unique_epoch[ev]=(len(np.unique(eeg_seg_interest_pos_np[ev,[0,1]]))==1)
        unique_sl_st[ev]=(len(np.unique(eeg_seg_interest_pos_np[ev,[2,3]]))==1)
        
    return unique_epoch, unique_sl_st