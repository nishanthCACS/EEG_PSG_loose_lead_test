#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 22:56:04 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved
"""
import numpy as np
import logging

from copy import deepcopy

from sleep_EEG_loose_lead_detect.preprocessing.sp_sw_rel_index_conversion import find_sp_se_relative_postion_in_epoch

logger = logging.getLogger("sp_sw_one_hot")
while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
logger.propagate = False



# --------------------------------------------------------------------------
# to make the annotation for all epcohes
# --------------------------------------------------------------------------

def find_interest_index_for_given_epoch_and_channel(eeg_seg_interest_pos_com,group_of_interest_por_com, int_ep=32, ch_interested=7):
        
    # # 186 index belongs to 32nd epoch
    # int_ep=32# good ids
    # ch_interested=7
    
    sel_indexes_for_epoch = np.where(eeg_seg_interest_pos_com[:,0]==int_ep)[0]
    sel_int_challel_occuring = np.where(group_of_interest_por_com[:,-1]==ch_interested)[0]
    # index=186
    intersted_for_channel=[]
    for i in sel_indexes_for_epoch:
        if i in sel_int_challel_occuring:
            intersted_for_channel.append(i)
    return intersted_for_channel

def find_interested_segments_seconds(intersted_for_channel, group_of_interest_por_com, ep_length=30,Fs=256,boundry_cond_sec_frac=0.5,
                                     end_loc_skip=False):

    #then using the ep_index_int to find the exact sec poistion of sp/ sw
    # first create the general time sec boundries for epoch
    time_boundries = np.array(range(0,Fs*(ep_length+1),Fs))
    
    
    interset_seconds = []
    carry_out_first=True
    carry_out = 0

    for ep_index_int in intersted_for_channel:
        logger.debug('ep_index_int %i',ep_index_int)

        logger.debug('start postion of interset portion %i',group_of_interest_por_com[ep_index_int,0])
        logger.debug('end postion of interset portion %i',group_of_interest_por_com[ep_index_int,1])
        loc_of_int_por_sec=0
        boundry_cond_sec_cont_int_por_sel = Fs*boundry_cond_sec_frac
        while group_of_interest_por_com[ep_index_int,0]>=time_boundries[loc_of_int_por_sec]:
            s_loc= loc_of_int_por_sec
            loc_of_int_por_sec=loc_of_int_por_sec+1
        
        if not end_loc_skip:
            loc_of_int_por_sec=-1
            if group_of_interest_por_com[ep_index_int,1]>time_boundries[loc_of_int_por_sec]:
                if carry_out_first:
                    # e_loc =  s_loc
                    e_loc= ep_length+loc_of_int_por_sec
                    # logger.info('end postion of interset portion %i is assigned as last index of the ep_index_int %i'% (group_of_interest_por_com[ep_index_int,1],ep_index_int))
                    carry_out = group_of_interest_por_com[ep_index_int,1]-time_boundries[loc_of_int_por_sec]
                    carry_out_first=False
                else:
                    raise("issue with tow carry outs")
            else:
                while group_of_interest_por_com[ep_index_int,1]<=time_boundries[loc_of_int_por_sec]:
                    e_loc= ep_length+loc_of_int_por_sec
                    loc_of_int_por_sec=loc_of_int_por_sec-1

            
        s_loc_th = False
        e_loc_th = False
        
        
        if (time_boundries[s_loc]-group_of_interest_por_com[ep_index_int,0])/Fs >= boundry_cond_sec_cont_int_por_sel:
            s_loc_th = True
            
            
        if not end_loc_skip:
            if  (group_of_interest_por_com[ep_index_int,1])/Fs-time_boundries[e_loc] < boundry_cond_sec_cont_int_por_sel:
                e_loc_th = True
            
            if s_loc_th and e_loc_th:
                interset_seconds.append(deepcopy(helper_for_interest_sec_map(s_loc,e_loc+1)))
            elif s_loc_th:
                interset_seconds.append(deepcopy(interset_seconds = helper_for_interest_sec_map(s_loc,e_loc)))
            elif e_loc_th:
                interset_seconds.append(deepcopy(helper_for_interest_sec_map(s_loc+1,e_loc+1)))
            else:
                interset_seconds.append(deepcopy(helper_for_interest_sec_map(s_loc+1,e_loc)))
        else:
            #if skipping the ending criteria
            if s_loc_th:
                interset_seconds.append(deepcopy(interset_seconds = helper_for_interest_sec_map(s_loc,s_loc)))
            else:
                interset_seconds.append(deepcopy(helper_for_interest_sec_map(s_loc+1,s_loc+1)))
    interset_seconds =  list(sum(interset_seconds,[]))

    return interset_seconds, carry_out



def helper_for_interest_sec_map(ch_st,ch_en):
    '''
        this is teh helper for create the interested segments 
    '''
    if ch_st == ch_en:
        return [ch_st]
    else:
       return list(range(ch_st,ch_en))

def epoch_annot_sec_wise_for_sp_sw(eeg_seg_interest_pos_com, group_of_interest_por_com, 
                                   sleep_stages, cont_EEG_segments_np, Fs, 
                                   ch_interested=7, ep_length=30, boundry_cond_sec_frac=0.5,
                                   end_loc_skip=False):
    '''
    This function takes the  group_of_interest_por_com, and eeg_seg_interest_pos_com
        sec wise annotate the each epoch 
    '''
    intersted_seconds_dic= {}
    carry_out_holder=np.zeros((len(sleep_stages)))
    
    
    for cn in range(0,np.size(cont_EEG_segments_np,axis=0)):
        # --------------------------------------------------------------------------
        # this is the continious starting position epoch index
        # --------------------------------------------------------------------------

        rel_ep_indx_start = cont_EEG_segments_np[cn][0]
        for int_ep in range(cont_EEG_segments_np[cn][0],cont_EEG_segments_np[cn][1]):
            intersted_for_channel  = find_interest_index_for_given_epoch_and_channel(eeg_seg_interest_pos_com,group_of_interest_por_com, int_ep=int_ep, ch_interested=ch_interested)
            _intersted_seconds, carry_out = find_interested_segments_seconds(intersted_for_channel, group_of_interest_por_com, ep_length=ep_length,Fs=Fs,boundry_cond_sec_frac=boundry_cond_sec_frac,
                                                                             end_loc_skip=end_loc_skip)
            intersted_seconds_dic[int_ep]=deepcopy(_intersted_seconds)
            carry_out_holder[int_ep]=carry_out
        
        #since the carry out is present in the continious blocks
        for int_ep in range(rel_ep_indx_start+1,cont_EEG_segments_np[cn][1]):
            devide_op=carry_out_holder[int_ep-1]/Fs
            if devide_op>0:
                if devide_op>boundry_cond_sec_frac and devide_op<1:
                    intersted_seconds_dic[int_ep].insert(0,0)
                else:
                    chk = deepcopy(intersted_seconds_dic[int_ep])
                    if (devide_op%1)>=boundry_cond_sec_frac:
                        chk_add = list(range(0,int(np.ceil(devide_op))))
                    else:
                        chk_add = list(range(0,int(np.floor(devide_op))))
                      
                    chk_fin=chk_add+chk
                    intersted_seconds_dic[int_ep]=deepcopy(chk_fin)

    return intersted_seconds_dic


#%%

def MT_targeted_intersted_segs(interset_seconds,max_ep_len=26,o_p_adjuster=0):
    '''
        Just limit the maximum of imterest portion occurance 
    '''
    interset_seconds_deep_copy= []
    if  o_p_adjuster>0:
        for i_s in interset_seconds:
            if i_s>o_p_adjuster:
                interset_seconds_deep_copy = MT_id_check_above_helper(i_s-o_p_adjuster, interset_seconds_deep_copy, max_ep_len)
            else:
                interset_seconds_deep_copy = MT_id_check_above_helper(i_s, interset_seconds_deep_copy, max_ep_len)
    else:
        for i_s in interset_seconds:
            interset_seconds_deep_copy = MT_id_check_above_helper(i_s, interset_seconds_deep_copy, max_ep_len)
    return interset_seconds_deep_copy




def one_hot_encode_interested_por(intersted_seconds, ep_len_finally_in_represnt = 27, tail_len=0):
    '''
    This function helps to create the one-hot intersted portion
    
    tail_len: the length in seconds considered to be annotated after the annotation
    '''
    intersted_seconds_as_one_hot =  np.zeros((ep_len_finally_in_represnt))
    #when we include the tail annotation include the carryout portion
    carry_seconds=0
    
    if tail_len==0:
        intersted_seconds_as_one_hot[intersted_seconds]=1
    else:
        #when we include the tail annotation include the carryout portion
        carry_seconds=0
        for int_sec in intersted_seconds:
            if (int_sec-tail_len)<ep_len_finally_in_represnt:
                intersted_seconds_as_one_hot[int_sec:int_sec+tail_len]=1
            else:
                carry_seconds = tail_len- (int_sec - ep_len_finally_in_represnt)
                
    return intersted_seconds_as_one_hot, carry_seconds

def MT_id_check_above_helper(chk_ind, interset_seconds_deep_copy, max_ep_len):
    '''
    This function helps to check the index is above the boundry and avoid it
    '''
    if chk_ind>max_ep_len:
        interset_seconds_deep_copy.append(max_ep_len)
    else:
        interset_seconds_deep_copy.append(chk_ind)
    return interset_seconds_deep_copy

def convert_to_full_one_hot_mapping_based_cont_seg_sp_sw(intersted_seconds_dic, MT_orgin_spec_full,
                                                        cont_EEG_segments_np,
                                                        epoch_length=30, o_p_adjuster = 0, tail_len=3):
    '''
    this function returns the o/p as one-hot mapping 
    Extend the one_hot encoding to 3 secs if we know the annotation is for 3 seconds and they are fell in continious
    
        tail_len=3 this will decide howlong the annotation will be carried

    '''    
    
    present_length =epoch_length-o_p_adjuster
    intersted_seconds_as_one_hot =  np.zeros((np.shape(MT_orgin_spec_full)[2]))
    carry_seconds=0

    #to incoperate the continious segments length
    rel_length=0
    
    for cn in range(0,np.size(cont_EEG_segments_np,axis=0)):
        #this is the continious starting position epoch index
        rel_ep_indx=0
        for int_ep in range(cont_EEG_segments_np[cn][0],cont_EEG_segments_np[cn][1]-1):
            _intersted_seconds =  intersted_seconds_dic[int_ep]
            intersted_seconds=  MT_targeted_intersted_segs(_intersted_seconds,o_p_adjuster=0)
                
                  
            s_indx= rel_length+  int(rel_ep_indx*epoch_length)
            e_indx= rel_length + int((rel_ep_indx+1)*epoch_length)
            if carry_seconds>0:
                intersted_seconds_as_one_hot[s_indx:s_indx+carry_seconds]=1
              
            # intersted_seconds_as_one_hot[s_indx:e_indx] = one_hot_encode_interested_por(intersted_seconds)
            intersted_seconds_as_one_hot_t, carry_seconds = one_hot_encode_interested_por(intersted_seconds, ep_len_finally_in_represnt = epoch_length, tail_len=tail_len)
            
            #to incoperate the carry out and the interested seconds
            intersted_seconds_as_one_hot[s_indx:e_indx] = intersted_seconds_as_one_hot[s_indx:e_indx] + intersted_seconds_as_one_hot_t
            rel_ep_indx = rel_ep_indx+1
        '''
        the go through the borders here the seconds are cut on the last
        '''
        int_ep = cont_EEG_segments_np[cn][1]-1    
    
        _intersted_seconds =  intersted_seconds_dic[int_ep]
        intersted_seconds=  MT_targeted_intersted_segs(_intersted_seconds,o_p_adjuster=o_p_adjuster)
        
        
        
        s_indx= rel_length+int(rel_ep_indx*present_length)
        e_indx= rel_length+int((rel_ep_indx+1)*present_length)
        
        intersted_seconds_as_one_hot[s_indx:e_indx], _ = one_hot_encode_interested_por(intersted_seconds)
        
        #this is going to be the stop point of the current continious segment
        rel_length = e_indx
    
    return intersted_seconds_as_one_hot



def convert_to_full_one_hot_mapping_based_cont_seg_sp_sw_all_channels(sp_sw_dic_format,
                                                                      sleep_stages_or,cont_EEG_segments_np,Fs,
        MT_spec_not_db,start_time_idx,
        ep_length=30,
        o_p_adjuster=3,  boundry_cond_sec_frac=0.5,combine_intersted_channels_to_all=False,
        
        ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'], spindle_interest_channel=[]):
    '''
    check the spindles in all channels instead of only central channels
    
    '''
    
    MT_orgin_spec_full =  np.concatenate(MT_spec_not_db,axis=2)
    sp_intersted_seconds_one_hot_six_ch =  np.zeros((np.shape(MT_orgin_spec_full)[2],len(ch_names)))
    
    if  sp_sw_dic_format['sp_sat']:
        sp_eeg_seg_interest_pos_com = sp_sw_dic_format['sp_eeg_seg_interest_pos_com']
        sp_group_of_interest_por_com = sp_sw_dic_format['sp_group_of_interest_por_com']
        sp_group_of_interest_por_com = find_sp_se_relative_postion_in_epoch(sp_group_of_interest_por_com,sp_eeg_seg_interest_pos_com,start_time_idx)
    
        
        '''
        spindle and MT-coorelatiion one hot encoding
        '''
        if len(spindle_interest_channel)==0:
            spindle_interest_channel= ch_names
            
        spindle_info_hold={}
        for ch_interested in range(0,len(ch_names)):
            # --------------------------------------------------------------------------
            # 
            # If we gave th eintersted channels to check the spindles only apply YASA only on those channels
            # 
            # --------------------------------------------------------------------------

            if ch_names[ch_interested] in spindle_interest_channel:
                    
                    
                '''
                    ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']
                   
                    the spindle detcion algorithm YASA mainly targetd on the central channels 
                    ch_interested=2
                        just considering the channel of C3-M2 for spindle detection
                '''
                # ch_interested=2
                # '''
                # this can be generalised for in_ep and channels
                # '''
                # # sp_intersted_for_channel  = find_interest_index_for_given_epoch_and_channel(sp_eeg_seg_interest_pos_com,sp_group_of_interest_por_com, int_ep=int_ep, ch_interested=ch_interested)
                # # _sp_intersted_seconds,_ = find_interested_segments_seconds(sp_intersted_for_channel, sp_group_of_interest_por_com, ep_length=ep_length,Fs=Fs,boundry_cond_sec_frac=boundry_cond_sec_frac)
                            
                # --------------------------------------------------------------------------
                # 
                #  sp_intersted_seconds_dic will hold the dictionary of indexes from the EEG_root with the predicted spindle locations
                # -------------------------------------------------------------------------- 
                sp_intersted_seconds_dic = epoch_annot_sec_wise_for_sp_sw(sp_eeg_seg_interest_pos_com, sp_group_of_interest_por_com, \
                                                                            sleep_stages_or, cont_EEG_segments_np, int(Fs), ep_length=ep_length, ch_interested=ch_interested, boundry_cond_sec_frac=boundry_cond_sec_frac)
                
                # --------------------------------------------------------------------------
                # this eill hold the predicted information to event annotation
                # --------------------------------------------------------------------------
                spindle_info_hold[ch_names[ch_interested]] = deepcopy(sp_intersted_seconds_dic)
                # --------------------------------------------------------------------------
                # using the dictionary gothrough the continious segments one by one and annotate the locations based on the dictionary
                #  such that cont_EEG_segments_np values represent the continious segments, e.g: (5,10) contioius segmemr should be same as 
                #   the sp_intersted_seconds_dic's keys akso have to have the same index 5 
                #  all are indexed to the start_time_idx epoches 
                # --------------------------------------------------------------------------

                sp_intersted_seconds_one_hot = convert_to_full_one_hot_mapping_based_cont_seg_sp_sw(sp_intersted_seconds_dic, MT_orgin_spec_full, \
                                                     cont_EEG_segments_np,
                                                      epoch_length=ep_length, o_p_adjuster = o_p_adjuster, tail_len=0)
                sp_intersted_seconds_one_hot_six_ch[:,ch_interested]=deepcopy(sp_intersted_seconds_one_hot)
    # --------------------------------------------------------------------------
    # preserve the intereted spindle channel locations 
    # --------------------------------------------------------------------------
    if combine_intersted_channels_to_all:
        sum_sp = sp_intersted_seconds_one_hot_six_ch.sum(axis=0)
        sum_sp =np.where(sum_sp<1,sum_sp,1)
        for ch_interested in range(0,len(ch_names)):
            sp_intersted_seconds_one_hot_six_ch[:,ch_interested]= sum_sp
    
    return sp_intersted_seconds_one_hot_six_ch, spindle_info_hold