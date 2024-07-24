#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:29:57 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved
"""

import os
import logging
import numpy as np
from copy import deepcopy

from sleep_EEG_loose_lead_detect.loose_lead_events.unify_outliers_to_loose_lead import unify_outliers_via_conv

#%% logger intialisation
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("loose_lead_to_events")
while logger.handlers:
      logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
logger.propagate = False



def event_annotations_origin_for_edf(whole_annotations, ep_loose_lead_or, ep_loose_lead_dur,
                                     in_place_event=True,  along_with_original_event_id=False):
    '''
    then insert inplace new event annotation
    
    this can be done by keep the current annotation like sleep-stage and combine the predictions as string or 
    or just replace with the predicted outcomes with the preprocess and/ or loose-lead outcome
    along_with_original_event_id
    
        for the subject with only one loose-lead rest of the information or sleep-stage would be useful
    '''
    
    '''
    then finalise the events-events ids based on the 
    preprcess, and loose-lead combined information
    '''
    
    ep_loose_lead = deepcopy(ep_loose_lead_or)
    
    whole_annot_cp = deepcopy(whole_annotations)   
   
    modified_indexes = np.sort(np.array(list(ep_loose_lead.keys())))
    
    if in_place_event:
        for i in modified_indexes:
            #since the start time remain same by tracking upto that point
            whole_annot_cp[i,1]= np.max(ep_loose_lead_dur[i])
            # logger.debug( np.max(ep_loose_lead_dur[i]))
            if along_with_original_event_id:
                ##just insert the event from the orginal file to keep track the original annoatated event
                ep_loose_lead[i].insert(0,whole_annotations[i][2])
            else:
                pass
            ep_tm = ';'.join(ep_loose_lead[i])
            # logger.debug(ep_tm)
            whole_annot_cp[i,2]=ep_tm
    
    else:
        '''
        since inserting the indices going to change the order 
        if we insert from the bottom, it is not going to affect the actual index
        '''
        modified_indexes_f =  np.flip(modified_indexes)
        whole_annot_cp_t = np.vsplit(whole_annot_cp,np.shape(whole_annot_cp)[0])
    
        for i in modified_indexes_f:
            whole_annot_cp_t_sel = deepcopy(whole_annot_cp_t[i])
            if len(ep_loose_lead[i])==1:
                ep_tm = ep_loose_lead[i][0]
                
                whole_annot_cp_t_sel[0][1]=ep_loose_lead_dur[i][0]
                whole_annot_cp_t_sel[0][2]=ep_tm
                
                whole_annot_cp_t.insert(i,deepcopy(whole_annot_cp_t_sel))
                # logger.debug(i)
            else:
        
                for l_e in range(0,len(ep_loose_lead[i])):
                    whole_annot_cp_t_sel= deepcopy(whole_annot_cp_t[i])
   
                    ep_tm = ep_loose_lead[i][l_e]
                    
                    whole_annot_cp_t_sel[0][1]=ep_loose_lead_dur[i][l_e]
                    whole_annot_cp_t_sel[0][2]=ep_tm
                    whole_annot_cp_t.insert(i,deepcopy(whole_annot_cp_t_sel))
    
        whole_annot_cp =   np.row_stack(whole_annot_cp_t)
    return whole_annot_cp




def main_loose_lead_to_origin_events_dic(arousal_annot_all, sel_index_sl_st, cont_EEG_segments_np,    
                                          ep_loose_lead={},   ep_loose_lead_dur={},even_id_unique=[],
                                          epoch_sec=30, T=4, sliding_size=1, ch_names=   ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'],
                                          assign_comment= 'loose lead', with_conv=True,
                                      outlier_presene_con_lenth_th=0, thresh_min_conv=5, 
                                      thresh_in_sec= True, conv_type = "same",
                                      with_fill_period=False,  len_period_tol_min=5/60, 
                                      show_single_outliers_before_combine_tol=True,
                                      con_seg_wise=False,
                                      verbose=False):
    '''
    
    arousal_annot_all: should have inthe format of outlier annotation x channels
    sel_index_sl_st: are the selected sleep-ids/ events from the events origin (ev_or) index
    
    cont_EEG_segments_np: have the selected continious segements relative to the start_id index (events extracted by mne)
        (Not relative to the good_ids by preprocess) so this index can be mapped via the sel_index_sl_st
    
    '''
    # --------------------------------------------------------------------------
    # in continious segmentation endoing epoch 
    # --------------------------------------------------------------------------
    
    last_epoch_size=int(epoch_sec-(T-sliding_size))

    # --------------------------------------------------------------------------
    #to map-back we need this information
    # 
    #   First get the number of epoches in the continious segment , to find the length of the continious segment
    # then assign the secwise legth for the continious segmentation
    # --------------------------------------------------------------------------
    con_diff = cont_EEG_segments_np[:,1]-cont_EEG_segments_np[:,0]
    con_gap =((con_diff-1)*epoch_sec)+last_epoch_size
    con_gap_cum_sum = np.cumsum(con_gap)
    con_gap_cum_sum_st =con_gap_cum_sum- con_gap
        
    # --------------------------------------------------------------------------
    # then go through the channels and unify them
    #  like fill the gapbetween the predicted outcomes, etc..
    # --------------------------------------------------------------------------
    for ch in range(0,np.shape(arousal_annot_all)[1]):
        arousal_annot =arousal_annot_all[:,ch]
        loose_lead_annot = unify_outliers_via_conv(arousal_annot, with_conv=with_conv,
                              outlier_presene_con_lenth_th=outlier_presene_con_lenth_th, thresh_min_conv=thresh_min_conv, 
                              thresh_in_sec= thresh_in_sec, conv_type =conv_type,
                                                          with_fill_period=with_fill_period,  len_period_tol_min=len_period_tol_min,
                                                          show_single_outliers_before_combine_tol=show_single_outliers_before_combine_tol, verbose=verbose)
        
        if np.shape(loose_lead_annot)[0]!=np.sum(con_gap):
            logger.error("Issue with the sliding window size and continious based annotation recovered")
        
        '''
        if the any loose-led present only we need to annoatet the events
        '''
        if np.sum(loose_lead_annot)>0:
            ep_tm = ch_names[ch]+' '+ assign_comment
        
            ep_loose_lead, ep_loose_lead_dur = loose_lead_to_origin_events_dic(ep_loose_lead,ep_loose_lead_dur, ep_tm,
                                            loose_lead_annot, sel_index_sl_st,epoch_sec,
                                            cont_EEG_segments_np, con_gap_cum_sum_st, con_gap_cum_sum, con_diff,con_seg_wise=con_seg_wise)
            if ep_tm not in even_id_unique:
                even_id_unique.append(ep_tm)
        else:
          pass

    return ep_loose_lead, ep_loose_lead_dur, even_id_unique


def loose_lead_to_origin_events_dic(ep_loose_lead,ep_loose_lead_dur, ep_tm,
                                    loose_lead_annot, sel_index_sl_st, epoch_sec,
                                    cont_EEG_segments_np, con_gap_cum_sum_st, con_gap_cum_sum, con_diff,con_seg_wise=False):
    '''
    ep_loose_lead: previously annotated loose-lead with the event origin ids as keys with the channel information
    ep_loose_lead_dur: the annotation duration
    
    ep_tm: annotation goind to placed in the ep_loose_lead dictionary


    sel_index_sl_st: are the selected sleep-ids/ events from the events origin (ev_or) index
    
    cont_EEG_segments_np: have the selected continious segements relative to the start_id index (events extracted by mne)
        (Not relative to the good_ids by preprocess) so this index can be mapped via the sel_index_sl_st
        
        con_seg_wise: annotate the whole segment as bad when the 50% of the epoches present in the continious segment is bad else only mark the epochs
    '''
    # --------------------------------------------------------------------------
    # to hold the other channels marked loose -lead events
    # --------------------------------------------------------------------------

    indexes_already_nannotted_loose_leads= list(ep_loose_lead.keys())
    
    for cn_grp in range(0,np.shape(cont_EEG_segments_np)[0]):
          # --------------------------------------------------------------------------  
          # to check any loose-lead annoatation presnet in the continious segment
          # --------------------------------------------------------------------------

          if np.sum(loose_lead_annot[con_gap_cum_sum_st[cn_grp]:con_gap_cum_sum[cn_grp]])>0:
    
              if con_diff[cn_grp]==1:
                  or_index_key= sel_index_sl_st[cont_EEG_segments_np[cn_grp][0]]
                  if or_index_key not in indexes_already_nannotted_loose_leads:
                      # --------------------------------------------------------------------------
                      # the starting potion is encoded with the 
                      # to store the comment with the original index or_index_key
                      # which can be obtained from the ev_or
                      # --------------------------------------------------------------------------
                      ep_loose_lead[or_index_key]= [ep_tm]
                      # --------------------------------------------------------------------------
                      # to store the duration with the original index
                      # --------------------------------------------------------------------------

                      ep_loose_lead_dur[or_index_key] =[np.sum(loose_lead_annot[con_gap_cum_sum_st[cn_grp]:con_gap_cum_sum[cn_grp]])]
    
                  else:
                      ep_loose_lead[or_index_key].append(ep_tm)
                      ep_loose_lead_dur[or_index_key].append(np.sum(loose_lead_annot[con_gap_cum_sum_st[cn_grp]:con_gap_cum_sum[cn_grp]]))
              else:
                  
                  or_index_key= sel_index_sl_st[cont_EEG_segments_np[cn_grp][0]]
                  # --------------------------------------------------------------------------  
                  #we are going to have continious group of annotation
                  # --------------------------------------------------------------------------
                  mark_all=False # mark all the epoches in the continious segment
                  first_cont=True# to make only first time check the condition
                  for sm_p in range(con_gap_cum_sum_st[cn_grp],con_gap_cum_sum[cn_grp],epoch_sec):
                      if first_cont and con_seg_wise:
                          loose_sec = np.sum(loose_lead_annot[sm_p:con_gap_cum_sum[cn_grp]])
                          if loose_sec > (con_gap_cum_sum[cn_grp]-con_gap_cum_sum_st[cn_grp])/2:
                              mark_all=True
                          first_cont = False

                      range_check =sm_p+epoch_sec
                      if con_gap_cum_sum[cn_grp]>range_check:
                          loose_sec = np.sum(loose_lead_annot[sm_p:range_check])
                      else:
                          loose_sec = np.sum(loose_lead_annot[sm_p:con_gap_cum_sum[cn_grp]])
                      # --------------------------------------------------------------------------  
                      # if we are marking all the epoches just mark them
                      # all as the epoches in the continious group with the predicted annotation
                      # --------------------------------------------------------------------------  
                      if mark_all:
                          if or_index_key not in indexes_already_nannotted_loose_leads:
                              # which can be obtained from the ev_or
                              ep_loose_lead[or_index_key]= [ep_tm]
                              #to store the duration with the original index
                              ep_loose_lead_dur[or_index_key] =[loose_sec]
            
                          else:
                              ep_loose_lead[or_index_key].append(ep_tm)
                              ep_loose_lead_dur[or_index_key].append(loose_sec)
                      else: 
                          # --------------------------------------------------------------------------  
                          #  only mark the epochs with loose-sec present for longer than 0 period
                          # --------------------------------------------------------------------------  
                          if loose_sec>0:
                             
                              if or_index_key not in indexes_already_nannotted_loose_leads:
                                  # which can be obtained from the ev_or
                                  ep_loose_lead[or_index_key]= [ep_tm]
                                  #to store the duration with the original index
                                  ep_loose_lead_dur[or_index_key] =[loose_sec]
                
                              else:
                                  ep_loose_lead[or_index_key].append(ep_tm)
                                  ep_loose_lead_dur[or_index_key].append(loose_sec)
                      or_index_key+=1
          else:
              pass
    return ep_loose_lead, ep_loose_lead_dur




'''
grab only the preprocess step-found out comes
'''


def epoch_status_to_events_annot(epoch_status, sel_index_sl_st, epoch_sec=30, ep_loose_lead={},    ep_loose_lead_dur={},    even_id_unique=[]):

    '''
    this function will asssign the preprocess event information
    '''  
    # --------------------------------------------------------------------------
    # since we knew the    
    #   sel_index_sl_st: are the selected sleep-ids/ events from the events origin (ev_or) index
    #   so the sel_index_sl_st[i] will map back to the origin
    # and we know the duration is epoch time default 30 sec
    # --------------------------------------------------------------------------

    epo_stat_not_norm_rel_sel_idx= np.where(epoch_status!='normal')[0]    

    # --------------------------------------------------------------------------
    # even though this no-need to be in dictionary format later loose-lead need to 
    # use this kind of adaptable format inorder to unify them 
    # better to use the same format in the pipeline
    # --------------------------------------------------------------------------

    
    for i in epo_stat_not_norm_rel_sel_idx:
       
        ep_tm = str(epoch_status[i])
        ep_loose_lead[sel_index_sl_st[i]]=[ep_tm]
        ep_loose_lead_dur[sel_index_sl_st[i]]=[epoch_sec]
        if ep_tm not in even_id_unique:
            even_id_unique.append(ep_tm)
    return ep_loose_lead, ep_loose_lead_dur, even_id_unique