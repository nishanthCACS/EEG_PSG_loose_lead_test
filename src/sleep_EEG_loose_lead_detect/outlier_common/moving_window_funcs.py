#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 11:20:22 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

this have the functions related to moving window

Modified on Tue Apr 18 15:37:46 2023
this to check the global threhold/ consucutive bock's threholds

Modified on Sat Apr 22 09:41:28 2023
to accomodate to include the global threhold calculated based on good-medians (not artifact medians)

Modified on Mon May  8 17:05:54 2023
the bug indentified by one of the subjects,
 the break_cont_segments_intersetd_sleep_stages is fixed

Modified on Fri Jun 16 15:48:43 2023 to accomodate the results of average correlation-coefficient 
the detection is obtained based on the grp to represent the single channel (ealeir this used for combintaion)
and for 6 channels six annottaion they are obtained from accessed each annotations seperately via grp

Modified on Mon Jul 10 23:54:17 2023 to accomdate the moving window channel

"""
import logging
import numpy as np

from copy import deepcopy

from sleep_EEG_loose_lead_detect.channel_correlation_outlier.poolers_Z_standardization_funcs import obtain_mean_corr
from sleep_EEG_loose_lead_detect.channel_correlation_outlier.poolers_Z_standardization_funcs import correlation_pooler, inter_mediate_mapping_correlation,z_standardization


from sleep_EEG_loose_lead_detect.outlier_common.cont_segment_artifact_annotation import cont_EEG_segments_np_len
from sleep_EEG_loose_lead_detect.GUI_interface.percentage_bar_vis  import percent_complete


logger = logging.getLogger("moving_window_local_outlier_funcs")
while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
# Set logging level to the logger
# logger.setLevel(logging.DEBUG) # <-- THIS!
logger.setLevel(logging.INFO)
logger.propagate = False

def break_cont_segments_intersetd_sleep_stages(cont_EEG_segments_np, sleep_stages_annot_flatten,intersted_sleep_stages,
                                                o_p_adjuster=3, ep_length= 30):
        
    '''
    annotate continiius segment-np with 
    this will annotatte the sleep-pre-processed-fragment (SPPF): the fragmnet term may confuce so we use this term to avoid the ambiguity
    
    this function will select the epoches from the SPFFs with the intersted sleep-stages continious segments
    this function will further group inside the SPFFs into fragmented contioius group of intersted sleep-stages
    
    For an example if we intersted in NREM sleep stages:
        such that this function can output preprocessed NREM continious fragements
        
        interseted_sleep_stage_cont_groups_start_end_index: contian the start and end point of the preprocessed continoius NREM segments
        len_interseted_sleep_stage_cont_groups: length of the preprocessed continoius NREM segments
    '''

    len_cont_seg, cont_seg_start_end_index =  cont_EEG_segments_np_len(cont_EEG_segments_np, o_p_adjuster=o_p_adjuster, ep_length=ep_length)
    # --------------------------------------------------------------------------
    # then break further into goups the SPPF segments
    # now we have the cont_seg_start_end_index of starting and ending positions
    # using this we can break the continious segments
    # --------------------------------------------------------------------------

    interseted_sleep_stage_cont_groups_start_end_index =[]   
    for cn_idx in range(0,len(len_cont_seg)):
        # --------------------------------------------------------------------------
        # Just assign the starting and ending position of the prepreoc
        # --------------------------------------------------------------------------

        s_p = cont_seg_start_end_index[cn_idx][0]
        e_p = cont_seg_start_end_index[cn_idx][1]
        # --------------------------------------------------------------------------
        # this will check the sl_st_index poistion relative to continoius segment
        # --------------------------------------------------------------------------

        sl_st_index_interset_cont_start = s_p
        # sl_st_index_interset_cont_end: SPFFs end point
        # --------------------------------------------------------------------------
        # boundry conditions between the SPFFs
        # the carry over is used for continius blocks boundry condition
        # --------------------------------------------------------------------------
        start_check = True
        carry=False
        
        # --------------------------------------------------------------------------
        #  go through one SPFF and break further
        # the groups only contian the interested sleep-stages e.g: like NREM (N1, N2, N3)
        # --------------------------------------------------------------------------
        for sl_st_index in range(s_p,e_p,ep_length):
            logger.debug('sl_st_index %i:', sl_st_index)
            # --------------------------------------------------------------------------
            # the following line is added to check intiate ending point after first full-for-loop
            # --------------------------------------------------------------------------
            sl_st_index_interset_cont_end=sl_st_index

            # --------------------------------------------------------------------------
            # Go through the sleep_stages_annot_flatten and check the 
            # SPFF's selected epoch indexes fell into the interested sleep-stages or not
            # if the sl_st_index not fell break the group of continuity and append the continous list
            # 
            # since the sl_st_index is not in the intersted group this can be safely ignored not considered for next starting point
            # --------------------------------------------------------------------------
            if not (sleep_stages_annot_flatten[sl_st_index] in intersted_sleep_stages):
                # --------------------------------------------------------------------------
                #  this will skip the first index in the current for loop
                #  since it is assigned as
                #        sl_st_index_interset_cont_end=sl_st_index
                # --------------------------------------------------------------------------
                if not (sl_st_index_interset_cont_end==sl_st_index_interset_cont_start):
                    # --------------------------------------------------------------------------
                    # the following line is added to check whether the broken interested SPFF 
                    # belong to the interested sleep-stage
                    # --------------------------------------------------------------------------
                    if sleep_stages_annot_flatten[sl_st_index_interset_cont_start] in intersted_sleep_stages:
                        interseted_sleep_stage_cont_groups_start_end_index.append([sl_st_index_interset_cont_start,sl_st_index_interset_cont_end])
                        # --------------------------------------------------------------------------
                        # since the sleep stage fell into intersted keep on carry
                        # to avoid the boundry condtion fell agin in the continous broken group
                        # --------------------------------------------------------------------------
                        carry=False
                # --------------------------------------------------------------------------
                #  assign the starting poition to new group to avoid continiously adding the unwanted sleep-continous SPFFS
                # --------------------------------------------------------------------------
                sl_st_index_interset_cont_start =sl_st_index
                # --------------------------------------------------------------------------
                # to capture the new continious groups strating position
                # --------------------------------------------------------------------------
                start_check = True

            else:
                # --------------------------------------------------------------------------
                # since the sleep stage fell into intersted keep on carry
                # --------------------------------------------------------------------------
                carry=True
                
                # --------------------------------------------------------------------------
                #the following portion is attached to assign the SPFF's starting position
                # first time fell into the group we have hold for starting position  
                # --------------------------------------------------------------------------
                if  start_check:
                    sl_st_index_interset_cont_start =sl_st_index
                    # --------------------------------------------------------------------------
                    # when this goes through first time with the sleep-stage interested this is turn offed
                    #  to maintian the contiusly check
                    # --------------------------------------------------------------------------
                    start_check = False
                
        # --------------------------------------------------------------------------
        # if it carry only the atleast once it goes through the loop
        # --------------------------------------------------------------------------

        if  carry:
            if sleep_stages_annot_flatten[sl_st_index_interset_cont_start] in intersted_sleep_stages:
                interseted_sleep_stage_cont_groups_start_end_index.append([sl_st_index_interset_cont_start,e_p])
            

    interseted_sleep_stage_cont_groups_start_end_index = np.array(interseted_sleep_stage_cont_groups_start_end_index,dtype=int)
    len_interseted_sleep_stage_cont_groups =interseted_sleep_stage_cont_groups_start_end_index[:,1]-interseted_sleep_stage_cont_groups_start_end_index[:,0]
    return interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups


def break_flat_MT_from_broken_sleep_st_segments_intersetd_sleep_stages(interseted_sleep_stage_cont_groups_start_end_index_or,
                                                                       len_interseted_sleep_stage_cont_groups_or,
                                                                       flat_MT_flatten):
    # --------------------------------------------------------------------------
    #  using the already broken sleep-continiuos blocks to break them further to 
    #  avoid the loose-lead kind of events
    # --------------------------------------------------------------------------
    interseted_sleep_stage_cont_groups_start_end_index_cp = deepcopy(interseted_sleep_stage_cont_groups_start_end_index_or)
    len_interseted_sleep_stage_cont_groups_cp = deepcopy(len_interseted_sleep_stage_cont_groups_or)
    
    # --------------------------------------------------------------------------
    #  then find the continious region of flat MT
    # --------------------------------------------------------------------------
    privious_annot_break_points = np.where(flat_MT_flatten>0)[0]
    if len(privious_annot_break_points)>0:

        privious_annot_break_points_diff = privious_annot_break_points[1:]-privious_annot_break_points[0:-1]
        cont_flat_MT_t = np.where(privious_annot_break_points_diff==1)[0]
        cont_flat_MT_t2 = [[privious_annot_break_points[cont_flat_MT_t[x]],privious_annot_break_points[cont_flat_MT_t[x]+1]] for x in range(0,len(cont_flat_MT_t))]
        
        cont_flat_MT=[]
        # --------------------------------------------------------------------------
        # this will avoid the continious flat MT there or not 
        # --------------------------------------------------------------------------
        if len(cont_flat_MT_t2)>0:
            cont_flat_MT_h=cont_flat_MT_t2[0][0]
            for i in range(1,len(cont_flat_MT_t2)):
                if cont_flat_MT_t2[i][0]-cont_flat_MT_t2[i-1][1] !=0:
                    # inorder to force the index included by python 0 index
                    cont_flat_MT.append([cont_flat_MT_h,cont_flat_MT_t2[i-1][1]+1])
                    cont_flat_MT_h = cont_flat_MT_t2[i][0]
                
        cont_flat_MT_flatten=[]
        for cn_flat_MT_inx in range(0,len(cont_flat_MT)):
            cont_flat_MT_flatten.append(list(range(cont_flat_MT[cn_flat_MT_inx][0],cont_flat_MT[cn_flat_MT_inx][1])))
        cont_flat_MT_flatten = list(sum(cont_flat_MT_flatten,[]))
                
        # --------------------------------------------------------------------------
        #  then using the broken interested sleep-stages are further breaked based on the flat_MT annotation
        # --------------------------------------------------------------------------
        

    
        # --------------------------------------------------------------------------
        # this will avoid the continious flat MT there or not 
        # --------------------------------------------------------------------------
        cn_flat_MT_inx=0    
        
        # --------------------------------------------------------------------------
        # to cehk the the continoius break points exists
        #  once we stratied to check th econtinious block check the next continoiuly annoptated 
        #  flatten channel exists to check further in continious blocks else leave it
        # --------------------------------------------------------------------------
        cont_exists=True
        break_point_inx=0
        # break_point_inx_track=0
        break_temp=[]
        for cn_idx in range(0,len(len_interseted_sleep_stage_cont_groups_cp)):
            # --------------------------------------------------------------------------
            # Just assign the starting and ending position of the prepreoc
            s_p = interseted_sleep_stage_cont_groups_start_end_index_cp[cn_idx][0]
            e_p = interseted_sleep_stage_cont_groups_start_end_index_cp[cn_idx][1]
            # 
            # logger.debug(break_point_inx)
            # --------------------------------------------------------------------------
            # just need to make sure the flat_MT is above the current satrting point
            #  for increasing thatws why -1
            # --------------------------------------------------------------------------
            while break_point_inx<(len(privious_annot_break_points)-1) and privious_annot_break_points[break_point_inx]<s_p:
                break_point_inx+=1
                # logger.debug('Here: %i',break_point_inx)
        
            if  s_p<= privious_annot_break_points[break_point_inx]<e_p:
                while  break_point_inx<len(privious_annot_break_points) and s_p<= privious_annot_break_points[break_point_inx]<e_p:
                    # --------------------------------------------------------------------------
                    #  check the break points in the continuius flat -MT like previuosly detect ed loose-lead, artifact, etc.
                    # --------------------------------------------------------------------------
                    break_temp, break_point_inx, break_return, cont_exists, cn_flat_MT_inx, s_p = helper_break_sleep_stage_flat_MT(cont_exists,privious_annot_break_points,break_point_inx,cont_flat_MT_flatten,
                                     cont_flat_MT,cn_flat_MT_inx,s_p,e_p,break_temp)
                    if break_return:
                        break

                # --------------------------------------------------------------------------
                # check the last privious_annot_break_points index make sure 
                #  boundry condition related continious sgement not missed
                # --------------------------------------------------------------------------
                if  break_point_inx>0 and s_p <=privious_annot_break_points[break_point_inx-1]+1<e_p:
    
                    if len(break_temp)>0 and (s_p != break_temp[-1][0]):
                     break_temp, break_point_inx, break_return, cont_exists, cn_flat_MT_inx, s_p = helper_break_sleep_stage_flat_MT_boundry_check(cont_exists,privious_annot_break_points,break_point_inx,cont_flat_MT_flatten,
                                          cont_flat_MT,cn_flat_MT_inx,s_p,e_p,break_temp)            
            else:
                if s_p != e_p:
                    break_temp.append([s_p,e_p])
        interseted_sleep_stage_cont_groups_start_end_index = np.array(break_temp,dtype=int)
        len_interseted_sleep_stage_cont_groups =interseted_sleep_stage_cont_groups_start_end_index[:,1]-interseted_sleep_stage_cont_groups_start_end_index[:,0]
    else:
        interseted_sleep_stage_cont_groups_start_end_index = interseted_sleep_stage_cont_groups_start_end_index_cp
        len_interseted_sleep_stage_cont_groups = len_interseted_sleep_stage_cont_groups_cp
    return interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups



def helper_break_sleep_stage_flat_MT(cont_exists,privious_annot_break_points,break_point_inx,cont_flat_MT_flatten,
                                     cont_flat_MT,cn_flat_MT_inx,s_p,e_p,break_temp):
    
    break_return = False
    # --------------------------------------------------------------------------
    # to cehk the the continoius break points exists
    #  once we stratied to check th econtinious block check the next continoiuly annoptated 
    #  flatten channel exists to check further in continious blocks else leave it
    # --------------------------------------------------------------------------
    if cont_exists  and privious_annot_break_points[break_point_inx] in cont_flat_MT_flatten:
        # --------------------------------------------------------------------------
        # if the continous block fell inside one full lsepp-group break into two pieces
        #
        # since we know it is in continious part
        # we can safely increase
        # 
        # when we necounter there is no continious blocks exists with the flatten annotation 
        #  just turn off the cont_exists
        # 
        #  if any of the already annotation not fell in the 
        # just an extra check 
        # --------------------------------------------------------------------------
        while cont_exists and s_p > cont_flat_MT[cn_flat_MT_inx][0]:
            cn_flat_MT_inx+=1
            if len(cont_flat_MT)==cn_flat_MT_inx:
                cont_exists=False
        # --------------------------------------------------------------------------
        # if the size is small only we assign else that block is left
        # --------------------------------------------------------------------------
        if cont_exists:
            # --------------------------------------------------------------------------
            #  if the continious segment interfered by the given annotation 
            #  break into two pieces
            # --------------------------------------------------------------------------

            if s_p <= cont_flat_MT[cn_flat_MT_inx][0]:
                break_temp.append([s_p,cont_flat_MT[cn_flat_MT_inx][0]])
            
            if cont_flat_MT[cn_flat_MT_inx][1] < e_p:
                # --------------------------------------------------------------------------
                #  since the next strating point will be the end of the current continiuos block
                # --------------------------------------------------------------------------
                s_p = cont_flat_MT[cn_flat_MT_inx][1]
                # then assign the hext index after the brak-point in the continiou block to check 
                cn_flat_MT_inx+=1
                # --------------------------------------------------------------------------
                #   while assigning the cn_flat_MT_inx
                #  make sure the continious blocks exists
                # --------------------------------------------------------------------------
                if len(cont_flat_MT)==cn_flat_MT_inx:
                    cont_exists=False
                    # --------------------------------------------------------------------------
                    #  if this is the last block just attach
                    # --------------------------------------------------------------------------

                    break_temp.append([cont_flat_MT[cn_flat_MT_inx-1][1],e_p])

              
            else:
                # --------------------------------------------------------------------------
                #  Just assign the increase the breaking points
                # --------------------------------------------------------------------------
                if cont_exists:
                    break_point_inx = np.where(privious_annot_break_points == (cont_flat_MT[cn_flat_MT_inx][1]-1))[0][0]+1
                break_return = True


    else:
        # --------------------------------------------------------------------------
        # just need to check the region 
        # --------------------------------------------------------------------------
        if  s_p <=privious_annot_break_points[break_point_inx]<e_p:
            break_temp.append([s_p,privious_annot_break_points[break_point_inx]])
            if (privious_annot_break_points[break_point_inx]+1)< e_p:
                s_p= privious_annot_break_points[break_point_inx]+1
                
                # --------------------------------------------------------------------------
                #  make sure the starting point not fell in the flat_MT annotation    
                # --------------------------------------------------------------------------
                while (s_p in privious_annot_break_points) and s_p<e_p:
                    s_p+=1
                    break_point_inx+=1

    break_point_inx+=1
    if break_point_inx==len(privious_annot_break_points):
        # --------------------------------------------------------------------------
        # lets break the while-loop 
        # --------------------------------------------------------------------------
        break_point_inx= len(privious_annot_break_points)-1
        break_return = True

    return break_temp, break_point_inx, break_return, cont_exists, cn_flat_MT_inx, s_p



def helper_break_sleep_stage_flat_MT_boundry_check(cont_exists,privious_annot_break_points,break_point_inx,cont_flat_MT_flatten,
                                     cont_flat_MT,cn_flat_MT_inx,s_p,e_p,break_temp):
    
    break_return = False
    # --------------------------------------------------------------------------
    # to cehk the the continoius break points exists
    #  once we stratied to check th econtinious block check the next continoiuly annoptated 
    #  flatten channel exists to check further in continious blocks else leave it
    # --------------------------------------------------------------------------
    if cont_exists  and privious_annot_break_points[break_point_inx] in cont_flat_MT_flatten:
        # --------------------------------------------------------------------------
        # if the continous block fell inside one full lsepp-group break into two pieces
        #
        # since we know it is in continious part
        # we can safely increase
        # 
        # when we necounter there is no continious blocks exists with the flatten annotation 
        #  just turn off the cont_exists
        # 
        #  if any of the already annotation not fell in the 
        # just an extra check 
        # --------------------------------------------------------------------------
        while cont_exists and s_p > cont_flat_MT[cn_flat_MT_inx][0]:
            cn_flat_MT_inx+=1
            if len(cont_flat_MT)==cn_flat_MT_inx:
                cont_exists=False
        # --------------------------------------------------------------------------
        # if the size is small only we assign else that block is left
        # --------------------------------------------------------------------------
        if cont_exists:
            # --------------------------------------------------------------------------
            #  if the continious segment interfered by the given annotation 
            #  break into two pieces
            # --------------------------------------------------------------------------

            if s_p <= cont_flat_MT[cn_flat_MT_inx][0]:
                break_temp.append([s_p,cont_flat_MT[cn_flat_MT_inx][0]])
            
            if cont_flat_MT[cn_flat_MT_inx][1] < e_p:
                # --------------------------------------------------------------------------
                #  since the next strating point will be the end of the current continiuos block
                # --------------------------------------------------------------------------
                s_p = cont_flat_MT[cn_flat_MT_inx][1]
                # then assign the hext index after the brak-point in the continiou block to check 
                cn_flat_MT_inx+=1
                # --------------------------------------------------------------------------
                #   while assigning the cn_flat_MT_inx
                #  make sure the continious blocks exists
                # --------------------------------------------------------------------------
                if len(cont_flat_MT)==cn_flat_MT_inx:
                    cont_exists=False
                    # --------------------------------------------------------------------------
                    #  if this is the last block just attach
                    # --------------------------------------------------------------------------

                    break_temp.append([cont_flat_MT[cn_flat_MT_inx-1][1],e_p])

              
            else:
                # --------------------------------------------------------------------------
                #  Just assign the increase the breaking points
                # --------------------------------------------------------------------------
                if cont_exists:
                    break_point_inx = np.where(privious_annot_break_points == (cont_flat_MT[cn_flat_MT_inx][1]-1))[0][0]+1
                break_return = True


    else:
        # --------------------------------------------------------------------------
        # just need to check the region 
        # --------------------------------------------------------------------------
        if  s_p <=privious_annot_break_points[break_point_inx]+1<e_p:
            break_temp.append([s_p,e_p])
         
    break_point_inx+=1
    if break_point_inx==len(privious_annot_break_points):
        # --------------------------------------------------------------------------
        # lets break the while-loop 
        # --------------------------------------------------------------------------
        break_point_inx= len(privious_annot_break_points)-1
        break_return = True

    return break_temp, break_point_inx, break_return, cont_exists, cn_flat_MT_inx, s_p







def median_with_moving_window_in_broken_cont_groups(corr_given, grp, 
    interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups,
    moving_window_size=60):
    '''
    this sort the elements is continious blocks in the moving window
        corr_given: is the correlation elements of the different channels either this can be obtained from Z-mapped or the direct values
     
    moving_window_size= 60#based on the 1996 paper
    
    the algorithm is implemented efficiently to moving window sorting once the moving window started
    since we are sliding in one index only that element is added to assign the new median
    
    
        
    take the group of values and sort them 
    
    when the len_grp_size - moving_window_size will be the size of above 
    window_size; due to once sec advance ovet the moving window size
    
    grp: is used interchangely as channel combintation for grp (optional)
        or 
        for average correlation this is used to slected the specific channel
        
        Direct correlation in signle vector can avoid the need of providing the grp information for selection
    '''
    
    # --------------------------------------------------------------------------
    # if the length is less than the moving window 
    # For the default value window with size 60 
    #      then the median is assigned only from that epoch/ two epochs like 
    #        27 sec's or 57 secs else the moving window is used till the staring to end of the continious blocks
    # --------------------------------------------------------------------------
    sorted_median_cont_grp =[]
    
    for cn_indx in range(0,len(len_interseted_sleep_stage_cont_groups)):
        
        sorted_median_cont_grp_t=[]
        # --------------------------------------------------------------------------
        #  getiing the tsrting and ending potion of interested continious group
        # --------------------------------------------------------------------------       
        s_p_intial = interseted_sleep_stage_cont_groups_start_end_index[cn_indx][0]
        e_p_intial = interseted_sleep_stage_cont_groups_start_end_index[cn_indx][1]
        
        # --------------------------------------------------------------------------
        # modified to accept correlation coefficient maps or single correlation coefficient maps
        # even though the correlation are fed sperately by the main functions 
        # user can provide the input via the grp or directly
        # --------------------------------------------------------------------------
        if len(np.shape(corr_given))==3:
            # --------------------------------------------------------------------------
            #  For approach 1 with all the correlation groups together
            # --------------------------------------------------------------------------
            
            group_check_intial = corr_given[s_p_intial:e_p_intial,grp[0][0],grp[0][1]]
        elif len(np.shape(corr_given))==2:
            # --------------------------------------------------------------------------
            #  For approach 2 with all the mean of the channels together
            # feed the channle information via the group
            # --------------------------------------------------------------------------
            group_check_intial = corr_given[s_p_intial:e_p_intial,grp]
            if np.isnan(group_check_intial).all():
                logger.debug('s_p_intial %i:',s_p_intial)
                logger.debug('e_p_intial %i:',e_p_intial)
        else:
            group_check_intial = corr_given[s_p_intial:e_p_intial]

        s_p = 0
        if len_interseted_sleep_stage_cont_groups[cn_indx]<=moving_window_size:
            e_p = e_p_intial - s_p_intial
        else:
            e_p = moving_window_size
        
        logger.debug('s_p %i',s_p) 
        
        # --------------------------------------------------------------------------
        # select the portion to slide the moving window
        # --------------------------------------------------------------------------
        group_check = deepcopy(group_check_intial[s_p:e_p])
        
        # --------------------------------------------------------------------------
        # do the first time sorting    
        # find the median value from the sorted list
        # --------------------------------------------------------------------------       
        goup_sorted = np.sort(group_check,kind='quicksort')
        sorted_median_cont_grp_t.append(deepcopy(np.median(goup_sorted)))
        
        
        # --------------------------------------------------------------------------
        # the folowing while loop can be presented as for loop but need to have litlle more variables
        # with 
        # if e_p< len_interseted_sleep_stage_cont_groups[cn_indx]:    

        # to adjust the while statement e_p condition
        # --------------------------------------------------------------------------

        e_p = e_p+1
        while e_p < len_interseted_sleep_stage_cont_groups[cn_indx]:
            
            # --------------------------------------------------------------------------
            # remove the element that matches from the sorted list    
            #    we need to remove the s_p related elements and
            # --------------------------------------------------------------------------
            for indx in range(0,len(goup_sorted)):
                if goup_sorted[indx] == group_check_intial[s_p]:
                    logger.debug('indx %i:',indx)
                    break
                
            s_p = s_p+1
            logger.debug("goup_sorted are: {}".format(' '.join(map(str, goup_sorted))))
            goup_sorted = np.delete(goup_sorted, [indx])
            logger.debug("goup_sorted are: {}".format(' '.join(map(str, goup_sorted))))

            # --------------------------------------------------------------------------
            # since the sorting list already avaliable just do the part of insertion sort with 
            # pivot element as the new in the group
            # --------------------------------------------------------------------------
            pivot_ele = group_check_intial[e_p-1]
            for indx in range(0,len(goup_sorted)):
                # --------------------------------------------------------------------------
                # to accomadate the python indexing
                # --------------------------------------------------------------------------
                if goup_sorted[indx]<=pivot_ele:
                    break
            goup_sorted = np.insert(goup_sorted, indx, pivot_ele)    
        
            sorted_median_cont_grp_t.append(deepcopy(np.median(goup_sorted)))
        
            e_p=e_p+1
            
        sorted_median_cont_grp.append(deepcopy(sorted_median_cont_grp_t))

    return sorted_median_cont_grp

def local_outlier_detection(corr_given, grp, sorted_median_cont_grp,
    interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups,
    arousal_annot, moving_window_size=60, th_fact = 1.5):  
    '''
    local outlier detection
    
    annotate arousal based on the sliding medians 
    six threhold conditions evaluated 1996 paper
    such factors as
    40,10,4,3,2, 1.5
    but here we check the factor with smmaler values that can be inverse of these factors 
    
    these factors are applied seperately for each continious segments seperately
    
    the values prsent in the continious segmets beggining part
      less than the moving_vindow is checked with only the first median value 
      (consider as belonging to the first moving window)
    '''

    for cn_indx in range(0,len(len_interseted_sleep_stage_cont_groups)):
    
        s_p_intial = interseted_sleep_stage_cont_groups_start_end_index[cn_indx][0]
        e_p_intial = interseted_sleep_stage_cont_groups_start_end_index[cn_indx][1]
        
        sorted_median_cont_grp_t = np.array(sorted_median_cont_grp[cn_indx])/th_fact

        '''
        modified to accept correlation coefficient maps or single correlation coefficient maps
        '''
        if len(np.shape(corr_given))==3:
            group_check_intial = corr_given[s_p_intial:e_p_intial,grp[0][0],grp[0][1]]
        elif len(np.shape(corr_given))==2:
            group_check_intial = corr_given[s_p_intial:e_p_intial,grp]
        else:
            group_check_intial = corr_given[s_p_intial:e_p_intial]
            
            
        s_p = 0
        if len_interseted_sleep_stage_cont_groups[cn_indx]<=moving_window_size:
            e_p = e_p_intial - s_p_intial
        else:
            e_p = moving_window_size
            
            
        for chk_val_indx in range(s_p,e_p):
            if group_check_intial[chk_val_indx] < sorted_median_cont_grp_t[0]:
                arousal_annot[chk_val_indx+s_p_intial]= 1

        
        if e_p< len_interseted_sleep_stage_cont_groups[cn_indx]:    
            for s_p in range(1,len_interseted_sleep_stage_cont_groups[cn_indx]-moving_window_size):
                chk_val_indx = e_p +s_p
                if group_check_intial[chk_val_indx] < sorted_median_cont_grp_t[s_p]:
                    arousal_annot[chk_val_indx+s_p_intial]= 1

    return arousal_annot



def local_and_global_based_outlier_detection_with_loc_good(corr_given, grp, sorted_median_cont_grp,
    interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups,
    arousal_annot,  moving_window_size=60, th_fact = 4, 
    sorted_median_cont_grp_comp_sort_median_cond=[True, 10],
    sorted_median_cont_grp_comp_sort_max_cond=[True,10],
    sorted_median_cont_grp_comp_sort_quan_cond=[True,10,0.75],
    verbose=False, warn_first=True,GUI_percentile=True):  
    '''
    local outlier detection
    
    annotate arousal based on the sliding medians 
    six threhold conditions evaluated 1996 paper
    such factors as
    40,10,4,3,2, 1.5
    but here we check the factor with smmaler values that can be inverse of these factors 
    
    these factors are applied seperately for each continious segments seperately
    
    the values prsent in the continious segmets beggining part
      less than the moving_vindow is checked with only the first median value 
      (consider as belonging to the first moving window)
     
      sorted_median_cont_grp_comp_sort_median_cond=[True, 10] these conditions give an option whether to consider the threhold or not with their corresponding factor
      sorted_median_cont_grp_comp_sort_quan=[True,10,0.75] the last one is the quantile value to decide the main value for threhold condition
        for correlation based analysis above 0.5 is great
    '''
    
    '''
    define global threhold different of ways
    since mean is interfered by the outliers
    
    '''
    if warn_first:
        logger.warning('The good pool of values obtained by the local window %i is used for global threshold', moving_window_size)

    
    dist_good_medians =[]
    for cn_indx in range(0,len(len_interseted_sleep_stage_cont_groups)):
        if GUI_percentile:    
            percent_complete(cn_indx, len(len_interseted_sleep_stage_cont_groups), bar_width=60, title="local-mov-wind-chan-"+str(grp), print_perc=True)

        s_p_intial = interseted_sleep_stage_cont_groups_start_end_index[cn_indx][0]
        e_p_intial = interseted_sleep_stage_cont_groups_start_end_index[cn_indx][1]
        
        # --------------------------------------------------------------------------
        #  selected contious grouos threhold is claultaed
        # --------------------------------------------------------------------------
        sorted_median_cont_grp_t = np.array(sorted_median_cont_grp[cn_indx])/th_fact

        # --------------------------------------------------------------------------
        # modified to accept correlation coefficient maps or single correlation coefficient maps
        # --------------------------------------------------------------------------
        if len(np.shape(corr_given))==3:
            group_check_intial = corr_given[s_p_intial:e_p_intial,grp[0][0],grp[0][1]]
        elif len(np.shape(corr_given))==2:
            group_check_intial = corr_given[s_p_intial:e_p_intial,grp]
        else:
            group_check_intial = corr_given[s_p_intial:e_p_intial]
            
        # --------------------------------------------------------------------------
        #  solve the issue with the segments with the lesser period than the moving window
        # --------------------------------------------------------------------------
        s_p = 0
        if len_interseted_sleep_stage_cont_groups[cn_indx]<=moving_window_size:
            e_p = e_p_intial - s_p_intial
        else:
            e_p = moving_window_size
            
        # --------------------------------------------------------------------------
        #  take each value and compare with the median thrhold assigned and 
        #  one hot encode the correlation values drop below the given threhold as
        #  bad segments in arousal_annot
        # --------------------------------------------------------------------------

        for chk_val_indx in range(s_p,e_p):
            if group_check_intial[chk_val_indx] < sorted_median_cont_grp_t[0]:
                arousal_annot[chk_val_indx+s_p_intial]= 1
            else:
                # --------------------------------------------------------------------------
                # here we get the kind of goodpoints only  not the medians, to represent the good segements
                # --------------------------------------------------------------------------
                dist_good_medians.append(group_check_intial[chk_val_indx])
                
        # --------------------------------------------------------------------------
        #  boundry condition check
        # --------------------------------------------------------------------------
        if e_p< len_interseted_sleep_stage_cont_groups[cn_indx]:    
            for s_p in range(1,len_interseted_sleep_stage_cont_groups[cn_indx]-moving_window_size):
                chk_val_indx = e_p +s_p
                if group_check_intial[chk_val_indx] < sorted_median_cont_grp_t[s_p]:
                    arousal_annot[chk_val_indx+s_p_intial]= 1
                else:
                    dist_good_medians.append(group_check_intial[chk_val_indx])


    # --------------------------------------------------------------------------
    # calculate the global condition based on the good-ids
    # 
    # three options are presented but this can be done in many different ways 
    #  those can be easily accomdated by the last option
    # --------------------------------------------------------------------------

    # Further instead of good-pools we can use the medians it self is used to obtain 
    #  the threshold but that is  implemented 
    # in local_and_global_based_outlier_detection_with_all function
    # --------------------------------------------------------------------------
    
    
    # --------------------------------------------------------------------------     
    # global_conditions option-1
    #   choose the median of good points to obtain the threhold
    # --------------------------------------------------------------------------     

    global_conditions = []
    if sorted_median_cont_grp_comp_sort_median_cond[0]:
        sorted_median_cont_grp_comp_sort_median = np.median(np.sort(np.array(dist_good_medians)))
        global_conditions.append(sorted_median_cont_grp_comp_sort_median/sorted_median_cont_grp_comp_sort_median_cond[1])
        if verbose:
            logger.info("Global threhold median with the factor %i",sorted_median_cont_grp_comp_sort_median_cond[1] )

    # --------------------------------------------------------------------------     
    # global_conditions option-2
    #   choose the maximum of good points  to obtain the threhold
    # most of the time this value corresponds to one so this is not a good option
    # --------------------------------------------------------------------------     
    
    if sorted_median_cont_grp_comp_sort_max_cond[0]:
        sorted_median_cont_grp_comp_sort_max = np.max(np.array(dist_good_medians))
        global_conditions.append(sorted_median_cont_grp_comp_sort_max/sorted_median_cont_grp_comp_sort_max_cond[1])
        if verbose:
            logger.info("Global threhold with maximum with the factor %i",sorted_median_cont_grp_comp_sort_max_cond[1] )
            

    # --------------------------------------------------------------------------     
    # global_conditions option-3
    #   choose the quantile value of good points  to obtain the threhold
    # --------------------------------------------------------------------------     
    
    if sorted_median_cont_grp_comp_sort_quan_cond[0]:
        sorted_median_cont_grp_comp_sort_quan = np.quantile(np.array(dist_good_medians),sorted_median_cont_grp_comp_sort_quan_cond[2])
        global_conditions.append(sorted_median_cont_grp_comp_sort_quan/sorted_median_cont_grp_comp_sort_quan_cond[1])
        if verbose:
             logger.info("Global threhold with maximum with the factor %i",sorted_median_cont_grp_comp_sort_max_cond[1] )
    
 
    
    # --------------------------------------------------------------------------
    # then use the global conditions to obtain the results
    # --------------------------------------------------------------------------
    global_conditions = np.array(global_conditions)
    for cn_indx in range(0,len(len_interseted_sleep_stage_cont_groups)):
    
        s_p_intial = interseted_sleep_stage_cont_groups_start_end_index[cn_indx][0]
        e_p_intial = interseted_sleep_stage_cont_groups_start_end_index[cn_indx][1]
        
        sorted_median_cont_grp_t = np.array(sorted_median_cont_grp[cn_indx])/th_fact

        # --------------------------------------------------------------------------
        # modified to accept correlation coefficient maps or single correlation coefficient maps
        # --------------------------------------------------------------------------
        if len(np.shape(corr_given))==3:
            group_check_intial = corr_given[s_p_intial:e_p_intial,grp[0][0],grp[0][1]]
        elif len(np.shape(corr_given))==2:
            group_check_intial = corr_given[s_p_intial:e_p_intial,grp]
        else:
            group_check_intial = corr_given[s_p_intial:e_p_intial]
            
            
        s_p = 0
        if len_interseted_sleep_stage_cont_groups[cn_indx]<=moving_window_size:
            e_p = e_p_intial - s_p_intial
        else:
            e_p = moving_window_size
            
            
        for chk_val_indx in range(s_p,e_p):
            # --------------------------------------------------------------------------
            # check the global conditions
            # --------------------------------------------------------------------------
            if np.sum(group_check_intial[chk_val_indx] < global_conditions)>0:
                arousal_annot[chk_val_indx+s_p_intial]= 1
     
        # --------------------------------------------------------------------------
        #  boundry condition check
        # --------------------------------------------------------------------------
        if e_p< len_interseted_sleep_stage_cont_groups[cn_indx]:    
            for s_p in range(1,len_interseted_sleep_stage_cont_groups[cn_indx]-moving_window_size):
                chk_val_indx = e_p +s_p
                if np.sum(group_check_intial[chk_val_indx] < global_conditions)>0:
                    arousal_annot[chk_val_indx+s_p_intial]= 1
     
    return arousal_annot

def local_and_global_based_outlier_detection_with_all(corr_given, grp, sorted_median_cont_grp,
    interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups,
    arousal_annot, moving_window_size=60, th_fact = 4, 
    sorted_median_cont_grp_comp_sort_median_cond=[True, 10],
    sorted_median_cont_grp_comp_sort_max_cond=[True,10],
    sorted_median_cont_grp_comp_sort_quan_cond=[True,10,0.75], verbose=False, warn_first=True):  
    '''
    local outlier detection
    
    annotate arousal based on the sliding medians 
    six threhold conditions evaluated 1996 paper
    such factors as
    40,10,4,3,2, 1.5
    but here we check the factor with smmaler values that can be inverse of these factors 
    
    these factors are applied seperately for each continious segments seperately
    
    the values prsent in the continious segmets beggining part
      less than the moving_vindow is checked with only the first median value 
      (consider as belonging to the first moving window)
     
          sorted_median_cont_grp_comp_sort_median_cond=[True, 10] these conditions give an option whether to consider the threhold or not with their corresponding factor
        sorted_median_cont_grp_comp_sort_quan=[True,10,0.75] the last one is the quantile value to decide the main value for threhold condition
        for correlation based analysis above 0.5 is great
    '''
    
    '''
    functionality wise lmost same local_and_global_based_outlier_detection_with_loc_good
        this is differenct only on how we use the samples to decide the outliers
 
     instead of good-pool here we are using local-moving window distribution to decide the global threhold
         such that global-detection  can be paralley run with  local-moving-window   
    '''
    if warn_first:
        logger.warning('The median values obtained by the local window %i used for global threshold'. moving_window_size)
        
    global_conditions = []
    if sorted_median_cont_grp_comp_sort_median_cond[0]:
        sorted_median_cont_grp_comp_sort_median = np.median(np.sort(np.array(list(sum(sorted_median_cont_grp,[])))))
        global_conditions.append(sorted_median_cont_grp_comp_sort_median/sorted_median_cont_grp_comp_sort_median_cond[1])
        if verbose:
            logger.info("Global threhold median with the factor %i",sorted_median_cont_grp_comp_sort_median_cond[1] )
            
    if sorted_median_cont_grp_comp_sort_max_cond[0]:
        sorted_median_cont_grp_comp_sort_max = np.max(np.array(list(sum(sorted_median_cont_grp,[]))))
        global_conditions.append(sorted_median_cont_grp_comp_sort_max/sorted_median_cont_grp_comp_sort_max_cond[1])
        if verbose:
            logger.info("Global threhold with maximum with the factor %i",sorted_median_cont_grp_comp_sort_max_cond[1] )
            
    if sorted_median_cont_grp_comp_sort_quan_cond[0]:
        sorted_median_cont_grp_comp_sort_quan = np.quantile(np.array(list(sum(sorted_median_cont_grp,[]))),sorted_median_cont_grp_comp_sort_quan_cond[2])
        global_conditions.append(sorted_median_cont_grp_comp_sort_quan/sorted_median_cont_grp_comp_sort_quan_cond[1])
        if verbose:
             logger.info("Global threhold with maximum with the factor %i",sorted_median_cont_grp_comp_sort_max_cond[1] )

    if len(global_conditions)>0:
        if warn_first:
            logger.warning('Even you assign many conditions only the mnium threhold going to be effect the final detection')


    global_conditions = np.array(global_conditions)
    
    for cn_indx in range(0,len(len_interseted_sleep_stage_cont_groups)):
    
        s_p_intial = interseted_sleep_stage_cont_groups_start_end_index[cn_indx][0]
        e_p_intial = interseted_sleep_stage_cont_groups_start_end_index[cn_indx][1]
        
        sorted_median_cont_grp_t = np.array(sorted_median_cont_grp[cn_indx])/th_fact

        # --------------------------------------------------------------------------
        # modified to accept correlation coefficient maps or single correlation coefficient maps
        # --------------------------------------------------------------------------
        if len(np.shape(corr_given))==3:
            group_check_intial = corr_given[s_p_intial:e_p_intial,grp[0][0],grp[0][1]]
        elif len(np.shape(corr_given))==2:
            group_check_intial = corr_given[s_p_intial:e_p_intial,grp]
        else:
            group_check_intial = corr_given[s_p_intial:e_p_intial]
            
            
        s_p = 0
        if len_interseted_sleep_stage_cont_groups[cn_indx]<=moving_window_size:
            e_p = e_p_intial - s_p_intial
        else:
            e_p = moving_window_size
            
            
        for chk_val_indx in range(s_p,e_p):
            if group_check_intial[chk_val_indx] < sorted_median_cont_grp_t[0]:
                arousal_annot[chk_val_indx+s_p_intial]= 1
            # --------------------------------------------------------------------------
            # check the global conditions
            #   if any of the wnated global condition 
            # --------------------------------------------------------------------------

            if np.sum(group_check_intial[chk_val_indx] < global_conditions)>0:
                  arousal_annot[chk_val_indx+s_p_intial]= 1

            logger.debug('chk_val_indx+s_p_intial %i',chk_val_indx+s_p_intial)
        
        if e_p< len_interseted_sleep_stage_cont_groups[cn_indx]:    
            for s_p in range(1,len_interseted_sleep_stage_cont_groups[cn_indx]-moving_window_size):
                chk_val_indx = e_p +s_p
                if group_check_intial[chk_val_indx] < sorted_median_cont_grp_t[s_p]:
                    arousal_annot[chk_val_indx+s_p_intial]= 1
                    
                #check the global conditions
                if np.sum(group_check_intial[chk_val_indx] < global_conditions)>0:
                      arousal_annot[chk_val_indx+s_p_intial]= 1

                logger.debug('chk_val_indx+s_p_intial %i',chk_val_indx+s_p_intial)

    return arousal_annot



def get_deep_copies(corr_check_given):
    
    # --------------------------------------------------------------------------
    # to avoid the issue of running sleep-stages sperately 
    # like running NREM and REM seperately
    # --------------------------------------------------------------------------
    corr_check  = deepcopy(corr_check_given)


    # --------------------------------------------------------------------------
    #WHILE CHECKING THE OUTLIERS ONLY CONCENTRATING THE CORRECT INDEX POSITION OF THE TRANSFORMED NREM AND REM
    # --------------------------------------------------------------------------

    bm_corr_check_mapped =deepcopy(corr_check)# to avoid the issue of running NREM and REM seperately

    return corr_check, bm_corr_check_mapped

    
def moving_window_based_out_lier_annotator_channel_mean(corr_check_given,sleep_stages_annot_flatten, intersted_sleep_stages,
                      ch_names, arousal_annot, 
                      cont_EEG_segments_np, 
                      b_val=0.0001,inter_mediate_transform=True,z_transform=True,Fisher_based=True,
                      flat_MT_consider=True,flat_MT_ch=[],intersted_sleep_stages_term='',
                      break_spindle_flatten=True,break_flat_MT_flatten=True,
                      avoid_spindle_loc=False, spindle_enc_ch=[],
                      moving_window_size=60, th_fact=1.5, o_p_adjuster=3, ep_length=30,
                      global_check=False, only_good_points=False,
                      sorted_median_cont_grp_comp_sort_median_cond=[True, 10],
                      sorted_median_cont_grp_comp_sort_max_cond=[True,10],
                      sorted_median_cont_grp_comp_sort_quan_cond=[True,10,0.75],
                      loose_lead_channels=[],
                      SPFF_return_int_sleep=False,
                      verbose=False, GUI_percentile=True):

    
    '''
    corr_check : this contain the correlation value flatten x channel x channel
    intersted_sleep_stages : the interested sleep stages that going to be selected for outlier annotations
     
     
    only_good_points: means the global threshold is obtained based only from the good-distribution( that are not anotated as local condition as bad segment)   
    
    o/p:  arousal_annot : anotated as 1 if they fell in outlier based condition
    
    break_spindle_flatten=True,break_flat_MT_flatten==True,
        this will break on the given flat_MT conditions and spindle condition as continious blocks
    '''
    if verbose:
        logger.info("Moving window based outlier detection intitated")
    if GUI_percentile:
        logger.warning('Intiation warning')

        if global_check:
            if only_good_points:
                logger.warning('The good pool of values obtained by the local window %i is used for global threshold', moving_window_size)
            else:
                logger.warning('The median values obtained by the local window %i used for global threshold', moving_window_size)
        # percent_complete(1, 100, bar_width=60, title="mov-window-outlier-detection", print_perc=True)
    # --------------------------------------------------------------------------
    # get the copies to avoid the edition later
    # all channels are used 
    # --------------------------------------------------------------------------
    corr_check, bm_corr_check_mapped = get_deep_copies(corr_check_given)

    corr_check_mapped = obtain_mean_corr(bm_corr_check_mapped, ch_names = ch_names, loose_lead_channels=loose_lead_channels)

    # --------------------------------------------------------------------------
    # only Z-transform mapping influenced by the spindle inclusion

    #the following obtained before the mean of the correlation, judt check the minimum numexist
    # --------------------------------------------------------------------------

    _, correlation_pool_sat = correlation_pooler(corr_check, sleep_stages_annot_flatten, intersted_sleep_stages,
                                                          flat_MT_consider=flat_MT_consider, flat_MT_ch=flat_MT_ch,
                                                          avoid_spindle_loc = avoid_spindle_loc, spindle_enc_ch =spindle_enc_ch)
    # --------------------------------------------------------------------------
    # Z-standardisation is performed on the full data
    # arousal annotation handle the wanted sleep-stage or not
    # --------------------------------------------------------------------------
    if correlation_pool_sat:
        if verbose:
            logger.info("Pooling the correltaion is done ")
        if GUI_percentile:    
            percent_complete(10, 100, bar_width=60, title="mov-window-outlier-detection", print_perc=True)
        if inter_mediate_transform:
            corr_check_mapped =  inter_mediate_mapping_correlation(corr_check_mapped,b_val=b_val)
        if z_transform:
            corr_check_mapped =  z_standardization(corr_check_mapped,Fisher_based=Fisher_based)


        # --------------------------------------------------------------------------
        # then using the SPFF's continious segments start end position to check the windowing kind of threhold
        # then based on the continious groups check the sleep-stages groups seperatey
        # --------------------------------------------------------------------------
        
        interseted_sleep_stage_cont_groups_start_end_index_or, len_interseted_sleep_stage_cont_groups_or = break_cont_segments_intersetd_sleep_stages(cont_EEG_segments_np, sleep_stages_annot_flatten,intersted_sleep_stages,
                                                        o_p_adjuster=o_p_adjuster, ep_length= ep_length)
        
        # --------------------------------------------------------------------------
        #  if not having flat_MT p[revious annpotation only using the continious segments to do the moving window
        # --------------------------------------------------------------------------
        if not (flat_MT_consider and break_flat_MT_flatten) or (avoid_spindle_loc and break_spindle_flatten):
            interseted_sleep_stage_cont_groups_start_end_index = deepcopy(interseted_sleep_stage_cont_groups_start_end_index_or)
            len_interseted_sleep_stage_cont_groups = deepcopy(len_interseted_sleep_stage_cont_groups_or)
        # --------------------------------------------------------------------------
        # to show the selcetd functions 
        # local_and_global_based_outlier_detection_with_loc_good, local_and_global_based_outlier_detection_with_all
        #  to global threhold  condition only once
        # --------------------------------------------------------------------------
        warn_first=True
        if GUI_percentile:
            # --------------------------------------------------------------------------
            # since this is warning is implemented above 
            # --------------------------------------------------------------------------

            warn_first=False

        # --------------------------------------------------------------------------
        # here the grp is used to represent the single channel's correlation coefficient
        # --------------------------------------------------------------------------
        for grp in range(0,len(ch_names)):

            if  (flat_MT_consider and break_flat_MT_flatten) or (avoid_spindle_loc and break_spindle_flatten):
                if (not avoid_spindle_loc) and break_flat_MT_flatten:
                    flat_MT_flatten = flat_MT_ch[:,grp]
                elif (not flat_MT_consider) and break_spindle_flatten:
                    flat_MT_flatten = spindle_enc_ch[:,grp]
                else:
                    # --------------------------------------------------------------------------
                    #  this will force all the annotation greatert than one to one
                    #    since the spindle need to be avoid in the influence in mobving window detection that place is broken in the continious segementation
                    # 
                    # --------------------------------------------------------------------------

                    _flat_MT_flatten =  flat_MT_ch[:,grp]+spindle_enc_ch[:,grp]
                    flat_MT_flatten = np.where(_flat_MT_flatten<1,_flat_MT_flatten,1)

                interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups = break_flat_MT_from_broken_sleep_st_segments_intersetd_sleep_stages(interseted_sleep_stage_cont_groups_start_end_index_or,
                                                                                       len_interseted_sleep_stage_cont_groups_or,
                                                                                       flat_MT_flatten)
                
            interseted_sleep_stage_cont_groups_start_end_index_new, len_interseted_sleep_stage_cont_groups_new  = check_the_break_segment_len(interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups)

                
            # --------------------------------------------------------------------------
            # one time effectively sort the medians and keep them
            # 
            # the function median_with_moving_window_in_broken_cont_groups effciently sort the elements one time
            #   to find the medians via the moving window
            # 
            # --------------------------------------------------------------------------
            sorted_median_cont_grp = median_with_moving_window_in_broken_cont_groups(corr_check_mapped, grp, 
                # interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups,
                interseted_sleep_stage_cont_groups_start_end_index_new, len_interseted_sleep_stage_cont_groups_new,
                moving_window_size=moving_window_size)
            
            if global_check:
                if only_good_points:
                    # --------------------------------------------------------------------------
                    #  only using the good -points obtained by the local moving window to decide the global based outlier finalisation
                    # --------------------------------------------------------------------------
                    arousal_annot[:,grp] =  local_and_global_based_outlier_detection_with_loc_good(corr_check_mapped, grp, sorted_median_cont_grp,
                    # interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups,
                    interseted_sleep_stage_cont_groups_start_end_index_new, len_interseted_sleep_stage_cont_groups_new,
                    arousal_annot[:,grp], moving_window_size=moving_window_size, th_fact = th_fact,
                    sorted_median_cont_grp_comp_sort_median_cond=sorted_median_cont_grp_comp_sort_median_cond,
                    sorted_median_cont_grp_comp_sort_max_cond=sorted_median_cont_grp_comp_sort_max_cond,
                    sorted_median_cont_grp_comp_sort_quan_cond=sorted_median_cont_grp_comp_sort_quan_cond, verbose=verbose, warn_first=warn_first,
                    GUI_percentile=GUI_percentile)
                else:
                    # --------------------------------------------------------------------------
                    #  only using the medians of the local moving winodw to decide theglobal threshold
                    #  since we already obtianed the medians just local and global can be annotate
                    #  together
                    # --------------------------------------------------------------------------

                    arousal_annot[:,grp]  =  local_and_global_based_outlier_detection_with_all(corr_check_mapped, grp, sorted_median_cont_grp,
                    # interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups,
                    interseted_sleep_stage_cont_groups_start_end_index_new, len_interseted_sleep_stage_cont_groups_new,
                    arousal_annot[:,grp] , moving_window_size=moving_window_size, th_fact = th_fact,
                    sorted_median_cont_grp_comp_sort_median_cond=sorted_median_cont_grp_comp_sort_median_cond,
                    sorted_median_cont_grp_comp_sort_max_cond=sorted_median_cont_grp_comp_sort_max_cond,
                    sorted_median_cont_grp_comp_sort_quan_cond=sorted_median_cont_grp_comp_sort_quan_cond, verbose=verbose, warn_first=warn_first)
                warn_first =False
            else:
                # --------------------------------------------------------------------------
                #  no global condition just follow the same 1996 paper kind of moving window method to detect the outliers
                # --------------------------------------------------------------------------
                arousal_annot[:,grp]  =  local_outlier_detection(corr_check_mapped, grp, sorted_median_cont_grp,
                    # interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups,
                    interseted_sleep_stage_cont_groups_start_end_index_new, len_interseted_sleep_stage_cont_groups_new,
                    arousal_annot[:,grp] , moving_window_size=moving_window_size, th_fact = th_fact)
                
            if verbose:
                logger.info("Outlier dectecion done on "+ch_names[grp])
            if GUI_percentile:
                percent_complete(10+int(90*((grp+1)/len(ch_names))), 100, bar_width=60, title="mov-window-outlier-detection", print_perc=True)
    else:
        logger.warning("correlation pool of "+intersted_sleep_stages_term+" not satisfied return the given arousal_annot")
    
    # --------------------------------------------------------------------------
    # rteturn th einterested sleep-stage continious index information
    #   interseted_sleep_stage_cont_groups_start_end_index: contian the start and end positions of intersted sleep-groups in SPFFs
    #   len_interseted_sleep_stage_cont_groups: length of that gruops
    # --------------------------------------------------------------------------

    if SPFF_return_int_sleep:
        # return arousal_annot, interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups
        return arousal_annot, interseted_sleep_stage_cont_groups_start_end_index_new, len_interseted_sleep_stage_cont_groups_new 
    else:
        return arousal_annot
    
def check_the_break_segment_len(interseted_sleep_stage_cont_groups_start_end_index, len_interseted_sleep_stage_cont_groups):
    # --------------------------------------------------------------------------    
    #  sanity check the size of the break segmentation
    # --------------------------------------------------------------------------
    len_interseted_sleep_stage_cont_groups_new =[]
    interseted_sleep_stage_cont_groups_start_end_index_new= []
    for cn_indx in range(0,len(len_interseted_sleep_stage_cont_groups)):
        if len_interseted_sleep_stage_cont_groups[cn_indx]>0:
            interseted_sleep_stage_cont_groups_start_end_index_new.append(interseted_sleep_stage_cont_groups_start_end_index[cn_indx,:])
            len_interseted_sleep_stage_cont_groups_new.append(len_interseted_sleep_stage_cont_groups[cn_indx])
            
    return interseted_sleep_stage_cont_groups_start_end_index_new, len_interseted_sleep_stage_cont_groups_new