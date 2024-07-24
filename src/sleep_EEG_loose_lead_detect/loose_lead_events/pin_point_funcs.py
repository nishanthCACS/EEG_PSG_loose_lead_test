#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 23:09:33 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

this has the function to pinpoint the loose-lead occurance with the given condition
"""
import numpy as np
from copy import deepcopy
from sleep_EEG_loose_lead_detect.loose_lead_events.loose_lead_annaot_funcs import loose_lead_per_period
from sleep_EEG_loose_lead_detect.loose_lead_events.unify_outliers_to_loose_lead import unify_outliers_via_conv

def pin_point_loose_lead(arousal_annot_NREM_REM,
            apply_conv_window=True, thresh_min_conv=5,  thresh_in_sec=True,outlier_presene_con_lenth_th=4,
            loose_check_with_fill_period=True, len_period_tol_min=5/60, 
            loose_lead_period_min =1,# percentage_on_given_period_while_sliding=False only this will be used
            percentage_on_given_period_while_sliding=False,
            overall_percent_check =False,
            num_occurance=3, percent_check= 5,
            loose_conv_wind=20,     stride_size=5, conv_type='same',
            ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']):
    
    
    '''
    arousal_annot_NREM_REM: the value of the obtained outliers
    apply_conv_window: whether apply the convolution (moving sum) to check the continious outliers
            thresh_min_conv=5,
            thresh_in_sec=True,
            outlier_presene_con_lenth_th=4
            
    loose_check_with_fill_period: whether consider the tolerence period

    percentage_on_given_period_while_sliding:    assign the loose leads based on the percentage of presence in the selected portion of window     
            loose_conv_wind=20,     stride_size=5, conv_type='same'
            percent_check= 5% to check the percentage of the outliers presence in the selected window
            
            after the sliding from the obtained outcome
              num_occurance=3, says the percentage_on_given_period_while_sliding has occured more than 3 times
            this can be implemented in plenty of ways like check the continuity etc.
            prefered to have (loose_conv_wind/stride_size)-1 to check the outcome in the continious period detected by the moving window

    loose_lead_period_min =1

    
    '''
    #this will decide the output presence period
    loose_lead_period_sec = int(loose_lead_period_min*60)
         
    loose_channel_pin_point={}
    loose_lead_channels =[]

    loose_lead_once=False
          

    # we can decide whether apply the convolution window or use the direct outliers
    if apply_conv_window:
        if thresh_in_sec:
            conv_window=thresh_min_conv
        else:
            conv_window = int(np.ceil(2*30*thresh_min_conv)) # here the 2 come from 60/30; 27 time instance in MT-spectrum

    for grp in range(0,len(ch_names)):
        
        # # we can decide whether apply the convolution window or use the direct outlier
        # if apply_conv_window:
        #     sel_ch_arousl_annot = np.convolve(arousal_annot_NREM_REM[:,grp],np.ones(int(conv_window)), "same")
        #     if len(sel_ch_arousl_annot)==6:
        #         raise
        #     sel_ch_arousl_annot= np.where(sel_ch_arousl_annot>outlier_presene_con_lenth_th,sel_ch_arousl_annot,0)
        #     sel_ch_arousl_annot= np.where(sel_ch_arousl_annot<=outlier_presene_con_lenth_th,sel_ch_arousl_annot,1)
        # else:
        #     sel_ch_arousl_annot= arousal_annot_NREM_REM[:,grp]
        arousal_annot =arousal_annot_NREM_REM[:,grp]
        # print('intial ',grp,': ',np.sum(arousal_annot))

        # show_single_outliers_before_combine_tol: must be True to show the intial outliers 
        sel_ch_arousl_annot = unify_outliers_via_conv(arousal_annot, with_conv=apply_conv_window,
                              outlier_presene_con_lenth_th=outlier_presene_con_lenth_th, thresh_min_conv=thresh_min_conv, 
                              thresh_in_sec= thresh_in_sec, conv_type =conv_type,
                            with_fill_period=loose_check_with_fill_period,  len_period_tol_min=len_period_tol_min,
                            show_single_outliers_before_combine_tol=True, verbose=False)



        # then assign the loose-leads based on the percentage_on_given_period_while_sliding
        if overall_percent_check:
            # print(np.sum(sel_ch_arousl_annot), ' and persent: ',100*np.sum(sel_ch_arousl_annot)/len(sel_ch_arousl_annot))
            if (100*np.sum(sel_ch_arousl_annot)/len(sel_ch_arousl_annot))>percent_check:
                loose_channel_pin_point[ch_names[grp]]=deepcopy(sel_ch_arousl_annot)
                loose_lead_once=True
                loose_lead_channels.append(grp)
        elif percentage_on_given_period_while_sliding:
                #this can be done in plenty of ways, here we are just checkin their percentage
            percent = loose_lead_per_period(sel_ch_arousl_annot, loose_conv_wind=loose_conv_wind, stride_size=stride_size, conv_type=conv_type)
        
            # print("here: ",grp," len: ",len(np.where(percent> percent_check)[0]),' sum: ',np.sum(sel_ch_arousl_annot))
            if len(list(np.where(percent> percent_check)[0]))>num_occurance: 
                loose_channel_pin_point[ch_names[grp]]=deepcopy(sel_ch_arousl_annot)
                loose_lead_once=True
                loose_lead_channels.append(grp)
        
        else:
            if np.sum(sel_ch_arousl_annot)>loose_lead_period_sec:
                loose_channel_pin_point[ch_names[grp]]=deepcopy(sel_ch_arousl_annot)
                loose_lead_once=True
                loose_lead_channels.append(grp)
    return loose_lead_once, loose_channel_pin_point, loose_lead_channels

# def pin_point_loose_lead(arousal_annot_NREM_REM,
#                          apply_conv_window=True,
#             thresh_min_conv=5,
#             thresh_in_sec=True,
#             outlier_presene_con_lenth_th=4,
#             loose_lead_period_min =1,# percentage_on_given_period_while_sliding=False only this will be used
#             percentage_on_given_period_while_sliding=False,
#             num_occurance=3, percent_check= 5,
#             loose_conv_wind=20,     stride_size=5, conv_type='same',
#             ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']):
    
    
#     '''
#     arousal_annot_NREM_REM: the value of the obtained outliers
#     apply_conv_window: whether apply the convolution (moving sum) to check the continious outliers
#             thresh_min_conv=5,
#             thresh_in_sec=True,
#             outlier_presene_con_lenth_th=4
            
#     percentage_on_given_period_while_sliding:    assign the loose leads based on the percentage of presence in the selected portion of window     
#             loose_conv_wind=20,     stride_size=5, conv_type='same'
#             percent_check= 5% to check the percentage of the outliers presence in the selected window
            
#             after the sliding from the obtained outcome
#               num_occurance=3, says the percentage_on_given_period_while_sliding has occured more than 3 times
#             this can be implemented in plenty of ways like check the continuity etc.
#             prefered to have (loose_conv_wind/stride_size)-1 to check the outcome in the continious period detected by the moving window

#     loose_lead_period_min =1

    
#     '''
#     #this will decide the output presence period
#     loose_lead_period_sec = int(loose_lead_period_min*60)
         
#     loose_channel_pin_point={}
#     loose_lead_channels =[]

#     loose_lead_once=False
    
#     # we can decide whether apply the convolution window or use the direct outliers
#     if apply_conv_window:
#         if thresh_in_sec:
#             conv_window=thresh_min_conv
#         else:
#             conv_window = int(np.ceil(2*30*thresh_min_conv)) # here the 2 come from 60/30; 27 time instance in MT-spectrum

#     for grp in range(0,len(ch_names)):
        
#         # we can decide whether apply the convolution window or use the direct outlier
#         if apply_conv_window:
#             sel_ch_arousl_annot = np.convolve(arousal_annot_NREM_REM[:,grp],np.ones(int(conv_window)), "same")
#             if len(sel_ch_arousl_annot)==6:
#                 raise
#             sel_ch_arousl_annot= np.where(sel_ch_arousl_annot>outlier_presene_con_lenth_th,sel_ch_arousl_annot,0)
#             sel_ch_arousl_annot= np.where(sel_ch_arousl_annot<=outlier_presene_con_lenth_th,sel_ch_arousl_annot,1)
#         else:
#             sel_ch_arousl_annot= arousal_annot_NREM_REM[:,grp]

#         # then assign the loose-leads based on the percentage_on_given_period_while_sliding
#         if percentage_on_given_period_while_sliding:
#             #this can be done in plenty of ways, here we are just checkin their percentage
#             percent = loose_lead_per_period(sel_ch_arousl_annot, loose_conv_wind=loose_conv_wind, stride_size=stride_size, conv_type=conv_type)
        
#             # print("here: ",grp," len: ",len(np.where(percent> percent_check)[0]),' sum: ',np.sum(sel_ch_arousl_annot))
#             if len(list(np.where(percent> percent_check)[0]))>num_occurance: 
#                 loose_channel_pin_point[ch_names[grp]]=deepcopy(sel_ch_arousl_annot)
#                 loose_lead_once=True
#                 loose_lead_channels.append(grp)
        
#         else:
#             if np.sum(sel_ch_arousl_annot)>loose_lead_period_sec:
#                 loose_channel_pin_point[ch_names[grp]]=deepcopy(sel_ch_arousl_annot)
#                 loose_lead_once=True
#                 loose_lead_channels.append(grp)
#     return loose_lead_once, loose_channel_pin_point, loose_lead_channels

