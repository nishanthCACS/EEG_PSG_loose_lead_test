#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:34:07 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

this script contains the functions related to correlation-coefficient calcultaions
Editted on Mon Jan 23 09:56:45 2023 to run on the relative index files

Modified to have only the outlier based functions
on Tue Jan 31 16:00:29 2023

Modified on Mon Feb  6 09:38:38 2023 to include the
 Fishers normalisation (arctanh)
 
 Modified on Thu Feb  9 11:31:08 2023 another major bug fixed, the arousals are annotated on the same mapped data the stats found

Modified on Created on Mon Mar  6 10:32:08 2023
 to chaek the variance based analysis to find the vertical spikes potential loose-lead before correlation based analysis
Modified on Thu Mar 16 08:46:00 2023
-to consider the miimum 100 samples, approximately 4-epochs

Modified on Mon Mar 20 09:53:53 2023
to avoid the spindle occuarance places participate in the correlation analysis (this can be either only considering central/ all channels predicted spindles)


Modified on Sat Mar 25 10:07:37 2023
to  check the z-mapped distribution 
    unimodal or multimodal distribution
    
    if multimodal distribution lets handle the outlier detection
    based on the higher corre-mapped value peak and that should be the highest peak to detect the outliers
    Else whole channel is annotated as bad channel since majority of the data 
     neglecting the skewness
     Further warn the multimodal presence as well as number of modes present 
     
     
 main differences 
     between 6_1 ver vs 6_2 i inter_mediate_mapping_correlation function mapp all 15 comb
find_loose_leads updated with uninodal vs multinodal distribution consideration

Modified on Thu Apr 27 08:43:32 2023
the function find_loose_leads is accomodateed with moving window based outlier detection 


Modified on Fri Jun 16 11:13:39 2023 to accomodate the mean of the correlation track the loose lead
Modified on Mon Jul 10 23:54:17 2023 to accomdate the moving window channel

"""
import logging

import numpy as np
from copy import deepcopy


from sleep_EEG_loose_lead_detect.outlier_common.moving_window_funcs import moving_window_based_out_lier_annotator_channel_mean#moving_window_based_out_lier_annotator
from sleep_EEG_loose_lead_detect.outlier_common.outlier_based_on_distribution_fucns import out_lier_annotator_channel_mean


# from outlier_common.cont_segment_artofact_annotation import calc_artifact_start_end, finalise_artifact_start_end, calc_per_loose_lead, loose_con_seg_main_index

# from outlier_common.moving_window_funcs import break_cont_segments_intersetd_sleep_stages, median_with_moving_window_in_broken_cont_groups#, local_outlier_detection

logger = logging.getLogger("outlier_funcs")

while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
# Set logging level to the logger
# logger.setLevel(logging.DEBUG) # <-- THIS!
logger.setLevel(logging.INFO)
logger.propagate = False

# --------------------------------------------------------------------------
#this is main function to cover both NREM and REM together
#  or just one sleep-stage group for mean based outlier detection
# --------------------------------------------------------------------------
def find_loose_leads_based_mean(correlation_flatten_MT_not_opt, sleep_stages_annot_flatten, 
                     ch_names,cross_correlation_ref_dic,
                     intersted_sleep_stages_NREM_REM, 
                     outlier_basic_con=3, 
                     b_val=0.0001,inter_mediate_transform=True,
                     z_transform=True,Fisher_based=True,
                    flat_MT_consider=True,
                    flat_MT=[],
                    avoid_spindle_loc=False, spindle_enc=[], spindle_possible_channels =[],
                    break_spindle_flatten=True,break_flat_MT_flatten=True,
                    num_bins=20, density=True,#unimodal vs multinodal based parmaeters
                    no_smoothing=False, persent_consider =20,
                    tail_check_bin=True, factor_check=10,
                    factor_check_consider_second=100,
                    GMM_based=True, threh_prob = 0.01,
                    cont_EEG_segments_np=[], cont_threh=4,
                    o_p_adjuster=3, ep_length=30,threh_prob_artf_cont_seg=0.5,
                    moving_window_based=False,#moving window based parameters
                    moving_window_size=60, th_fact=4,
                    global_check=False, only_good_points=False,
                    sorted_median_cont_grp_comp_sort_median_cond=[True, 10],
                    sorted_median_cont_grp_comp_sort_max_cond=[True,10],
                    sorted_median_cont_grp_comp_sort_quan_cond=[True,10,0.75],
                    loose_lead_channels=[], verbose=False,GUI_percentile=True):

    """
    Main function predicts the outliers based on the mean of correlation
    function to cover given sleep stage group (both NREM and REM together
    or just one sleep-stage group)
    
    Such that this function should be called two seperate time to 
        process the NREM and REM seperately
    """

    # --------------------------------------------------------------------------
    # while avoiding already presented outlier annotations
    #  for an example this can be the varaince based outlier flat_MT
    # --------------------------------------------------------------------------
    if flat_MT_consider:
        flat_MT_ch =flat_MT
        arousal_annot =deepcopy(flat_MT)
    else:
        arousal_annot = np.zeros((len(sleep_stages_annot_flatten),len(ch_names)))
        flat_MT_ch=[]    

    # --------------------------------------------------------------------------
    # while avoiding the spindle occuring locations
    # --------------------------------------------------------------------------
    if avoid_spindle_loc:
        # --------------------------------------------------------------------------
        # correlation_pooler will avoid the location of the spindles when it pools so this will be omitted in the distribution based detectipn
        # --------------------------------------------------------------------------
        # default consider the spindles of all the channels
        # --------------------------------------------------------------------------
        if len(spindle_possible_channels)==0:
            spindle_enc_ch =spindle_enc
        else:
            # --------------------------------------------------------------------------
            # if the user provide the 
            #   spindle_possible_channels 
            #  then concentrate only on these spindle_possible_channels whether the spindles exist or not
            # asssign the reest of the channels no-spidles exist
            # --------------------------------------------------------------------------
            spindle_enc_ch= np.zeros((len(sleep_stages_annot_flatten),len(ch_names)))
            for ch1 in range(0,len(ch_names)):
                if (ch1 in spindle_possible_channels):
                    spindle_enc_ch[:,ch1] = spindle_enc[:,ch1] 

    else:
        spindle_enc_ch=[]


    # --------------------------------------------------------------------------
    # then the outlier detection choises
    # of how we are going to use the mean to find the outliers using 
    # either the moving window based approacgh or the distribution based approach
    # --------------------------------------------------------------------------
    if moving_window_based:
        if len(cont_EEG_segments_np)==0:
            raise ValueError("this needs to atleast one contious-segments {sleep-pre-processed-fragments (SPPF)}  ")
        arousal_annot=  moving_window_based_out_lier_annotator_channel_mean(correlation_flatten_MT_not_opt,sleep_stages_annot_flatten, intersted_sleep_stages_NREM_REM,
                              ch_names, arousal_annot, 
                              cont_EEG_segments_np, 
                            b_val=b_val,inter_mediate_transform=inter_mediate_transform,z_transform=z_transform,
                            flat_MT_consider=flat_MT_consider,flat_MT_ch=flat_MT_ch,intersted_sleep_stages_term='NREM_REM ',
                            avoid_spindle_loc=avoid_spindle_loc, spindle_enc_ch=spindle_enc_ch,
                            break_spindle_flatten=break_spindle_flatten,break_flat_MT_flatten=break_flat_MT_flatten,
                            moving_window_size=moving_window_size, th_fact=th_fact, o_p_adjuster=o_p_adjuster, ep_length=ep_length,
                            global_check = global_check, only_good_points=only_good_points,
                            sorted_median_cont_grp_comp_sort_median_cond=sorted_median_cont_grp_comp_sort_median_cond,
                sorted_median_cont_grp_comp_sort_max_cond=sorted_median_cont_grp_comp_sort_max_cond,                    

                sorted_median_cont_grp_comp_sort_quan_cond=sorted_median_cont_grp_comp_sort_quan_cond,loose_lead_channels=loose_lead_channels, verbose=verbose, GUI_percentile=GUI_percentile)
       
    else:

        arousal_annot = out_lier_annotator_channel_mean(correlation_flatten_MT_not_opt,sleep_stages_annot_flatten, intersted_sleep_stages_NREM_REM,
                              ch_names, arousal_annot,outlier_basic_con=outlier_basic_con,
                              b_val=b_val,inter_mediate_transform=inter_mediate_transform,z_transform=z_transform,Fisher_based=Fisher_based,
                              flat_MT_consider=flat_MT_consider,flat_MT_ch=flat_MT_ch, intersted_sleep_stages_term='NREM_REM ',
                              avoid_spindle_loc=avoid_spindle_loc, spindle_enc_ch=spindle_enc_ch,
                              num_bins = num_bins, density=density,
                              no_smoothing = no_smoothing, persent_consider =persent_consider,
                              tail_check_bin= tail_check_bin, factor_check=factor_check,                                         
                              factor_check_consider_second=factor_check_consider_second,                                 
                              GMM_based=GMM_based, threh_prob =threh_prob, loose_lead_channels=loose_lead_channels)          

    # --------------------------------------------------------------------------
    # make sure avoiding the spindle occuring locations
    # --------------------------------------------------------------------------

    if avoid_spindle_loc:            
        '''
            to skip the spindle present in the arousal_annot anotation
        '''
        #first find the intersept
        _arousal_annot= spindle_enc_ch*arousal_annot
    
        #then negate the common occurances
        arousal_annot = arousal_annot -  _arousal_annot

    return arousal_annot
    
# --------------------------------------------------------------------------
# 
# inorder to make the life easier this implemented function become handly
# to use the NREM and REM seperately
# 
# --------------------------------------------------------------------------
def find_loose_leads_based_mean_seperately(correlation_flatten_MT_not_opt, sleep_stages_annot_flatten, 
                      ch_names,cross_correlation_ref_dic,
                      intersted_sleep_stages_NREM, intersted_sleep_stages_REM,
                      outlier_basic_con=3, 
                      b_val=0.0001,inter_mediate_transform=True,
                      z_transform=True,Fisher_based=True,
                      thresh_min=5,
                    flat_MT_consider=True,
                    flat_MT=[],
                    avoid_spindle_loc=False, spindle_enc=[], spindle_possible_channels =[],
                    break_spindle_flatten=True,break_flat_MT_flatten=True,
                    num_bins=20, density=True,#unimodal vs multinodal based parmaeters
                    no_smoothing=False, persent_consider =20,
                    tail_check_bin=True, factor_check=10,
                    factor_check_consider_second=100,
                    GMM_based=True, threh_prob = 0.01,
                    cont_EEG_segments_np=[], cont_threh=4,
                    o_p_adjuster=3, ep_length=30,threh_prob_artf_cont_seg=0.5,
                    moving_window_based=False,#moving window based parameters
                    moving_window_size=60, th_fact=4,
                    global_check=False, only_good_points=False,
                    sorted_median_cont_grp_comp_sort_median_cond=[True, 10],
                    sorted_median_cont_grp_comp_sort_max_cond=[True,10],
                    sorted_median_cont_grp_comp_sort_quan_cond=[True,10,0.75],
                    loose_lead_channels=[]):

    """
    thresh: since the MT endup in 27 sec per epoch if more than 5 minute there is an outliers present
    27 x (5 x 60 sec / 30 sec) = 270
    
    cont_threh:4 sec means check the tail upto 4 sec; from the 1st encountered outlier's period lasts
    if the outlier continiously present more than 4 sec then place that portion as loose-lead suspect
    
    """
    thresh = np.ceil(2*27*thresh_min) # here the 2 come from 60/30; 27 time instance in MT-spectrum
    #this is the threshold value to check the number of outliers to detect potential outlier
    # thresh=len(sleep_stages_annot_flatten)/threh_frac

   
    #while avoiding the flat_MT based on the variance
    if flat_MT_consider:
        flat_MT_ch =flat_MT
        arousal_annot =deepcopy(flat_MT)
    else:
        arousal_annot = np.zeros((len(sleep_stages_annot_flatten),len(ch_names)))
        flat_MT_ch=[]    
    
    #while avoiding the spindle occuring locations
    if avoid_spindle_loc:
        #in default consider the spindles of all the channels
        if len(spindle_possible_channels)==0:
            spindle_enc_ch =spindle_enc
        else:
            spindle_enc_ch= np.zeros((len(sleep_stages_annot_flatten),len(ch_names)))
            for ch1 in range(0,len(ch_names)):
                if (ch1 in spindle_possible_channels):
                    spindle_enc_ch[:,ch1] = spindle_enc[:,ch1] 

    else:
        spindle_enc_ch=[]
            
    if not moving_window_based:
        # logger.warning("rnunning on NREM intiated")
        arousal_annot = out_lier_annotator_channel_mean(correlation_flatten_MT_not_opt,sleep_stages_annot_flatten, intersted_sleep_stages_NREM,
                              ch_names, arousal_annot,outlier_basic_con=outlier_basic_con,
                              b_val=b_val,inter_mediate_transform=inter_mediate_transform,z_transform=z_transform,Fisher_based=Fisher_based,
                              flat_MT_consider=flat_MT_consider,flat_MT_ch=flat_MT_ch, intersted_sleep_stages_term='NREM ',
                              avoid_spindle_loc=avoid_spindle_loc, spindle_enc_ch=spindle_enc_ch,
                              num_bins = num_bins, density=density,
                              no_smoothing = no_smoothing, persent_consider =persent_consider,
                              tail_check_bin= tail_check_bin, factor_check=factor_check,                                         
                              factor_check_consider_second=factor_check_consider_second,                                 
                              GMM_based=GMM_based, threh_prob =threh_prob, loose_lead_channels=loose_lead_channels)            # logger.warning("rnunning on NREM Done")
        # logger.warning("rnunning on REM intiated")
        arousal_annot = out_lier_annotator_channel_mean(correlation_flatten_MT_not_opt,sleep_stages_annot_flatten, intersted_sleep_stages_REM,
                          ch_names, arousal_annot,outlier_basic_con=outlier_basic_con,
                          b_val=b_val,inter_mediate_transform=inter_mediate_transform,z_transform=z_transform,Fisher_based=Fisher_based,
                        flat_MT_consider=flat_MT_consider,flat_MT_ch=flat_MT_ch,intersted_sleep_stages_term='REM ',
                        avoid_spindle_loc=avoid_spindle_loc, spindle_enc_ch=spindle_enc_ch,
                        num_bins = num_bins, density=density,
                        no_smoothing = no_smoothing, persent_consider =persent_consider,
                        tail_check_bin= tail_check_bin, factor_check=factor_check,                                         
                        factor_check_consider_second=factor_check_consider_second,                                 
                        GMM_based=GMM_based, threh_prob =threh_prob, loose_lead_channels=loose_lead_channels)
        # logger.warning("rnunning on REM Done")


    else:
        if len(cont_EEG_segments_np)==0:
            raise("this needs the contious-segments {sleep-pre-processed-fragments (SPPF)}  ")
        # grp = cross_correlation_ref_dic[ch]

        arousal_annot=  moving_window_based_out_lier_annotator_channel_mean(correlation_flatten_MT_not_opt,sleep_stages_annot_flatten, intersted_sleep_stages_NREM,
                              ch_names, arousal_annot, 
                              cont_EEG_segments_np, 
                            b_val=b_val,inter_mediate_transform=inter_mediate_transform,z_transform=z_transform,
                            flat_MT_consider=flat_MT_consider,flat_MT_ch=flat_MT_ch,intersted_sleep_stages_term='NREM ',
                            avoid_spindle_loc=avoid_spindle_loc, spindle_enc_ch=spindle_enc_ch,
                              break_spindle_flatten=break_spindle_flatten,break_flat_MT_flatten=break_flat_MT_flatten,
                            moving_window_size=moving_window_size, th_fact=th_fact, o_p_adjuster=o_p_adjuster, ep_length=ep_length,
                            global_check = global_check, only_good_points=only_good_points,
                            sorted_median_cont_grp_comp_sort_median_cond=sorted_median_cont_grp_comp_sort_median_cond,
                sorted_median_cont_grp_comp_sort_max_cond=sorted_median_cont_grp_comp_sort_max_cond,
                sorted_median_cont_grp_comp_sort_quan_cond=sorted_median_cont_grp_comp_sort_quan_cond, loose_lead_channels=loose_lead_channels)
        
        arousal_annot=  moving_window_based_out_lier_annotator_channel_mean(correlation_flatten_MT_not_opt,sleep_stages_annot_flatten, intersted_sleep_stages_REM,
                          ch_names, arousal_annot, 
                          cont_EEG_segments_np, 
                        b_val=b_val,inter_mediate_transform=inter_mediate_transform,z_transform=z_transform,
                        flat_MT_consider=flat_MT_consider,flat_MT_ch=flat_MT_ch,intersted_sleep_stages_term='REM ',
                        avoid_spindle_loc=avoid_spindle_loc, spindle_enc_ch=spindle_enc_ch,
                          break_spindle_flatten=break_spindle_flatten,break_flat_MT_flatten=break_flat_MT_flatten,
                        moving_window_size=moving_window_size, th_fact=th_fact, o_p_adjuster=o_p_adjuster, ep_length=ep_length,
                        global_check = global_check, only_good_points=only_good_points,
                        sorted_median_cont_grp_comp_sort_median_cond=sorted_median_cont_grp_comp_sort_median_cond,
                sorted_median_cont_grp_comp_sort_max_cond=sorted_median_cont_grp_comp_sort_max_cond,
                sorted_median_cont_grp_comp_sort_quan_cond=sorted_median_cont_grp_comp_sort_quan_cond,
                loose_lead_channels=loose_lead_channels)

    
        
        
    if avoid_spindle_loc:            
        '''
            to skip the spindle present in the arousal_annot anotation
        '''
        #first find the intersept
        _arousal_annot= spindle_enc_ch*arousal_annot
    
        #then negate the common occurances
        arousal_annot = arousal_annot -  _arousal_annot
        
    
    return arousal_annot
    

'''
function wi/o mean
'''

# def find_loose_leads(correlation_flatten_MT_not_opt, sleep_stages_annot_flatten, 
#                      interested_correlation_groups,cross_correlation_ref_dic,
#                      intersted_sleep_stages_NREM, intersted_sleep_stages_REM,
#                      outlier_basic_con=3, 
#                      b_val=0.0001,inter_mediate_transform=True,
#                      z_transform=True,Fisher_based=True,
#                      thresh_min=5,
#                     flat_MT_consider=True,
#                     flat_MT=[],
#                     avoid_spindle_loc=False, spindle_enc=[], spindle_possible_channels =[],
#                     num_bins=20, density=True,#unimodal vs multinodal based parmaeters
#                     no_smoothing=False, persent_consider =20,
#                     tail_check_bin=True, factor_check=10,
#                     factor_check_consider_second=100,
#                     GMM_based=True, threh_prob = 0.01,
#                     cont_seg_wise=False, cont_EEG_segments_np=[], cont_threh=4,
#                     o_p_adjuster=3, ep_length=30,threh_prob_artf_cont_seg=0.5,
#                     moving_window_based=False,#moving window based parameters
#                     moving_window_size=60, th_fact=4,
#                     global_check=False, only_good_points=False,
#                     sorted_median_cont_grp_comp_sort_median_cond=[True, 10],
#                     sorted_median_cont_grp_comp_sort_max_cond=[True,10],
#                     sorted_median_cont_grp_comp_sort_quan_cond=[True,10,0.75]):

#     """
#     thresh: since the MT endup in 27 sec per epoch if more than 5 minute there is an outliers present
#     27 x (5 x 60 sec / 30 sec) = 270
    
#     cont_threh:4 sec means check the tail upto 4 sec; from the 1st encountered outlier's period lasts
#     if the outlier continiously present more than 4 sec then place that portion as loose-lead suspect
    
#     """
#     thresh = np.ceil(2*27*thresh_min) # here the 2 come from 60/30; 27 time instance in MT-spectrum
#     #this is the threshold value to check the number of outliers to detect potential outlier
#     # thresh=len(sleep_stages_annot_flatten)/threh_frac

#     #% then based on the losse lead values find the loose potential lead 
#     arousal_annot_7= np.zeros((len(interested_correlation_groups), len(sleep_stages_annot_flatten)))

#     for g in range(0,len(interested_correlation_groups)):
#         # g=0
#         arousal_annot= np.zeros((len(sleep_stages_annot_flatten)))
#         ch =interested_correlation_groups[g]
#         channel_combintaions_considered=[]
#         channel_combintaions_considered.append(cross_correlation_ref_dic[ch])
        
#         ch1=cross_correlation_ref_dic[ch][0][0]
#         ch2=cross_correlation_ref_dic[ch][0][1]
        
#         #while avoiding the flat_MT based on the variance
#         if flat_MT_consider:
#             flat_MT_ch =np.sum(flat_MT[:,[ch1,ch2]],axis=1)
#             flat_MT_ch = np.where(flat_MT_ch==0,flat_MT_ch,1)
#             arousal_annot =deepcopy(flat_MT_ch)
#         else:
#             arousal_annot= np.zeros((len(sleep_stages_annot_flatten)))
#             flat_MT_ch=[]    
        
#         #while avoiding the spindle occuring locations
#         if avoid_spindle_loc:
#             #in default consider the spindles of all the channels
#             if len(spindle_possible_channels)==0:
#                 spindle_enc_ch = np.sum(spindle_enc[:,[ch1,ch2]],axis=1) 
#                 spindle_enc_ch = np.where(spindle_enc_ch==0,spindle_enc_ch,1)
#             else:
#                 if (ch1 in spindle_possible_channels) and (ch2 in spindle_possible_channels):
#                     spindle_enc_ch = np.sum(spindle_enc[:,[ch1,ch2]],axis=1) 
#                     spindle_enc_ch = np.where(spindle_enc_ch==0,spindle_enc_ch,1)
#                 elif (ch1 in spindle_possible_channels):
#                     spindle_enc_ch = spindle_enc[:,ch1] 
#                 elif (ch2 in spindle_possible_channels):
#                     spindle_enc_ch = spindle_enc[:,ch2] 
#                 else:
#                     spindle_enc_ch= np.zeros((len(sleep_stages_annot_flatten)))

#         else:
#             spindle_enc_ch=[]
            
#         if not moving_window_based:
#         # if not only_flat_MT:
#             # logger.warning("rnunning on NREM intiated")
#             arousal_annot = out_lier_annotator(correlation_flatten_MT_not_opt,sleep_stages_annot_flatten, intersted_sleep_stages_NREM,
#                                   channel_combintaions_considered, arousal_annot,outlier_basic_con=outlier_basic_con,
#                                   b_val=b_val,inter_mediate_transform=inter_mediate_transform,z_transform=z_transform,Fisher_based=Fisher_based,
#                                   flat_MT_consider=flat_MT_consider,flat_MT_ch=flat_MT_ch, intersted_sleep_stages_term='NREM '+ch,
#                                   avoid_spindle_loc=avoid_spindle_loc, spindle_enc_ch=spindle_enc_ch,
#                                   num_bins = num_bins, density=density,
#                                   no_smoothing = no_smoothing, persent_consider =persent_consider,
#                                   tail_check_bin= tail_check_bin, factor_check=factor_check,                                         
#                                   factor_check_consider_second=factor_check_consider_second,                                 
#                                   GMM_based=GMM_based, threh_prob =threh_prob)            # logger.warning("rnunning on NREM Done")
#             # logger.warning("rnunning on REM intiated")
#             arousal_annot = out_lier_annotator(correlation_flatten_MT_not_opt,sleep_stages_annot_flatten, intersted_sleep_stages_REM,
#                               channel_combintaions_considered, arousal_annot,outlier_basic_con=outlier_basic_con,
#                               b_val=b_val,inter_mediate_transform=inter_mediate_transform,z_transform=z_transform,Fisher_based=Fisher_based,
#                             flat_MT_consider=flat_MT_consider,flat_MT_ch=flat_MT_ch,intersted_sleep_stages_term='REM '+ch,
#                             avoid_spindle_loc=avoid_spindle_loc, spindle_enc_ch=spindle_enc_ch,
#                             num_bins = num_bins, density=density,
#                             no_smoothing = no_smoothing, persent_consider =persent_consider,
#                             tail_check_bin= tail_check_bin, factor_check=factor_check,                                         
#                             factor_check_consider_second=factor_check_consider_second,                                 
#                             GMM_based=GMM_based, threh_prob =threh_prob)
#             # logger.warning("rnunning on REM Done")


#         else:
#             if len(cont_EEG_segments_np)==0:
#                 raise("this needs the contious-segments {sleep-pre-processed-fragments (SPPF)}  ")
#             grp = cross_correlation_ref_dic[ch]

#             arousal_annot=  moving_window_based_out_lier_annotator(correlation_flatten_MT_not_opt,sleep_stages_annot_flatten, intersted_sleep_stages_NREM,
#                                   channel_combintaions_considered, arousal_annot, grp,
#                                   cont_EEG_segments_np, 
#                                 b_val=b_val,inter_mediate_transform=inter_mediate_transform,z_transform=z_transform,
#                                 flat_MT_consider=flat_MT_consider,flat_MT_ch=flat_MT_ch,intersted_sleep_stages_term='NREM '+ch,
#                                 avoid_spindle_loc=avoid_spindle_loc, spindle_enc_ch=spindle_enc_ch,
#                                 moving_window_size=moving_window_size, th_fact=th_fact, o_p_adjuster=o_p_adjuster, ep_length=ep_length,
#                                 global_check = global_check, only_good_points=only_good_points,
#                                 sorted_median_cont_grp_comp_sort_median_cond=sorted_median_cont_grp_comp_sort_median_cond,
#                     sorted_median_cont_grp_comp_sort_max_cond=sorted_median_cont_grp_comp_sort_max_cond,
#                     sorted_median_cont_grp_comp_sort_quan_cond=sorted_median_cont_grp_comp_sort_quan_cond)
            
#             arousal_annot=  moving_window_based_out_lier_annotator(correlation_flatten_MT_not_opt,sleep_stages_annot_flatten, intersted_sleep_stages_REM,
#                               channel_combintaions_considered, arousal_annot, grp,
#                               cont_EEG_segments_np, 
#                             b_val=b_val,inter_mediate_transform=inter_mediate_transform,z_transform=z_transform,
#                             flat_MT_consider=flat_MT_consider,flat_MT_ch=flat_MT_ch,intersted_sleep_stages_term='REM '+ch,
#                             avoid_spindle_loc=avoid_spindle_loc, spindle_enc_ch=spindle_enc_ch,
#                             moving_window_size=moving_window_size, th_fact=th_fact, o_p_adjuster=o_p_adjuster, ep_length=ep_length,
#                             global_check = global_check, only_good_points=only_good_points,
#                             sorted_median_cont_grp_comp_sort_median_cond=sorted_median_cont_grp_comp_sort_median_cond,
#                     sorted_median_cont_grp_comp_sort_max_cond=sorted_median_cont_grp_comp_sort_max_cond,
#                     sorted_median_cont_grp_comp_sort_quan_cond=sorted_median_cont_grp_comp_sort_quan_cond)

            
#         if avoid_spindle_loc:            
#             '''
#                 to skip the spindle present in the arousal_annot anotation
#             '''
#             #first find the intersept
#             _arousal_annot= spindle_enc_ch*arousal_annot
        
#             #then negate the common occurances
#             arousal_annot = arousal_annot -  _arousal_annot
            
#         arousal_annot_7[g,:]=deepcopy(arousal_annot)
#     summation_out_lier = np.sum(arousal_annot_7,axis=1)
    
#     '''
#     for the time being this is piked based on the overall presence of outliers
#     mean above 5 minutes annotated are marked as detected outlier group
#     '''
#     detected_outliers =[]
#     for g in range(0,len(interested_correlation_groups)):  
#         if summation_out_lier[g]>thresh:
#             detected_outliers.append(interested_correlation_groups[g])
            
#     loose_lead_sus = detect_looseleads_bassed_on_continious_outliers(arousal_annot_7, thrh=cont_threh)
#     if cont_seg_wise:
#         loose_lead_sub_cont, len_cont_seg = calc_artifact_start_end(loose_lead_sus, cont_EEG_segments_np, o_p_adjuster=3, ep_length=30)
#         arouse_check_dic_main = finalise_artifact_start_end(loose_lead_sub_cont)
#         per_check_main =calc_per_loose_lead(len_cont_seg,arouse_check_dic_main)
#         loose_con_seg_main, arousal_annot_7_cp = loose_con_seg_main_index(per_check_main, len_cont_seg, arousal_annot_7,threh_prob_artf_cont_seg=threh_prob_artf_cont_seg)       
              
        
#         return summation_out_lier, arousal_annot_7, detected_outliers, loose_lead_sus, arousal_annot_7_cp

#     else:
#         return summation_out_lier, arousal_annot_7, detected_outliers, loose_lead_sus

