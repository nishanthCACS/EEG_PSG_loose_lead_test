#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:04:23 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved
"""
# import os
import logging
# import collections

from math import comb

import numpy as np
# import matplotlib.pyplot as plt

from copy import deepcopy
from sleep_EEG_loose_lead_detect.channel_correlation_outlier.poolers_Z_standardization_funcs import correlation_pooler, inter_mediate_mapping_correlation,z_standardization
from sleep_EEG_loose_lead_detect.channel_correlation_outlier.poolers_Z_standardization_funcs import obtain_mean_corr

# --------------------------------------------------------------------------
# need to be careful to use this functions, since these are in checking phase
# --------------------------------------------------------------------------
from sleep_EEG_loose_lead_detect.outlier_common.fun_uni_mul_temp_out_lier import unimodal_multimodal_distribution_mode_checker, tail_check_dist
from sleep_EEG_loose_lead_detect.outlier_common.fun_uni_mul_temp_out_lier import GMM_based_binomial_assum_thresh, multimodal_z_map_corr_distribution_handler



logger = logging.getLogger("outlier_dist_funcs")

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
# to use the mean of the correlation values to detect the outliers
# --------------------------------------------------------------------------
def out_lier_annotator_channel_mean(corr_check_given,sleep_stages_annot_flatten, intersted_sleep_stages,
                      ch_names, arousal_annot,outlier_basic_con=3,
                      b_val=0.0001,inter_mediate_transform=True,z_transform=True,Fisher_based=True,
                      flat_MT_consider=True,flat_MT_ch=[],intersted_sleep_stages_term='',
                      avoid_spindle_loc=False, spindle_enc_ch=[],
                        num_bins=20, density=True,
                        no_smoothing=False, persent_consider =20,
                        tail_check_bin= True, factor_check=10,
                        factor_check_consider_second=100,
                        GMM_based=True, threh_prob = 0.01,
                        loose_lead_channels=[]):

    
    # --------------------------------------------------------------------------
    # corr_check : this contain the correlation value flatten x channel x channel
    # intersted_sleep_stages : the interested sleep stages
    # arousal_annot : anotated as 1 if they fell in outlier based condition   
    # --------------------------------------------------------------------------

    
    corr_check  = deepcopy(corr_check_given)
    # --------------------------------------------------------------------------
    # to avoid the issue when we  are running NREM and REM seperately
    # WHILE CHECKING THE OUTLIERS ONLY CONCENTRATING THE CORRECT INDEX POSITION OF THE TRANSFORMED NREM AND REM
    # 
    # to avoid the issue of running NREM and REM seperately
    # 
    # --------------------------------------------------------------------------

    bm_corr_check_mapped =deepcopy(corr_check_given)
    
    # --------------------------------------------------------------------------
    # only Z-transform mapping influenced by the spindle inclusion

    #the following obtained before the mean of the correlation
    # --------------------------------------------------------------------------

    bm_corr_pool, correlation_pool_sat = correlation_pooler(corr_check, sleep_stages_annot_flatten, intersted_sleep_stages,
                                                          flat_MT_consider=flat_MT_consider, flat_MT_ch=flat_MT_ch,
                                                          avoid_spindle_loc = avoid_spindle_loc, spindle_enc_ch =spindle_enc_ch)
    # --------------------------------------------------------------------------
    #the obtain the mean of the correlation for specific channel
    # here all the combination of other channels are considered
    # --------------------------------------------------------------------------

    corr_pool = obtain_mean_corr(bm_corr_pool, ch_names = ch_names, loose_lead_channels=loose_lead_channels)
    corr_check_mapped = obtain_mean_corr(bm_corr_check_mapped, ch_names = ch_names, loose_lead_channels= loose_lead_channels)

    
    # --------------------------------------------------------------------------
    # Z-standardisation is performed on the full data
    # 
    # A-starndadisatiobn is perfered for distribution based approch
    # arousal annotation handle the wanted sleep-stage or not
    # --------------------------------------------------------------------------
    if correlation_pool_sat:
        if inter_mediate_transform:
            corr_pool = inter_mediate_mapping_correlation(corr_pool,b_val=b_val)
            corr_check_mapped =  inter_mediate_mapping_correlation(corr_check_mapped,b_val=b_val)
        if z_transform:
            corr_pool = z_standardization(corr_pool,Fisher_based=Fisher_based)
            corr_check_mapped =  z_standardization(corr_check_mapped,Fisher_based=Fisher_based)

        # --------------------------------------------------------------------------
        # to use the mean of the correlation values to detect the outliers
        # --------------------------------------------------------------------------
        summ_stat = summerty_stat_retriever_main(corr_pool,outlier_basic_con,ch_names, num_bins = num_bins, density=density,
                                                  no_smoothing = no_smoothing, persent_consider =persent_consider,
                                          tail_check_bin = tail_check_bin, factor_check = factor_check,
                                          factor_check_consider_second=factor_check_consider_second,
                                          GMM_based = GMM_based, threh_prob = threh_prob,     
                                          average_of_corr=True)
        logger.warning("Z-mapped correlation pool mean is used to find the summery-statistics for outliers")

        for sl_st_in in range(0,len(sleep_stages_annot_flatten)):
            if sleep_stages_annot_flatten[sl_st_in] in intersted_sleep_stages:
                for ch1 in range(0,len(ch_names)):
                    #TO COMPARE THE RIGHT TRANSFORMED DATA
                    if corr_check_mapped[sl_st_in,ch1] <summ_stat[ch1,0] or corr_check_mapped[sl_st_in,ch1] >summ_stat[ch1,1]:
                        arousal_annot[sl_st_in,ch1]=1
                    if arousal_annot[sl_st_in,ch1]>1:
                        arousal_annot[sl_st_in,ch1]=1
    else:
        logger.warning("correlation pool of "+intersted_sleep_stages_term+" not satisfied return the given arousal_annot")
        

    return arousal_annot

def outlier_stat_finder(correlation_sel_channel_flattened, outlier_basic_con=3):
    '''
        outlier_basic_con : this is the value we have given to select the outlier how far
            The default is 3.
    '''
    # --------------------------------------------------------------------------
    #first calculate the quantiles
    # --------------------------------------------------------------------------

    Q1 = np.quantile(correlation_sel_channel_flattened,0.25)
    Q3 = np.quantile(correlation_sel_channel_flattened,0.75)

    # --------------------------------------------------------------------------  
    #inter quantile range
    IQR = Q3 - Q1
    
    #this choose the basic factor condition to calculate the outlier
    OL_cond_base_IQR = outlier_basic_con*IQR

    max_cutoff = (Q3 + OL_cond_base_IQR)
        
    min_cutoff = (Q1 - OL_cond_base_IQR)
    
    return min_cutoff, max_cutoff


def summerty_stat_retriever_main(corr_pool,outlier_basic_con,ch_names,
                                  num_bins=20, density=True,
                        no_smoothing=False, persent_consider =20,
                        tail_check_bin=True,  factor_check=10, check_all_factors= True,
                        factor_check_consider_second=100,
                        GMM_based=True, threh_prob = 0.01,
                        average_of_corr=False):
    '''
    corr_pool : pooled values of correlation to check the 
    
    tail_check_bin: the points may fell in the tail
    
    average_of_corr: If true, this means all combinations average combinely used to find
    the average of the correlation is used to find the final summery-stat for the given channel
    '''
    
    # --------------------------------------------------------------------------
    # To run the mean of the correlation
    # --------------------------------------------------------------------------
    if not average_of_corr:
        
        a=0
        # --------------------------------------------------------------------------
        # to assign the combinations
        # 
        # 6C2 = 15 combination with 2 coloumns
        # 
        # --------------------------------------------------------------------------
        summery_stat = np.zeros((comb(len(ch_names),2),2))
        for ch1 in range(0,len(ch_names)-1): 
            for ch2 in range(ch1+1,len(ch_names)):
    
                given_single_temp_correlation_flattened = corr_pool[:,ch1,ch2]
    
                
                summery_stat[a,:]= summerty_stat_retriever_signle_corr(given_single_temp_correlation_flattened,outlier_basic_con,
                              num_bins=num_bins, density=density,
                    no_smoothing=no_smoothing, persent_consider =persent_consider,
                    tail_check_bin=tail_check_bin,  factor_check=factor_check, check_all_factors= check_all_factors,
                    factor_check_consider_second=factor_check_consider_second,
                    GMM_based=GMM_based, threh_prob =threh_prob)
    
                a=a+1
                
    else:
        #here we are considering the all combinations reltaive to specif channel to find the details
        summery_stat = np.zeros((len(ch_names),2))
        for ch1 in range(0,len(ch_names)): 
            '''
            this can be easily modified to obtain the only the intersetd channel groups 
            
            '''
            #get the mean among the correlation of the interested chanenl with other channels
            given_single_temp_correlation_flattened = corr_pool[:,ch1]
            summery_stat[ch1,:]= summerty_stat_retriever_signle_corr(given_single_temp_correlation_flattened,outlier_basic_con,
                                          num_bins=num_bins, density=density,
                                no_smoothing=no_smoothing, persent_consider =persent_consider,
                                tail_check_bin=tail_check_bin,  factor_check=factor_check, check_all_factors= check_all_factors,
                                factor_check_consider_second=factor_check_consider_second,
                                GMM_based=GMM_based, threh_prob =threh_prob)

    return summery_stat

# --------------------------------------------------------------------------
# the function summerty_stat_retriever_main 
# mainly adapted and only focus on one correlation group
# --------------------------------------------------------------------------
def summerty_stat_retriever_signle_corr(given_single_temp_correlation_flattened,outlier_basic_con,
                                  num_bins=20, density=True,
                        no_smoothing=False, persent_consider =20,
                        tail_check_bin=True,  factor_check=10, check_all_factors= True,
                        factor_check_consider_second=100,
                        GMM_based=True, threh_prob = 0.01):
    '''
    given_single_temp_correlation_flattened : pooled values of correlation to check the 
    
    tail_check_bin: the points may fell in the tail
    '''

    '''
    then check the distribution
    whthether it is uninodal/ multinodal distribution
    '''

    _ind_max, x, y_mapped = unimodal_multimodal_distribution_mode_checker(given_single_temp_correlation_flattened, 
                                                      num_bins = num_bins, density=density,
                                                      no_smoothing = no_smoothing, persent_consider =persent_consider,
                                                      plot_on = False, title = '',
                                                      save_fig = False, save_fig_name = '')
                                                      # save_fig_name+ ch_names[ch1]+'_'+ch_names[ch2])


    '''
    check the indexes to avoid the tails
    
    '''
    if tail_check_bin:
        ind_max = tail_check_dist(_ind_max, y_mapped,  factor_check=factor_check, check_all_factors=check_all_factors)
    else:
        ind_max=_ind_max 
    
    #when the number of modes greater than 1 only check the multimodal_z_map_corr_distribution_handler
    if len(ind_max)==1:
        # # print("Skipping multi distribution ",title)
        # myPlot = sns.histplot(given_single_temp_correlation_flattened,kde=True)
        # if save_fig:
        #     plt.savefig(_save_fig_name+'.png', bbox_inches='tight', pad_inches=0.05)
        # plt.show()          
        '''
        then choose the final selected distribution for summery stat calculation
        '''
        # summery_stat[a,:]= np.nanpercentile(correlation_sel_channel_flattened, (25,50,75,90))
        summery_stat_sing = outlier_stat_finder(given_single_temp_correlation_flattened, outlier_basic_con=outlier_basic_con)
        
    else:
        if GMM_based:
            # the summery_stat contain two index for min_cutt_off, max_cutt_off 
            # here the max_cutoff is assigned as maximum value of given_single_temp_correlation_flattened + 1
            summery_stat_sing = GMM_based_binomial_assum_thresh(given_single_temp_correlation_flattened,
                                                    ind_max, x, y_mapped, threh_prob=threh_prob,                                 
                                                    plot_on = False, title = '', save_fig = False, save_fig_name = '')
            #tail_peak_warn_onturn off will avoid the warning of tail peak skipped
            # print(os.getcwd(),save_fig_name+ ch_names[ch1]+'_'+ch_names[ch2])
        else:
            correlation_sel_channel_flattened = multimodal_z_map_corr_distribution_handler(given_single_temp_correlation_flattened,
                                                    ind_max, x, y_mapped, 
                                                    factor_check=factor_check,factor_check_consider_second=factor_check_consider_second,
                                                    plot_on = False, title = '', save_fig = False, save_fig_name = '',
                                                    tail_peak_warn_on=False)              
            #tail_peak_warn_onturn off will avoid the warning of tail peak skipped
                        
            '''
            then choose the final selected distribution for summery stat calculation
            '''
            summery_stat_sing= outlier_stat_finder(correlation_sel_channel_flattened, outlier_basic_con=outlier_basic_con)
           
    return summery_stat_sing

def out_lier_annotator(corr_check_given,sleep_stages_annot_flatten, intersted_sleep_stages,
                      channel_combintaions_considered, arousal_annot,outlier_basic_con=3,
                      b_val=0.0001,inter_mediate_transform=True,z_transform=True,Fisher_based=True,
                      flat_MT_consider=True,flat_MT_ch=[],intersted_sleep_stages_term='',
                      avoid_spindle_loc=False, spindle_enc_ch=[],    ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'],
                        num_bins=20, density=True,
                        no_smoothing=False, persent_consider =20,
                        tail_check_bin= True, factor_check=10,
                        factor_check_consider_second=100,
                        GMM_based=True, threh_prob = 0.01):

    
    '''
    corr_check : this contain the correlation value flatten x channel x channel
    intersted_sleep_stages : the interested sleep stages
    arousal_annot : anotated as 1 if they fell in outlier based condition
    
    '''
    corr_check  = deepcopy(corr_check_given)# to avoid the issue of running NREM and REM seperately
    #WHILE CHECKING THE OUTLIERS ONLY CONCENTRATING THE CORRECT INDEX POSITION OF THE TRANSFORMED NREM AND REM
    corr_check_mapped =deepcopy(corr_check)# to avoid the issue of running NREM and REM seperately
    
    '''
    only Z-transform mapping influenced by the spindle inclusion
    '''
    corr_pool, correlation_pool_sat = correlation_pooler(corr_check, sleep_stages_annot_flatten, intersted_sleep_stages,
                                                          flat_MT_consider=flat_MT_consider, flat_MT_ch=flat_MT_ch,
                                                          avoid_spindle_loc = avoid_spindle_loc, spindle_enc_ch =spindle_enc_ch)
    '''
    Z-standardisation is performed on the full data
    arousal annotation handle the wanted sleep-stage or not
    '''
    # print(correlation_pool_sat)
    if correlation_pool_sat:
        if inter_mediate_transform:
            corr_pool = inter_mediate_mapping_correlation(corr_pool,b_val=b_val)
            corr_check_mapped =  inter_mediate_mapping_correlation(corr_check_mapped,b_val=b_val)
        if z_transform:
            corr_pool = z_standardization(corr_pool,Fisher_based=Fisher_based)
            corr_check_mapped =  z_standardization(corr_check_mapped,Fisher_based=Fisher_based)
        '''
        below part need to be totally new if we accomodate the moving window
        '''

        # print("channel_combintaions_considered: ",channel_combintaions_considered)
        summ_stat = summerty_stat_retriever_main(corr_pool,outlier_basic_con,ch_names, num_bins = num_bins, density=density,
                                                  no_smoothing = no_smoothing, persent_consider =persent_consider,
                                          tail_check_bin = tail_check_bin, factor_check = factor_check,
                                          factor_check_consider_second=factor_check_consider_second,
                                          GMM_based = GMM_based, threh_prob = threh_prob,    ch_names =ch_names)
        
        # print('corr_check_mapped: ',np.shape(corr_check_mapped))
        for sl_st_in in range(0,len(sleep_stages_annot_flatten)):
            if sleep_stages_annot_flatten[sl_st_in] in intersted_sleep_stages:
                for grp in channel_combintaions_considered:
                    
                    if corr_check_mapped[sl_st_in,grp[0][0],grp[0][1]] <summ_stat[grp[1],0] or corr_check_mapped[sl_st_in,grp[0][0],grp[0][1]] >summ_stat[grp[1],1]:
                        arousal_annot[sl_st_in]=1
            if arousal_annot[sl_st_in]>1:
                arousal_annot[sl_st_in]=1
    else:
        logger.warning("correlation pool of "+intersted_sleep_stages_term+" not satisfied return the given arousal_annot")
          
       
    return arousal_annot

