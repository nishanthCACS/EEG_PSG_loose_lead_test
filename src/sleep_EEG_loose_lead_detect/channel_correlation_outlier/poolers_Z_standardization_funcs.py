#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:27:44 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

this part is seperated from the out_lier_based_fun_vin6_4_2
 
poolers_Z_standardization_funcs

modified on Fri Jun 16 12:41:49 2023
to accomodate the meean of the correlation via the function inter_mediate_mapping_correlation
the obtain_mean_corr added to obtain the mean of the correlation while avoiding self correlation

modified on Tue Jul 11 00:05:31 2023 to accomadate the loose-leads consideration in phase two mean

"""
import logging
import numpy as np

logger = logging.getLogger("poolers_z")

while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
# Set logging level to the logger
# logger.setLevel(logging.DEBUG) # <-- THIS!
logger.setLevel(logging.INFO)
logger.propagate = False



def cross_correltion_dic(ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']):
    '''
    Parameters
    ----------
    ch_names : TYPE, optional
        DESCRIPTION. The default is ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'].

    Returns
    -------
    cross_correlation_ref_dic : relaive correlation as dictionary format with indexes

    '''
    cross_correlation_ref_dic={}
    a=0
    for ch1 in range(0,len(ch_names)-1): 
        for ch2 in range(ch1+1,len(ch_names)):
            cross_correlation_ref_dic[ch_names[ch1]+'-'+ch_names[ch2]]=[[ch1,ch2],a]
            a=a+1

    return cross_correlation_ref_dic


# --------------------------------------------------------------------------
# based on the sleep stages pool all correlation values together
# --------------------------------------------------------------------------
def correlation_pooler(correlation_flatten, sleep_stages_annot_flatten, intersted_sleep_stages,
                        flat_MT_consider=False,flat_MT_ch=[],threhold_pooled_samples=99,
                        avoid_spindle_loc = False, spindle_enc_ch =[]):
    '''
    This function pool the selected sleep-stages with their correlation values
    
    threhold_pooled_samples: if the pooles samples above the given threhold only considered 
    here 99 means atleast 100 samples should be exist to consider in outlier detection
    
    avoid_spindle_loc: True will omit the correlation values occured in the spindle occured places
    in this case already skipped flat_MT_ch (like presented other algorithms/ known annotated outliers/ artifacts)
    '''
    
    corr_pool=[]
    first=True
    for sl_st_in in range(0,len(sleep_stages_annot_flatten)):
        if sleep_stages_annot_flatten[sl_st_in] in intersted_sleep_stages:
            # --------------------------------------------------------------------------
            # if we considering both variance (/ already presented outlier annotations) 
            # and avoid spindle location
            # --------------------------------------------------------------------------

            if avoid_spindle_loc and flat_MT_consider:
                if np.sum(flat_MT_ch[sl_st_in,:])==0 and np.sum(spindle_enc_ch[sl_st_in,:])==0:
                    corr_pool.append(correlation_flatten[sl_st_in])    
            # --------------------------------------------------------------------------
            # if we only avoid spindle location
            # --------------------------------------------------------------------------
            elif avoid_spindle_loc:
                if np.sum(spindle_enc_ch[sl_st_in,:])==0:
                    corr_pool.append(correlation_flatten[sl_st_in])   
                    if first:
                        logger.warning("considering any single present in any channel leave that option out, this can be developed further")
                        first=False
            # --------------------------------------------------------------------------
            # if we considering variance/ alreday presented outlier annotations
            # --------------------------------------------------------------------------
            elif flat_MT_consider:
                if np.sum(flat_MT_ch[sl_st_in,:])==0:
                    corr_pool.append(correlation_flatten[sl_st_in])
            # --------------------------------------------------------------------------
            # if we do not consider neither variance nor avoid spindle location
            # --------------------------------------------------------------------------

            else:
                corr_pool.append(correlation_flatten[sl_st_in])

    if len(corr_pool)>threhold_pooled_samples:
        corr_pool = np.stack(corr_pool,axis=0)
        correlation_pool_sat = True
    else:
        correlation_pool_sat = False
        logger.warning("Due to lack of samples (%i) the pooling not satisfied the samples need to be above %i",len(corr_pool),threhold_pooled_samples)

    return corr_pool, correlation_pool_sat


def inter_mediate_mapping_correlation(corr_coeff_t,b_val=0.0001):
    '''
        corr_coeff_t: dimention time x ch x ch
            or
        corr_coeff_t: dimention time x ch 
            or
        corr_coeff_t: dimention time
        
    this function is can map the correlation values depends on the input given
    '''
    # --------------------------------------------------------------------------
    #this is modified to accept the single group
    # --------------------------------------------------------------------------
    if len(np.shape(corr_coeff_t))==3:
        for a in range(0,np.shape(corr_coeff_t)[0]):
            for ch1 in range(0,np.shape(corr_coeff_t)[1]):
                for ch2 in range(0,np.shape(corr_coeff_t)[2]):
                    if corr_coeff_t[a,ch1,ch2]==1:
                        corr_coeff_t[a,ch1,ch2]=corr_coeff_t[a,ch1,ch2]-b_val
                    elif corr_coeff_t[a,ch1,ch2]==-1:
                        corr_coeff_t[a,ch1,ch2]=corr_coeff_t[a,ch1,ch2]+b_val
                        
    elif  len(np.shape(corr_coeff_t))==2:
        for a in range(0,np.shape(corr_coeff_t)[0]):
            for ch1 in range(0,np.shape(corr_coeff_t)[1]):
                if corr_coeff_t[a,ch1]==1:
                    corr_coeff_t[a,ch1]=corr_coeff_t[a,ch1]-b_val
                elif corr_coeff_t[a,ch1]==-1:
                    corr_coeff_t[a,ch1]=corr_coeff_t[a,ch1]+b_val
    else:
        for a in range(0,np.shape(corr_coeff_t)[0]):
            if corr_coeff_t[a]==1:
                corr_coeff_t[a]=corr_coeff_t[a]-b_val
            elif corr_coeff_t[a]==-1:
                corr_coeff_t[a]=corr_coeff_t[a]+b_val
    return corr_coeff_t


def z_standardization(corr_pool,Fisher_based=True):
    '''
            corr_pool: dimention time x ...
            such that this z-standize along the time axis

            Fisher_based: standardize with arctanh function
    '''
    if Fisher_based:
        correlation_z=np.arctanh(corr_pool)
    else:
        standard_dev = np.std(corr_pool,axis=0)
        mean = np.mean(corr_pool,axis=0)
        correlation_z = (corr_pool-mean)/standard_dev
    return correlation_z

def loose_lead_channel_count_check(ch_names,loose_lead_channels):
    if len(list(set(ch_names)))<(len(list(set(loose_lead_channels)))+2):
        raise ValueError("Atleast one good-channel need to be create the mean for each channel")


def obtain_mean_corr(bm_corr_pool_given, ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'],loose_lead_channels=[]):
    '''
    this function creats the final cmean of the correlation pool group 
    while avoiding the diagonal correlation
    '''
    corr_pool_mean_ob = np.zeros((np.shape(bm_corr_pool_given)[0],len(ch_names)))
    for ch1 in range(0,len(ch_names)):
        temp_ch_indexes = list(range(0,len(ch_names)))
        # --------------------------------------------------------------------------
        # to avod the self correlation
        # --------------------------------------------------------------------------
        temp_ch_indexes.remove(ch1)
        # --------------------------------------------------------------------------
        #  avoid teh mean of correlations with the loose-leads
        # --------------------------------------------------------------------------
        for l_ch in loose_lead_channels:
            if l_ch in temp_ch_indexes:
                temp_ch_indexes.remove(l_ch)
        corr_pool_mean_ob[:,ch1] = np.mean(bm_corr_pool_given[:,ch1,temp_ch_indexes], axis=1)
    
    return corr_pool_mean_ob
