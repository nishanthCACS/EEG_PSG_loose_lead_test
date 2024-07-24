#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:56:40 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

once the outliers are detected we can assign the loose-leads as
in diffrent ways, since some major arousals fell near to sleep-stage transition on some channels may 
mislead for loose-lead condition.

Oneway to overcome this is checking, the continious period of the outliers can be cobtained by the convolutional window, and step
the convolutional window combine the predicted outliers in the given convolutional period,
    
    And the portion of outlier exists depends on the output (dot product/ summation) of the convolutional windows
        Then we can annotate the portion as loose-lead, if the dot-product of convolutional window above the 
        ``outlier_presene_con_lenth_th'' (threshold annotation condition loose-lead)
    
    If we want obtain only outliers presence for continious period just need to assign the 
    ``outlier_presene_con_lenth_th'' the value as convoutional-period

Even after using the continious period some sumbjects' arousals mislead the outcome of loose-lead
so user can assign this value in three different ways to find the loose lead
    1> Check the overall period in time of the detected outliers exists, with/ w/o
     the continious period (convolutional window) checking the outliers. 
    
    2> Check the overall percentage of the full sleep (with the considered portion like NREM/ REM)
    
    3> Check the moving window portion to how uch percentage present as outliers, this is like convolutional window with 
    20 minutes and 5 minute sliding  (15 minutes overlap), this is almost like first continoius period finder, just re-iterate 
    or check the direct outliers
"""

import logging
import numpy as np

logger = logging.getLogger("loose_lead_finaliser")

while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
# Set logging level to the logger
logger.setLevel(logging.INFO)
logger.propagate = False



def get_con_window(thresh_min_conv=5, thresh_in_sec=False):
    '''
    to obtain the convolutional window
    thresh_min_conv: this assign the period of the convolutional window period 
    thresh_in_sec: if the value of the convolutional period is assigned in seconds
    '''
    if thresh_in_sec:
        conv_window=thresh_min_conv
    else:
        # --------------------------------------------------------------------------
        # here the 2 come from 60/30; 27 time instance in MT-spectrum
        # conv_window = int(np.ceil(2*27*thresh_min_conv)) 
        # --------------------------------------------------------------------------

        conv_window = int(np.ceil(60*thresh_min_conv)) 
    return conv_window

def conv_window_based_annotation(checking_annot,conv_window,
                                 outlier_presene_con_lenth_th= 1):
    
    '''
    
    Just assigin the parameters fo obtained outliers based on the given condition combined via
     the convolutional window check the period of exist
    
    
    '''
    # --------------------------------------------------------------------------
    #assign the convolutional window size
    # --------------------------------------------------------------------------

    sel_ch_arousl_annot = np.convolve(checking_annot,np.ones(int(conv_window)), "same")

    sel_ch_arousl_annot= np.where(sel_ch_arousl_annot>=outlier_presene_con_lenth_th,sel_ch_arousl_annot,0)
    sel_ch_arousl_annot= np.where(sel_ch_arousl_annot<outlier_presene_con_lenth_th,sel_ch_arousl_annot,1)
    return sel_ch_arousl_annot


def check_period_time(sel_ch_arousl_annot):
    '''
         Check the overall period in time of the detected outliers exists, with/ w/o
         the continious period (convolutional window) checking the outliers. 
         
         sel_ch_arousl_annot: thjis suppose to in seconds
    '''
    return np.sum(sel_ch_arousl_annot)
    
def check_period_perentage(sel_ch_arousl_annot, full=True, 
                            sleep_stages_annot_flatten=[], intersted_sleep_stages=[],
                             flat_MT_consider=False,flat_MT_ch=[],threhold_pooled_samples=99,
                             avoid_spindle_loc = False, spindle_enc_ch =[]):
    '''
      Check the overall percentage of the full sleep (with/ without the considered portion like NREM/ REM)
          full: means we are considering all the sleep-stages like 

    '''
    if full:
       chk_period =  np.sum(sel_ch_arousl_annot)
       tot_len = len(sel_ch_arousl_annot)
    else:
       
       pool_annot, pool_sat = value_pooler(sel_ch_arousl_annot, sleep_stages_annot_flatten, intersted_sleep_stages,
                        flat_MT_consider=flat_MT_consider,flat_MT_ch=flat_MT_ch,threhold_pooled_samples=threhold_pooled_samples,
                        avoid_spindle_loc = avoid_spindle_loc, spindle_enc_ch =spindle_enc_ch)
       if pool_sat:
           chk_period =  np.sum(pool_annot)
           tot_len = len(pool_annot)
       else:
             chk_period=1
             tot_len=1
    return 100* chk_period/tot_len
  

def loose_lead_per_period(sel_ch_arousl_annot, loose_conv_wind=20,     stride_size=5, conv_type='same'):
    '''
       loose_conv_wind=20# in minutes
       stride_size=5 # in minutes
       conv_type: if not same considered as valid
     
        this function uses convolution with ones window so it is just a sum with the sliding window
   
            Check the moving window portion to how uch percentage present as outliers, this is like convolutional window with 
        20 minutes and 5 minute sliding  (15 minutes overlap), this is almost like first continoius period finder, just re-iterate 
        or check the direct outliers
    '''

    
    conv_window = int(loose_conv_wind*60)
    # -------------------------------------------------------------------------- 
    # stride size
    # --------------------------------------------------------------------------
    s=int(stride_size*60)
  
    overall_percent = []
    for c in list(range(0,len(sel_ch_arousl_annot),s)):
        overall_percent.append(100*np.sum(sel_ch_arousl_annot[c:c+conv_window])/conv_window)
    if conv_type =='same':
        if len(sel_ch_arousl_annot)>(c+1):
           overall_percent.append(100*np.sum(sel_ch_arousl_annot[c:len(sel_ch_arousl_annot)])/conv_window)
           
        
    return np.array(overall_percent)

# based on the sleep stages pool all correlation values together
def value_pooler(correlation_flatten, sleep_stages_annot_flatten, intersted_sleep_stages,
                       flat_MT_consider=False,flat_MT_ch=[],threhold_pooled_samples=99,
                       avoid_spindle_loc = False, spindle_enc_ch =[]):
    '''
    This is a directly copied or adapted from 
        poolers_Z_standardization_funcs import correlation_pooler

    This function pool the selected sleep-stages with their values
    
    threhold_pooled_samples: if the pooles samples above the given threhold only considered 
    here 99 means atleast 100 samples should be exist to consider in outlier detection
    
    avoid_spindle_loc: True will omit the correlation values occured in the spindle occured places
    in this case already skipped flat_MT_ch 
    '''
    
    corr_pool=[]
    first=True
    for sl_st_in in range(0,len(sleep_stages_annot_flatten)):
        if sleep_stages_annot_flatten[sl_st_in] in intersted_sleep_stages:
            # --------------------------------------------------------------------------
            # if we considering both variance and avoid spindle location
            # --------------------------------------------------------------------------
            if avoid_spindle_loc and flat_MT_consider:
                if np.sum(flat_MT_ch[sl_st_in,:])==0 and np.sum(spindle_enc_ch[sl_st_in,:])==0:
                    corr_pool.append(correlation_flatten[sl_st_in])             
            # if we only avoid spindle location
            elif avoid_spindle_loc:
                if np.sum(spindle_enc_ch[sl_st_in,:])==0:
                    corr_pool.append(correlation_flatten[sl_st_in])   
                    if first:
                        logger.warning("considering any single present in any channel leave that option out, this can be developed further")
                        first=False
            # --------------------------------------------------------------------------
            # if we considering variance 
            # --------------------------------------------------------------------------
            elif flat_MT_consider:
                if np.sum(flat_MT_ch[sl_st_in,:])==0:
                    corr_pool.append(correlation_flatten[sl_st_in])
                    
            # --------------------------------------------------------------------------
            # if we do not consider neither variance nor avoid spindle location
            # --------------------------------------------------------------------------
            else:
                corr_pool.append(correlation_flatten[sl_st_in])

    if len(corr_pool)>99:
        corr_pool = np.stack(corr_pool,axis=0)
        correlation_pool_sat = True
    else:
        correlation_pool_sat = False
        logger.warning("Due to lack of samples the pooling not satisfied")
    return corr_pool, correlation_pool_sat



