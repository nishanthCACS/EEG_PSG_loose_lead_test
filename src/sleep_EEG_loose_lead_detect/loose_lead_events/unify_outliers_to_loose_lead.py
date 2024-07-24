#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:01:50 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

this function is created to unify the outliers present in the single channel
"""
import logging
import numpy as np
from copy import deepcopy

# --------------------------------------------------------------------------
#logger intialisation
# --------------------------------------------------------------------------
logger = logging.getLogger("loose_lead_finaliser")

while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
# Set logging level to the logger
# logger.setLevel(logging.DEBUG) # <-- THIS!
logger.setLevel(logging.INFO)
logger.propagate = False


def unify_outliers_via_conv(arousal_annot_or, with_conv=True,
                             outlier_presene_con_lenth_th=0, thresh_min_conv=5, thresh_in_sec= True, conv_type = "same",
                             with_fill_period=False,  len_period_tol_min=5/60, show_single_outliers_before_combine_tol=True,
                             verbose=False):
    '''
     show_single_outliers_before_combine_tol = False; this will only show the filled outcome 
     This should be True all the time to avoid biased outcome
    '''
    #to avoid the issue of modifing the orginal annotation
    #due to this in intial/ second run possible error 
    arousal_annot = deepcopy(arousal_annot_or)
    if verbose:
        logger.info('unify_outliers_via_conv is intiated')

    if with_conv:
            
        # --------------------------------------------------------------------------
        #  we can decide whether apply the convolution window or use the direct outliers
        # --------------------------------------------------------------------------
        if thresh_in_sec:
            conv_window=thresh_min_conv
        else:
            conv_window = int(np.ceil(60*thresh_min_conv)) # here the 2 come from 60/30; 27 time instance in MT-spectrum
            
        if outlier_presene_con_lenth_th>=conv_window:
            raise ValueError("outlier_presene_con_lenth_th should be lower than the thresh_min_conv(/conv_window) ")
    
        sel_ch_arousl_annot = np.convolve(arousal_annot,np.ones(int(conv_window)), conv_type)
    else:
        sel_ch_arousl_annot = arousal_annot
        outlier_presene_con_lenth_th=0
        

        
    if not with_fill_period:
        sel_ch_arousl_annot= np.where(sel_ch_arousl_annot>outlier_presene_con_lenth_th,sel_ch_arousl_annot,0)
        sel_ch_arousl_annot= np.where(sel_ch_arousl_annot<=outlier_presene_con_lenth_th,sel_ch_arousl_annot,1)
        if verbose:
            logger.info('unify_outliers_via_conv is intiated')
        return sel_ch_arousl_annot

    else:
             
        # --------------------------------------------------------------------------
        #first get the index of the number of outliers per the given period
        # --------------------------------------------------------------------------
        if with_conv:
            sel_pos_index =  np.where(sel_ch_arousl_annot>outlier_presene_con_lenth_th)[0]
        else:
            sel_pos_index =  np.where(sel_ch_arousl_annot>0)[0]
            logger.debug("Here")
        # --------------------------------------------------------------------------
        #then combine the all outliers to gether to presnet the bad loose period with tolerance
        #the following variable will get the diffrecne between the anntated outlier and next outlier
        # --------------------------------------------------------------------------

        sel_pos_index_differ = sel_pos_index[1:]-sel_pos_index[0:-1]
        
        len_period_tol =len_period_tol_min*60
        sel_sat_pos_refer_index =  np.where(sel_pos_index_differ<len_period_tol)[0]
        
        
        sel_ch_arousl_annot_period_fill = np.zeros((len(sel_ch_arousl_annot)))
        # --------------------------------------------------------------------------
        #the interval need to be filled can be found by the 
        # --------------------------------------------------------------------------

        for i in sel_sat_pos_refer_index:
            sel_ch_arousl_annot_period_fill[sel_pos_index[i]:sel_pos_index[i+1]]=1
        
        if show_single_outliers_before_combine_tol:
            
            for i in sel_pos_index:
                sel_ch_arousl_annot_period_fill[i]=1
        if verbose:
            logger.info('unify_outliers_via_conv is intiated')

        return sel_ch_arousl_annot_period_fill