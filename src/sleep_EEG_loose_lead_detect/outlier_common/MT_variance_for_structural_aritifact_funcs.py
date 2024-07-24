#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:43:55 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

This functions are intended to find the structural artifacts based on the variance
vertical spikes can be captured by the variance among the frequency axis
"""
import numpy as np
import logging
import matplotlib.pyplot as plt


from copy import deepcopy
from sleep_EEG_loose_lead_detect.outlier_common.outlier_based_on_distribution_fucns import outlier_stat_finder



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

def find_statistics_for_vertical_spikes(MT_spec_db,outlier_basic_con):
    
    '''
    take the MT_raw and check their variance among the frequency distribution
    
    Here we check among the frequency band
    Further we are checking the variance didtribution while pooling all the channels information together (without Z-standisation)
    
    '''
    #inorder to pick the data using only one subject then combine different of subjects finalised threhold can be derived
    pooled_variance = np.transpose(MT_spec_db.std(axis=1),(1,0))
    
    #if any of the channels have higher standard deviation means there is a continious block has an issue
    # standard_dev = np.std(pooled_variance,axis=0)
    mean = np.mean(pooled_variance,axis=0)
    
    #here we are not deviding by standard deviation, but subtract the mean to centralise due to noise channel mean value going to be changed
    standardized_var = (pooled_variance-mean)#/standard_dev
    # standardized_var = pooled_variance
    
    standardized_var_pooled =standardized_var.flatten()
    min_cutt, max_cutt = outlier_stat_finder(standardized_var_pooled, outlier_basic_con=outlier_basic_con)
    # standardized_var_pooled
    channel_dic={}
    for ch in range(0,6):
        check=[]
        for sl_pos in range(0,len(pooled_variance)):
            if standardized_var[sl_pos,ch] <min_cutt or standardized_var[sl_pos,ch] >max_cutt:
                check.append(pooled_variance[sl_pos,ch])

        channel_dic[ch]=deepcopy(check)
    return channel_dic

def find_vertical_spikes(MT_spec_db,MT_spec_raw,std_thres=5,cont_seg_wise=False, return_var=False):
    '''
        std_thres=5#mostly if that below ~8 looks concerns
    '''
    logger.warning("choose the std threhold wisely we don't gurantee the default value provided will find the spikes")
    
    #this can be done by just pooling all channels variance and find the outlier
    pooled_variance = np.transpose(MT_spec_db.std(axis=1),(1,0))

    # #here we are not deviding by standard deviation, but subtract the mean to 
    # # centralise due to noise channel mean value going to be changed
    # mean = np.mean(pooled_variance,axis=0)
    # pooled_variance = (pooled_variance-mean)#/standard_dev
    if  cont_seg_wise:
        # flat_MT = np.zeros((len(MT_spec_raw),6))
        flat_MT=np.ones((np.shape(pooled_variance)[0],np.shape(pooled_variance)[1]))
        s_pos =0
        for cont_seg in range(0,len(MT_spec_raw)):
            MT_db=  10*np.log10(MT_spec_raw[cont_seg])
            e_pos = s_pos+np.shape(MT_db)[2]
            flat_MT[s_pos:e_pos,:] = np.any(MT_db.std(axis=1)<=std_thres, axis=1)
            s_pos = s_pos+np.shape(MT_db)[2]
    else:
        
        """
        considering all the MT-frequency's variance
        """
        flat_MT=np.ones((np.shape(pooled_variance)[0],np.shape(pooled_variance)[1]))
        flat_MT = np.where(pooled_variance<std_thres,flat_MT,0)
    if return_var:
        return flat_MT, pooled_variance
    else:
        return flat_MT

def plot_vertical_spike_founder(fname,MT_spec_db,
                                # channel_combintaions_considered,
                                flat_MT, ch1,ch_names,
                                sleep_stages_annot_flatten, 
                                markersize=[0.25,0.1],
                                f_min_interst=0.5, f_max_interst=32.5, vmin=-15, vmax=15, save_fig=True, db_scale_given=False):
    
    sleep_stages_annot_plot_flatten=sleep_stages_annot_flatten/5

    
    # arousal_annot= np.zeros((len(sleep_stages_annot_flatten)))
       
    
    tt = np.arange(len(sleep_stages_annot_flatten))/3600
    
    fig = plt.figure(figsize=(12,5.5))
    
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5,2,1])
    
    ax_ss = fig.add_subplot(gs[0])
        
    ax_ss.step(tt, sleep_stages_annot_plot_flatten, color='r',linewidth=markersize[0])
    ax_ss.yaxis.grid(True)
    ax_ss.set_ylim([0.05,1.5])
    ax_ss.set_xlim([0,tt[-1]])
    
    sleep_bound =list(np.array([1,2,3,4,5,6])/5)
    ax_ss.set_yticks(sleep_bound)
    ax_ss.set_yticklabels(['N3', 'N2', 'N1', 'R', 'W','U'])
    # ax_ss.plot(tt,correlation_flatten[:,ch1,ch2],'x',markersize=markersize[1],color='b',label='F')
    ax_ss.legend()
    
    
    ax_mt1 = fig.add_subplot(gs[1])
    ax_mt1.imshow(MT_spec_db[ch1,:,:], cmap='jet', origin='lower', extent=[0,tt[-1],f_min_interst,f_max_interst], interpolation='none',aspect='auto',vmin=vmin,vmax=vmax)
    
    ax_mt1.set_ylabel('Frequency/ Hz')
    ax_mt1.set_xlabel('time (hour)')
    ax_mt1.set_title(ch_names[ch1]+' MT-spectrum')

    
    ax_s1 = fig.add_subplot(gs[2])
    ax_s1.step(tt, flat_MT[:,ch1], color='k')
    ax_s1.yaxis.grid(True)
    ax_s1.set_ylim([0.05,1.5])
    ax_s1.set_yticks([1])
    ax_s1.set_yticklabels(['Arousal'])
    ax_s1.set_xlim([0,tt[-1]])
    
    if save_fig:
        plt.savefig(ch_names[ch1]+'_vertical_spike_'+texfile_name+'.png', bbox_inches='tight', pad_inches=0.05)