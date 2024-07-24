#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:34:07 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

this script contains the functions related to correlation-coefficient calcultaions
Editted on Mon Jan 23 09:56:45 2023 to run on the relative index files

"""
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from copy import deepcopy

from sleep_EEG_loose_lead_detect.channel_correlation_outlier.poolers_Z_standardization_funcs import inter_mediate_mapping_correlation
from sleep_EEG_loose_lead_detect.MT_spectrum.Multitaper_class import taper_eigen_extractor_optim_bandwidth, taper_eigen_extractor
from sleep_EEG_loose_lead_detect.MT_spectrum.Multitaper_class import overlap_window_1sec_fixed_slide_spectrogram_given_freq_res
from sleep_EEG_loose_lead_detect.GUI_interface.percentage_bar_vis  import percent_complete

# from out_lier_based_fun_vin5 import inter_mediate_mapping_correlation
'''
https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html on Mon Jan 16 09:00:49 2023
the pearson product-moment correlation coefficients are only calaultaed by numpy

when we calculatethe correlation coefficient via numpy 
the rows should be variables (in EEG here herethe rows should be different EEG-channels)
coloumns are the observations of these variables

'''
logger = logging.getLogger("correlation_funcs")
while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
# Set logging level to the logger
logger.setLevel(logging.INFO)
logger.propagate = False

def time_correlation_coefficient_retriever_for_cont_seg(ex_b,Fs,T=4,sliding_size=1,
                                                        pearson=False,spearman=True):
    """
    ex_b = given signal to calculate the correlation, shape should be 
        channels  x flatten axis
        
    Fs= sampling frquencncy


    T=4; if we take the 4 sec window for calculating the correlation coefficient
    sliding_size=1; 1sec sliding the window is sliding with 1 sec, means 4 sec window going to have 3 sec overlap   

    number of combinations
    if we consider all 6 channels for check correlation calcultaion end up in 15
    6C2 = 6!/(4!x2!)=15
    if we consider the only the frontal and central 4C2 = 6 combinations, but in this case another question arises 
    are we not considering the occipital channels arousals
    """
    # --------------------------------------------------------------------------
    # window size
    # --------------------------------------------------------------------------
    N=int(T*Fs)
    Fs=int(Fs)


    # --------------------------------------------------------------------------
    # this is sliding cretion with the sampling size (Fs)
    # --------------------------------------------------------------------------
    corr_coeff_time_t=np.zeros((len(list(range(0,np.size(ex_b,axis=1)+1-N,sliding_size*Fs))),np.size(ex_b,axis=0),np.size(ex_b,axis=0)))
    a=0
    for j in range(0,np.size(ex_b,axis=1)+1-N,sliding_size*Fs):
        if pearson:
            corr_coeff_time_t[a,:,:]=np.corrcoef(ex_b[:,j:j+N])
        elif spearman:
            # --------------------------------------------------------------------------
            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html when each row represnet the variable
            # --------------------------------------------------------------------------
            corr_coeff_time_t[a,:,:],_= spearmanr(ex_b[:,j:j+N],axis=1)

        a=a+1
    return corr_coeff_time_t



def time_correlation_coefficient_retriever_for_cont_seg_main(EEG_data_given,sleep_stages, cont_EEG_segments,  Fs,start_time_idx=[],
                                                             padding=0,  window_time=30,    T=4, sliding_size=1,
                                                             interested_channels_index=[],root_data=True, 
                                                             leaving_part_com=False, leaving_indexes_count=3,
                                                             b_val=0.0001,inter_mediate_transform=True,
                                                             return_flatten=True,verbose=False):
    
    '''
    root_data: Uses the main six channels
    
    the last index of the continius segmentation is not part of the continious segmenation 
    use regular python index directly
    
    Inorder to compare MT and time correlation in same plane
        leaving_part_com=This decide the last three indexes are going to be 
        leaving_indexes_count=3 how many indexes left in the continious segment 
    '''

    window_size = int(round(window_time*Fs))
    if verbose:
        logger.info("EEG_data_given shape: {}".format(' x '.join(map(str, np.shape(EEG_data_given)))))
    if root_data:
        logger.warning('Initiated on Root EEG with no-referenced chanels')
        if len(start_time_idx)==0:
            raise ValueError("Please assign start_time_idx")
        EEG_segs = EEG_data_given[:,list(map(lambda x:np.arange(x-padding,x+window_size+padding), start_time_idx))]#.transpose(1,0,2)  # (#window, #ch, window_size+2padding)
    else:
        logger.warning('Initiated on processed channels with referenced chanels')

        EEG_segs = EEG_data_given[:,interested_channels_index,:].transpose(1,0,2)
    if leaving_part_com:
        logger.warning('Purposely leaving %i sec in each continiou segments for comparison'%leaving_indexes_count)
       
    if verbose:
        logger.info('Time window is with %f sec'%T)
    corr_coeff_time=[]
        
    start_point =0
    end_point=0
    sleep_stages_annot = []


    if verbose:
        logger.info("EEG_segs shape: {}".format(' x '.join(map(str, np.shape(EEG_segs)))))
        logger.info("cont_EEG_segments shape: {}".format(' x '.join(map(str, np.shape(cont_EEG_segments)))))


    for i in range(0,np.size(cont_EEG_segments,axis=0)):

        sel_chk=EEG_segs[:,cont_EEG_segments[i][0]:cont_EEG_segments[i][1],:]
        end_point+=(cont_EEG_segments[i][1]-cont_EEG_segments[i][0])
        logger.debug("start point: %i  end point: %i", start_point, end_point)
    
        sleep_stages_annot_t=[]
        for k in range(0,(end_point-start_point)-1):
            sleep_stages_annot_t.append(np.ones((window_time-(T-sliding_size))) *sleep_stages[start_point+k])
            if sleep_stages[start_point+k]== sleep_stages[start_point+k+1]:
                sleep_stages_annot_t.append(np.ones(((T-sliding_size))) *sleep_stages[start_point+k])#assign unknown
            else:
                sleep_stages_annot_t.append(np.ones(((T-sliding_size))) *((sleep_stages[start_point+k]+sleep_stages[start_point+k+1])/2))#assign unknown
            logger.debug("start_point+k:' %i  ",start_point+k)
        logger.debug("end_point-1:' %i  ",end_point-1)

        if leaving_part_com:
            sleep_stages_annot_t.append(np.ones((window_time-leaving_indexes_count)) *sleep_stages[end_point-1])
        else:    
            sleep_stages_annot_t.append(np.ones((window_time-(T-sliding_size))) *sleep_stages[end_point-1])
    
        # --------------------------------------------------------------------------
        # then hstack the continious segmentations
        # --------------------------------------------------------------------------
        ex_b = np.squeeze(np.concatenate(np.hsplit(sel_chk, np.size(sel_chk, axis=1)),axis=2))
        corr_coeff_time_t = time_correlation_coefficient_retriever_for_cont_seg(ex_b,Fs,T=T,sliding_size=sliding_size)
        if leaving_part_com:
            corr_coeff_time.append(deepcopy(corr_coeff_time_t[0:(-leaving_indexes_count),:,:]))
        else:
            corr_coeff_time.append(deepcopy(corr_coeff_time_t))

        sleep_stages_annot.append(deepcopy(np.concatenate(sleep_stages_annot_t,axis=0)))
        
        start_point+=(cont_EEG_segments[i][1]-cont_EEG_segments[i][0])

    if return_flatten:
        correlation_flatten = np.concatenate(corr_coeff_time,axis=0)
        if inter_mediate_transform:
            correlation_flatten = inter_mediate_mapping_correlation(correlation_flatten,b_val=b_val)
        
        sleep_stages_annot_flatten = np.concatenate(sleep_stages_annot,axis=0)
        return correlation_flatten, sleep_stages_annot_flatten
    else:
        return corr_coeff_time, sleep_stages_annot


def MT_based_correltion_calc_in_continious_segs(EEG_data_given, sleep_stages, cont_EEG_segments, 
                                                Fs,ch_names, start_time_idx=[], T=4, 
                                                f_min_interst=0.5, f_max_interst=32.5, 
                                                window_time=30, padding=0, 
                                                sleep_stage_anoot_ext=False,
                                                db_scale=False, save_db_scale=False,
                                                pearson=False,spearman=True,
                                                interested_channels_index=[],root_data=True,
                                                b_val=0.0001,inter_mediate_transform=True,optim_bandwidth=True,
                                                save_MT_spectrum = False, return_flatten=True, verbose=False,
                                                GUI_percentile=True):
    '''
    
    obtain the MT-spectrum for continious segmentations seperately and then obtain the correlation value
            sleep_stage_anoot_ext=False#like extracting the sleep_stage_anoot or not 
    save_db_scale: return the obtained MT_spectrum db scale whether the correlation vlues calculated in dB or not
    '''
    if  verbose:
        logger.info("intialising the MT-extraction")
    # if GUI_percentile:
    #     percent_complete(1, 100, bar_width=60, title="Ini-MT-correlation-extraction", print_perc=True)

    if optim_bandwidth:
        tapers, eigen, d_f, N = taper_eigen_extractor_optim_bandwidth(4,Fs,verbose=verbose)
    else:
        tapers, eigen, d_f, N = taper_eigen_extractor(4,Fs,verbose=verbose)
        
    if  verbose:
        logger.info("MT spectrums' applying tapers calculated ")
    # if GUI_percentile:
    #     percent_complete(25, 100, bar_width=60, title="Ini-MT-correlation-extraction", print_perc=True)
    # --------------------------------------------------------------------------
    # since we are using the overlap_window_1sec_fixed_slide_spectrogram_given_freq_res function 
    # that is fixed for 1 sec sliding window
    # --------------------------------------------------------------------------
    sliding_size=1    

    N=int(T*Fs)#window size
    Fs=int(Fs)
    
    d_f=1/T
    # --------------------------------------------------------------------------
    # The correltation is calaultaed based on th egiven range of the frequency
    # --------------------------------------------------------------------------
    f_min_index = int(f_min_interst/d_f)
    f_max_index = int(f_max_interst/d_f)
    
    
    window_size = int(round(window_time*Fs))
    
    if root_data:
        EEG_segs = EEG_data_given[:,list(map(lambda x:np.arange(x-padding,x+window_size+padding), start_time_idx))]#.transpose(1,0,2)  # (#window, #ch, window_size+2padding)
    else:
        EEG_segs = EEG_data_given[:,interested_channels_index,:].transpose(1,0,2)
   
    if  verbose:
        logger.info("EEG data prepared for MT-extraction, correlation calculation intiated")

    # if GUI_percentile:
    #     percent_complete(100, 100, bar_width=60, title="Ini-MT-correlation-extraction", print_perc=True)
        
    corr_coeff_time=[]
        
    start_point =0
    end_point=0
    sleep_stages_annot = []

    if save_MT_spectrum:
        spectrogram_col_p_all=[]
    # --------------------------------------------------------------------------
    #  calculate teh eprcentile for GUI interface
    # to ensure the prediction satisfies only with the single continious segment
    # --------------------------------------------------------------------------
    if GUI_percentile:
        if np.size(cont_EEG_segments,axis=0)==1:
            size_for_per=1
        else:
            size_for_per = np.size(cont_EEG_segments,axis=0)-1
        
    for co in range(0,np.size(cont_EEG_segments,axis=0)):

        if GUI_percentile:
            percent_complete(co,size_for_per , bar_width=60, title="    MT-correlation-extraction", print_perc=True)

        sel_chk=EEG_segs[:,cont_EEG_segments[co][0]:cont_EEG_segments[co][1],:]
        ex_b = np.squeeze(np.concatenate(np.hsplit(sel_chk, np.size(sel_chk, axis=1)),axis=2))
        
        # --------------------------------------------------------------------------
        # first claculte the MT-spectrum for six channels
        # since the six cahnnels information is needed to check the correlation between them
        # --------------------------------------------------------------------------

        channel=0
        c=ex_b[channel,:]#.flatten()
        spectrogram_col_g, t, xf = overlap_window_1sec_fixed_slide_spectrogram_given_freq_res(N,c,tapers,eigen,d_f,Fs)
            
        spectrogram_col_t=spectrogram_col_g[f_min_index:f_max_index,:]
        spectrogram_col_p = np.zeros((len(ch_names),f_max_index-f_min_index,np.size(spectrogram_col_t,axis=1)))
        spectrogram_col_p[channel,:,:]=deepcopy(spectrogram_col_t)
        
        for channel in range(1,len(ch_names)):
            c=ex_b[channel,:]#.flatten()
            spectrogram_col_g, t, xf = overlap_window_1sec_fixed_slide_spectrogram_given_freq_res(N,c,tapers,eigen,d_f,Fs)
            spectrogram_col_p[channel,:,:]=deepcopy(spectrogram_col_g[f_min_index:f_max_index,:])
                
        if save_MT_spectrum and (not save_db_scale):
            spectrogram_col_p_all.append(deepcopy(spectrogram_col_p))
        
        # --------------------------------------------------------------------------
        # convert the obtained MT-spectrum to log-scale to calculate the correlation
        # --------------------------------------------------------------------------
        if db_scale:
            spectrogram_col_p = 10*np.log10(spectrogram_col_p)

            if save_MT_spectrum and save_db_scale:
                spectrogram_col_p_all.append(deepcopy(spectrogram_col_p))
        else:
             if save_MT_spectrum and save_db_scale:
                 spectrogram_col_p_all.append(deepcopy(10*np.log10(spectrogram_col_p)))
        # --------------------------------------------------------------------------
        # then using the pre-calculated MT-spectrum values to calcultate the pearson correlation coefficient
        # --------------------------------------------------------------------------

        corr_coeff_time_t=np.zeros((len(list(range(0,np.size(ex_b,axis=1)+1-N,sliding_size*Fs))),np.size(ex_b,axis=0),np.size(ex_b,axis=0)))
        for cor in range(0,np.size(spectrogram_col_p,axis=2)):
            if pearson:
                corr_coeff_time_t[cor,:,:]=np.corrcoef(spectrogram_col_p[:,:,cor])
            elif spearman:
                # --------------------------------------------------------------------------
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html when each row represnet the variable
                # --------------------------------------------------------------------------
                corr_coeff_time_t[cor,:,:],_=spearmanr(spectrogram_col_p[:,:,cor],axis=1)

        corr_coeff_time.append(deepcopy(corr_coeff_time_t))
        
        # --------------------------------------------------------------------------
        # relevant to to sleep-stage annotation   
        # --------------------------------------------------------------------------

        if sleep_stage_anoot_ext:
            end_point+=(cont_EEG_segments[co][1]-cont_EEG_segments[co][0])
        
            sleep_stages_annot_t=[]
            for k in range(0,(end_point-start_point)-1):
                sleep_stages_annot_t.append(np.ones((window_time-(T-sliding_size))) *sleep_stages[start_point+k])
                # --------------------------------------------------------------------------
                # assign the next sleep-stage
                # --------------------------------------------------------------------------
                sleep_stages_annot_t.append(np.ones(((T-sliding_size))) *sleep_stages[start_point+k])

            sleep_stages_annot_t.append(np.ones((window_time-(T-sliding_size))) *sleep_stages[end_point-1])
            sleep_stages_annot.append(deepcopy(np.concatenate(sleep_stages_annot_t,axis=0)))
            start_point+=(cont_EEG_segments[co][1]-cont_EEG_segments[co][0])
    if  verbose:
        logger.info("EEG data prepared for MT-extraction, correlation calculation done")
        
    if return_flatten:
        correlation_flatten = np.concatenate(corr_coeff_time,axis=0)
        if inter_mediate_transform:
            correlation_flatten = inter_mediate_mapping_correlation(correlation_flatten,b_val=b_val)
    else:
        if inter_mediate_transform:
            raise Exception("inter_mediate_mapping_correlation is built for flatten please do it explicitly make inter_mediate_transform=False")
    

    if save_MT_spectrum:
        if sleep_stage_anoot_ext:
            if return_flatten:
                sleep_stages_annot_flatten = np.concatenate(sleep_stages_annot,axis=0)
                return correlation_flatten, sleep_stages_annot_flatten, spectrogram_col_p_all
            else:
                return corr_coeff_time, sleep_stages_annot, spectrogram_col_p_all

        else:
            if return_flatten:
                return correlation_flatten, spectrogram_col_p_all
            else:
                return corr_coeff_time, spectrogram_col_p_all
    else:
        if sleep_stage_anoot_ext:
            if return_flatten:
                sleep_stages_annot_flatten = np.concatenate(sleep_stages_annot,axis=0)
                return correlation_flatten, sleep_stages_annot_flatten
            else:
                return corr_coeff_time, sleep_stages_annot
        else:
            if return_flatten:
                return correlation_flatten
            else:
                return corr_coeff_time
                
def sleep_annot_retriever_sep_MT(cont_EEG_segments,sleep_stages, T=4, 
                                                window_time=30):
    
    '''
    This function assign the sleep-stage annotation for MT
    
    '''
    # --------------------------------------------------------------------------
    # since we are using the overlap_window_1sec_fixed_slide_spectrogram_given_freq_res function
    # that is fixed for 1 sec sliding window
    # --------------------------------------------------------------------------
    sliding_size=1
    start_point =0
    end_point=0
    sleep_stages_annot = []
    for co in range(0,np.size(cont_EEG_segments,axis=0)):
        # --------------------------------------------------------------------------
        # relevant to sleep-stage annotation
        # --------------------------------------------------------------------------

        end_point+=(cont_EEG_segments[co][1]-cont_EEG_segments[co][0])
    
        sleep_stages_annot_t=[]
        for k in range(0,(end_point-start_point)-1):
            sleep_stages_annot_t.append(np.ones((window_time-(T-sliding_size))) *sleep_stages[start_point+k])
            # --------------------------------------------------------------------------
            # assign unknown
            # --------------------------------------------------------------------------
            sleep_stages_annot_t.append(np.ones(((T-sliding_size))) *sleep_stages[start_point+k])

        sleep_stages_annot_t.append(np.ones((window_time-(T-sliding_size))) *sleep_stages[end_point-1])
        sleep_stages_annot.append(deepcopy(np.concatenate(sleep_stages_annot_t,axis=0)))
        start_point+=(cont_EEG_segments[co][1]-cont_EEG_segments[co][0])
    
    sleep_stages_annot_flatten = np.concatenate(sleep_stages_annot,axis=0)

    return sleep_stages_annot_flatten
