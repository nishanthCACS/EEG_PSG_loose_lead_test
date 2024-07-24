''#!/usr/bin/env python3
# -*- coding: utf-8 -*-'''
"""***********************************
Created on Mon Sep 13 21:46:46 2021

@author: yeungds
@modified by: yuanyuan.li@nih.gov
***********************************"""

import subprocess
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import detrend
from joblib import Parallel, delayed
#import nitime.algorithms as tsa
#from multitaper_spectrogram import *
from mne.time_frequency import psd_array_multitaper
from bandpower import *
# this is python version of sample entropy which is very slow but runs on different OSs
#from sampen_python.sampen2 import sampen2  # https://sampen.readthedocs.io/en/latest/
from sampen import sampen2
# this is C++ version of sample entropy which is not slow but needs compilation on Linux
#SAMPEN_PATH = '/ddn/gs1/home/liy19/tb/cp_bi/sampen/1.0.0/c/sampen'  # https://www.physionet.org/physiotools/sampen/


def compute_power_each_seg(eeg_seg, Fs, NW, band_freq, window_length):

    window_starts = np.arange(0,eeg_seg.shape[1]-window_length+1,window_step)
    EEGs = detrend(eeg_seg[:,list(map(lambda x:np.arange(x,x+window_length), window_starts))], axis=2)


    BW = NW*2./(window_length/Fs)   # BW=1.0 Double check on 4sec? 

    spec, freq = psd_array_multitaper(EEGs, Fs, fmin=bandpass_freq[0], fmax=bandpass_freq[1], 
                                    adaptive=False, low_bias=True, n_jobs=1, verbose='ERROR', 
                                    bandwidth=BW, normalization='length')
                                    #bandwidth=BW, normalization='full')
    spec = spec.transpose(1,2,0)

    return spec, freq






def compute_features_each_seg(eeg_seg, Fs, NW, band_freq, band_names, total_freq_range, 
                              n_jobs, combined_channel_names=None, window_length=None, 
                              window_step=None, need_sampen=False):


    assert len(band_freq)==len(band_names), 'band_names must have same length as band_freq'
    if window_length is None or window_step is None:
        window_length = eeg_seg.shape[-1]
        window_step = eeg_seg.shape[-1]
    

    
    window_starts = np.arange(0,eeg_seg.shape[1]-window_length+1,window_step)
    #window_num = len(window_starts)
    EEGs = detrend(eeg_seg[:,list(map(lambda x:np.arange(x,x+window_length), window_starts))], axis=2)


    BW = NW*2./(window_length/Fs)   # BW=1.0 Double check on 4sec? 

    bf_np=np.array(band_freq) # band_freq is 2D vector with all power band ranges
    #print(bf_np.min())
    #print(bf_np.max())
    spec, freq = psd_array_multitaper(EEGs, Fs, fmin=bf_np.min(), fmax=bf_np.max(), 
                                    adaptive=False, low_bias=True, n_jobs=n_jobs, verbose='ERROR', 
                                    bandwidth=BW, normalization='length')
                                    #bandwidth=BW, normalization='full')
    spec = spec.transpose(1,2,0)


    
    if combined_channel_names is not None:
        spec = (spec[:,:,::2]+spec[:,:,1::2])/2.0

    # Calculate band power per sub window
    #  - use relative to quantify by total power within a sub-window
    #  - no need to specify band_freq, it will take min & max
    bp, band_findex = bandpower(spec, freq, band_freq, relative=True)


    ## time domain features
    
    # signal line length
    f1 = np.abs(np.diff(eeg_seg,axis=1)).sum(axis=1)*1.0/eeg_seg.shape[-1]
    # signal kurtosis
    f2 = stats.kurtosis(eeg_seg,axis=1,nan_policy='propagate')
    if need_sampen:
        # signal sample entropy
        f3 = []
        for ci in range(len(eeg_seg)):
            #Bruce, E. N., Bruce, M. C., & Vennelaganti, S. (2009).
            #Sample entropy tracks changes in EEG power spectrum with sleep state and aging. Journal of clinical neurophysiology, 26(4), 257.
            # python version (slow, for multiple OSs)
            f3.append(sampen2(list(eeg_seg[ci]),mm=2,r=0.2,normalize=True)[-1][1])  # sample entropy
            # C++ version (not slow, for Linux only)
            #sp = subprocess.Popen([SAMPEN_PATH,'-m','2','-r','0.2','-n'],stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.STDOUT)#
        

    ## frequency domain features
    f4 = [];f5 = [];f6 = [];f7 = [];f9 = []
    band_num = len(band_freq)
    for bi in range(band_num):
        if band_names[bi].lower()!='sigma': # no need for sigma band
            if len(spec)>1:  # this segment is split into multiple sub-windows
                # max, min, std of band power inside this segment
                f4.extend(np.percentile(bp[bi],95,axis=0))
                f5.extend(bp[bi].min(axis=0))
                f7.extend(bp[bi].std(axis=0)) 
            # mean band power inside this segment
            f6.extend(bp[bi].mean(axis=0))

        if len(spec)>1:
            # spectral kurtosis as a rough density measure of transient events such as spindle
            spec_flatten = spec[:,band_findex[bi],:].reshape(spec.shape[0]*len(band_findex[bi]),spec.shape[2])
            f9.extend(stats.kurtosis(spec_flatten, axis=0, nan_policy='propagate'))

    f10 = []
    delta_theta = bp[0]/(bp[1]+1)
    if len(spec)>1:  # this segment is split into multiple sub-windows
        # max, min, std, mean of delta/theta ratios inside this segment
        f10.extend(np.percentile(delta_theta,95,axis=0))
        f10.extend(np.min(delta_theta,axis=0))
    f10.extend(np.mean(delta_theta,axis=0))
    if len(spec)>1:
        f10.extend(np.std(delta_theta,axis=0))
    
    f11 = []
    delta_alpha = bp[0]/(bp[2]+1)
    if len(spec)>1:  # this segment is split into multiple sub-windows
        # max, min, std, mean of delta/alpha ratios inside this segment
        f11.extend(np.percentile(delta_alpha,95,axis=0))
        f11.extend(np.min(delta_alpha,axis=0))
    f11.extend(np.mean(delta_alpha,axis=0))
    if len(spec)>1:
        f11.extend(np.std(delta_alpha,axis=0))
    
    f12 = []
    theta_alpha = bp[1]/(bp[2]+1)
    # max, min, std, mean of theta/alpha ratios inside this segment
    if len(spec)>1:  # this segment is split into multiple sub-windows
        f12.extend(np.percentile(theta_alpha,95,axis=0))
        f12.extend(np.min(theta_alpha,axis=0))
    f12.extend(np.mean(theta_alpha,axis=0))
    if len(spec)>1:
        f12.extend(np.std(theta_alpha,axis=0))

    #----new features----
    # 0='delta',1='theta',2='alpha',3='sigma',4='beta'

    f13 = []
    theta_sigma = bp[1]/(bp[3]+1)
    # max, min, std, mean of theta/sigma ratios inside this segment
    if len(spec)>1:  # this segment is split into multiple sub-windows
        f13.extend(np.percentile(theta_sigma,95,axis=0))
        f13.extend(np.min(theta_sigma,axis=0))
    f13.extend(np.mean(theta_sigma,axis=0))
    if len(spec)>1:
        f13.extend(np.std(theta_sigma,axis=0))


    f14 = []
    alpha_sigma = bp[2]/(bp[3]+1)
    # max, min, std, mean of alpha/sigma ratios inside this segment
    if len(spec)>1:  # this segment is split into multiple sub-windows
        f14.extend(np.percentile(alpha_sigma,95,axis=0))
        f14.extend(np.min(alpha_sigma,axis=0))
    f14.extend(np.mean(alpha_sigma,axis=0))
    if len(spec)>1:
        f14.extend(np.std(alpha_sigma,axis=0))


    f15 = []
    alpha_beta = bp[2]/(bp[4]+1)
    # max, min, std, mean of alpha/beta ratios inside this segment
    if len(spec)>1:  # this segment is split into multiple sub-windows
        f15.extend(np.percentile(alpha_beta,95,axis=0))
        f15.extend(np.min(alpha_beta,axis=0))
    f15.extend(np.mean(alpha_beta,axis=0))
    if len(spec)>1:
        f15.extend(np.std(alpha_beta,axis=0))


    f16 = []
    sigma_beta = bp[3]/(bp[4]+1)
    # max, min, std, mean of sigma/beta ratios inside this segment
    if len(spec)>1:  # this segment is split into multiple sub-windows
        f16.extend(np.percentile(sigma_beta,95,axis=0))
        f16.extend(np.min(sigma_beta,axis=0))
    f16.extend(np.mean(sigma_beta,axis=0))
    if len(spec)>1:
        f16.extend(np.std(sigma_beta,axis=0))


    if need_sampen:
        return np.r_[f1,f2,f3,f4,f5,f6,f7,f9,f10,f11,f12,f13,f14,f15,f16]
    else:
        return np.r_[f1,f2,f4,f5,f6,f7,f9,f10,f11,f12,f13,f14,f15,f16]






def extract_features(eeg_segs, sleep_stages, Fs, channel_names, NW, sub_window_time=None, 
                     sub_window_step=None, combined_channel_names=None, 
                     need_sampen=False, n_jobs=1, verbose=True):
    
    
    seg_num = len(eeg_segs)
    if seg_num <= 0:
        return []

    band_names = ['delta','theta','alpha','sigma','beta']
    band_freq = [[0.25,4],[4,8],[8,12],[12,15],[15,30]]  # [Hz] yyl check 30?
    tostudy_freq = [0.3, 30.]  # [Hz]           yyl check 30?  

    sub_window_size = int(round(sub_window_time*Fs))
    sub_step_size = int(round(sub_window_step*Fs))          
    
    
    #print('-----sub_window_size=',str(sub_window_size),'-------')
    
    # Populate feature names
    feature_names = ['line_length_%s'%chn for chn in channel_names]
    feature_names += ['kurtosis_%s'%chn for chn in channel_names]
    if need_sampen:
        feature_names += ['sample_entropy_%s'%chn for chn in channel_names]

    if sub_window_time is None or sub_window_step is None:
        feats = ['mean']
    else:
        feats = ['max','min','mean','std','kurtosis']
    for ffn in feats:
        for bn in band_names:
            if ffn=='kurtosis' or bn!='sigma': # no need for sigma band
                feature_names += ['%s_bandpower_%s_%s'%(bn,ffn,chn) for chn in combined_channel_names]

    power_ratios = ['delta/theta','delta/alpha','theta/alpha','theta/sigma','alpha/sigma',
                    'alpha/beta','sigma/beta']
    for pr in power_ratios:
        if not (sub_window_time is None or sub_window_step is None):
            feature_names += ['%s_max_%s'%(pr,chn) for chn in combined_channel_names]
            feature_names += ['%s_min_%s'%(pr,chn) for chn in combined_channel_names]
        feature_names += ['%s_mean_%s'%(pr,chn) for chn in combined_channel_names]
        if not (sub_window_time is None or sub_window_step is None):
            feature_names += ['%s_std_%s'%(pr,chn) for chn in combined_channel_names]

    
    # Extract features for epoch by epoch (col:feature x row:epochs)
    df = pd.DataFrame(columns=feature_names,index=sleep_stages)
    for i in range(seg_num):
        #print(segi)
        fea = compute_features_each_seg(eeg_segs[i], Fs, NW, band_freq, band_names, tostudy_freq,
                n_jobs, combined_channel_names, sub_window_size, sub_step_size, need_sampen)

        # Build feature matrix
        df.iloc[i]=fea.tolist()
        

    return df, feature_names

