#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:47:00 2021

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved
"""
import numpy as np
from scipy.fft import fft

def powerline_noice_magnificant_checker(EEG_segs_sel,Fs,noice_f=60,min_in_f=5,max_in_f=25):
    '''
    EEG_segs_sel: this the selected normal epoches of one channel in far away
    noice_f: notch frequency; 60 Hz for the US powerline noice

    to be more carefull avoid the noice by the DC shift and focus on the EEG sleep data interest portion
    
    min_in_f: 5 
    max_in_f: 25

    '''
    avg_powers =np.zeros((np.size(EEG_segs_sel,axis=0),2))
    for e in range(0,np.size(EEG_segs_sel,axis=0)):
        y=EEG_segs_sel[e]
        N=len(y)
        dt=1/Fs
        # --------------------------------------------------------------------------
        # classical FFT: since we are only cheking the power line noise exist so we can safely apply FFT
        #   seems the scipy.fft is faster only that is chosen
        # --------------------------------------------------------------------------
        yf = fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*dt), N//2)
        
        avg_powers[e,:] = helper_for_fft_power_calc(xf,yf,noice_f, min_in_f, max_in_f)
    
    avg_power_near_powerline, avg_power_not_powerline = np.mean(avg_powers,axis=0)
    
    # --------------------------------------------------------------------------
    # then finally  says the signal need to go through the notch filter
    # --------------------------------------------------------------------------
    if avg_power_near_powerline > avg_power_not_powerline:
        return True 
    else:
        return False

def helper_for_fft_power_calc(xf,yf,noice_f, min_in_f, max_in_f):
    '''
    This function is for check the average power near to the powerline/ given frequnecy
    and the intended frequecy range

    xf: the ffts frequency axis
    yf: the ffts's powers
    ''' 
    indexes_near_to_powerline=np.where((xf>(noice_f-1)) & (xf<(noice_f+1)))[0]
    avg_power_near_powerline = sum(abs(yf[indexes_near_to_powerline])**2)/len(indexes_near_to_powerline)
    
    # --------------------------------------------------------------------------
    # this will sleect the inetersetd region default is designed to avoid the noice by the DC shift in EEG sleep analysis
    # --------------------------------------------------------------------------

    indexes_out_of_powerline_dc_shift_sleep_EEG = np.where((xf>min_in_f) & (xf < max_in_f))[0]
    avg_power_not_powerline = sum(abs(yf[indexes_out_of_powerline_dc_shift_sleep_EEG])**2)/len(indexes_out_of_powerline_dc_shift_sleep_EEG)
    
    return avg_power_near_powerline, avg_power_not_powerline
