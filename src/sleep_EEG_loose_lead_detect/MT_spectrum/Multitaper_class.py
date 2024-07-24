#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:38:17 2021 

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved
    sp_len : the signal used to calcualte the spectrum in seconds
    ov_len :  while moving the window to create the spectrogram choose howmuch seconds 
    sover_all_length : full signal length, if it is one epoch then 30 sec 
    y : EEG signal for one epoch 
    
    ov_len = 0.5 #considering 0.5Sec overlap with the previous samples, if 256 sampling rate 128 samples from the previous window
    average_bands=[0.5,2,4,8,12,15,20,25]

    Editted on Thu Jun  2 17:44:34 2022
Include the optima bandwidth selection criteria for MT- tapers selection


Modified on Thu Nov 10 08:33:53 2022 to accomadate the logger info, etc...

"""
import matplotlib.pyplot as plt
import logging
import os
# import tensorflow as tf
import pickle
#import pickle5 as pickle


from spectrum import pmtm,dpss#*
import numpy as np
from scipy.fft import fft
from copy import deepcopy
'''
to include the depreciated function to satify the spectral estimation
'''
from scipy.signal._signaltools import _centered
import  scipy.signal.signaltools
scipy.signal.signaltools._centered = _centered


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("MT_spectrum")
while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def taper_eigen_extractor(T=4,Fs=256, f_max_interst=0, bw=2,verbose=False):
    '''
    T: in seconds to choose the window

    N (int) – desired window length
    L (int) – returns the first L Slepian sequences.
    
    The default values of this taper in chosen from somewhat near value based on the window size (4 x 256 =1024 ~ 1000)
    Such a way the Time half bandwth is near to 15
    
    And this 4 sec window with 2Hz badwidth is chosen based on the paper
        EEG also show the sleep data characterstics as mentioned in the paper
    
    A Review of Multitaper Spectral Analysis
        Behtash Babadi∗, Member, IEEE, and Emery N. Brown, Fellow, IEEE
    
    '''

    # --------------------------------------------------------------------------
    # when the f_max interest is not chosen this is automatically the maximum vlaue of Niquist samopling frequency
    # --------------------------------------------------------------------------

    if f_max_interst==0:
        f_max_interst=int(Fs/2)
    
    # --------------------------------------------------------------------------
    # d_f: frequency resolution
    # --------------------------------------------------------------------------
    d_f = 1/T
    
    # --------------------------------------------------------------------------
    # window length used to calculate the spectrum
    # --------------------------------------------------------------------------
    N = int(Fs*T)
    # --------------------------------------------------------------------------
    # The multitapered method
    # --------------------------------------------------------------------------

    TW= bw*T/2
    L= int(2*TW-1)
    [tapers, eigen] = dpss(N, TW, L)
    if verbose:
        logger.info("number of samples: %i",N)
        logger.info('TW calculataed based on the Time half bandwidth: %f',TW)
        logger.info('number of tapers going to be %i',  L)

    return tapers, eigen, d_f, N



def taper_eigen_extractor_optim_bandwidth(T,Fs, f_max_interst=0,TW=4,TW_opt_default=True,verbose=False):
    '''
    Since the value sletion for taper extractor impact on the spectrum thus these values may chosen with right idea
    Please check the paper before you choose your default values, since this may cause unotended outputs in the spectrum
       https://www.osti.gov/pages/servlets/purl/1402465
       
       Optimal Bandwidth for Multitaper Spectrum Estimation
                Charlotte L. Haley, Member, IEEE, and Mihai Anitescu, Member, IEEE
    
    TW: Time  bandwidth product

    N (int) – desired window length
    L (int) – returns the first L Slepian sequences.
    
    The default values of this taper in chosen from somewhat near value based on the window size (4 x 256 =1024 ~ 1000)
    Such a way the Time half bandwth is near to 15
    
    And this 4 sec window with 2Hz badwidth is chosen based on the paper
        EEG also show the sleep data characterstics as mentioned in the paper

    '''
    # --------------------------------------------------------------------------
    # when the f_max interest is not chosen this is automatically the maximum value of Niquist sampling frequency
    # --------------------------------------------------------------------------

    if f_max_interst==0:
        f_max_interst=int(Fs/2)
    # --------------------------------------------------------------------------
    #    d_f: frequency resolution
    # --------------------------------------------------------------------------

    d_f = 1/T

    # --------------------------------------------------------------------------
    # window length used to calculate the spectrum
    # --------------------------------------------------------------------------
    N = int(Fs*T)

    if TW_opt_default:
        logging.warning("these default value not gurantee optimal solution just a rough calculation to find the near optimal spectral estimation")
        # --------------------------------------------------------------------------
        # following values are roughly calaculated based on the Fig.2 of the paper
        # --------------------------------------------------------------------------
        if 100<=N<=500:
            TW= N/50
        elif 500<N<=1500:
            TW= ((7*N)/500)+1
        elif 1500<N:
            logging.warning("Not derived from the paper; just using the rough calculation of in range 500-1500 assume NW is lie on that line")
            TW= ((7*N)/500)+1
        else:
            raise("The N should be greater than or equal to 100")
    # --------------------------------------------------------------------------
    # The multitapered method
    # TW= bw*T/2
    # --------------------------------------------------------------------------

    L= int(2*TW-1)
    [tapers, eigen] = dpss(N, TW, L)

    if verbose:
        logger.info("number of samples: %i",N)
        logger.info('Time half bandwidth: %f',TW)
        logger.info('Band width calculated based on the Time half bandwidth: %f',TW*2/T)
        logger.info('Number of tapers going to be %i',  L)
    

    return tapers, eigen, d_f, N

def overlap_window_1sec_fixed_slide_spectrogram_given_freq_res(N,c,tapers,eigen,d_f,Fs, extracted_spectrums_of_tapers=False, f_maximum_inters=30,method='unity',normalisation_full=True):
    '''
    To create the overlapping window
    d_f: spectral resolution in (Hz)
    c: raw time domian signal need to extract the frequency features 
    f_maximum_inters: 30Hz this is a good choice for sleep data
       
    method= 'eigen' 
    method= 'unity'
    extracted_spectrums_of_tapers: If we want the spectrums as it is from the tapers so we can use deep learning to weight each spectrums 
    and frequency accordingly
    '''
    xf = np.linspace(0.0, int((Fs/2)/d_f), int((N//2)/d_f))
    spectrogram_col=np.zeros((N//2,int((len(c)-N)/Fs)+1))
    if extracted_spectrums_of_tapers:
        spectrogram_col=np.zeros((int((len(c)-N)/Fs),len(eigen),N//2))
    
    dt=1/Fs
    a=0
    t=[]
    for i in range(0,len(c)+1-N,Fs):
        y=c[i:i+N]

        Sk_complex, weights, eigenvalues=pmtm(y, e=eigen, v=tapers, NFFT=N, show=False,method=method)
        Sk = abs(Sk_complex)**2
        if extracted_spectrums_of_tapers:
            if normalisation_full:
                spectrogram_col[a,:,:] = deepcopy(Sk[:,0:N//2])* dt
            else:
                spectrogram_col[a,:,:] = deepcopy(Sk[:,0:N//2])

        else:
            # --------------------------------------------------------------------------
            # adapted from the library https://pyspectrum.readthedocs.io/en/latest/_modules/spectrum/mtm.html on Feb-10-2022 at 15.22p.m
            # --------------------------------------------------------------------------
            if method == "adapt":
                Sk = Sk.transpose()
                Sk = np.mean(Sk * weights, axis=1)
            else:
                Sk = np.mean(Sk * weights, axis=0)
                           
            newpsd  = Sk[0:N//2] * 2
            if normalisation_full:
                spectrogram_col[:,a] = deepcopy(newpsd)*dt
            else:
                spectrogram_col[:,a] = deepcopy(newpsd)
        a=a+1
        t.append(a)
        
    return spectrogram_col, t, xf



def build_for_comp_purpose_spectrogram_given_freq_res(N,c,tapers,eigen,d_f,Fs, extracted_spectrums_of_tapers=False, f_maximum_inters=30,method='adapt'):
    '''
    To create the overlapping window
    d_f: spectral resolution in (Hz)
    c: raw time domian signal need to extract the frequency features 
    f_maximum_inters: 30Hz this is a good choice for sleep data

    extracted_spectrums_of_tapers: If we want the spectrums as it is from the tapers so we can use deep learning to weight each spectrums and frequency accordingly
    '''
    xf = np.linspace(0.0, int((Fs/2)/d_f), int((N//2)/d_f))
    spectrogram_col=np.zeros((N//2,int((len(c)-N)/Fs)))

    dt=1/Fs
    a=0
    y=c
    Sk_complex, weights, eigenvalues=pmtm(y, e=eigen, v=tapers, NFFT=N, show=False,method=method)
    Sk = abs(Sk_complex)**2
    if extracted_spectrums_of_tapers:
        spectrogram_col[a,:,:] = deepcopy(Sk[:,0:N//2])* dt
    else:
        # --------------------------------------------------------------------------
        # adapted from the library https://pyspectrum.readthedocs.io/en/latest/_modules/spectrum/mtm.html on Feb-10-2022 at 15.22p.m
        # --------------------------------------------------------------------------
        if method == "adapt":
            Sk = Sk.transpose()
            Sk = np.mean(Sk * weights, axis=1)
        else:
            Sk = np.mean(Sk * weights, axis=0)
                       
        newpsd  = (Sk[0:N//2] * 2)*dt
        
    return newpsd, xf


