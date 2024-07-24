#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:03:17 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved


this will gothrough the continious segments and annotate the whole segment as artifact based on the persentage of artifcat presesnce in the 
continious segment
"""
from copy import deepcopy
import numpy as np

def cont_EEG_segments_np_len(cont_EEG_segments_np, o_p_adjuster=3, ep_length=30):
    '''
    this will calcultate the length of each continiuos segments
    obtained based on the strat_idx
     Purpose since the preprocessed one will have an array of preprocessed and MT-extimation w/o kmowing the breaking leads to unintended outcome
        this will extract the sleep-pre-processed-fragment (SPPF): the fragmnet term may confuce so we use this term to avoid the ambiguity

    
    len_cont_seg: length of the preprocessed continious segments
    cont_seg_start_end_index: the start and end point of the prprocessed continius segments
    '''
    len_cont_seg = np.zeros(np.size(cont_EEG_segments_np,axis=0),dtype=int)
    cont_seg_start_end_index =  np.zeros((np.size(cont_EEG_segments_np,axis=0),2),dtype=int)
    s_p =0
    for cn in range(0,np.size(cont_EEG_segments_np,axis=0)):
        # --------------------------------------------------------------------------
        # length of continious segmentations is calaulcted based on the starting postion and end position
        # the index difference shows howmany epochee fell in the continious segment
        #   and the final epoch is adjusted by o_p_adjuster
        # the o_p_adjuster just adapt the space created due to the MT-spectrum window en
        # --------------------------------------------------------------------------
        len_cont_seg[cn] =ep_length*(cont_EEG_segments_np[cn][1]-cont_EEG_segments_np[cn][0])-o_p_adjuster

        # --------------------------------------------------------------------------
        # Now re-assign the index or length via the 
        # newly assigned length from the len_cont_seg
        # as newly assigned length + s_p going to be the next contnious segment's starting point
        # --------------------------------------------------------------------------

        cont_seg_start_end_index[cn,0]=s_p
        s_p = s_p + len_cont_seg[cn]
        cont_seg_start_end_index[cn,1]=s_p
    return len_cont_seg, cont_seg_start_end_index

