#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:18:18 2022

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved


Modified on Mon Mar 20 08:57:32 2023
to to accomadate the coreelation coefficient pickles
"""

import os
import numpy as np
import pickle
import logging



def load_pickle(f,ext, load_corr_dir):
    '''
    to load the correlation values
    '''
    with open(load_corr_dir+f+ext+'.pickle', 'rb') as handle:
        loaded = pickle.load(handle)
    del handle
    
    return loaded


def load_data_sleep_stages(f,load_EEG_ext_loc,load_sleep_stages):
    '''
    load the segmented EEG_data and sleep stages
    '''
    # sl_stages_np = np.load(f+'_sleep_stages_cat.npy')
    os.chdir('/')
    os.chdir(load_EEG_ext_loc)
    extracted_numpy = np.load(f + "_filtered_EEG.npy")

    os.chdir('/')
    os.chdir(load_sleep_stages)
    sleep_stages = np.load(f + "_sleep_stages.npy")
    
    return sleep_stages,extracted_numpy

def load_data_sp_ini(f,load_spindle_loc,sel_id_name,load_spindle_loc_re_index,sel_id_name_rel):

    '''       
    sp_columns_ordered =  ['Start', 'End', 'Duration', 'Peak', 'Amplitude', 'RMS', 'AbsPower',
            'RelPower', 'Frequency', 'Oscillations', 'Symmetry', 'IdxChannel']
    '''
    os.chdir('/')
    os.chdir(load_spindle_loc)
    
    #first load the obtained spindle/ slowwaves with whole EEG segments
    sp_group_of_interest_por_com = np.load(f+'_sp_fea_'+sel_id_name+'.npy')
    sp_eeg_seg_interest_pos_com = np.load(f+'_sp_sl_st_info_'+sel_id_name+'.npy')   
    
    os.chdir('/')
    os.chdir(load_spindle_loc_re_index)
    sp_group_of_interest_por_com_re =  np.load(f+'_sp_fea_'+sel_id_name_rel+'.npy')
    #since the continious segments_np is from the sel_ids these groups are going to be same
    if np.array_equal(sp_group_of_interest_por_com[:,2:],sp_group_of_interest_por_com_re[:,2:]):
        return sp_eeg_seg_interest_pos_com, sp_group_of_interest_por_com_re
    else:
        raise("Issue with re-index and normal segmented groups")
        
        
def load_data_sleep_stage_origin(f,load_sleep_stages):
    '''
    load the segmented EEG_data and sleep stages
    '''

    os.chdir('/')
    os.chdir(load_sleep_stages)
    sleep_stages = np.load(f + "_sleep_stages_origin.npy")
    
    return sleep_stages
        
def load_data_sp_sw(f,load_spindle_loc,sel_id_name):

    '''
    sw_columns_ordered =  ['Start', 'End', 'MidCrossing','Duration', 'NegPeak', 'PosPeak',  'ValNegPeak', 'ValPosPeak', 
                           'PTP', 'Slope', 'Frequency',  'IdxChannel']
        
    sp_columns_ordered =  ['Start', 'End', 'Duration', 'Peak', 'Amplitude', 'RMS', 'AbsPower',
            'RelPower', 'Frequency', 'Oscillations', 'Symmetry', 'IdxChannel']
    '''
    os.chdir('/')
    os.chdir(load_spindle_loc)
    
    #first load the obtained spindle/ slowwaves with whole EEG segments
    sw_group_of_interest_por_com = np.load(f+'_sw_fea_'+sel_id_name+'.npy')
    sw_eeg_seg_interest_pos_com = np.load(f+'_sw_sl_st_info_'+sel_id_name+'.npy')
    
    sp_group_of_interest_por_com = np.load(f+'_sp_fea_'+sel_id_name+'.npy')
    sp_eeg_seg_interest_pos_com = np.load(f+'_sp_sl_st_info_'+sel_id_name+'.npy')
    
    sp_cont_EEG_segments = np.load(f+'_sp_cont_EEG_seg_info__'+sel_id_name+'.npy')
    sw_cont_EEG_segments = np.load(f+'_sw_cont_EEG_seg_info__'+sel_id_name+'.npy')
    return sw_group_of_interest_por_com, sw_eeg_seg_interest_pos_com, \
            sp_group_of_interest_por_com, sp_eeg_seg_interest_pos_com, \
            sp_cont_EEG_segments,sw_cont_EEG_segments
            
def load_MT(f,load_MT_loc,epoch_wise, load_part ='_spec_all_re_ref_chan'):
    '''
    load the MT-spectrum
    '''
    if epoch_wise:
        os.chdir('/')
        os.chdir(load_MT_loc)
        with open(f+load_part+'.pickle', 'rb') as handle:
            subject_spectrum_dic = pickle.load(handle)
        del handle
        return subject_spectrum_dic
    else:
        raise("still building for continious groups")
        
        
def load_normalised_MT_spec_sleep_stages(f,out_loc_extracted_spec_db,out_loc_sleep):
    '''
    load the spectrums and sleep stages
    '''
    os.chdir('/')
    os.chdir(out_loc_extracted_spec_db)
    extracted_numpy_flipped =  np.load(f+'_spliced_norm_db_multispectrum.npy')
    extracted_numpy =  np.load(f+'_spliced_norm_db_multispectrum_not_flipped.npy')

    # sl_stages_np = np.load(f+'_sleep_stages_cat.npy')
    os.chdir('/')
    os.chdir(out_loc_sleep)
    sleep_stages = np.load(f + "_sleep_stages.npy")
    
    return extracted_numpy_flipped,sleep_stages,extracted_numpy


