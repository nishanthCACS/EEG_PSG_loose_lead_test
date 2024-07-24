#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:00:11 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved
modifeied on Tue Oct 31 17:44:18 2023 to accomodate Correlation combination in onbe plot in adaptive plot function

"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from copy import deepcopy

from sleep_EEG_loose_lead_detect.loose_lead_events.unify_outliers_to_loose_lead import unify_outliers_via_conv

logger = logging.getLogger("plotting_outliers")
while logger.handlers:
     logger.handlers.pop()
     
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)
# Set logging level to the logger
# logger.setLevel(logging.DEBUG) # <-- THIS!
logger.setLevel(logging.INFO)
logger.propagate = False

# def plot_single_mean_coorelation_with_MT_flex(in_name_temp, arousal_annot_all,
#                                          correlation_flatten, sleep_stages_annot_flatten, spectrogram_col_p_all_db,
#                                          cross_correlation_ref_dic,
#                                          grp='C3',    ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'],
#                                               markersize=[0.5,0.1], f_min_interst=0.5,  f_max_interst=32.5,
#                                               vmin=-15,  vmax=15, save_fig=False, saving_fig_dirs='',
#                                               special_save='raw_corr_',comm= 'outliers',comm2='sus-loose-lead',    ss_label =    'Correlation-coefficient ',
#                                            with_conv=True,
#                                               outlier_presene_con_lenth_th=4,
#                                               thresh_min_conv=30, thresh_in_sec= True, conv_type = "same",
#                                               with_fill_period=False,  len_period_tol_min=5,show_single_outliers_before_combine_tol=True,
#                                               figsize=(12,8.5)  ,  
#                                               gridspec_dic={'ss':10,'ss2':1,'s2':1,'s1':10,'mt1':10},
#                                               index_plots={'ss': 1,'ss2':0,'s1':2,'mt1':3,'s2':4},
#                                               plot_on_por ={'ss2':True,'s1':True,'mt1':True,'s2':True}):
#     '''
#     check_continious_present by s2
#         figsize=(12,8.5)
#         gridspec=[1,10,1,10,1]   
#     else
#         figsize=(12,8)  
#         gridspec=[1,10,1,10]   

#     '''
    
#     plot_on_por['ss']=True
#     axiss_count=0
#     for axises  in list(plot_on_por.keys()):
#         if plot_on_por[axises]:
#             axiss_count+=1
#     gridspec=np.zeros(axiss_count)

#     for axises  in list(plot_on_por.keys()):
#         if plot_on_por[axises]:
#             gridspec[index_plots[axises]] = gridspec_dic[axises]
#     ch1=cross_correlation_ref_dic[grp]
#     arousal_annot= arousal_annot_all[:,ch1]
#     #converting the sleep-stages to NREM and REM
#     sleep_stages_annot_flatten_NREM=np.where(sleep_stages_annot_flatten>3,sleep_stages_annot_flatten,1)
#     sleep_stages_annot_flatten_NREM=np.where(sleep_stages_annot_flatten_NREM<2,sleep_stages_annot_flatten_NREM,0)
    
#     sleep_stages_annot_flatten_REM=np.where(sleep_stages_annot_flatten==4,sleep_stages_annot_flatten,0)
#     sleep_stages_annot_flatten_REM=np.where(sleep_stages_annot_flatten_REM==0,sleep_stages_annot_flatten_REM,1)
#     # print(np.max(sleep_stages_annot_flatten_REM))

#     sleep_stages_annot_flatten=sleep_stages_annot_flatten/5
    
    
#     tt = np.arange(len(sleep_stages_annot_flatten))/3600

#     fig = plt.figure(figsize=figsize)  
#     gs = fig.add_gridspec(len(gridspec), 1, height_ratios=gridspec)   
    
#     ax_ss = fig.add_subplot(gs[index_plots['ss']])
    
    
#     ax_ss.yaxis.grid(True)
#     ax_ss.set_ylim([-0.2,1.5])
#     ax_ss.set_ylabel(ss_label)      
#     indx_non_zero = np.where(arousal_annot!=0)[0]
    

    
#     # ax_ss.plot(tt[indx_non_zero],correlation_flatten[indx_non_zero,ch1],'x',markersize=markersize[1],color='r',label='F-corr-outliers')
#     if plot_on_por['ss2']:
#         ax_ss2 = fig.add_subplot(gs[index_plots['ss2']],sharex=ax_ss)
#         ax_ss2.fill_between(tt, sleep_stages_annot_flatten_NREM, linewidth=markersize[0],color='w',alpha=1,edgecolor='black', hatch="////",label='NREM')
#         ax_ss2.fill_between(tt, sleep_stages_annot_flatten_REM, linewidth=markersize[0],color='k',alpha=1,label='REM')
#         ax_ss2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left',ncol=2, borderaxespad=0)
    
#     markers=['.','x']
#     ax_ss.plot(tt,correlation_flatten[:,ch1],markers[0],markersize=markersize[1],color='b')
#     ax_ss.plot(tt[indx_non_zero],correlation_flatten[indx_non_zero,ch1],'x',markersize=markersize[1],color='r',label='F-corr-outliers')


    
#     ax_ss_st= ax_ss.twinx()
#     ax_ss_st.step(tt, sleep_stages_annot_flatten, color='k',linewidth=markersize[0])
#     ax_ss_st.set_ylim([-0.2,1.5])
#     sleep_bound =list(np.array([0,1,2,3,4,5])/4)
#     ax_ss_st.set_yticks(sleep_bound)
#     ax_ss_st.set_yticklabels(['N3', 'N2', 'N1', 'R', 'W','U'])
#     ax_ss_st.set_ylabel('sleep-stages ')            

#     #annoatte the loose-leads
#     if plot_on_por['s1']:
    
#         ax_s1 = fig.add_subplot(gs[index_plots['s1']],sharex=ax_ss)
#         ax_s1.fill_between(tt, arousal_annot,color='k')
    
#         ax_s1.yaxis.grid(True)
#         ax_s1.set_ylim([0.05,1.5])
#         ax_s1.set_yticks([1])
#         ax_s1.set_yticklabels([comm])
#         ax_s1.set_xlim([0,tt[-1]])

#     if plot_on_por['mt1']:
    
#         #plot the MT-spectrum
#         ax_mt1 = fig.add_subplot(gs[index_plots['mt1']],sharex=ax_ss)
#         ax_mt1.imshow(spectrogram_col_p_all_db[ch1,:,:], cmap='jet', origin='lower', extent=[0,tt[-1],f_min_interst,f_max_interst], interpolation='none',aspect='auto',vmin=vmin,vmax=vmax)
        
#         ax_mt1.set_ylabel('Frequency/ Hz')
#         ax_mt1.set_xlabel('time (hour)')
#         ax_mt1.set_title(ch_names[ch1]+' MT-spectrum')
    
#     if plot_on_por['s2']:
#         loose_lead_annot = unify_outliers_via_conv(arousal_annot, with_conv=with_conv,
#                              outlier_presene_con_lenth_th=outlier_presene_con_lenth_th, thresh_min_conv=thresh_min_conv, 
#                              thresh_in_sec= thresh_in_sec, conv_type =conv_type,
#                                                           with_fill_period=with_fill_period,  len_period_tol_min=len_period_tol_min,
#                                                           show_single_outliers_before_combine_tol=show_single_outliers_before_combine_tol)
#         ax_s2 = fig.add_subplot(gs[index_plots['s2']],sharex=ax_ss)
#         ax_s2.fill_between(tt, loose_lead_annot,color='k')
    
#         ax_s2.yaxis.grid(True)
#         ax_s2.set_ylim([0.05,1.5])
#         ax_s2.set_yticks([1])
#         ax_s2.set_yticklabels([comm2])
#         ax_s2.set_xlim([0,tt[-1]])

#     fig_name=ch_names[ch1]+'_corr_mean_'+special_save+in_name_temp+'.png'

#     if save_fig:
#         os.chdir('/') 
#         os.chdir(saving_fig_dirs)    
#         plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.05)
#     return fig_name


def plot_single_with_MT_flex(in_name_temp, arousal_annot_all,
                                          correlation_flatten, sleep_stages_annot_flatten, spectrogram_col_p_all_db,
                                          cross_correlation_ref_dic,
                                          grp='C3',    ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'],
                                              markersize=[0.5,0.1], f_min_interst=0.5,  f_max_interst=32.5,
                                              vmin=-15,  vmax=15, save_fig=False, saving_fig_dirs='',
                                              special_save='raw_corr_',comm= 'outliers',comm2='sus-loose-lead',  
                                              ss_label =    'Correlation-coefficient ',
                                            with_conv=True,special_MT_legend=False,
                                              outlier_presene_con_lenth_th=4,
                                              thresh_min_conv=30, thresh_in_sec= True, conv_type = "same",
                                              with_fill_period=False,  len_period_tol_min=5,show_single_outliers_before_combine_tol=True,
                                              figsize=(12,8.5)  ,  
                                              gridspec_dic={'ss':10,'ss2':1,'s2':1,'s1':10,'mt1':10},
                                              index_plots={'ss': 1,'ss2':0,'s1':2,'mt1':3,'s2':4},
                                              plot_on_por ={'ss2':True,'s1':True,'ss':True,'s2':True,'cv':False}):
    '''
    check_continious_present by s2
        figsize=(12,8.5)
        gridspec=[1,10,1,10,1]   
    else
        figsize=(12,8)  
        gridspec=[1,10,1,10]   

    this funtion skip the correltations in the plot
    '''

    plot_on_por['mt1']=True

            
    axiss_count=0
    for axises  in list(plot_on_por.keys()):
        if plot_on_por[axises]:
            axiss_count+=1
            # print(axises)
    gridspec=np.zeros(axiss_count)
    # print('axiss_count: ',axiss_count)
    for axises  in list(plot_on_por.keys()):
        if plot_on_por[axises]:
            # print('index_plots[axises]:',index_plots[axises])
            gridspec[index_plots[axises]] = gridspec_dic[axises]
    ch1=cross_correlation_ref_dic[grp]
    arousal_annot= arousal_annot_all[:,ch1]
    #converting the sleep-stages to NREM and REM
    sleep_stages_annot_flatten_NREM=np.where(sleep_stages_annot_flatten>3,sleep_stages_annot_flatten,1)
    sleep_stages_annot_flatten_NREM=np.where(sleep_stages_annot_flatten_NREM<2,sleep_stages_annot_flatten_NREM,0)
    
    sleep_stages_annot_flatten_REM=np.where(sleep_stages_annot_flatten==4,sleep_stages_annot_flatten,0)
    sleep_stages_annot_flatten_REM=np.where(sleep_stages_annot_flatten_REM==0,sleep_stages_annot_flatten_REM,1)
    # print(np.max(sleep_stages_annot_flatten_REM))

    sleep_stages_annot_flatten=sleep_stages_annot_flatten/5
    
    
    tt = np.arange(len(sleep_stages_annot_flatten))/3600

    fig = plt.figure(figsize=figsize)  
    gs = fig.add_gridspec(len(gridspec), 1, height_ratios=gridspec)   
    


    #plot the MT-spectrum
    ax_mt1 = fig.add_subplot(gs[index_plots['mt1']])
    ax_mt1.imshow(spectrogram_col_p_all_db[ch1,:,:], cmap='jet', origin='lower', extent=[0,tt[-1],f_min_interst,f_max_interst], interpolation='none',aspect='auto',vmin=vmin,vmax=vmax)
    
    ax_mt1.set_ylabel('Frequency/ Hz')
    
    
    if special_MT_legend:
           ax_mt1.set_ylabel('Frequency (Hz)')
           special_mt = [0.25,(f_max_interst*3/4),40,'white']
           ax_mt1.text(special_mt[0],special_mt[1],ch_names[ch1],fontsize=special_mt[2],color=special_mt[3],weight='bold')

    else:
        ax_mt1.set_ylabel('Channel-'+ch_names[ch1]+'\n MT-spectrum \n'+'Frequency/ Hz')
    ax_mt1.set_xlabel('time (hour)')
   
    if plot_on_por['s2']:
        loose_lead_annot = unify_outliers_via_conv(arousal_annot, with_conv=with_conv,
                              outlier_presene_con_lenth_th=outlier_presene_con_lenth_th, thresh_min_conv=thresh_min_conv, 
                              thresh_in_sec= thresh_in_sec, conv_type =conv_type,
                                                          with_fill_period=with_fill_period,  len_period_tol_min=len_period_tol_min,
                                                          show_single_outliers_before_combine_tol=show_single_outliers_before_combine_tol)
        ax_s2 = fig.add_subplot(gs[index_plots['s2']],sharex=ax_mt1)
        ax_s2.fill_between(tt, loose_lead_annot,color='k')
    
        ax_s2.yaxis.grid(True)
        ax_s2.set_ylim([0.05,1.5])
        ax_s2.set_yticks([1])
        ax_s2.set_yticklabels([comm2])
        ax_s2.set_xlim([0,tt[-1]])



    if plot_on_por['ss']:

        ax_ss = fig.add_subplot(gs[index_plots['ss']],sharex=ax_mt1)
         
        ax_ss.step(tt, sleep_stages_annot_flatten, color='k',linewidth=markersize[0])
        ax_ss.set_ylim([-0.2,1.5])
        sleep_bound =list(np.array([0,1,2,3,4,5])/4)
        ax_ss.set_yticks(sleep_bound)
        ax_ss.set_yticklabels(['N3', 'N2', 'N1', 'R', 'W','U'])
        ax_ss.set_ylabel('sleep-stages ')            
    
        ax_ss.yaxis.grid(True)
        
    # if plot_on_por['ss'] and plot_on_por['ss_corr']:
    #     ax_ss_corr= ax_ss.twinx()
    #     ax_ss_corr.yaxis.grid(True)
    #     ax_ss_corr.set_ylabel(ss_label)          
    #     markers=['.','x']
    #     ax_ss_corr.plot(tt,correlation_flatten,markers[0],markersize=markersize[1],color='b')
    
    if plot_on_por['ss2']:
        ax_ss2 = fig.add_subplot(gs[index_plots['ss2']],sharex=ax_mt1)
        ax_ss2.fill_between(tt, sleep_stages_annot_flatten_NREM, linewidth=markersize[0],color='w',alpha=1,edgecolor='black', hatch="////",label='NREM')
        ax_ss2.fill_between(tt, sleep_stages_annot_flatten_REM, linewidth=markersize[0],color='k',alpha=1,label='REM')
        ax_ss2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left',ncol=2, borderaxespad=0)

    #annoatte the loose-leads
    if plot_on_por['s1']:
    
        ax_s1 = fig.add_subplot(gs[index_plots['s1']],sharex=ax_mt1)
        ax_s1.fill_between(tt, arousal_annot,color='k')
    
        ax_s1.yaxis.grid(True)
        ax_s1.set_ylim([0.05,1.5])
        ax_s1.set_yticks([1])
        ax_s1.set_yticklabels([comm])
        ax_s1.set_xlim([0,tt[-1]])

    if plot_on_por['cv']:

        ax_cv = fig.add_subplot(gs[index_plots['cv']],sharex=ax_mt1)

        if thresh_in_sec:
            conv_window=thresh_min_conv
        else:
            conv_window = int(np.ceil(2*30*thresh_min_conv)) # here the 2 come from 60/30; 27 time instance in MT-spectrum
            
        if outlier_presene_con_lenth_th>=conv_window:
            raise("this should be low")
    
        sel_ch_arousl_annot = np.convolve(arousal_annot,np.ones(int(conv_window)), conv_type)       
        ax_cv.fill_between(tt, sel_ch_arousl_annot,color='k',alpha=1,linewidth=markersize[0])
        ax_cv.set_xlim([0,tt[-1]])
        # ax_cv.set_yticks([1])
        # ax_cv.set_yticklabels(['sus-artifact'])
        ax_cv.set_ylabel('conv o/p')

    fig_name=ch_names[ch1]+'_corr_mean_'+special_save+in_name_temp+'.png'

    if save_fig:
        os.chdir('/') 
        os.chdir(saving_fig_dirs)    
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.05)
    return fig_name


def plot_single_with_MT_flex_adapt(min_hour,max_hour,in_name_temp, arousal_annot_all,
                                         correlation_flatten, sleep_stages_annot_flatten, spectrogram_col_p_all_db,
                                         cross_correlation_ref_dic,color_hl='r', corr_axis_givenn_range=[],
                                         grp='C3',grp2='O1', ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'],
                                        markersize=[0.5,0.1], f_min_interst=0.5,  f_max_interst=32.5,
                                        vmin=-15,  vmax=15, save_fig=False, saving_fig_dirs='',full_sleep=False,
                                        special_save='raw_corr_',comm= 'outliers',comm2='sus-loose-lead',  
                                        special_annot_comm3=False,special_MT_legend=False,
                                        comm3='sus-loose-lead ', comm_ex='expert \n annotation',
                                        s4_comm='', 
                                        ss_label =    'Correlation-coefficient ',special_ss_legend=False,ss_corr_text='',
                                        special_corr = [0.25,1,40,'black'],
                                        with_conv=True,
                                        outlier_presene_con_lenth_th=4,
                                        plot_outlier=True, plot_legend_corr=True,
                                        thresh_min_conv=30, thresh_in_sec= True, conv_type = "same",legend_upper_NREM=True,
                                        with_fill_period=False,  len_period_tol_min=5,show_single_outliers_before_combine_tol=True,
                                        arousal_annot_ex=[],
                                        figsize=(12,8.5)  ,  
                                        gridspec_dic={'ss':10,'ss2':1,'s2':1,'s1':10,'mt1':10,'mt2':10,'s3':1,'ex_annot':1},plot_not_avoid_sl_st=True,
                                        index_plots={'ss': 1,'ss2':0,'s1':2,'mt1':3,'mt2':3,'s2':4,'ex_annot':5},
                                        plot_on_por ={'ss2':True,'s1':True,'ss':True,'ss_corr':False,'s2':True,'cv':False,'mt1':True,'s3':False,'ex_annot':False, 'mt2':False},
                                        plot_on_def ={'ss2':True,'s1':True,'ss':True,'ss_corr':False,'s2':True,'cv':False,'mt1':True,'s3':False,'ex_annot':False, 'mt2':False}):
    '''
    check_continious_present by s2
        figsize=(12,8.5)
        gridspec=[1,10,1,10,1]   
    else
        figsize=(12,8)  
        gridspec=[1,10,1,10]   

    this funtion skip the correltations in the plot
    this function will help to plot the outcomes
    '''
    
    axiss_count=0
    for axises  in list(plot_on_por.keys()):
        if plot_on_por[axises]:
            axiss_count+=1

    gridspec=np.zeros(axiss_count)
    for keyi in list(plot_on_def.keys()):
        if keyi not in list(plot_on_por.keys()):
            plot_on_por[keyi]=False
            print(keyi, 'is forcefully made as False')
            
    for axises  in list(plot_on_por.keys()):
        if plot_on_por[axises]:
            if axises!='ss_corr':
                # print(axises)
                gridspec[index_plots[axises]] = gridspec_dic[axises]
            
            
    ch1=cross_correlation_ref_dic[grp]
    #for correlation combination second group for MT_spectrum 2
    ch2=cross_correlation_ref_dic[grp2]

    arousal_annot= arousal_annot_all[:,ch1]
    #converting the sleep-stages to NREM and REM
    sleep_stages_annot_flatten_NREM=np.where(sleep_stages_annot_flatten>3,sleep_stages_annot_flatten,1)
    sleep_stages_annot_flatten_NREM=np.where(sleep_stages_annot_flatten_NREM<2,sleep_stages_annot_flatten_NREM,0)
    
    sleep_stages_annot_flatten_REM=np.where(sleep_stages_annot_flatten==4,sleep_stages_annot_flatten,0)
    sleep_stages_annot_flatten_REM=np.where(sleep_stages_annot_flatten_REM==0,sleep_stages_annot_flatten_REM,1)

    sleep_stages_annot_flatten=sleep_stages_annot_flatten/5
    
    
    tt = np.arange(len(sleep_stages_annot_flatten))/3600
    
    '''
    converting sleep-time to minutes format in selected range
    '''
    if not full_sleep:
        tt_sel= np.where(np.logical_and(tt>=min_hour, tt<=max_hour))
        if len(tt_sel)==1:
            tt_sel=tt_sel[0]
    else:
        tt_sel=[x for x in range(0,len(tt))]
    tt_sel_range = tt[tt_sel]
 
    fig = plt.figure(figsize=figsize)  
    gs = fig.add_gridspec(len(gridspec), 1, height_ratios=gridspec)   
    used_axces=[]
    default_axis_sel=False
    if plot_on_por['mt1']:
    
    
        #plot the MT-spectrum
        ax_mt1 = fig.add_subplot(gs[index_plots['mt1']])

        # ax_mt1.imshow(spectrogram_col_p_all_db[ch1,:,tt_sel][0,:,:].T, cmap='jet', origin='lower', extent=[tt_sel_range[0],tt_sel_range[-1],f_min_interst,f_max_interst], interpolation='none',aspect='auto',vmin=vmin,vmax=vmax)

        ax_mt1.imshow(spectrogram_col_p_all_db[ch1,:,tt_sel].T, cmap='jet', origin='lower', extent=[tt_sel_range[0],tt_sel_range[-1],f_min_interst,f_max_interst], interpolation='none',aspect='auto',vmin=vmin,vmax=vmax)
        if special_MT_legend:
            ax_mt1.set_ylabel('Frequency (Hz)')
            special_mt = [0.25,(f_max_interst*3/4),40,'white']
            ax_mt1.text(special_mt[0],special_mt[1],ch_names[ch1],fontsize=special_mt[2],color=special_mt[3],weight='bold')

        else:
            ax_mt1.set_ylabel('Channel-'+ch_names[ch1]+'\n MT-spectrum \n'+'Frequency/ Hz')
        ax_mt1.set_xlabel('time (hour)')
        # ax_mt1.set_title(ch_names[ch1]+' MT-spectrum')
        default_axis_sel=True
        default_axis = ax_mt1
        used_axces.append(ax_mt1)
        ax_mt1.set_xlim([tt_sel_range[0],tt_sel_range[-1]])

    if plot_on_por['mt2']:
    
    
        #plot the MT-spectrum
        ax_mt2 = fig.add_subplot(gs[index_plots['mt2']])

        # ax_mt1.imshow(spectrogram_col_p_all_db[ch1,:,tt_sel][0,:,:].T, cmap='jet', origin='lower', extent=[tt_sel_range[0],tt_sel_range[-1],f_min_interst,f_max_interst], interpolation='none',aspect='auto',vmin=vmin,vmax=vmax)

        ax_mt2.imshow(spectrogram_col_p_all_db[ch2,:,tt_sel].T, cmap='jet', origin='lower', extent=[tt_sel_range[0],tt_sel_range[-1],f_min_interst,f_max_interst], interpolation='none',aspect='auto',vmin=vmin,vmax=vmax)
        if special_MT_legend:
            ax_mt2.set_ylabel('Frequency (Hz)')
            ax_mt2.text(special_mt[0],special_mt[1],ch_names[ch2],fontsize=special_mt[2],color=special_mt[3],weight='bold')       
        else:
            ax_mt2.set_ylabel('Channel-'+ch_names[ch2]+'\n MT-spectrum \n'+'Frequency/ Hz')
        ax_mt2.set_xlabel('time (hour)')
        # ax_mt1.set_title(ch_names[ch1]+' MT-spectrum')
        default_axis_sel=True
        default_axis = ax_mt2
        used_axces.append(ax_mt2)
        ax_mt2.set_xlim([tt_sel_range[0],tt_sel_range[-1]])


    if plot_on_por['s2']:
        loose_lead_annot = unify_outliers_via_conv(arousal_annot, with_conv=with_conv,
                             outlier_presene_con_lenth_th=outlier_presene_con_lenth_th, thresh_min_conv=thresh_min_conv, 
                             thresh_in_sec= thresh_in_sec, conv_type =conv_type,
                                                          with_fill_period=with_fill_period,  len_period_tol_min=len_period_tol_min,
                                                          show_single_outliers_before_combine_tol=show_single_outliers_before_combine_tol)
        if default_axis_sel:
            ax_s2 = fig.add_subplot(gs[index_plots['s2']],sharex=ax_mt1)
        else:
            ax_s2 = fig.add_subplot(gs[index_plots['s2']])
            default_axis = ax_s2
            default_axis_sel=True

        ax_s2.fill_between(tt_sel_range, loose_lead_annot[tt_sel],color='k')
    
        ax_s2.yaxis.grid(True)
        ax_s2.set_ylim([0.05,1.5])
        ax_s2.set_yticks([1])
        ax_s2.set_yticklabels([comm2])
        ax_s2.set_xlim([tt_sel_range[0],tt_sel_range[-1]])
        used_axces.append(ax_s2)



    if plot_on_por['ss']:
        if default_axis_sel:
            ax_ss = fig.add_subplot(gs[index_plots['ss']],sharex=default_axis)
        else:
            ax_ss = fig.add_subplot(gs[index_plots['ss']])
            default_axis_sel=True
            default_axis = ax_ss
        

        if plot_on_por['ss_corr']:
            if plot_not_avoid_sl_st:
                ax_ss_corr= ax_ss.twinx()
            else:
                ax_ss_corr=ax_ss
                
  
               
            # ax_ss_corr.yaxis.grid(True)
            if special_ss_legend:
                ax_ss_corr.set_ylabel(ss_label)          
                
                
                ax_ss_corr.text(special_corr[0],special_corr[1],ss_corr_text,fontsize=special_corr[2],color=special_corr[3],weight='bold')
                

            else:
                ax_ss_corr.set_ylabel(ss_label)          
            markers=['.','x']
            if plot_outlier:
                ax_ss_corr.plot(tt_sel_range,correlation_flatten[tt_sel],markers[0],markersize=markersize[1],color='b',label='good-correlation')
            else:
                ax_ss_corr.plot(tt_sel_range,correlation_flatten[tt_sel],markers[0],markersize=markersize[1],color='b',label='correlation')


            ax_ss_corr.set_xlim([tt_sel_range[0],tt_sel_range[-1]])
            used_axces.append(ax_ss_corr)
            
            indx_non_zero = np.where(arousal_annot!=0)[0]
            
            indx_non_zero_sel=[]
            for x in  list(indx_non_zero):
                # if x in list(tt_sel[0])
                if x in list(tt_sel):
                    indx_non_zero_sel.append(x)
                    
            if plot_outlier:
                ax_ss_corr.plot(tt[indx_non_zero_sel],correlation_flatten[indx_non_zero_sel],'x',markersize=markersize[1],color=color_hl,label='bad-correlation')
            ax_ss_corr.set_xlabel('time (hour)')

            # ax_ss_corr.legend(bbox_to_anchor=(0, -2.5, 1, 0.2), loc='lower left',ncol=2, borderaxespad=0)
            # ax_ss_corr.legend( loc='center left',borderaxespad=0)
            if plot_legend_corr:
                # ax_ss_corr.legend( loc='lower center',borderaxespad=0)
                ax_ss_corr.legend( loc='lower left',borderaxespad=0)

        if plot_not_avoid_sl_st:
            ax_ss.step(tt_sel_range, sleep_stages_annot_flatten[tt_sel], color='k',linewidth=markersize[0])
            ax_ss.set_ylim([-0.2,1.5])
            sleep_bound =list(np.array([0,1,2,3,4,5])/4)
            ax_ss.set_yticks(sleep_bound)
            ax_ss.set_yticklabels(['N3', 'N2', 'N1', 'R', 'W','U'])
            ax_ss.set_ylabel('sleep-stages ')            
            ax_ss.set_xlabel('time (hour)')

        ax_ss.set_ylim([-0.2,1.25])
        if len(corr_axis_givenn_range)>0:
            ax_ss_corr.set_ylim(corr_axis_givenn_range)
            if not plot_not_avoid_sl_st:    
                ax_ss.set_ylim(corr_axis_givenn_range)

        ax_ss.yaxis.grid(True)
        if not plot_on_por['ss_corr']:
            
            used_axces.append(ax_ss)
            ax_ss.set_xlim([tt_sel_range[0],tt_sel_range[-1]])
        

    # if plot_on_por['ss'] and plot_on_por['ss_corr']:
        

    if plot_on_por['ss2']:
    
        if default_axis_sel:
            ax_ss2 = fig.add_subplot(gs[index_plots['ss2']],sharex=default_axis)
        else:
            ax_ss2 = fig.add_subplot(gs[index_plots['ss2']])
            default_axis_sel=True
            default_axis = ax_ss2

        
        ax_ss2.fill_between(tt_sel_range, sleep_stages_annot_flatten_NREM[tt_sel], linewidth=markersize[0],color='w',alpha=1,edgecolor='black', hatch="////",label='NREM')
        ax_ss2.fill_between(tt_sel_range, sleep_stages_annot_flatten_REM[tt_sel], linewidth=markersize[0],color='k',alpha=1,label='REM')
        if legend_upper_NREM:
            ax_ss2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left',ncol=2, borderaxespad=0)
        else:
            ax_ss2.legend(bbox_to_anchor=(0, -2.5, 1, 0.2), loc='lower left',ncol=2, borderaxespad=0)
      

        used_axces.append(ax_ss2)

    #annoatte the loose-leads
    if plot_on_por['s1']:
        if default_axis_sel:
            ax_s1 = fig.add_subplot(gs[index_plots['s1']],sharex=default_axis)

        else:
            ax_s1 = fig.add_subplot(gs[index_plots['s1']])
            default_axis_sel=True
            default_axis = ax_s1
            
        ax_s1.fill_between(tt_sel_range, arousal_annot[tt_sel],color='k')
    
        ax_s1.yaxis.grid(True)
        ax_s1.set_ylim([0.05,1.5])
        ax_s1.set_yticks([1])
        ax_s1.set_yticklabels([comm])
        ax_s1.set_xlim([tt_sel_range[0] ,tt_sel_range[-1]])
        used_axces.append(ax_s1)
        ax_s1.set_xlabel('time (hour)')

    if plot_on_por['ex_annot']:
        ax_ex = fig.add_subplot(gs[index_plots['ex_annot']],sharex=default_axis)

        ax_ex.fill_between(tt_sel_range, arousal_annot_ex[tt_sel],color='k')
    
        ax_ex.yaxis.grid(True)
        ax_ex.set_ylim([0.05,1.5])
        ax_ex.set_yticks([1])
        ax_ex.set_yticklabels([comm_ex])
        ax_ex.set_xlim([tt_sel_range[0] ,tt_sel_range[-1]])
        used_axces.append(ax_ex)
        ax_ex.set_xlabel('time (hour)')



    if plot_on_por['cv']:

        ax_cv = fig.add_subplot(gs[index_plots['cv']],sharex=default_axis)

        if thresh_in_sec:
            conv_window=thresh_min_conv
            conv_prefix ='Unit weight \n '+str(thresh_min_conv)+' sec \n'
        else:
            conv_window = int(np.ceil(2*30*thresh_min_conv)) # here the 2 come from 60/30; 27 time instance in MT-spectrum
            conv_prefix ='Unit weight \n '+str(thresh_min_conv)+' min \n'

        if outlier_presene_con_lenth_th>conv_window:
            raise("this should be low")
    
        sel_ch_arousl_annot = np.convolve(arousal_annot,np.ones(int(conv_window)), conv_type)       
        ax_cv.fill_between(tt_sel_range, sel_ch_arousl_annot[tt_sel],color='k',alpha=1,linewidth=markersize[0])
        ax_cv.axhline(y=outlier_presene_con_lenth_th, color='r', linestyle='-')

        
        ax_cv.set_xlim([tt_sel_range[0],tt_sel_range[-1]])
            
        ax_cv.set_ylabel(conv_prefix+'conv o/p')
        used_axces.append(ax_cv)


    if plot_on_por['s3']:
        loose_lead_annot3 = unify_outliers_via_conv(arousal_annot, with_conv=with_conv,
                             outlier_presene_con_lenth_th=outlier_presene_con_lenth_th, thresh_min_conv=thresh_min_conv, 
                             thresh_in_sec= thresh_in_sec, conv_type =conv_type,
                                                          with_fill_period=True,  len_period_tol_min=len_period_tol_min,
                                                          show_single_outliers_before_combine_tol=show_single_outliers_before_combine_tol)
        ax_s3 = fig.add_subplot(gs[index_plots['s3']],sharex=default_axis)
       

        ax_s3.fill_between(tt_sel_range, loose_lead_annot3[tt_sel],color='k')
    
        ax_s3.yaxis.grid(True)
        ax_s3.set_ylim([0.05,1.5])
        ax_s3.set_yticks([1])
        if not special_annot_comm3:
            if len_period_tol_min>0:
                ax_s3.set_yticklabels([comm3+'\n'+'with '+str(len_period_tol_min)+' min fill'])
            else:
                ax_s3.set_yticklabels([comm3+'\n'+'with '+str(len_period_tol_min*60)+' sec fill'])
        else:
            ax_s3.set_yticklabels([comm3])
 
        ax_s3.set_xlim([tt_sel_range[0],tt_sel_range[-1]])

        used_axces.append(ax_s3)
        ax_s3.set_xlabel('time (hour)')


    if plot_on_por['s4']:
        loose_lead_annot4 = unify_outliers_via_conv(arousal_annot, with_conv=with_conv,
                             outlier_presene_con_lenth_th=outlier_presene_con_lenth_th, thresh_min_conv=thresh_min_conv, 
                             thresh_in_sec= thresh_in_sec, conv_type =conv_type,
                                                          with_fill_period=True,  len_period_tol_min=5,
                                                          show_single_outliers_before_combine_tol=show_single_outliers_before_combine_tol)
        ax_s4 = fig.add_subplot(gs[index_plots['s4']],sharex=default_axis)
       

        ax_s4.fill_between(tt_sel_range, loose_lead_annot4[tt_sel],color='k')
    
        ax_s4.yaxis.grid(True)
        ax_s4.set_ylim([0.05,1.5])
        ax_s4.set_yticks([1])
        ax_s4.set_yticklabels([s4_comm+' \n with 5 min fill'])
        ax_s4.set_xlim([tt_sel_range[0],tt_sel_range[-1]])
        ax_s4.set_xlabel('time (hour)')
        used_axces.append(ax_s4)

    '''
    the following portion need to be updated to support the bigger general;isation
    '''
    if not plot_on_por['ss_corr']:

        for ax in used_axces:
            if plot_on_por['mt1']:
                if ax!=ax_mt1:
                    ax.grid(True,axis = 'x')
            else:
                ax.grid(True,axis = 'x')
    
            ax.set_xlabel('time (hour)')
            # ax.set_xlabel('.')
    
            ax.label_outer()
    
    else:
        '''
        ;label_outer force the corrsponding axis label to the outer once
        '''
        if len(used_axces)==1:
            used_axces[0].set_xlabel('time (hour)')

        elif  plot_on_por['mt1']:

            ax_ss.label_outer()
            ax_ss.grid(True,axis = 'x')
            if plot_on_por['mt2']:
                ax_mt1.label_outer()
                ax_mt2.set_xlabel('time (hour)')   
    
            else:
                if  plot_on_por['mt1']:
                    ax_mt1.set_xlabel('time (hour)')   
        else:
            
            for ax in used_axces:
       
                ax.set_xlabel('time (hour)')
                # ax.set_xlabel('.')
   
            ax.label_outer()
        # ax_mt1.label_outer()
    if  plot_on_por['mt2']:
        fig_name=ch_names[ch1]+ch_names[ch2]+'_corr_comb_'+special_save+in_name_temp+'.png'
    else:
        fig_name=ch_names[ch1]+'_corr_mean_'+special_save+in_name_temp+'.png'

    if save_fig:
        os.chdir('/') 
        os.chdir(saving_fig_dirs)    
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.05)
    return fig_name

