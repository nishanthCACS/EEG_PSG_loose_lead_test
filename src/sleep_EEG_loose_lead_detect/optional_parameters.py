#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:16:34 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

this script contain the direct run to directly run on the servers once assign the i/p output paramters
"""
import argparse
import logging
# Import sys module
import sys
import os

sys.path.append("src")

from sleep_EEG_loose_lead_detect.directory_utils import EEG_sleep_dir

from sleep_EEG_loose_lead_detect.GUI_interface.percentage_bar_vis  import percent_complete

# --------------------------------------------------------------------------
#logger intialisation
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("parsing_arguments")
while logger.handlers:
      logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
logger.propagate = False




class parameter_assignment:
    
    def __init__(self, GUI_percentile=True, save_events_origin = True, sleep_stage_preprocess_origin = True,
             channel_specific_preprocess=True,    
             pred_slow_waves=False,
             pred_spindles =False, T=4,
             amplitude_high_same_all_age=False,
             avoid_spindle_loc=False,
             verbose=False,
             f_min_interst=0.5,f_max_interst=32.5,
             epoch_length = 30,  line_freq = 60,        
             ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'],
             b_val=0.001,
             intersted_sleep_stages_REM=[4],
             intersted_sleep_stages_NREM=[0,1,2,3],#N1,N2,N3,N4
             intersted_sleep_stages_NREM_REM_comb=[0,1,2,3,4]):#N1,N2,N3,N4 and R):
        
        self.GUI_percentile=GUI_percentile


        # --------------------------------------------------------------------------
        # developments optional to user
        # --------------------------------------------------------------------------

        self.save_events_origin= save_events_origin
        self.sleep_stage_preprocess_origin= sleep_stage_preprocess_origin

        # --------------------------------------------------------------------------
        # channel_specific_preprocess: Bool, optional
        # to return the preprocessed events with the channel information that has changed from the normal to other status with the channel information
        
        # Even this condition is true this doen't gurantee all the other channels are good for this specific epoch 
        # That need to be varified seperately 
        
        
        # if not channel_specific_preprocess just return the epoch status without channel specific information
        # like nan value, high/lower amplitude etc.
        # --------------------------------------------------------------------------
        self.channel_specific_preprocess=channel_specific_preprocess
        self.ch_names = ch_names
        
        # --------------------------------------------------------------------------
        # intialisng whether consider the variance
        # --------------------------------------------------------------------------
        self.flat_MT_consider = False
        # --------------------------------------------------------------------------
        # spindle related functions
        # --------------------------------------------------------------------------

        self.pred_slow_waves=pred_slow_waves
        self.pred_spindles =pred_spindles 
    
        self.amplitude_high_same_all_age=amplitude_high_same_all_age
        self.avoid_spindle_loc=avoid_spindle_loc
        self.verbose = verbose

        # --------------------------------------------------------------------------
        #  correlaton coefficient transition boundry comdition
        # --------------------------------------------------------------------------
        self.b_val = b_val
  
        self.intersted_sleep_stages_REM=intersted_sleep_stages_REM#R
        self.intersted_sleep_stages_NREM=intersted_sleep_stages_NREM#N1,N2,N3,N4
        self.intersted_sleep_stages_NREM_REM_comb=intersted_sleep_stages_NREM_REM_comb#N1,N2,N3,N4 and R
    
        cross_correlation_ref_dic={}
        a=0
        for ch1 in range(0,len(ch_names)-1): 
            for ch2 in range(ch1+1,len(ch_names)):
                cross_correlation_ref_dic[ch_names[ch1]+'-'+ch_names[ch2]]=[[ch1,ch2],a]
                a=a+1
    
        self.cross_correlation_ref_dic = cross_correlation_ref_dic
        
        self.epoch_length = epoch_length  
        self.line_freq = line_freq
       
                    
        # --------------------------------------------------------------------------
        # MT and corelation related parameters
        # --------------------------------------------------------------------------
        self.T=T
        self.f_min_interst=f_min_interst
        self.f_max_interst=f_max_interst



    # Declare function to define command-line arguments
    def readOptions(self,docker_image,args=sys.argv[1:]):
        '''
        this function is created for parsing arguments via the docker
        
        '''
        parser = argparse.ArgumentParser(description="The parsing commands lists.")
        if docker_image:
            parser.add_argument("-i", "--inloc", help="Input directory relative to the mount using docker file, this directory holds the .edf file or .edf files")
            parser.add_argument("-o", "--outloc", help="Main o/p directory relative to the mount using docker file, this directory will hold the tool obtained results")
        else:
            parser.add_argument("-i", "--inloc", help="Input directory in ablsolute path, this directory holds the edf file or files")
            parser.add_argument("-o", "--outloc", help="Main o/p directory in ablsolute path, this directory holds the edf file or filess")
        # parser.add_argument("-i", "--inloc", help="Input directory in ablsolute path, this directory holds the edf file or files")
        
        
        # --------------------------------------------------------------------------
        # provide a txt file to feed through the tool to process the edf files
        # --------------------------------------------------------------------------
        parser.add_argument("-edf", "--edflist", help="file name for list of edf file/ files need to detect the artifacts, in  a text file with comma-separated")
        
        
        
        # --------------------------------------------------------------------------
        # provide a txt file to feed the tool specfications to parse the argumnets for the tool
        #  chekc the length if no input is provided uses the defaults
        # --------------------------------------------------------------------------
        parser.add_argument("-opt", "--options", default='option.txt',help="file name for options to the tools this can be used to vary the default parameters  under construction")

            
        # --------------------------------------------------------------------------
        # Options to feed the i/ps related to the 
        # --------------------------------------------------------------------------
        parser.add_argument("-ev", "--event", default=1, type=int,choices=[0,1],help="file name for saving the final results in events with sleep-stage-related epochs")
        parser.add_argument("-dic", "--dictionary", default=0, type=int, choices=[0,1],help="Save the metadata from the edf in dictionary format as pickle")
        parser.add_argument("-b", "--bad_epochs", default=1, type=int, choices=[0,1],help="Check the bad epochs in the provided edf to focus only on the sleep-related signals")
        parser.add_argument("-sleep", "--sleep_annot", default=0, type=int, choices=[0,1],help="save the sleep annotation in npy file")
        parser.add_argument("-out", "--outlier", default=0, type=int, choices=[0,1],help="save the predicted outliers")
        parser.add_argument("-MT", "--MultiTaper", default=0, type=int, choices=[0,1],help="save the predicted multitaper outcome")
        parser.add_argument("-outNREM", "--outlierNREM_REM", default=0, type=int, choices=[0,1],help="save the predicted outliers as NREM and REM") 
        parser.add_argument("-sp", "--spindles", default=0, type=int, choices=[0,1],help="First, predict the spindles via the YASA, then avoid the spindles while predicting the outliers") 
        parser.add_argument("-la", "--Latex", default=0, type=int, choices=[0,1],help="Save the figures with the predicted outcome for Latex") 
        parser.add_argument("-tag", "--tag", default='', type=str,help= "Add this tag to save the pickle file to track the changes in different parameter choices") 
        # --------------------------------------------------------------------------
        # whether consider the standard deviation value
        # --------------------------------------------------------------------------
        parser.add_argument("-sep", "--NREM_REM_sep", default=0, type=int, choices=[0,1],help="Feed NREM and REM separately") 
        parser.add_argument("-var", "--flat_MT_consider", default=0, type=int, choices=[0,1],help="Consider the variance") 
        parser.add_argument("-std", "--std_thres", default=5, type=int, help="Standard deviation threshold ") 


        # --------------------------------------------------------------------------
        # Moving window based parameter choises
        # --------------------------------------------------------------------------
        parser.add_argument("-mv_win", "--moving_window_size", default=60, type=int, help="Moving window size") 
        parser.add_argument("-th_fact", "--th_fact", default=2, type=int, help="Local moving window threshold") 
        parser.add_argument("-gl_fact", "--sorted_median_cont_grp_comp_sort_quan_cond_th_fac", default=4, type=int, help="Local moving window threshold") 


        # --------------------------------------------------------------------------
        # finalise the output of the convolutional window
        # --------------------------------------------------------------------------
        parser.add_argument("-with_conv", "--with_convolution_for_check",default=0, type=int, choices=[0,1],  help="Whether considering the moving window for unifying the outliers") 

        parser.add_argument("-th_out", "--outlier_presene_con_lenth_th",default=4, type=int,  help="Threhold length in seconds for unifying the outliers") 
        parser.add_argument("-th_conv", "--thresh_min_conv",default=5, type=int,  help="Convolutional window length in seconds for unifying the outliers") 

        parser.add_argument("-fill", "--with_fill_period",default=0, type=int, choices=[0,1], help="Fill the predicted unified gaps") 


        # --------------------------------------------------------------------------
        # combinely select all the option or leaving all 
        # --------------------------------------------------------------------------
        parser.add_argument("-all", "--sel_all", default=0, type=int, choices=[0,1],help="select all the options for saving the directory") 


        opts = parser.parse_args(args)
        return opts
    
    def assign_user_inputs_for_directories(self,docker_image=True, direct_run=False):
        
        #just skipping the reading i/ps
        if not direct_run:
            print("not direct_run")
            # Call the function to read the argument values
            options = self.readOptions(docker_image,sys.argv[1:])

            # --------------------------------------------------------------------------
            # get the parsed argumnet s and aassign accordingly
            # --------------------------------------------------------------------------
            self.flat_MT_consider = options.flat_MT_consider
            self.std = options.std_thres
            if options.NREM_REM_sep:
                self.sep = True
            else:
                self.sep = False

            self.moving_window_size = options.moving_window_size
            self.th_fact = options.th_fact
            self.gl_th_fac = options.sorted_median_cont_grp_comp_sort_quan_cond_th_fac


            # --------------------------------------------------------------------------
            # Finalisation output related arguments
            # --------------------------------------------------------------------------
            self.with_fill_period = options.with_fill_period
            self.thresh_min_conv = options.thresh_min_conv
            self.tag = options.tag
            self.outlier_presene_con_lenth_th = options.outlier_presene_con_lenth_th

            # --------------------------------------------------------------------------
            loading_dir_pre = EEG_sleep_dir()

            loading_dir_pre.in_loc = str(options.inloc)
            loading_dir_pre.out_loc = str(options.outloc)

        
            if self.GUI_percentile:
                percent_complete(25, 100, bar_width=60, title="Assiging user i/ps", print_perc=True)  
            # --------------------------------------------------------------------------
            # get the values from the check box
            # --------------------------------------------------------------------------
            if direct_run or (not direct_run and options.sel_all):
                
                loading_dir_pre.keep_signature_dic['dic']=True
                loading_dir_pre.keep_signature_dic['evtxt']=True
                loading_dir_pre.keep_signature_dic['bad_epochs']=True
                loading_dir_pre.keep_signature_dic['out_loc_outlier']=True
                loading_dir_pre.keep_signature_dic['sleep_anot']=True
                loading_dir_pre.keep_signature_dic['MT_spec']=True
                loading_dir_pre.keep_signature_dic['annota_NREM_REM']=True
                loading_dir_pre.keep_signature_dic['splidle_inc']=True
                loading_dir_pre.keep_signature_dic['tex_files']=True

            else:
                loading_dir_pre.keep_signature_dic['dic'] = options.dictionary
                loading_dir_pre.keep_signature_dic['evtxt'] =  options.event
                loading_dir_pre.keep_signature_dic['bad_epochs'] =  options.bad_epochs
                loading_dir_pre.keep_signature_dic['out_loc_outlier'] =  options.outlier
                loading_dir_pre.keep_signature_dic['sleep_anot'] = options.sleep_annot
                loading_dir_pre.keep_signature_dic['MT_spec'] =  options.MultiTaper
                loading_dir_pre.keep_signature_dic['annota_NREM_REM'] =  options.outlierNREM_REM
                loading_dir_pre.keep_signature_dic['splidle_inc'] =  options.spindles
                loading_dir_pre.keep_signature_dic['tex_files'] =  options.Latex
            # print(options.name)
            # print(options.email

            if self.GUI_percentile:
                percent_complete(50, 100, bar_width=60, title="Assiging user i/ps", print_perc=True)  
            print("not direct_run")

            # logger.warning("Need to assign directories")
            loading_dir_pre.assign_directories()
            if self.GUI_percentile:
                percent_complete(60, 100, bar_width=60, title="Assiging user i/ps", print_perc=True)  
            # --------------------------------------------------------------------------
            # to assign the edf files 
            #  handle signele .edf or multiple edfs based on the condition
            # --------------------------------------------------------------------------
            try:
                # --------------------------------------------------------------------------
                #  based on the users provided i/p this will hadle
                # --------------------------------------------------------------------------
                edf_files=[]
                if options.edflist.endswith('.txt') or options.edflist.endswith('.csv') :
                    import csv
                    with open(options.edflist, 'r') as file:
                        csvreader = csv.reader(file)
                        for rows in csvreader:
                            for row in rows:
                                if len(row)>0:
                                    edf_files.append(row)
                                
                    file.close()
                    
                elif options.edflist.endswith('.edf'):
                    edf_files=[options.edflist]
                elif  options.edflist.endswith(']'):
                    # logger.error("Need to assign the edf files list in in the txt file")

                    # edf_files=list(options.edflist)
                    edf_files_t = options.edflist.split(',')
                    if len(edf_files_t)==1:
                        edf_files = [edf_files_t[0][1:-1]]
                    elif len(edf_files_t)==2:
                        edf_files = [edf_files_t[0][1:],edf_files_t[1][:-1]]
                    else:
                        edf_files = [edf_files_t[0][1:],edf_files_t[1][:-1]]+edf_files_t[1:-1]
            except:
                # --------------------------------------------------------------------------
                #  if the user doen't provided any i/ps use all the edf files in the inloc
                # --------------------------------------------------------------------------
                logger.warning("")
                logger.warning("")
                logger.warning("")
                logger.warning("User edf not worked or user not provided any edfs via the commandline")
                logger.warning("so using all edf from in_loc")
                logger.warning("")
                logger.warning("")
                logger.warning("")
                os.chdir('/')
                edf_files=[]
                for file in os.listdir(loading_dir_pre.in_loc):
                    # check only edf files
                    if file.endswith('.edf'):
                        edf_files.append(file)
            
            loading_dir_pre.edf_files = edf_files
            if self.GUI_percentile:
                percent_complete(100, 100, bar_width=60, title="Assiging user i/ps", print_perc=True)  
        
        else:
            loading_dir_pre = EEG_sleep_dir()
        
        self.loading_dir_pre = loading_dir_pre
        return loading_dir_pre


    def preprocess_par(self):

        epoch_length = self.epoch_length   # [s]
           
        # --------------------------------------------------------------------------
        # Assigning prprocess paramter values
        # since the bandpass filter is applied from 0.5-32.5Hz
        # --------------------------------------------------------------------------
        f_min_interst = self.f_min_interst
        f_max_interst =self.f_max_interst

        # --------------------------------------------------------------------------
        # sleep-related events the region of interst going to be 0.5-32.5Hz
        # --------------------------------------------------------------------------
        bandpass_freq = [f_min_interst, f_max_interst]  # [Hz]
        
        # --------------------------------------------------------------------------
        # power line frequency
        # --------------------------------------------------------------------------
        line_freq = self.line_freq # [Hz]
        # --------------------------------------------------------------------------
        #this will check the essentially of notch filter usage in the pipeline
        # if the power line noise is less the notch fileter is not going to be applied
        # --------------------------------------------------------------------------
        notch_freq_essential_checker=False

        # --------------------------------------------------------------------------
        # Obtain the epochs only related to annoatted sleep-stages
        # --------------------------------------------------------------------------
        normal_only = True
        
        # --------------------------------------------------------------------------
        #  the values assgined the amplitude threhold value for subject
        # --------------------------------------------------------------------------
        amplitude_thres = 2000
        
        # --------------------------------------------------------------------------
        # hold to use the value for MT-range 
        # --------------------------------------------------------------------------
        self.bandpass_freq = bandpass_freq
        
        return epoch_length, line_freq, bandpass_freq, normal_only, notch_freq_essential_checker, amplitude_thres
    
    
    def intial_parameters_outlier_vertical_spikes(self,  break_spindle_flatten=True, break_flat_MT_flatten=True,
                                            z_transform=False,  inter_mediate_transform = True,  Fisher_based = True,
                                                flat_MT_consider= False):
    
        # --------------------------------------------------------------------------
        #  break the contuniious grops while considering the given spindle or 
        #  already predicted i/p given
        # --------------------------------------------------------------------------
        break_spindle_flatten=break_spindle_flatten
        break_flat_MT_flatten=break_flat_MT_flatten
        
        # --------------------------------------------------------------------------
        #if we used the Z-transform we need this for approach-1
        # --------------------------------------------------------------------------
        z_transform=z_transform   
        inter_mediate_transform = inter_mediate_transform
        Fisher_based = Fisher_based
    
        # --------------------------------------------------------------------------
        #here we haven't obtain any flat signal based on the variance based analysis
        # one way generally apply to get the flat signal based on the variance of the spectrum
        # this also can be used to input the already  expternally detected artifacts
        # --------------------------------------------------------------------------
        flat_MT_consider= flat_MT_consider

        return break_spindle_flatten, break_flat_MT_flatten, z_transform,  inter_mediate_transform,  Fisher_based, flat_MT_consider 
    
        
    def methodology_related_paramaters_for_outlier_detection(self,
                                                             tail_check_bin=False, GMM_based=False,# distribution based parameters
                                                             factor_check=10, threh_prob = 0.01, outlier_basic_con=3,
                                                             moving_window_based=True, moving_window_size=60, th_fact=2,
                                                             sep=False, global_check=True, only_good_points=True,
                                                             sorted_median_cont_grp_comp_sort_median_cond=[False,10],
                                                             sorted_median_cont_grp_comp_sort_max_cond=[False,10],
                                                             sorted_median_cont_grp_comp_sort_quan_cond=[True,4,0.75],
                                                             cont_seg_wise=False, cont_threh=4,threh_prob_artf_cont_seg=0.5,
                                                             with_fill_period=True,
                                                             with_conv=False,
                                                             thresh_min_conv=5,thresh_in_sec=True,outlier_presene_con_lenth_th=4):
        # --------------------------------------------------------------------------
        # this is like using methology based on the distribution 
        # in this script we are not checking the distribution based outlier detection
        # 
        # Methodology of distribution based
        # --------------------------------------------------------------------------
        self.tail_check_bin = tail_check_bin
        self.GMM_based = GMM_based

    
        self.factor_check = factor_check
        self.threh_prob =  threh_prob
        self.outlier_basic_con = outlier_basic_con
    
    
        # --------------------------------------------------------------------------
        # this is like using methology based on the moving window 
        # in this script we are checking based on the moving 
        # 
        # Methodology of moving window based
        # --------------------------------------------------------------------------  
        self.moving_window_based=moving_window_based
        # --------------------------------------------------------------------------
        # moving window based parameters
        # --------------------------------------------------------------------------       
        # this is the twice the size of epoch 30 x 2 = 60
        self.moving_window_size = moving_window_size
        self.th_fact = th_fact
        #consider the NREM and REM combinely or not
        self.sep=sep
                  
        # here the global threhold is used
        self.global_check=global_check
        #means only good-points are used for global_check
        self.only_good_points=only_good_points

        # global check parameters
        self.sorted_median_cont_grp_comp_sort_median_cond= sorted_median_cont_grp_comp_sort_median_cond
        self.sorted_median_cont_grp_comp_sort_max_cond=sorted_median_cont_grp_comp_sort_max_cond
        self.sorted_median_cont_grp_comp_sort_quan_cond=sorted_median_cont_grp_comp_sort_quan_cond

    
        # --------------------------------------------------------------------------
        #  finalise the detected outliers on the preprocessed based continious segments
        #  this is common to both approaches
        # --------------------------------------------------------------------------
        self.cont_seg_wise=cont_seg_wise
        
        # --------------------------------------------------------------------------
        #  this is for distibution
        # --------------------------------------------------------------------------
        self.cont_threh=cont_threh
        self.threh_prob_artf_cont_seg=threh_prob_artf_cont_seg
        
        # --------------------------------------------------------------------------
        # pin point the loose lead detection based on the convolutional window
        # --------------------------------------------------------------------------
        self.with_conv = with_conv
        self.thresh_min_conv=thresh_min_conv#the size of the convoultionbal window
        self.thresh_in_sec=thresh_in_sec#provied sizw in seconds
        self.outlier_presene_con_lenth_th=outlier_presene_con_lenth_th
        return tail_check_bin, GMM_based,    factor_check, threh_prob, outlier_basic_con,\
            moving_window_based, moving_window_size, th_fact,\
            sep, global_check, only_good_points,\
            sorted_median_cont_grp_comp_sort_median_cond, sorted_median_cont_grp_comp_sort_max_cond, sorted_median_cont_grp_comp_sort_quan_cond,\
            cont_seg_wise, cont_threh, threh_prob_artf_cont_seg,\
            with_conv, thresh_min_conv,thresh_in_sec,outlier_presene_con_lenth_th,\
            with_fill_period

    def assign_par_finalise_lead_loose_due_to_amoun_of_presented_artifacts(self,
       loose_lead_period_min=5,
            percentage_on_given_period_while_sliding=False,
            overall_percent_check=True,
            apply_conv_window=False,num_occurance=3, percent_check=5, 
            loose_conv_wind=20,stride_size=5, conv_type='same'):
        # loose_lead_period_min=1,
        # percentage_on_given_period_while_sliding=False,
        # overall_percent_check=False,
        # apply_conv_window=True,num_occurance=3, percent_check=5, loose_conv_wind=20,stride_size=5, conv_type='same'):
   
        
        # --------------------------------------------------------------------------
        #finalise the loose lead detection the following parameters are just annotate the whole lead as loose-lead based on the conditions
        # len_period_tol_min or loose_lead_period_min to fill the gaps
        # --------------------------------------------------------------------------
        self.loose_lead_period_min =loose_lead_period_min
        self.overall_percent_check = overall_percent_check
    
        # --------------------------------------------------------------------------
        # pin-point is done, following parameter part is used to find the loose-lead not pointing the loose-lead
        # --------------------------------------------------------------------------
        self.percentage_on_given_period_while_sliding = percentage_on_given_period_while_sliding
        self.apply_conv_window=apply_conv_window
        self.num_occurance=num_occurance
        self.percent_check=percent_check
        self.loose_conv_wind=loose_conv_wind
        self.stride_size=stride_size
        self.conv_type_loose_lead_desision=conv_type
        
        return   loose_lead_period_min,\
        percentage_on_given_period_while_sliding,\
            overall_percent_check,\
            apply_conv_window,num_occurance, percent_check, loose_conv_wind,stride_size, conv_type
