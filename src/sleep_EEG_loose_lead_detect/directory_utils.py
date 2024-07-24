#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 07:44:21 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

this class is created to just maintain all the directories at same place,
Just either getting from the user input or manually assign them
"""
import os
import logging

logger = logging.getLogger("EEG_directory")
while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

class EEG_sleep_dir:
        
    def __init__(self,splidle_inc=False, tex_files=False, assign_bad_events_rel_path=True):
        
        # --------------------------------------------------------------------------
        # Define the locations reltated to the find_bad_epochs functions
        # --------------------------------------------------------------------------
        self.current_wd = os.getcwd()
        self.tex_files = tex_files
        self.splidle_inc= splidle_inc
        # --------------------------------------------------------------------------
        # Define the locations reltated to the find_bad_epochs functions
        # --------------------------------------------------------------------------
        if assign_bad_events_rel_path:
            self.bad_events_file_path= '/'.join(self.current_wd.split('/')[:-1]+['docs','bad_events_bathroom.txt'])

        logger.debug(self.bad_events_file_path)
        self.keep_signature_dic={}
        self.keep_signature_dic['dic']=False
        self.keep_signature_dic['evtxt']=False
        self.keep_signature_dic['bad_epochs']=False
        self.keep_signature_dic['out_loc_outlier']=False
        self.keep_signature_dic['sleep_anot']=False
        self.keep_signature_dic['MT_spec']=False
        self.keep_signature_dic['annota_NREM_REM']=False
        self.keep_signature_dic['splidle_inc']=False
        self.keep_signature_dic['tex_files']=False
        
        # --------------------------------------------------------------------------
        #  those who run the direct assignment use this to assign 
        # Define the locations reltated to the given raw EEG data
        # --------------------------------------------------------------------------
        # self.in_loc
       
        # --------------------------------------------------------------------------
        # Define the locations reltated to save the obtained EEG data
        # --------------------------------------------------------------------------
        # self.out_loc
        
        
    def assign_directories(self):
               
        # --------------------------------------------------------------------------
        # lets create tthe directories based on the user need
        # --------------------------------------------------------------------------
        
        # to keep the meta information like age, Fs, ect.
        keep_signature_dic = self.keep_signature_dic
        self.check_directory_exist_and_create_files(self.out_loc +'_temp/')

        if keep_signature_dic['dic']:
            self.out_loc_dic= self.out_loc +'dictionaries/'
            #create the unavilable directories 
            self.check_directory_exist_and_create_files(self.out_loc_dic)
        else:
            self.out_loc_dic= self.out_loc +'_temp/'

        if keep_signature_dic['evtxt']:
            self.out_loc_txt= self.out_loc +'events_in_txt/'
            self.check_directory_exist_and_create_files(self.out_loc_txt)

        else:
            self.out_loc_txt= self.out_loc +'_temp/'

        if keep_signature_dic['bad_epochs']:
            self.bad_epochs_folder =self.out_loc +'bad_epochs/'
            self.check_directory_exist_and_create_files(self.bad_epochs_folder)

        else:
            self.bad_epochs_folder= self.out_loc +'_temp/'
        
    
        if keep_signature_dic['out_loc_outlier']:     
            self.out_loc_outlier = self.out_loc +'outlier_reref_op_tap/'
            self.check_directory_exist_and_create_files(self.out_loc_outlier)

        else:
            self.out_loc_outlier= self.out_loc +'_temp/'
            
            
        if keep_signature_dic['sleep_anot']:     
            self.out_loc_outlier_sleep_anot = self.out_loc +'sleep_anot/'
            self.check_directory_exist_and_create_files(self.out_loc_outlier_sleep_anot)

        else:
            self.out_loc_outlier_sleep_anot= self.out_loc +'_temp/'
            
            
        if keep_signature_dic['MT_spec']:     
            self.out_loc_outlier_MT_spec = self.out_loc +'MT_spec/'
            self.check_directory_exist_and_create_files(self.out_loc_outlier_MT_spec)

        else:
            self.out_loc_outlier_MT_spec= self.out_loc +'_temp/'

        if keep_signature_dic['annota_NREM_REM']:     
            self.out_loc_NREM_REM = self.out_loc + 'annota_NREM_REM/'
            self.check_directory_exist_and_create_files(self.out_loc_NREM_REM)

        else:
            self.out_loc_NREM_REM= self.out_loc +'_temp/'


        if self.splidle_inc and keep_signature_dic['splidle_inc']:     

            self.save_spindle_loc_main =  self.out_loc + 'sp_sw/'
            self.save_spindle_loc = self.save_spindle_loc_main + 'sp_sw_with_origin/'
            self.save_converted_index_loc = self.save_spindle_loc_main + 'sp_sw_with_converted_index/'
            #create the unavilable directories 
            self.check_directory_exist_and_create_files(self.save_spindle_loc_main)
            self.check_directory_exist_and_create_files(self.save_spindle_loc)
            self.check_directory_exist_and_create_files(self.save_converted_index_loc)

        else:
            
            self.save_spindle_loc_main =  self.out_loc +'_temp/'
            self.save_spindle_loc =self.out_loc +'_temp/'
            self.save_converted_index_loc =self.out_loc +'_temp/'
            
        if not self.tex_files and keep_signature_dic['tex_files']:      
            # --------------------------------------------------------------------------
            # Define the locations reltated to the tex directory
            #  this is for lictrature purpose only this
            #  only output the text command and out put location thst need to be compiled by Latex later
            # --------------------------------------------------------------------------


            # --------------------------------------------------------------------------
            # Define the locations reltated to save the obtained EEG data
            # --------------------------------------------------------------------------

            self.tex_main_dir =  self.out_loc +'_temp/'
            self.appendix_images_corr =self.out_loc +'_temp/'
            
        # logger.info('')
        # logger.info('GUI interface in development')
        # logger.info('')
                


    def check_directory_exist_and_create_files(self,direcotory_given):
        '''
        This function creates teh space for save the files
        '''
        os.chdir('/')
        if (not os.path.exists(direcotory_given)):
            if direcotory_given != "":
                os.makedirs(direcotory_given)

                logger.info('')
                logger.warning('path created to save the file'+ direcotory_given)
                logger.info('')
            else:
                logger.error("Path should atleast have one character")
          


        