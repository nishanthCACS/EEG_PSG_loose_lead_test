#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:27:10 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved
"""
import os

def save_texfile_single_combination(saving_tex_dirs,
                                    texfile_name, text_caption, scale,
                                 texfile_part):
    '''
    create a tex file for image

    '''
    os.chdir('/') 
    os.chdir(saving_tex_dirs)  

        
    with open(texfile_name+'.tex', 'w') as f:
        f.write('')
        f.write('\\begin{figure}[htp] \n')
        f.write('')
        
        f.write('\centering \n')
        f.write('\centerline{\includegraphics[scale='+str(scale)+']{'+texfile_part+texfile_name+'.png}}\n')
               
        f.write(' \n')

                                
        f.write(' \n')
        f.write('\caption{'+text_caption+'\label{'+texfile_name+'}} \n')
        f.write('\end{figure} \n')


    print('\input{'+texfile_part+texfile_name+'.tex}')



def save_texfile_double_col_plot(saving_tex_dirs,texfile_name,  figure_names,
                        texfile_part,text_main_extra,figure_names_dic,spcial_cap_part,
                        common_cap, common_cap_2, with_caption=True,
                        scale_first = 0.5):
    os.chdir('/') 
    os.chdir(saving_tex_dirs)  
    # scale_first = 0.5
    # i_grp=figure_names.pop(0)
    with open(texfile_name+'.tex', 'w') as f:
        f.write('')
        f.write('\\begin{figure}[htp] \n')
    
        scale=0.28
    
        while len(figure_names)>0:
            
            i_grp=figure_names.pop(0)
            f.write('\\begin{subfigure}[t]{0.48\\textwidth}\n')
            f.write('\centering \n')
            f.write('\centerline{\includegraphics[scale='+str(scale)+']{'+texfile_part+text_main_extra[i_grp]+figure_names_dic[i_grp]+'.png}}\n')
            if with_caption:
                f.write('\caption{'+spcial_cap_part[i_grp]+' \label{'+figure_names_dic[i_grp]+'}}\n')
    
            f.write('\end{subfigure}~\\begin{subfigure}[t]{0.48\\textwidth} \n')
            
            i_grp=figure_names.pop(0)
            
            f.write('\centering \n')
            f.write('\centerline{\includegraphics[scale='+str(scale)+']{'+texfile_part+text_main_extra[i_grp]+figure_names_dic[i_grp]+'.png}}\n')
            if with_caption:           
                f.write('\caption{'+spcial_cap_part[i_grp]+' \label{'+figure_names_dic[i_grp]+'}}\n')
            f.write('\end{subfigure} \n')
            f.write(' \n')
        f.write(' \n')
        f.write('\caption{'+common_cap+ common_cap_2)    
    
        f.write('\label{'+texfile_name+'}}\n')
        f.write('\\end{figure} \n')
    
    print('\input{'+texfile_part+texfile_name+'.tex}')



def save_texfile_single_col_plot(saving_tex_dirs,texfile_name,  figure_names,
                        texfile_part,text_main_extra,figure_names_dic,spcial_cap_part,
                        common_cap, common_cap_2, with_caption=True,
                        scale_first = 0.5):
    os.chdir('/') 
    os.chdir(saving_tex_dirs)  
    # scale_first = 0.5
    # i_grp=figure_names.pop(0)
    with open(texfile_name+'.tex', 'w') as f:
        f.write('')
        f.write('\\begin{figure}[htp] \n')
    
        while len(figure_names)>0:
            
            i_grp=figure_names.pop(0)
            f.write('\\begin{subfigure}[t]{0.95\\textwidth}\n')
            f.write('\centering \n')
            f.write('\centerline{\includegraphics[scale='+str(scale_first)+']{'+texfile_part+text_main_extra[i_grp]+figure_names_dic[i_grp]+'.png}}\n')
            if with_caption:
                f.write('\caption{'+spcial_cap_part[i_grp]+' \label{'+figure_names_dic[i_grp]+'}}\n')
    
            f.write('\end{subfigure} \n')
            f.write(' \n')


        f.write(' \n')
        f.write('\caption{'+common_cap+ common_cap_2)    
    
        f.write('\label{'+texfile_name+'}}\n')
        f.write('\\end{figure} \n')
    
    print('\input{'+texfile_part+texfile_name+'.tex}')

