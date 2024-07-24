# import all classes / functions from the tkinter
from tkinter import *
from sleep_EEG_loose_lead_detect.directory_utils import EEG_sleep_dir


 
# # Function for clearing the 
# # contents of all entry boxes  
# def clear_all() :
 
#     # whole content of entry boxes is deleted
#     principal_field.delete(0, END)  
#     rate_field.delete(0, END)
#     time_field.delete(0, END)
#     compound_field.delete(0, END)
   
#     # set focus on the principal_field entry box 
#     principal_field.focus_set()
 
 
# Function to find compound interest 
def assign_directories(loading_dir_pre):
 
    # --------------------------------------------------------------------------
    # get the directory information from the entry box
    # --------------------------------------------------------------------------
    loading_dir_pre.in_loc = str(inp_edf.get("1.0", "end-1c"))
    loading_dir_pre.out_loc = str(out_main.get("1.0", "end-1c"))
    # --------------------------------------------------------------------------
    if  Checkbutton_all.get():
        Checkbutton_de_all.set(0)
        Checkbutton_dic.set(1)
        Checkbutton_evt_txt.set(1)
        Checkbutton_bad_epochs.set(1)
        Checkbutton_out_loc_outlier.set(1)
        Checkbutton_sleep_anot.set(1)
        Checkbutton_MT_spec.set(1)
        Checkbutton_annota_NREM_REM.set(1)
        Checkbutton_splidle_inc.set(1)
        Checkbutton_tex_files.set(1)

    if  Checkbutton_de_all.get():
        Checkbutton_dic.set(0)
        Checkbutton_evt_txt.set(0)
        Checkbutton_bad_epochs.set(0)
        Checkbutton_out_loc_outlier.set(0)
        Checkbutton_sleep_anot.set(0)
        Checkbutton_MT_spec.set(0)
        Checkbutton_annota_NREM_REM.set(0)
        Checkbutton_splidle_inc.set(0)
        Checkbutton_tex_files.set(0)

    # get the values from the check box
    # --------------------------------------------------------------------------
    loading_dir_pre.keep_signature_dic['dic'] = Checkbutton_dic.get()
    loading_dir_pre.keep_signature_dic['evtxt'] = Checkbutton_evt_txt.get()
    loading_dir_pre.keep_signature_dic['bad_epochs'] = Checkbutton_bad_epochs.get()
    loading_dir_pre.keep_signature_dic['out_loc_outlier'] = Checkbutton_out_loc_outlier.get()
    loading_dir_pre.keep_signature_dic['sleep_anot'] = Checkbutton_sleep_anot.get()
    loading_dir_pre.keep_signature_dic['MT_spec'] = Checkbutton_MT_spec.get()
    loading_dir_pre.keep_signature_dic['annota_NREM_REM'] = Checkbutton_annota_NREM_REM.get()
    loading_dir_pre.keep_signature_dic['splidle_inc'] = Checkbutton_splidle_inc.get()
    loading_dir_pre.keep_signature_dic['tex_files'] = Checkbutton_tex_files.get()
    # --------------------------------------------------------------------------
    # create teh directory from the assigned check boxes
    # --------------------------------------------------------------------------
    loading_dir_pre.assign_directories()
    # print(loading_dir_pre.in_loc)
    # print(loading_dir_pre.out_loc)
    
    print("ASssign directories suceed")

    root.withdraw()
    root.destroy
 
# Driver code
if __name__ == "__main__" :
    b_g_color=  "light yellow"
    main_background_color='light green'
    
    loading_dir_pre = EEG_sleep_dir(splidle_inc=True)

    # Create a GUI window
    root = Tk()
   
    # Set the background colour of GUI window
    root.configure(background =main_background_color)
   
    # Set the configuration of GUI window
    # root.geometry("600x400")
    root.geometry("500x250")
    root.rowconfigure(9)
    root.columnconfigure(9)
    # set the name of tkinter GUI window
    root.title("sleep EEG major structural artifact detect") 
       
    
    label0 = Label(root, text = "This GUI is just helping for set up the working directory",
                    fg = 'black', bg =main_background_color)
   
    # --------------------------------------------------------------------------
    # Define the locations reltated to the given raw EEG data
    # --------------------------------------------------------------------------
    # self.in_loc
    label1 = Label(root, text = "EDF i/p directory: ",
                    fg = 'black', bg = 'red')
   
    # --------------------------------------------------------------------------
    # Create the locations reltated to save the obtained EEG data
    # --------------------------------------------------------------------------
    label2 = Label(root, text = "Main o/p directory : ",
                   fg = 'black', bg = 'red')

    label3 = Label(root, text ='Directorys needed', font = "50",bg=main_background_color) 

 
    # padx keyword argument used to set padding along x-axis .
    # pady keyword argument used to set padding along y-axis .
    label0.grid(row = 0,columnspan=8,sticky=E) 

    
    label1.grid(row = 1, column = 0, padx = 0,pady=0) 
    label2.grid(row = 2, column = 0, padx = 0,pady=0) 
    label3.grid(row = 3, column = 0, padx = 0,pady=0)
    # label4.grid(row = 5, column = 0, padx = 0,pady=0)
    
    # --------------------------------------------------------------------------
    # Create a entry box 
    # for filling or typing the information.
    # --------------------------------------------------------------------------

    inp_edf = Text(root, height = 1,
                width = 45,
                bg = b_g_color,   fg = 'black')
    out_main = Text(root, height = 1,
                width = 45,
                bg = b_g_color,    fg = 'black')    # time_field = Entry(root)
    # compound_field = Entry(root)
 
    # grid method is used for placing 
    # the widgets at respective positions 
    # in table like structure .
     
    # padx keyword argument used to set padding along x-axis .
    # pady keyword argument used to set padding along y-axis .
    inp_edf.grid(row = 1, column = 1, padx = 0,pady=0) 
    out_main.grid(row = 2, column = 1, padx = 0,pady=0) 
    '''
    earlier-1
    '''    

    # --------------------------------------------------------------------------
    # asiigning input taker for check box
    # --------------------------------------------------------------------------
    Checkbutton_evt_txt = IntVar()  
    Checkbutton_dic = IntVar()  
    Checkbutton_bad_epochs = IntVar()  
    Checkbutton_out_loc_outlier= IntVar()  
    Checkbutton_sleep_anot = IntVar()  
    Checkbutton_MT_spec = IntVar()  
    Checkbutton_annota_NREM_REM = IntVar()  
    Checkbutton_splidle_inc = IntVar()  
    Checkbutton_tex_files = IntVar()  

    Checkbutton_all = IntVar()  
    Checkbutton_de_all = IntVar()  

    # --------------------------------------------------------------------------
    # define default values
    # --------------------------------------------------------------------------
    Checkbutton_evt_txt.set(1)


    b_evtxt = Checkbutton(root, text = "Events txt", 
                          variable = Checkbutton_evt_txt,
                          onvalue = True,
                          offvalue = False,
                          height = 1,
                          width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)

    b_dic = Checkbutton(root, text = "Dictionary", 
                          variable = Checkbutton_dic,
                          onvalue = True,
                          offvalue = False,
                          height = 1,
                          width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
    
    b_bad_epochs = Checkbutton(root, text = "Bad-epochs", 
                          variable = Checkbutton_bad_epochs,
                          onvalue = True,
                          offvalue = False,
                          height = 1,
                          width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)

    b_out_loc_outlier = Checkbutton(root, text = "o/p outlier", 
                          variable = Checkbutton_out_loc_outlier,
                          onvalue = True,
                          offvalue = False,
                          height = 1,
                          width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
    
    b_sleep_anot = Checkbutton(root, text = "sleep annot", 
                          variable = Checkbutton_sleep_anot,
                          onvalue = True,
                          offvalue = False,
                          height = 1,
                          width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)

    
    b_MT_spec = Checkbutton(root, text = "MT -spectrum", 
                          variable = Checkbutton_MT_spec,
                          onvalue = True,
                          offvalue = False,
                          height = 1,
                           width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)

    b_annota_NREM_REM = Checkbutton(root, text = "Detected \n outliers pickle", 
                          variable = Checkbutton_annota_NREM_REM,
                          onvalue = True,
                          offvalue = False,
                          height = 2,
                          width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)

    b_splidle_inc= Checkbutton(root, text = "Splidle pickles", 
                          variable = Checkbutton_splidle_inc,
                          onvalue = True,
                          offvalue = False,
                          height = 2,
                          width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
    
    b_tex_files= Checkbutton(root, text = "Latex Fig", 
                          variable = Checkbutton_tex_files,
                          onvalue = True,
                          offvalue = False,
                          height = 1,
                          width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
    
    
    b_sel_all=  Checkbutton(root, text = "Select all", 
                          variable = Checkbutton_all,
                          onvalue = True,
                          offvalue = False,
                          height = 1,
                          width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
    b_de_sel_all=  Checkbutton(root, text = "De select all", 
                          variable = Checkbutton_de_all,
                          onvalue = True,
                          offvalue = False,
                          height = 1,
                          width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
    '''
    earlier-2
    '''     
    # Create a Submit Button and attached 
    # to calculate_ci function 
    submit_button1 = Button(root, text = "Submit", bg =b_g_color, 
                     fg = "black", command=lambda: assign_directories(loading_dir_pre))
    
    
    # --------------------------------------------------------------------------
    # assign buttons to the grid
    # --------------------------------------------------------------------------    
    b_evtxt.grid(row = 4, column = 0,padx=0,pady=0)
    b_dic.grid(row = 5, column = 0,padx=0,pady=0)
    b_sleep_anot.grid(row = 6, column = 0,padx=0,pady=0)

    b_bad_epochs.grid(row = 4, column =1, pady = 0,sticky=W)
    b_out_loc_outlier.grid(row = 5, column = 1,padx=0,pady=0,sticky=W)
    b_MT_spec.grid(row = 6, column = 1,padx=0,pady=0,sticky=W)


    # b_splidle_inc.grid(row = 4, column =2,padx=0,pady=0,sticky=W)
    # b_tex_files.grid(row =5, column =2,padx=0,pady=0,sticky=W)
    # b_annota_NREM_REM.grid(row =6, column =2,padx=0,sticky=W)
    b_splidle_inc.grid(row = 4, column =1,padx=0,pady=0,sticky=E)
    b_tex_files.grid(row =5, column =1,padx=0,pady=0,sticky=E)
    b_annota_NREM_REM.grid(row =6, column =1,padx=0,sticky=E)

    b_sel_all.grid(row =7, column =0,padx=0)
    b_de_sel_all.grid(row =7, column =1,padx=0,sticky=W)


    '''
    assign the directories
    '''
   
    submit_button1.grid(row = 3, column = 1,padx=0,pady=0)

    # Start the GUI 
    root.mainloop()