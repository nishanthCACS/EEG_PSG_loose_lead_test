# import all classes / functions from the tkinter
import tkinter as tk     ## Python 3.x
from tkinter import ttk

# import tkinter as tk
from sleep_EEG_loose_lead_detect.directory_utils import EEG_sleep_dir



class Assign_directories_loose_lead:
    def __init__(self, splidle_inc=True):

        b_g_color=  "light yellow"
        main_background_color='light green'
        
        loading_dir_pre = EEG_sleep_dir(splidle_inc=splidle_inc)
        
        # Create a GUI window
        self.root=tk.Tk()
           
        # Set the background colour of GUI window
        self.root.configure(background =main_background_color)
           
        # Set the configuration of GUI window
        # self.root.geometry("600x400")
        self.root.geometry("500x250")
        self.root.rowconfigure(9)
        self.root.columnconfigure(9)
        # set the name of tkinter GUI window
        self.root.title("sleep EEG major structural artifact detect") 
           
        
        label0 = tk.Label(self.root, text = "This GUI is just helping for set up the working directory",
                        fg = 'black', bg =main_background_color)
           
        # --------------------------------------------------------------------------
        # Define the locations reltated to the given raw EEG data
        # --------------------------------------------------------------------------
        # self.in_loc
        label1 = tk.Label(self.root, text = "EDF i/p directory: ",
                        fg = 'black', bg = 'red')
           
        # --------------------------------------------------------------------------
        # Create the locations reltated to save the obtained EEG data
        # --------------------------------------------------------------------------
        label2 = tk.Label(self.root, text = "Main o/p directory : ",
                       fg = 'black', bg = 'red')
        
        label3 = tk.Label(self.root, text ='Directorys needed', font = "50",bg=main_background_color) 
        
         
        # padx keyword argument used to set padding along x-axis .
        # pady keyword argument used to set padding along y-axis .
        label0.grid(row = 0,columnspan=8,sticky=tk.E) 
        
        
        label1.grid(row = 1, column = 0, padx = 0,pady=0) 
        label2.grid(row = 2, column = 0, padx = 0,pady=0) 
        label3.grid(row = 3, column = 0, padx = 0,pady=0)
        # label4.grid(row = 5, column = 0, padx = 0,pady=0)
        
        # --------------------------------------------------------------------------
        # Create a entry box 
        # for filling or typing the information.
        # --------------------------------------------------------------------------
        
        self.inp_edf = tk.Text(self.root, height = 1,
                    width = 45,
                    bg = b_g_color,fg = "black")
        self.out_main = tk.Text(self.root, height = 1,
                    width = 45,
                    bg = b_g_color,fg = "black")    # time_field = Entry(root)
        # compound_field = Entry(root)
         
        # grid method is used for placing 
        # the widgets at respective positions 
        # in table like structure .
         
        # padx keyword argument used to set padding along x-axis .
        # pady keyword argument used to set padding along y-axis .
        self.inp_edf.grid(row = 1, column = 1, padx = 0,pady=0) 
        self.out_main.grid(row = 2, column = 1, padx = 0,pady=0) 
        '''
        earlier-1
        '''    
        
        # --------------------------------------------------------------------------
        # asiigning input taker for check box
        # --------------------------------------------------------------------------
        self.Checkbutton_evt_txt = tk.IntVar()  
        self.Checkbutton_dic = tk.IntVar()  
        self.Checkbutton_bad_epochs = tk.IntVar()  
        self.Checkbutton_out_loc_outlier= tk.IntVar()  
        self.Checkbutton_sleep_anot = tk.IntVar()  
        self.Checkbutton_MT_spec = tk.IntVar()  
        self.Checkbutton_annota_NREM_REM = tk.IntVar()  
        self.Checkbutton_splidle_inc = tk.IntVar()  
        self.Checkbutton_tex_files = tk.IntVar()  
        
        self.Checkbutton_all = tk.IntVar()  
        self.Checkbutton_de_all = tk.IntVar()  
        
        # --------------------------------------------------------------------------
        # define default values
        # --------------------------------------------------------------------------
        self.Checkbutton_evt_txt.set(1)
        
        
        self.b_evtxt = tk.Checkbutton(self.root, text = "Events txt", 
                              variable = self.Checkbutton_evt_txt,
                              onvalue = True,
                              offvalue = False,
                              height = 1,
                              width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
        
        self.b_dic = tk.Checkbutton(self.root, text = "Dictionary", 
                              variable = self.Checkbutton_dic,
                              onvalue = True,
                              offvalue = False,
                              height = 1,
                              width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
        
        self.b_bad_epochs = tk.Checkbutton(self.root, text = "Bad-epochs", 
                              variable = self.Checkbutton_bad_epochs,
                              onvalue = True,
                              offvalue = False,
                              height = 1,
                              width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
        
        self.b_out_loc_outlier = tk.Checkbutton(self.root, text = "o/p outlier", 
                              variable = self.Checkbutton_out_loc_outlier,
                              onvalue = True,
                              offvalue = False,
                              height = 1,
                              width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
        
        self.b_sleep_anot = tk.Checkbutton(self.root, text = "sleep annot", 
                              variable = self.Checkbutton_sleep_anot,
                              onvalue = True,
                              offvalue = False,
                              height = 1,
                              width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
        
        
        self.b_MT_spec = tk.Checkbutton(self.root, text = "MT -spectrum", 
                              variable = self.Checkbutton_MT_spec,
                              onvalue = True,
                              offvalue = False,
                              height = 1,
                               width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
        
        self.b_annota_NREM_REM = tk.Checkbutton(self.root, text = "Detected \n outliers pickle", 
                              variable = self.Checkbutton_annota_NREM_REM,
                              onvalue = True,
                              offvalue = False,
                              height = 2,
                              width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
        
        self.b_splidle_inc= tk.Checkbutton(self.root, text = "Splidle pickles", 
                              variable = self.Checkbutton_splidle_inc,
                              onvalue = True,
                              offvalue = False,
                              height = 2,
                              width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
        
        self.b_tex_files= tk.Checkbutton(self.root, text = "Latex Fig", 
                              variable = self.Checkbutton_tex_files,
                              onvalue = True,
                              offvalue = False,
                              height = 1,
                              width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
        
        
        self.b_sel_all = tk.Checkbutton(self.root, text = "Select all", 
                              variable = self.Checkbutton_all,
                              onvalue = True,
                              offvalue = False,
                              height = 1,
                              width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
        self.b_de_sel_all = tk.Checkbutton(self.root, text = "De select all", 
                              variable = self.Checkbutton_de_all,
                              onvalue = True,
                              offvalue = False,
                              height = 1,
                              width = 15,    fg = 'black', bg  =main_background_color,padx=0,pady=0)
        '''
        earlier-2
        '''     
        # Create a Submit Button and attached 
        # to calculate_ci function 
        submit_button1 = tk.Button(self.root, text = "Submit", bg =b_g_color, 
                         fg = "black", command=lambda: self.assign_directories(loading_dir_pre))
        submit_button1.grid(row = 3, column = 1,padx=0,pady=0)
        
        # quit_button1 = tk.Button(self.root, text="QUIT", command=self.root.destroy)
        # quit_button1 = tk.Button(self.root, text="QUIT", command=self.root.withdraw)
        # quit_button1 = tk.Button(self.root, text="QUIT", command=self.root.quit)
        quit_button1 = tk.Button(self.root, text="QUIT", command=self.disable_button,fg = "black")

        
        quit_button1.grid(row = 3, column = 1,padx=0,pady=0,sticky=tk.E)

        
        # --------------------------------------------------------------------------
        # assign buttons to the grid
        # --------------------------------------------------------------------------    
        self.b_evtxt.grid(row = 4, column = 0,padx=0,pady=0)
        self.b_dic.grid(row = 5, column = 0,padx=0,pady=0)
        self.b_sleep_anot.grid(row = 6, column = 0,padx=0,pady=0)
        
        self.b_bad_epochs.grid(row = 4, column =1, pady = 0,sticky=tk.W)
        self.b_out_loc_outlier.grid(row = 5, column = 1,padx=0,pady=0,sticky=tk.W)
        self.b_MT_spec.grid(row = 6, column = 1,padx=0,pady=0,sticky=tk.W)
        
        
        # b_splidle_inc.grid(row = 4, column =2,padx=0,pady=0,sticky=tk.W)
        # b_tex_files.grid(row =5, column =2,padx=0,pady=0,sticky=tk.W)
        # b_annota_NREM_REM.grid(row =6, column =2,padx=0,sticky=tk.W)
        self.b_splidle_inc.grid(row = 4, column =1,padx=0,pady=0,sticky=tk.E)
        self.b_tex_files.grid(row =5, column =1,padx=0,pady=0,sticky=tk.E)
        self.b_annota_NREM_REM.grid(row =6, column =1,padx=0,sticky=tk.E)
        
        self.b_sel_all.grid(row =7, column =0,padx=0)
        self.b_de_sel_all.grid(row =7, column =1,padx=0,sticky=tk.W)
        
        

        # Start the GUI 
        self.root.mainloop()
        

    def disable_button(self):
        # quit_button1 = tk.Button(self.root, text="QUIT", command=self.root.destroy)
        # # quit_button1 = tk.Button(self.root, text="QUIT", command=self.root.withdraw)

        self.root.withdraw()
        self.root.destroy()
        # --------------------------------------------------------------------------
        # if the popup okay is not pressed and the pop-up exists close it
        # --------------------------------------------------------------------------
        try:
            self.popup.withdraw()
            self.popup.destroy()
        except:
            pass
        
        
    def popupmsg(self,msg):

    
        self.popup = tk.Tk()
        self.popup.wm_title("Safe to close the assign window!")
        label = tk.Label(self.popup, text=msg,fg='black', font= ('Helvetica', 10))
        # label = ttk.Label(self.popup, text=msg,fg='black', font= ('Helvetica', 10))

        label.pack(side="top", fill="x", pady=10)
        # B1 = ttk.Button(self.popup, text="Okay",fg = "black", command = self.popup.destroy)
        B1 = tk.Button(self.popup, text="Okay",fg = "black", command = self.popup.destroy)

        B1.pack()
        self.popup.mainloop()
    
    # Function to define the directories
    def assign_directories(self,loading_dir_pre):
        
        # --------------------------------------------------------------------------
        # get the directory information from the entry box
        # --------------------------------------------------------------------------
        loading_dir_pre.in_loc = str(self.inp_edf.get("1.0", "end-1c"))
        loading_dir_pre.out_loc = str(self.out_main.get("1.0", "end-1c"))
        # --------------------------------------------------------------------------
        if  self.Checkbutton_all.get():
            self.Checkbutton_de_all.set(0)
            self.Checkbutton_dic.set(1)
            self.Checkbutton_evt_txt.set(1)
            self.Checkbutton_bad_epochs.set(1)
            self.Checkbutton_out_loc_outlier.set(1)
            self.Checkbutton_sleep_anot.set(1)
            self.Checkbutton_MT_spec.set(1)
            self.Checkbutton_annota_NREM_REM.set(1)
            self.Checkbutton_splidle_inc.set(1)
            self.Checkbutton_tex_files.set(1)
    
        if  self.Checkbutton_de_all.get():
            self.Checkbutton_dic.set(0)
            self.Checkbutton_evt_txt.set(0)
            self.Checkbutton_bad_epochs.set(0)
            self.Checkbutton_out_loc_outlier.set(0)
            self.Checkbutton_sleep_anot.set(0)
            self.Checkbutton_MT_spec.set(0)
            self.Checkbutton_annota_NREM_REM.set(0)
            self.Checkbutton_splidle_inc.set(0)
            self.Checkbutton_tex_files.set(0)
    
        # get the values from the check box
        # --------------------------------------------------------------------------
        loading_dir_pre.keep_signature_dic['dic'] = self.Checkbutton_dic.get()
        loading_dir_pre.keep_signature_dic['evtxt'] = self.Checkbutton_evt_txt.get()
        loading_dir_pre.keep_signature_dic['bad_epochs'] = self.Checkbutton_bad_epochs.get()
        loading_dir_pre.keep_signature_dic['out_loc_outlier'] = self.Checkbutton_out_loc_outlier.get()
        loading_dir_pre.keep_signature_dic['sleep_anot'] = self.Checkbutton_sleep_anot.get()
        loading_dir_pre.keep_signature_dic['MT_spec'] = self.Checkbutton_MT_spec.get()
        loading_dir_pre.keep_signature_dic['annota_NREM_REM'] = self.Checkbutton_annota_NREM_REM.get()
        loading_dir_pre.keep_signature_dic['splidle_inc'] = self.Checkbutton_splidle_inc.get()
        loading_dir_pre.keep_signature_dic['tex_files'] = self.Checkbutton_tex_files.get()
        # --------------------------------------------------------------------------
        # create teh directory from the assigned check boxes
        # --------------------------------------------------------------------------
        loading_dir_pre.assign_directories()
        self.loading_dir_pre= loading_dir_pre

        self.popupmsg("Assign directories suceed")

