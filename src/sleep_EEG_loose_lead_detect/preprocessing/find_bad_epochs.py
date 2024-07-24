#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@ author AMlan Talukdar

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> 
Modified on Wed Jun 28 12:22:28 2023 to create the package
Modified on Mon Jul  3 07:50:39 2023 to obtain directory from the unified directory_utils

This bad epoch detection is optional for the EEG-server- for our use only
"""

from math import ceil
import numpy as np
import pdb, mne, os

from sleep_EEG_loose_lead_detect.directory_utils import EEG_sleep_dir

# --------------------------------------------------------------------------
def readFile(filename):

    fl = open(filename, "r")
    data = fl.readlines()
    fl.close()

    return data

# --------------------------------------------------------------------------
def createDir(d, is_file_path=False):
    
    if is_file_path: d = os.path.dirname(d)

    if d != "":
        if not os.path.exists(d):
            os.makedirs(d)

# --------------------------------------------------------------------------
def writeFile(filename, data, mode="w"):
    createDir(filename, is_file_path=True)

    fl = open(filename, mode)
    fl.write(data)
    fl.close()

# --------------------------------------------------------------------------
def formatDataTable(data, col_sep="\t", row_sep="\n"):
    return row_sep.join([col_sep.join([str(item1) for item1 in item]) for item in data])

# --------------------------------------------------------------------------
def writeDataTableAsText(data, filename, mode="w"):
    text = formatDataTable(data, "\t", "\n")

    writeFile(filename, text, mode)

# --------------------------------------------------------------------------
def loadSampleNames(edf_source):
        return ['.'.join(item.split('.')[:-1]) for item in os.listdir(edf_source)
                if (os.path.isfile(os.path.join(edf_source, item)) and item.split('.')[-1].lower() == 'edf')]

# --------------------------------------------------------------------------
def printDec(msg):
    
    horizontal_border = '_' * 50
    vertical_border = '|'

    l = len(horizontal_border)

    print(horizontal_border)
    print(' ' * l)
    msg_part = msg.strip()
    while len(msg_part) >= l - 4:
        print(vertical_border + ' ' + msg_part[:l - 4] + ' ' + vertical_border)
        msg_part = msg_part[l - 4:].strip()
    print(vertical_border + ' ' + msg_part + ' ' * (l - 3 - len(msg_part)) + vertical_border)
    print(horizontal_border)
    print("")

# --------------------------------------------------------------------------
def markBadEpochs(data_file_path_data, bad_events_file_path, epoch_sec=None, verbose=False):

    if epoch_sec is None: epoch_sec = 30

    # --------------------------------------------------------------------------
    # Relevant sleep stage events
    # --------------------------------------------------------------------------
    sleep_stage_events = {'sleep stage w',
                        'sleep stage r',
                        'sleep stage 1', 
                        'sleep stage 2', 
                        'sleep stage 3',
                        'sleep stage 4',
                        'sleep stage n1', 
                        'sleep stage n2', 
                        'sleep stage n3',
                        'sleep stage ?'}

    # --------------------------------------------------------------------------
    # Load bad events
    # --------------------------------------------------------------------------
    bad_events = set([item.strip() for item in readFile(bad_events_file_path)])

    # try:
    data = mne.io.read_raw_edf(data_file_path_data, verbose=False)
    # except:
    # data = mne.io.read_raw_edf(data_file_path_data,verbose=False,encoding='latin1')

    end_data = data.get_data().shape[1]

    sampling_freq = float(data.info['sfreq'])
    window_length = int(epoch_sec*sampling_freq)

    # --------------------------------------------------------------------------
    # Get all events
    # --------------------------------------------------------------------------
    events, event_ids = mne.events_from_annotations(data, verbose=False)
    event_id_rev = {val:key for key, val in event_ids.items()}
    events_rev = np.array([[item[0], item[1], event_id_rev[item[2]]] for item in events])

    #ignorable_epochs += list(events_rev[indices][0])

    ignorable_epochs = []

    for evt_id in event_ids.keys():

        # --------------------------------------------------------------------------
        # Check if an event is among the bad events
        # --------------------------------------------------------------------------
        if evt_id.lower() in bad_events:

            # --------------------------------------------------------------------------
            # Get bad event indices
            # --------------------------------------------------------------------------
            indices = np.where(events_rev[:, 2]==evt_id)[0]

            for i in range(len(indices)):
                
                # --------------------------------------------------------------------------
                # Get bad epoch
                # --------------------------------------------------------------------------
                bad_epoch = int(events_rev[indices[i]][0])

                # --------------------------------------------------------------------------
                # Find the earlier and next sleep events epochs to the bad epoch
                # --------------------------------------------------------------------------
                sleep_epoch_prev = None
                sleep_epoch_next = None
                for j_prev in range(indices[i]-1, -1, -1):
                    if events_rev[j_prev][2].lower() in sleep_stage_events:
                        sleep_epoch_prev = int(events[j_prev][0])
                        break
        
                for j_next in range(indices[i]+1, len(events_rev)):
                    if events_rev[j_next][2].lower() in sleep_stage_events:
                        sleep_epoch_next = int(events[j_next][0])
                        break
                
                # --------------------------------------------------------------------------
                # If no sleep event found before the bad event, the bad epoch is not within
                # sleep period, so no need to remove it
                # If no sleep event found after the bad event, mark all the epochs from the
                # previous sleep event to the epoch that contains the bad event
                # --------------------------------------------------------------------------
                if sleep_epoch_prev == None: continue
                if sleep_epoch_next == None: sleep_epoch_next = sleep_epoch_prev + ceil((bad_epoch-sleep_epoch_prev)/window_length + 1) * window_length 

                # --------------------------------------------------------------------------
                # Check if the previous sleep event has a full epoch without including the 
                # bad event. If so, the bad event epoch is not inside the previous sleep epoch
                # --------------------------------------------------------------------------
                while (bad_epoch - sleep_epoch_prev) > window_length:
                    sleep_epoch_prev += window_length

                # --------------------------------------------------------------------------
                # Put all the epochs between the earlier and next sleep events epochs in the
                # ignorable list
                # --------------------------------------------------------------------------
                ignorable_epochs += [epoch for epoch in range(sleep_epoch_prev, min(end_data, sleep_epoch_next), window_length)]
                
                if len(ignorable_epochs) == 0: pdb.set_trace()
    
    return sorted(set(ignorable_epochs))

# --------------------------------------------------------------------------
def extractBadEpochs(loading_dir):
    
    # --------------------------------------------------------------------------
    # Set subject folder and bad events folder
    # --------------------------------------------------------------------------
    eeg_folder = loading_dir.eeg_folder
    bad_events_file_path = loading_dir.bad_events_file_path
    bad_epochs_folder =loading_dir.bad_epochs_folder

    # --------------------------------------------------------------------------
    # Load subjects files, and these subject files need to be in one place
    # --------------------------------------------------------------------------
    subject_files = loadSampleNames(loading_dir.eeg_folder)
    
  
    # --------------------------------------------------------------------------
    # Extract bad epochs for every subject
    # --------------------------------------------------------------------------
    counter = 0
    for file_name in subject_files:

        #if file_name.split('_')[0] != '19-1930': continue

        counter += 1

        printDec(file_name + ', ' + str(counter))

        data_file_path = f"{eeg_folder}/{file_name}.edf"
        bad_epochs_file_path = f"{bad_epochs_folder}/{file_name}_be.txt"

        ignorable_epochs = markBadEpochs(data_file_path, bad_events_file_path)

        # --------------------------------------------------------------------------
        # Save the ignorable epochs for a subject
        # --------------------------------------------------------------------------
        if len(ignorable_epochs): writeDataTableAsText(np.array(ignorable_epochs)[:, None], bad_epochs_file_path)

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # Obtain the directories for bad-epoch identifincation
    # --------------------------------------------------------------------------
    loading_dir = EEG_sleep_dir()
    extractBadEpochs(loading_dir)