# Major extrinsic sleep EEG artifact detection-Image

## Abstract

Polysomnography is an overnight sleep study used for diagnosis of sleep disorders. In
polysomnography, a patient’s brain activity is measured using electroencephalography
(EEG) through six leads placed on the scalp: frontal (F3 and F4), central (C3 and C4),
and occipital (O1 and O2). Large extrinsic or non-physiological artifacts are caused by
movement and loose leads (channels). These can distort EEG measurements, but manual
checking for such artifacts is prohibitively time-consuming. We developed a method to
automatically identify large extrinsic artifacts in an EEG trace.


## Inputs

There are three possible ways provided, to feed the inputs. 
- Using the cmd to provide the i/p and o/p directory.
- Using the script via installing the package/ with directory assignment.
- Under construction, Based on GUI, for placing the i/p and o/p directories.

pip install sleep_EEG_loose_lead_detect-0.0-py3-none-any.whl

### Example code
Under the "test/" folder
```bash
python all_in_cmd.py -i /Documents/Data_Sleep/EDF_file_directory/ -o /Documents/output_directory/ -edf /Documents/check_loose_lead/edf_list.txt
```
 
If the users prefer the script to run or more EEG(.edf) files with more control.
- Install through the 
```bash
cd .../EEG_PSG_loose_lead_test/dist/
pip install sleep_EEG_loose_lead_detect-0.0-py3-none-any.whl
```
- Use the directory assignment.

Then use the "script_pip_installed_in_one_outlier_variance_inclusive.py''.

Let us first refer to the tool-related inputs provided in the "Example code".
- ***-i /Documents/Data_Sleep/EDF_file_directory/***: This is the  main EEG files (".edf") location 
- ***-o /Documents/output_directory/***: This is the main output location, all the results finally end up here. 
- ***-edf /Documents/check_loose_lead/edf_list.txt***: This contains the list of (".edf") files the tool is supposed to check. If this input not provided all the ".edf" files in the input directory going to be accessed by the tool.

To see the help related to possible inputs 
```bash
-h
```

You see something like this as an output 
```console
optional arguments:
  -h, --help            show this help message and exit
  -i INLOC, --inloc INLOC
                        Input directory relative to the mount using docker file, this directory holds the .edf file or .edf files
  -o OUTLOC, --outloc OUTLOC
                        Main o/p directory relative to the muount using docker file, this dierctory will hold the tool obtained results
  -edf EDFLIST, --edflist EDFLIST
                        file name for list of edf file/ files need to detect the artifacts, in a text file with comma seperated
  -opt OPTIONS, --options OPTIONS
                        file name for options to the tools this can be used to vary the default parmaters under construction
  -ev {0,1}, --event {0,1}
                        file name for saving the final results in events with sleep-stage related epoches
  -dic {0,1}, --dictionary {0,1}
                        Save the meta data from the edf in dictionary format as pickle
  -b {0,1}, --bad_epochs {0,1}
                        Check th ebad epoches in the provided edf to focus only in the sleep-related signals
  -sleep {0,1}, --sleep_annot {0,1}
                        save the sleep-annotation in npy file
  -out {0,1}, --outlier {0,1}
                        save the predicted outliers
  -MT {0,1}, --MultiTaper {0,1}
                        save the predicted multitapers outcome
  -outNREM {0,1}, --outlierNREM_REM {0,1}
                        save the predicted outliers as NREM and REM
  -sp {0,1}, --spindles {0,1}
                        First predict the spindles via the YASA, then avoid the spindles while predicting the outliers
  -la {0,1}, --Latex {0,1}
                        Save the figures with the predicted outcome for Latex
  -all {0,1}, --sel_all {0,1}
                        select all the options for saving the directory
 ```

## Docker
Apart from this we are also providing [Docker](https://hub.docker.com/r/nishyanand/loose_lead_test).

## save modified EEG file as EDF

Please check the files on EDF modifier project and docker documentation.

