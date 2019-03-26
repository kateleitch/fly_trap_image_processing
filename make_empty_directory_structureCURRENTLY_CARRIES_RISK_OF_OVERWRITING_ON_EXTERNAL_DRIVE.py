#! /usr/bin/python

from __future__ import print_function
import cv2 # opencv
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import itertools
import time
import json
from subprocess import call
from subprocess import check_output


path_to_external_drive = '/media/kate/My Passport/coyote_lake_field_data_only_on_external_drives/'

#call(["MP4Box", "-splitx", splitx_string, output_name, "-out", output_name.split('.')[0]+'_'+str(k)+'.'+output_name.split('.')[1]])
pwd = check_output(["pwd"]).split('\n')[0]
print ('')
print('your current working directory is: ' + pwd)


if pwd.split('/')[-1]== 'coyote_lake_field_data':
    keep_going = raw_input('good, looks like you are in the directory immediately above all the experiment-date directories; are you sure this is the case? (y/n) ')
    if keep_going == 'y':
        print ('')
        print ('looks like the following experiment-date directories already exist: ')
        folders = os.listdir(pwd)
        for folder in folders:
            if '201' in folder:
                print (folder)
        add_new_folder =raw_input('Do you want to make a new experiment-date directory? (y/n) ')
        if add_new_folder == 'y':
            new_folder_name = raw_input('OK, enter the name of the new experiment-date directory: (yyyy_mm_dd) ')
            call(["mkdir", new_folder_name])
            call(['mkdir', path_to_external_drive+new_folder_name])

        dict_of_traps_with_mini_vane_data={}
        mini_vane = True
        resp = raw_input('Did you acquire any mini-wind-vane data for this experiment? (y/n) ')
        if resp == 'n':
            mini_vane = False
        while mini_vane:
            print ('')
            trap_letter=raw_input('Enter the letter of a trap for which you have mini-wind-vane data (e.g. A): ')
            dict_of_traps_with_mini_vane_data['trap_'+trap_letter] = []
            resp = raw_input('Any more traps with wind data? (y/n) ')
            if resp == 'n':
                break

        default_trap_lettering = [str for str in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz']
        number_of_trapcam_traps = int(raw_input('How many traps did you set up to acquire timelapse data? '))
        trap_letter_list = default_trap_lettering[0:number_of_trapcam_traps]
        print ('So I assume you have traps named: ')
        for let in trap_letter_list:
            print ('trap_'+let)
        print ('')
        move_on = raw_input('Does that list of trap names look right? (y/n) ')
        if move_on == 'n':
            trap_letter_list = []
            while True:
                letter_to_append = raw_input('Enter a trap letter, or enter "finished" ')
                trap_letter_list.append(letter_to_append)
                if letter_to_append == "finished":
                    break

        list_of_all_traps_with_trapcam_timelapse_data= ['trap_'+letter for letter in trap_letter_list]

        external_drive_trapcam_timelapse_path = path_to_external_drive+ new_folder_name+'/trapcam_timelapse/'
        call(['mkdir', external_drive_trapcam_timelapse_path])
        for trap in list_of_all_traps_with_trapcam_timelapse_data:
            call(['mkdir', external_drive_trapcam_timelapse_path+trap])


        #now making symlink between external drive (where timelapse data will be stored) and my local directory structure
        call(['ln', '-s', external_drive_trapcam_timelapse_path, pwd+'/'+new_folder_name+'/'+'/trapcam_timelapse'])

        directory_dictionary = {'arrival_dynamics_figs':{},
                                'fly_rearing_records': {},
                                'ground_truth_rmse_figs':{},
                                'weather_data': {'metone_anemometer_data': {},
                                                'mini_wind_vane_data': dict_of_traps_with_mini_vane_data}}
        for key in directory_dictionary:
            call(['mkdir', new_folder_name+'/'+key])
            if len(directory_dictionary[key]) >0:
                for subkey in directory_dictionary[key]:
                    call(['mkdir', new_folder_name+'/'+key+'/'+subkey])
                    if len(directory_dictionary[key][subkey]) >0:
                        for subsubkey in directory_dictionary[key][subkey]:
                            call(['mkdir', new_folder_name+'/'+key+'/'+subkey+'/'+subsubkey])



    if keep_going == 'n':
        print ('OK, you need to make sure you are in that top-level directory.')

else:
    print ('looks like you need to get in the directory immediately above all the experiment-date dirs')
