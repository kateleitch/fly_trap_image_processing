#! /usr/bin/python

import cv2 # opencv
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import itertools
import time
import json
from pylab import *
from scipy.optimize import curve_fit
import random
font = cv2.FONT_HERSHEY_SIMPLEX

directory = sys.argv[1]

with open(directory+'/all_traps_ground_truth.json') as f:
    ground_truth_json = json.load(f)
with open(directory+'/all_traps_gaussian_analysis_params.json') as f:
    analysis_parameters_json = json.load(f)
with open(directory+'/field_parameters.json') as f:
    field_parameters = json.load(f)

def load_color_image(filename):
    header = "\xff\xd8"
    tail = "\xff\xd9"
    with open(filename, "rb") as image:
        data = image.read()
        try:
            start = data.index(header)
            end = data.index(tail, start) + 2
        except ValueError:
            print ("Can't find JPEG data!")
            return None
    img = cv2.imread(filename)
    return img

def load_mask(square_mask_path):
    mask = cv2.imread(square_mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = mask/255 #rescales mask to be 0s and 1s
    return mask

def get_filenames(path, contains, does_not_contain=['~', '.pyc']):
    cmd = 'ls ' + '"' + path + '"'
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    try:
        all_filelist.remove('')
    except:
        pass
    filelist = []
    for i, filename in enumerate(all_filelist):
        if contains in filename:
            fileok = True
            for nc in does_not_contain:
                if nc in filename:
                    fileok = False
            if fileok:
                filelist.append( os.path.join(path, filename) )
    return filelist

def get_time_since_release_from_filename (release_time='', name = '', offset = 0):
    frame_time_string = name.split('.')[-2].split('_')[-1]
    frame_hour = int(frame_time_string[0:2])
    frame_min = int(frame_time_string[2:4])
    frame_sec = int(frame_time_string[4:6])
    frame_seconds_timestamp = (frame_hour)*3600 + (frame_min)*60 + (frame_sec)
    frame_seconds = frame_seconds_timestamp - offset
    release_time_seconds = int(release_time.split(':')[0])*3600 +int(release_time.split(':')[1])*60 + int(release_time.split(':')[2])
    time_elapsed = frame_seconds - release_time_seconds
    return time_elapsed



def score_images_manually(index, samp_num, pre_release_filename_list, square_mask):
    try:
        filename = pre_release_filename_list[samp_num+index]
    except:
        print ('out of range!') # this error handling is not fully thought-through
        #break
    img = load_color_image(filename)
    if img is None:
        print ('img is None!')
        #continue
    else:
        masked_image=(cv2.bitwise_and(img,img,mask = square_mask))

    ##### here, want to do the following:
    # 1. show image, maybe annotated with "consec: str(samp_num)"
    masked_image_resized = cv2.resize(masked_image, (1296,972)) #halves image dimensions just for display purposes
    #masked_image_resized = masked_image_resized[:,150:-150].copy()

    # while True:

    print (str(index)+', '+str(samp_num+1))

    textstr = 'Number of flies ON trap: '
    img = cv2.putText(masked_image_resized.copy(), textstr, (0,30), font, 0.5, (255,255,255),1, cv2.LINE_AA)
    cv2.imshow("image", img)
    #wait_key_val = cv2.waitKey(0) & 0xFF
    wait_key_val = cv2.waitKey(0)% 256
    if wait_key_val == ord('q'):
        return 'break'
    if wait_key_val == ord('b'): # user wants to step backwards
        return 'step back'
    flies_on_trap = int(chr(int(wait_key_val)))
    print ('flies_on_trap: ' + str(flies_on_trap))

    new_textstr = "Number of flies ON trap: " +str(flies_on_trap)
    img = cv2.putText(masked_image_resized.copy(), new_textstr, (0,30), font, 0.5, (255,255,255),1, cv2.LINE_AA)
    new_textstr2 = "Number of flies IN trap: "
    img = cv2.putText(img, new_textstr2, (0,70), font, 0.5, (255,255,255),1, cv2.LINE_AA)
    cv2.imshow("image", img)
    wait_key_val = cv2.waitKey(0) & 0xFF
    if wait_key_val == ord('q'):
        return 'break'
    if wait_key_val == ord('b'): # user wants to step backwards
        return 'step back'
    flies_in_trap = int(chr(int(wait_key_val)))
    print ('flies_in_trap: ' + str(flies_in_trap))

    new_textstr = "Number of flies ON trap: " +str(flies_on_trap)
    img = cv2.putText(masked_image_resized.copy(), new_textstr, (0,30), font, 0.5, (255,255,255),1, cv2.LINE_AA)
    new_textstr2 = "Number of flies IN trap: "+str(flies_in_trap)
    img = cv2.putText(img, new_textstr2, (0,70), font, 0.5, (255,255,255),1, cv2.LINE_AA)
    img = cv2.putText(img, 'Number of non-drosophila: ', (0,110), font, 0.5, (255,255,255),1, cv2.LINE_AA)
    cv2.imshow("image", img)
    wait_key_val = cv2.waitKey(0) & 0xFF
    if wait_key_val == ord('q'):
        return 'break'
    if wait_key_val == ord('b'): # user wants to step backwards
        return 'step back'
    non_flies = int(chr(int(wait_key_val)))

    new_textstr = "Number of flies ON trap: " +str(flies_on_trap)
    img = cv2.putText(masked_image_resized.copy(), new_textstr, (0,30), font, 0.5, (255,255,255),1, cv2.LINE_AA)
    new_textstr2 = "Number of flies IN trap: "+str(flies_in_trap)
    img = cv2.putText(img, new_textstr2, (0,70), font, 0.5, (255,255,255),1, cv2.LINE_AA)
    new_textstr3 = 'Number of non-drosophila: ' + str(non_flies)
    img = cv2.putText(img, new_textstr3, (0,110), font, 0.5, (255,255,255),1, cv2.LINE_AA)
    img = cv2.putText(img, 'press any key for next frame', (0,150), font, 0.5, (255,255,255),1, cv2.LINE_AA)
    cv2.imshow("image", img)
    wait_key_val = cv2.waitKey(0) & 0xFF
    if wait_key_val == ord('q'):
        return 'break'

    return [flies_on_trap, flies_in_trap, non_flies]
######### first, to loop through sample frames for all traps and ask a user to score the number of flies in/on trap for each frame:
params = ground_truth_json["common to all traps"]

frame_sel_params = params["frame selection parameters"]
all_json_entries_for_this_trap = []
for trap_name in analysis_parameters_json: # for each trap name:
    analysis_params = analysis_parameters_json[trap_name]["analysis parameters"]
    print ('analyzing ' +trap_name)
    timelapse_directory = directory +'/trapcam_timelapse/'+trap_name
    square_mask = load_mask(square_mask_path = timelapse_directory+'/mask.jpg')
    pre_release_filename_list = [] # list of filenames in the specified pre-release time range; frames for manual ground truthing will be drawn from this list
    post_release_filename_list = [] #same as above, but for the specified post-release time range.
    full_filename_list = get_filenames(path = timelapse_directory, contains = "tl", does_not_contain = ['th']) # this is the full list of image filenames in the folder
    for filename in full_filename_list:
        print (filename)
        time_since_release = get_time_since_release_from_filename(release_time = field_parameters["time_of_fly_release"], name = filename, offset = analysis_params["camera time advanced by __ sec"])

        if time_since_release < 60* frame_sel_params["pre-release time range"][0]:
            continue
        if time_since_release < 60* frame_sel_params["pre-release time range"][1]:
            pre_release_filename_list.append(filename)
            print ('appending to pre list: ' + filename)
            continue
        if time_since_release < 60* frame_sel_params["post-release time range"][0]:
            continue
        if time_since_release < 60* frame_sel_params["post-release time range"][1]:
            post_release_filename_list.append(filename)
            print ('appending to post list: ' + filename)

        # {"filename": ".jpg", "time since release": 0, "flies on trap": 0, "flies in trap": 0, "non flies": 0, "reject frame entirely?": false}
    consec_frames = frame_sel_params["display __ consecutive frames per sample"]

    pre_samp_num  = frame_sel_params["number of pre-release samples"]
    starting_frame_indices = random.sample(np.arange(len(pre_release_filename_list)), pre_samp_num)
    break_toggle = 0
    step_back_toggle = 0
    samp_num_memory = 0
    for idx in np.arange(len(starting_frame_indices)):
        index = starting_frame_indices[idx]
        print ('')
        if break_toggle == 1:
            break
        if step_back_toggle ==1:
            index = starting_frame_indices[idx -1] # normally, step_back_num is 0, but when the user wants to step backwards it'll be -1
            print ('moving back to index: ' +str(index))
            #going back to previous starting frame and all its samples; want to delete all previous entries for the set of frames
            del all_json_entries_for_this_trap[-1*samp_num_memory:]
            step_back_toggle = 0

        for samp_num in np.arange(pre_samp_num):
            ret = score_images_manually(index, samp_num, pre_release_filename_list, square_mask)
            if ret == 'break':
                break_toggle = 1
                break
            if ret == 'step back':
                print ('revisiting previous frame')
                step_back_toggle = 1
                samp_num_memory = samp_num
                break
            else:
                json_entry = {"filename": filename,
                                "time since release": get_time_since_release_from_filename(release_time = field_parameters["time_of_fly_release"], name = filename, offset = analysis_params["camera time advanced by __ sec"]),
                                "flies on trap": ret[0],
                                "flies in trap": ret[1],
                                "non flies": ret[2]}
                all_json_entries_for_this_trap.append(json_entry)

    post_samp_num = frame_sel_params["number of post-release samples"]
    starting_frame_indices = random.sample(np.arange(len(post_release_filename_list)), post_samp_num)
    toggle = 0
    for index in starting_frame_indices:
        print ('')
        if toggle == 1:
            break
        for samp_num in np.arange(post_samp_num):
            ret = score_images_manually(index, samp_num, post_release_filename_list, square_mask)
            if ret == 'break':
                toggle = 1
                break
            else:
                json_entry = {"filename": filename,
                                "time since release": get_time_since_release_from_filename(release_time = field_parameters["time_of_fly_release"], name = filename, offset = analysis_params["camera time advanced by __ sec"]),
                                "flies on trap": ret[0],
                                "flies in trap": ret[1],
                                "non flies": ret[2]}
                all_json_entries_for_this_trap.append(json_entry)

    cv2.destroyAllWindows()
    print (trap_name)
    print (all_json_entries_for_this_trap) # a list of dictionaries, with one dictionary per frame



#
#
# "trap_X": {"ground truthed frames": [{"filename": ".jpg", "time since release": 0, "flies on trap": 0, "flies in trap": 0, "non flies": 0}], "in-trap counts per parameter": [], "on-trap counts per parameter": [], "in-trap RMSE":[], "on-trap RMSE":[]},


# {"frame selection parameters":{"number of pre-release samples":5,
#                                                         "number of post_release samples":20,
#                                                       "display __ consecutive frames per sample": 5,
#                                                       "pre-release time range": [-20, 0],
#                                                       "post-release time range": [1,40]},
#                           "parameters to test": {"mahalanobis squared threshold":[10,11,12,13,14,15,16,17,18,19,20,25],
#                                                 "minimum contour size":[5,6,7,8,9,10,11,12,13,14]}},
