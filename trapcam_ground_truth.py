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

def show_image_as_it_is_being_scored(image, request_str, request_index, response_str, wait_int):
    image_copy = cv2.putText(image, request_str+':  '+response_str, (0,30+40*request_index), font, 0.5, (255,255,255),1, cv2.LINE_AA)
    cv2.imshow("image", image_copy)
    wait_key_val = cv2.waitKey(wait_int) & 0xFF
    if wait_key_val == ord('q'):
        return 'break'
    if wait_key_val == ord('b'): # user wants to step backwards
        return 'step back'

    try:
        tens_place = int(chr(int(wait_key_val)))
        wait_key_val_2 = cv2.waitKey(wait_int) & 0xff
        if wait_key_val_2 == ord('q'):
            return 'break'
        if wait_key_val_2 == ord('b'): # user wants to step backwards
            return 'step back'
        try:
            ones_place = int(chr(int(wait_key_val_2)))
        except:
            return 'other key pressed'

        count = 10*tens_place + ones_place
        return count
    except:
        return 'other key pressed'

def score_images_manually(im, request_list):
    response_list = []
    for (i, request_str) in enumerate(request_list):
        ret = show_image_as_it_is_being_scored(image= im, request_str = request_str, request_index = i, response_str = '', wait_int = 0)
        if ret == 'break':
            return ret
        elif ret == 'step back':
            return ret
        elif ret == 'other key pressed':
            return ret
        else:
            response = str(ret)
            response_list.append(response)
            if len(response_list)>(len(request_list)-1): #if the user has answered all the questions in the request list, then...
                return response_list
            ret = show_image_as_it_is_being_scored(image= im, request_str = request_str, request_index = i, response_str = response, wait_int =1)

####################################################################################33
params = ground_truth_json[0]["common to all traps"]
frame_sel_params = params["frame selection parameters"]
trap_to_ground_truth = 'trap_'+raw_input("Enter letter of trap to ground-truth: ")

request_list = ['flies on trap', 'press any number for next frame '] # <----- likely to be invariant; this specifies what the ground-truthing person should be looking for.
#request_list = ['flies on trap', 'flies in trap', 'non flies', 'press any number for next frame ']

for trap_name in analysis_parameters_json: # for each trap name:
    if trap_name != trap_to_ground_truth:
        continue

    all_json_entries_for_this_trap = {}
    analysis_params = analysis_parameters_json[trap_name]["analysis parameters"]
    print ('analyzing ' +trap_name)
    timelapse_directory = directory +'/trapcam_timelapse/'+trap_name
    square_mask = load_mask(square_mask_path = timelapse_directory+'/mask.jpg')
    pre_release_filename_list = [] # list of filenames in the specified pre-release time range; frames for manual ground truthing will be drawn from this list
    post_release_filename_list = [] #same as above, but for the specified post-release time range.
    full_filename_list = get_filenames(path = timelapse_directory, contains = "tl", does_not_contain = ['th']) # this is the full list of image filenames in the folder
    for filename in full_filename_list:
        #print (filename)
        time_since_release = get_time_since_release_from_filename(release_time = field_parameters["time_of_fly_release"], name = filename, offset = analysis_params["camera time advanced by __ sec"])

        if time_since_release < 60* frame_sel_params["pre-release time range"][0]:
            continue
        if time_since_release < 60* frame_sel_params["pre-release time range"][1]:
            pre_release_filename_list.append(filename)
            continue
        if time_since_release < 60* frame_sel_params["post-release time range"][0]:
            continue
        if time_since_release < 60* frame_sel_params["post-release time range"][1]:
            post_release_filename_list.append(filename)

    consec_frames = frame_sel_params["display __ consecutive frames per sample"]

    pre_samp_num  = frame_sel_params["number of pre-release samples"]
    starting_frame_indices = random.sample(np.arange(len(pre_release_filename_list)-consec_frames), pre_samp_num)
    all_pre_filenames_to_score = []
    for i in starting_frame_indices:
        l = pre_release_filename_list[i:i+consec_frames]
        all_pre_filenames_to_score.append(l)

    post_samp_num  = frame_sel_params["number of post-release samples"]
    starting_frame_indices = random.sample(np.arange(len(post_release_filename_list)-consec_frames), post_samp_num)
    all_post_filenames_to_score = []
    for i in starting_frame_indices:
        l = post_release_filename_list[i:i+consec_frames]
        all_post_filenames_to_score.append(l)

    all_filenames_to_score = all_pre_filenames_to_score + all_post_filenames_to_score
    random.shuffle(all_filenames_to_score)
    ### kjl edit 10 March 2019
    all_filenames_to_score =  [item for sublist in all_filenames_to_score for item in sublist] #flattening list
    all_images_to_score = [load_color_image(x) for x in all_filenames_to_score]
    all_masked_images_to_score =[cv2.resize(cv2.bitwise_and(img,img,mask = square_mask), (1296,972)) for img in all_images_to_score]

    ######################################  now entering loop for displaying images and scoring ################################################

    save_data_for_this_trap = True
    frame_pos = 0
    while True:
        if frame_pos < 0:
            print ('index '+str(frame_pos)+ ' is out of range!')
            break
        try:
            current_image_orig = all_masked_images_to_score[frame_pos]
            current_image = current_image_orig.copy()
            f = all_filenames_to_score[frame_pos]
        except:
            print ('index '+str(frame_pos)+ ' is out of range!')
            break

        ret = score_images_manually(current_image, request_list)
        if ret == 'break':
            break
        if ret == 'step back':
            print ('stepping back')
            frame_pos += -1
            del all_json_entries_for_this_trap[all_filenames_to_score[all_filenames_to_score.index(f) -1]] #delete the previous entry
            continue
        if ret == 'other key pressed':
            print ('non-integer key pressed; stepping back')
            frame_pos += -1
            del all_json_entries_for_this_trap[all_filenames_to_score[all_filenames_to_score.index(f) -1]] #delete the previous entry ["foo", "bar", "baz"].index("bar")
            continue
        else:
            print ('')
            for response in ret:
                print response

            response_dict = {request_list[i]:ret[i] for i in range(len(ret)-1)}
            time_dict = {"time since release": get_time_since_release_from_filename(release_time = field_parameters["time_of_fly_release"], name = f, offset = analysis_params["camera time advanced by __ sec"])}
            response_dict.update(time_dict)
            all_json_entries_for_this_trap[f] = response_dict

            frame_pos += 1

    cv2.destroyAllWindows()
    print (trap_name)
    print (len(all_json_entries_for_this_trap)) # a dictionary of dictionaries, with one dictionary per frame, keyed by

    if save_data_for_this_trap:
        with open(directory+'/all_traps_ground_truth.json') as f:
            growing_json = json.load(f)

        growing_json.append({trap_name: all_json_entries_for_this_trap})
        with open(directory+'/all_traps_ground_truth.json', mode = 'w') as f:
            json.dump(growing_json,f)

###############################################################################################################################
