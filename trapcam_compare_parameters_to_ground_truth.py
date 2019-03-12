#! /usr/bin/python
from __future__ import print_function
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
import trapcam_analysis as trapcam_analysis
import seaborn as sns

font = cv2.FONT_HERSHEY_SIMPLEX

def unique_pairs(list1, list2):
    """Produce all pairs between these two lists, as a way to explore parameter space """
    for i1 in list1:
        for i2 in list2:
            yield i1, i2

directory = sys.argv[1]
with open(directory+'/all_traps_ground_truth.json') as f:
    ground_truth_json = json.load(f)
with open(directory+'/all_traps_gaussian_analysis_params.json') as f:
    analysis_parameters_json = json.load(f)
with open(directory+'/field_parameters.json') as f:
    field_parameters = json.load(f)

params_to_test = ground_truth_json[0]["common to all traps"]["parameters to test"]
mahalanobis_list = params_to_test["mahalanobis squared threshold"]
minimum_contour_list = params_to_test["minimum contour size"]

ground_truth_start_minutes_before_release = ground_truth_json[0]["common to all traps"]["frame selection parameters"]["pre-release time range"][0]
ground_truth_end_minutes_after_release = ground_truth_json[0]["common to all traps"]["frame selection parameters"]["post-release time range"][1]

analyzer = trapcam_analysis.TrapcamAnalyzer(directory)

################################ this is dorky but I think it'll become useful when I'm dealing with lots of data #######################################
while True:
    reference_trap_list = []
    while True:
        letter = raw_input("Enter a(nother) trap letter to analyze, or type 'go' to start batch analysis: ")
        if letter == 'go':
            break
        else:
            reference_trap_list.append('trap_'+letter)
    print ('you said you want to analyze: ')
    for ref_trap in reference_trap_list:
        print (ref_trap)
    user_go_ahead = raw_input("Are those the traps you'd like to analyze? (y/n) ")
    if user_go_ahead == 'y':
        break
    if user_go_ahead == 'n':
        continue
##########################################################################################################################################################


for trap_name in analysis_parameters_json: # for each trap name:
    namestr = directory+'/'+trap_name+'_on_trap_rmse.png'

    plt.close('all')
    if trap_name not in reference_trap_list: # this is a temporary measure for troubleshooting!!!!
        continue

    all_json_entries_for_this_trap = {} # output to populate

    print ('analyzing ' +trap_name)
    timelapse_directory = directory +'/trapcam_timelapse/'+trap_name
    analysis_params = analysis_parameters_json[trap_name]['analysis parameters']
    square_mask = analyzer.load_mask(square_mask_path = timelapse_directory+'/mask.jpg')

    ############################ possibly not relevant till later ##################################
    for index, dict in enumerate(ground_truth_json): # this is awkward but I need something clunky here because the ground_truth_json is actually a list at the top level
        if trap_name in dict.keys():
            current_index = index
            break
    ground_truthed_frames_for_this_trap = ground_truth_json[current_index][trap_name] #yields a dictionary of dictionaries, keyed by filename
    ground_truthed_filename_list = [str(key) for key in ground_truthed_frames_for_this_trap]
    ################################################################################################

    masked_image_stack = []
    filename_list = []
    full_filename_list = analyzer.get_filenames(path = timelapse_directory, contains = "tl", does_not_contain = ['th']) #  full list of image filenames in the folder
    print ('length of full filename list: ' +str(len(full_filename_list)))

    train_num = analysis_params['number of frames for bg model training']
    buffer_btw_training_and_test = analysis_params['frames between training and test']

    for ground_truthed_filename in ground_truthed_filename_list:
        all_json_entries_for_this_filename = {}

        full_filename_index = np.where(np.array(full_filename_list) == ground_truthed_filename)[0][0]
        sublist_to_analyze= (full_filename_list[full_filename_index-(train_num + buffer_btw_training_and_test+1):full_filename_index+1])
        masked_images_to_analyze = [cv2.bitwise_and(analyzer.load_color_image(i),analyzer.load_color_image(i), mask = square_mask) for i in sublist_to_analyze]

#        using this mahal and min_cont, analyze the subset of files that had been ground-truthed
        for mahal, min_cont in unique_pairs(mahalanobis_list, minimum_contour_list):
            print ('')
            print ('frame %d out of %d' %( ground_truthed_filename_list.index(ground_truthed_filename), len(ground_truthed_filename_list)))
            print ("mahal: "+str(mahal))
            print ("min cont: "+str(min_cont))
            all_flies_over_time, annotated_output_stack, time_since_release_list, analyzed_filename_stack, all_contrast_metrics = analyzer.find_contours_using_pretrained_backsub_MOG2(full_image_stack = masked_images_to_analyze,
                                                                       filename_stack = sublist_to_analyze,
                                                                       train_num = analysis_params['number of frames for bg model training'],
                                                                       mahalanobis_squared_thresh = mahal,
                                                                       buffer_btw_training_and_test = analysis_params['frames between training and test'], # if too high, shadows get bad
                                                                       minimum_contour_size = min_cont,
                                                                       maximum_contour_size = analysis_params['maximum_contour_size'],
                                                                       time_of_fly_release = field_parameters["time_of_fly_release"],
                                                                       camera_time_offset = analysis_params['camera time advanced by __ sec'],
                                                                       ontrap_intrap_threshold = analysis_params["threshold to differentiate in- and on- trap"])


            json_entry = { "mahalanobis squared thresh": mahal,
                            "minimum contour size": min_cont,
                            "flies in trap": len(all_flies_over_time[-1]['flies in trap']) ,
                            "flies on trap": len(all_flies_over_time[-1]['flies on trap']),
                            "non flies":     len(all_flies_over_time[-1]['not_flies']),
                            #"flies-in-trap square error": (len(all_flies_over_time[-1]['flies in trap'])  - int(ground_truthed_frames_for_this_trap[ground_truthed_filename]["flies in trap"]))**2,
                            #"non-flies square error":     (len(all_flies_over_time[-1]['not_flies'])      - int(ground_truthed_frames_for_this_trap[ground_truthed_filename]["non flies"]))**2,
                            "flies-on-trap square error": (len(all_flies_over_time[-1]['flies on trap'])  - int(ground_truthed_frames_for_this_trap[ground_truthed_filename]["flies on trap"]))**2}
            all_json_entries_for_this_filename[(mahal,min_cont)] = json_entry
        all_json_entries_for_this_trap[ground_truthed_filename] = all_json_entries_for_this_filename
    print (all_json_entries_for_this_trap)

    ##### Okayyyy now for this trap I have a dictionary with each frames' residuals (analyzed - ground truthed) for each parameter pair and for each designation (in trap, on trap, non fly)
    # should save this dictionary as a json, or maybe not

    # should calculate the RMSE per parameter pair, per designation (starting with on-trap, which is most interesting to me right now), across all ground-truthed frames.

    on_trap_squared_error_sum_dict = {}
    in_trap_squared_error_sum_dict = {} # for now, ignoring this
    non_fly_squared_error_sum_dict = {} # for now, ignoring this
    for f in all_json_entries_for_this_trap:
        for tuple_key in all_json_entries_for_this_trap[f]:
            print (tuple_key)
            if tuple_key not in on_trap_squared_error_sum_dict:
                on_trap_squared_error_sum_dict[tuple_key] = 0

            print('adding the following squared error: '+str(all_json_entries_for_this_trap[f][tuple_key]["flies-on-trap square error"]))
            on_trap_squared_error_sum_dict[tuple_key] += all_json_entries_for_this_trap[f][tuple_key]["flies-on-trap square error"]
    print (on_trap_squared_error_sum_dict)

    trap_rmse_array = np.zeros([len(mahalanobis_list), len(minimum_contour_list)])
    for tuple_key in on_trap_squared_error_sum_dict:
        mahal_idx = np.where(np.array(mahalanobis_list) ==tuple_key[0])[0][0]
        min_idx = np.where(np.array(minimum_contour_list) == tuple_key[1])[0][0]
        trap_rmse_array[mahal_idx][min_idx] = np.sqrt(on_trap_squared_error_sum_dict[tuple_key])/np.sqrt(len(all_json_entries_for_this_trap))

    print (trap_rmse_array)

    fig = plt.figure(figsize=(8,8), facecolor="white")
    ax = fig.add_subplot(111)
    ax = sns.heatmap(trap_rmse_array, linewidth =0.5, cmap = 'viridis', yticklabels=[str(x)for x in mahalanobis_list ], xticklabels= [str(x)for x in minimum_contour_list], vmax = np.percentile(trap_rmse_array,85))

    plt.ylabel('sq mahalanobis distance threshold')
    plt.xlabel('minimum fly contour size')
    plt.title(trap_name)
    ax.collections[0].colorbar.set_label("on-trap count RMSE")
    namestr = directory+'/ground_truth_rmse_figs/'+trap_name+'_on_trap_rmse_'+str(len(ground_truthed_filename_list))+'_gt_frames.png'
    plt.savefig(namestr, bbox_inches='tight')
    plt.show()

# 6. for all traps, generate 2 heat maps of RMSEs -- one for in-trap counts, one for on-trap counts.
# 7. by eye, look at all heat maps and decide for each trap what parameter pair minimizes error relative to ground truth; prioritize minimization of ON-TRAP error
# 8. using this fixed parameter pair, then iterate through each trap to automatically determine (by fitting histogram of contrast metrics) an optimal threshold to classify in-trap vs. on-trap flies
