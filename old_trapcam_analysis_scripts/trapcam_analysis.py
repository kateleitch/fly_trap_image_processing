#!/usr/bin/python
from __future__ import print_function
import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import scipy.stats
import json
import time
import itertools
import random

directory = sys.argv[1]
with open(directory+'/all_traps_analysis_params.json') as f:
    analysis_parameters = json.load(f)
with open(directory+'/field_parameters.json') as f:
    field_parameters = json.load(f)
with open(directory+'/trap_layout_parameters.json') as f:
    trap_layout_parameters = json.load(f)

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

def load_image_file_and_convert_to_gray(filename): #square mask is trap-specific
    img = cv2.imread(filename)
    if img is None:
        return img
    else:
        gimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # fill in top rows with black to hide timestamp
        gimg[0:100,:] = 0
        return gimg

def load_mask(square_mask_path):
    mask = cv2.imread(square_mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = mask/255 #rescales mask to be 0s and 1s
    return mask

def load_image_file(filename):
    img = cv2.imread(filename)
    img[0:100,:] = 0
    return img

def load_image_stack(filelist, square_mask_path):
    square_mask = load_mask(square_mask_path)
    image_stack = []
    for filename in filelist:
        image_stack.append(load_image_file_and_convert_to_gray(filename)*square_mask)
    return image_stack

def fit_ellipse_to_contour(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        area = cv2.contourArea(contour)
        return cx, cy, area
    else:
        return None
    '''
    ellipse = cv2.fitEllipse(contour)
    (x,y), (a,b), angle = ellipse
    a /= 2.
    b /= 2.
    ecc = np.min((a,b)) / np.max((a,b))
    '''
    return x, y, area

def smooth_image(thresh_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #looks like the same kernel is used for morphologyEx, dilate, and erode; maybe in the future it'll be useful to have different kernel sizes
    thresh_img_smooth = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations = 1)
    thresh_img_smooth = cv2.dilate(thresh_img_smooth, kernel, iterations=10) # make the contour bigger to join neighbors
    thresh_img_smooth = cv2.erode(thresh_img_smooth, kernel, iterations=4) # make contour smaller again
    return thresh_img_smooth

def get_contours_from_frame_and_stack(image_stack, frame_to_analyze, threshold=25):
    # get the median image
    med_img = np.median(image_stack, axis=0) # THIS IS THE BOTTLENECK. THIS TAKES AROUND 0.13 SECONDS, LIKE AN ORDER OF MAGNITUDE LONGER THAN ANYTHING ELSE IN THIS FUNCTION

    # find all the dark objects in the image compared to the median
    #adaptive_thresh = {'neighborhood': 141, 'offset': 40}
    # adaptive_thresh = {}
    # if adaptive_thresh:
    #     delta_img = cv2.compare(np.float32(image_stack[frame_to_analyze]), np.float32(med_img), cv2.CMP_LT)
    #     thresh_img = cv2.adaptiveThreshold(delta_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                                 cv2.THRESH_BINARY_INV,
    #                                                 adaptive_thresh["neighborhood"],
    #                                                 adaptive_thresh["offset"])

    thresh_img = cv2.compare(np.float32(image_stack[frame_to_analyze]), np.float32(med_img)-threshold, cv2.CMP_LT)

    # CMP_LT is less than
    # smooth the image a little. play around with the kernel size (specified above) to get desired effect
    thresh_img_smooth = smooth_image(thresh_img)
    # grab the outer contour of the blobs. Note, this is for opencv version 3.
    image, contours, hierarchy = cv2.findContours(thresh_img_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def count_flies(image_stack, frame_to_analyze, minimum_contour_size=50, maximum_contour_size=200, threshold=25):

    contours = get_contours_from_frame_and_stack(image_stack, frame_to_analyze, threshold=threshold)
    flies = []
    not_flies = []
    for contour in contours:
        if len(contour) > 5: # smaller contours can't be turned into ellipses.
            x, y, area = fit_ellipse_to_contour(contour)
            fly = {'x': x, 'y': y, 'area': area}
            if area < maximum_contour_size and area > minimum_contour_size:
                flies.append(fly)
            else:
                not_flies.append(fly)

    return flies, not_flies

def show_image_with_circles_drawn_around_putative_flies(color_image, flies, not_flies, square_mask_path):
    square_mask = load_mask(square_mask_path)
    fig = plt.figure(figsize=(20,12)) # change the figsize to increase/decrease the image size
    ax = fig.add_subplot(111) # one axis
    for fly in flies:
        print ('fly!')
        cv2.circle(color_image, (fly['x'], fly['y']), 50, [0,255,0], 5)
        ax.text(fly['x']+50, fly['y']+50, str(fly['area']), color = ([0,1,0]))
    for not_fly in not_flies:
        print ('not fly!')
        cv2.circle(color_image, (not_fly['x'], not_fly['y']), 50, [255,0,0], 5)
        ax.text(not_fly['x']+50, not_fly['y']+50, str(not_fly['area']), color =([1,0,0]))
    masked_color_image = cv2.bitwise_and(color_image,color_image,mask = square_mask)
    plt.imshow(masked_color_image)
    plt.show()
    time.sleep(.5)
    plt.close('all')
    print ()
    #cv2.destroyAllWindows()


def get_time_since_release_from_filename (release_time='13:30:27', name = ''):
    frame_time_string = name.split('.')[-2].split('_')[-1]
    frame_hour = int(frame_time_string[0:2])
    frame_min = int(frame_time_string[2:4])
    frame_sec = int(frame_time_string[4:6])
    frame_seconds = (frame_hour)*3600 + (frame_min)*60 + (frame_sec)
    release_time_seconds = int(release_time.split(':')[0])*3600 +int(release_time.split(':')[1])*60 + int(release_time.split(':')[2])
    time_elapsed = frame_seconds - release_time_seconds
    return time_elapsed

def find_all_frames_with_flies_in_them(dir='',
                                        filename_identifier = 'tl',
                                        does_not_contain = ['th', '.zip', 'mask'],
                                        minimum_contour_size = 120,
                                        maximum_contour_size = 650,
                                        threshold = 25,
                                        number_of_frames_for_median = 3,
                                        time_of_fly_release = '', # < ----  this will come from the "field_parameters.json" file
                                        analyze_how_many_minutes_prior_to_release = 2,
                                        analyze_how_many_minutes_post_release = 20,
                                        square_mask_path = ''):

    filelist = get_filenames(dir, filename_identifier, does_not_contain=does_not_contain)
    mask = load_mask(square_mask_path) # this takes 0.06 seconds
    results = {}
    image_stack = []
    filelist_stack = []
    timing_list = []
    for filename in filelist:
        time_since_release = get_time_since_release_from_filename(release_time = time_of_fly_release, name = filename)
        if time_since_release < -60* analyze_how_many_minutes_prior_to_release:
            continue
        if time_since_release > 60* analyze_how_many_minutes_post_release:
            break
        gimg = load_image_file_and_convert_to_gray(filename) # this takes 0.02 seconds
        if gimg is None:
            continue
        masked_gimg = gimg*mask # this takes 0.0015 seconds
        image_stack.append(masked_gimg)
        filelist_stack.append(filename)
        analyze_frame_number = int((number_of_frames_for_median-1)/2.)
        if len(image_stack) > number_of_frames_for_median:
            image_stack.pop(0) # remove the first frame to keep total at 3
            filelist_stack.pop(0)
        if len(image_stack) == number_of_frames_for_median:
            frame_number = int(filelist_stack[analyze_frame_number].split(filename_identifier)[1].split('_')[2])
            flies, not_flies = count_flies(image_stack, analyze_frame_number,  # <------------------------------ THIS TAKES 0.15 SECONDS EACH TIME
                                           minimum_contour_size, maximum_contour_size, threshold)
            results[filename] = {'seconds post release': time_since_release, 'nflies': len(flies), 'nother': len(not_flies)}
    return results  #, filelist_stack # <---- NOT SURE WHY IT WOULD BE USEFUL TO RETURN FILELIST_STACK. WHY DID I DO THIS ORIGINALLY?

def find_SOME_frames_with_flies_in_them(dir='',
                                        filename_identifier = 'tl',
                                        does_not_contain = ['th', '.zip', 'mask'],
                                        minimum_contour_size = 120,
                                        maximum_contour_size = 650,
                                        threshold = 25,
                                        number_of_frames_for_median = 3,
                                        time_of_fly_release = '', # < ----  this will come from the "field_parameters.json" file
                                        analyze_how_many_minutes_prior_to_release = 2,
                                        analyze_how_many_minutes_post_release = 20,
                                        square_mask_path = '',
                                        number_of_example_frames_wanted = 10):

    frames_with_flies_counter = 0
    frames_with_nonflies_counter = 0
    filelist = get_filenames(dir, filename_identifier, does_not_contain=does_not_contain)
    shuffled_filelist = random.sample(filelist, len(filelist))
    square_mask = load_mask(square_mask_path) # this takes 0.06 seconds
    ground_truth_sample = {}
    filelist_stack = []
    timing_list = []
    analyze_frame_number = int((number_of_frames_for_median-1)/2.)
    n = number_of_example_frames_wanted # renamed this to keep the following line short

    for filename in shuffled_filelist:
        time_since_release = get_time_since_release_from_filename (release_time= time_of_fly_release, name = filename)
        if time_since_release < -60* analyze_how_many_minutes_prior_to_release:
            continue
        if time_since_release > 60* analyze_how_many_minutes_post_release:
            continue
        image_stack = []
        original_index = filelist.index(filename)
        filelist_indices_for_median = original_index + (np.linspace(-analyze_frame_number, analyze_frame_number, 2*analyze_frame_number+1))
        print ('filelist indices for median: ' +str(filelist_indices_for_median))
        for idx in filelist_indices_for_median:
            image_stack.append(load_image_file_and_convert_to_gray(filelist[int(idx)])*square_mask)

        print (filename)

        #if frames_with_flies_counter < n or frames_with_nonflies_counter < n or frames_without_contours_counter < n: # if ANY counter is below n, control will go into this loop
        bool_list = [frames_with_flies_counter < n, frames_with_nonflies_counter < n]
        if np.any(bool_list): # if ANY counter is below n, control will go into this loop
            print (frames_with_flies_counter, frames_with_nonflies_counter)

            flies, not_flies = count_flies(image_stack, analyze_frame_number,  # <------------------------------ THIS TAKES 0.15 SECONDS EACH TIME
                                           minimum_contour_size, maximum_contour_size, threshold)

            ground_truth_sample[filename] = {'seconds post release': time_since_release, 'nflies': len(flies), 'nother': len(not_flies)}
            if len(flies) >0:
                frames_with_flies_counter += 1
            if len(not_flies) >0:
                frames_with_nonflies_counter +=1

            color_image = cv2.imread(filename)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            fig = plt.figure(figsize=(24,8)) # change the figsize to increase/decrease the image size

            ax = fig.add_subplot(132) # one axis
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            for fly in flies:
                cv2.circle(color_image, (fly['x'], fly['y']), 50, [0,255,0], 5)
                ax.text(fly['x']+50, fly['y']+50, str(fly['area']), color = ([0,1,0]))
            for not_fly in not_flies:
                cv2.circle(color_image, (not_fly['x'], not_fly['y']), 50, [255,0,0], 5)
                ax.text(not_fly['x']+50, not_fly['y']+50, str(not_fly['area']), color =([1,0,0]))
            masked_color_image = cv2.bitwise_and(color_image,color_image,mask = square_mask)
            plt.imshow(masked_color_image)
            plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
            ax.set_title('randomly drawn frame \n' + filename)
            ax.text(0.5, 0.05, ('threshold: '+ str(threshold) + ' min size: ' + str(minimum_contour_size) + ' max size: ' + str(maximum_contour_size)),horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color = 'white')

            ax2 = fig.add_subplot(131)
            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
            color_image = cv2.imread(filelist[original_index-1])
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            masked_color_image = cv2.bitwise_and(color_image,color_image,mask = square_mask)
            plt.imshow(masked_color_image)
            plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
            ax2.set_title('previous')

            ax3 = fig.add_subplot(133)
            ax3.axes.get_xaxis().set_visible(False)
            ax3.axes.get_yaxis().set_visible(False)
            ax3.set_title('next')
            color_image = cv2.imread(filelist[original_index+1])
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            masked_color_image = cv2.bitwise_and(color_image,color_image,mask = square_mask)
            plt.imshow(masked_color_image)
            plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

            plt.show()
            # time.sleep(.5)
            plt.close('all')
            print ()

        else: # if all the counters are greater than or equal to n
            break
    return ground_truth_sample

def analyze_and_display_an_image(filename,
                                 filename_identifier = 'tl_0000_',
                                 minimum_contour_size = 170,
                                 maximum_contour_size = 600,
                                 threshold = 25,
                                 number_of_frames_for_median = 3,
                                 square_mask_path = ''):

    analyze_frame_number = int((number_of_frames_for_median-1)/2.)
    directory = os.path.dirname(filename)
    filelist = get_filenames(directory, filename_identifier)
    filename_index = filelist.index(filename)
    filelist = filelist[filename_index-1:filename_index+2]
    image_stack = load_image_stack(filelist, square_mask_path)
    flies, not_flies = count_flies(image_stack, analyze_frame_number, minimum_contour_size, maximum_contour_size, threshold)
    color_image = cv2.imread(filelist[analyze_frame_number])
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    show_image_with_circles_drawn_around_putative_flies(color_image, flies, not_flies, square_mask_path)

    # also show the thresholded image
    # gimg = load_image_file_and_convert_to_gray(filename)
    # med_img = np.median(image_stack, axis=0)
    # thresh_img = cv2.compare(np.float32(gimg), np.float32(med_img)-threshold, cv2.CMP_LT)
    # thresh_img_smooth = smooth_image(thresh_img)
    # show_image(thresh_img_smooth)

################################### start the real shit #############################################################
d = analysis_parameters # just to make the lines more concise, I'm renaming the dictionary
for key in d:
    timelapse_directory = directory +'/trapcam_timelapse/'+key
    params = d[key]['analysis parameters']
    if params['ground truth'] == True: # here my aim is to iterate through a subset of the files, ideally in random order, until I've gathered n instances of frames with flies, n instances of frames with nonflies, and n instances of frames without any countours detected. These I will then examine manually for false positives and false negatives.
        ground_truthing_sample = find_SOME_frames_with_flies_in_them(dir = timelapse_directory,
                                            filename_identifier = 'tl',
                                            minimum_contour_size = params['minimum_contour_size'],
                                            maximum_contour_size = params['maximum_contour_size'],
                                            threshold = params['threshold'],
                                            number_of_frames_for_median = params['number_of_frames_for_median'],
                                            time_of_fly_release = field_parameters['time_of_fly_release'],
                                            analyze_how_many_minutes_prior_to_release = params["analyze_how_many_minutes_prior_to_release"],
                                            analyze_how_many_minutes_post_release = params["analyze_how_many_minutes_post_release"],
                                            square_mask_path = str(directory +'/trapcam_timelapse/'+key+'/mask.jpg'),
                                            number_of_example_frames_wanted = params['ground truth frames wanted'])
        d[key]['ground_truthing_sample']= ground_truthing_sample

        #now that I've obtained this randomly-drawn sample of images, providing at least n instances of a frame with a "fly" and n of a frame with a "nonfly object", I am going to iterate through this ground-truthing sample, displaying the image + circles drawn around putative flies and manually annotate, for each frame:
        #how many false positives: (fpos_nonfly = a non-fly object, fpos_edge = a contour created at the edge)
        #how many false negatives: (fneg_atop_deemed_nonfly = a fly atop the trap deemed a nonfly,
        #                             fneg_atop_not_even_a_contour=
        #                             fneg_inside_deemed_nonfly =
        #                             fneg_inside_not_even_a_contour =)
        #

    else:
        results= find_all_frames_with_flies_in_them(dir = timelapse_directory,
                                            filename_identifier = 'tl',
                                            minimum_contour_size = params['minimum_contour_size'],
                                            maximum_contour_size = params['maximum_contour_size'],
                                            threshold = params['threshold'],
                                            number_of_frames_for_median = params['number_of_frames_for_median'],
                                            time_of_fly_release = field_parameters['time_of_fly_release'],
                                            analyze_how_many_minutes_prior_to_release = params["analyze_how_many_minutes_prior_to_release"],
                                            analyze_how_many_minutes_post_release = params["analyze_how_many_minutes_post_release"],
                                            square_mask_path = str(directory +'/trapcam_timelapse/'+key+'/mask.jpg'))
        d[key]['analysis_results']= results

output_name = directory+'/all_traps_analysis_output.json'
with open(output_name,'w') as fid:
    to_write = d
    to_write_json = json.dumps(to_write)
    fid.write('{0}\n'.format(to_write_json))
# print ()
# print ('TO DO: calculation of time-since-release is 2 seconds higher than expected; might be an off-by-one-frame error related to the calculation of the median image')
