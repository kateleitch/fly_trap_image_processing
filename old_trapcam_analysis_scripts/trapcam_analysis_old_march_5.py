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

directory = sys.argv[1]

with open(directory+'/all_traps_gaussian_analysis_params.json') as f:
    analysis_parameters_json = json.load(f)
with open(directory+'/field_parameters.json') as f:
    field_parameters = json.load(f)
with open(directory+'/trap_layout_parameters.json') as f:
    trap_layout_parameters = json.load(f)

#-----------------------------------------------------------------------

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

def load_image_file(filename):
    img = cv2.imread(filename)
    img[0:100,:] = 0
    return img

def fit_ellipse_to_contour(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00']) # centroid, x
        cy = int(M['m01']/M['m00']) # centroid, y
        area = cv2.contourArea(contour)

        ellipse = cv2.fitEllipse(contour)
        (x,y), (a,b), angle = ellipse
        a /= 2.
        b /= 2.
        eccentricity = np.min((a,b)) / np.max((a,b))
        return cx, cy, area, eccentricity
    else:
        return None

    #return cx, cy, area

def smooth_image(thresh_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_img_smooth = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations = 1)
    thresh_img_smooth = cv2.dilate(thresh_img_smooth, kernel, iterations=10) # make the contour bigger to join neighbors
    thresh_img_smooth = cv2.erode(thresh_img_smooth, kernel, iterations=10) # make contour smaller again
    return thresh_img_smooth

def get_contours_from_frame_and_stack(image_stack, frame_to_analyze, threshold=25):
    # get the median image
    med_img = np.median(image_stack, axis=0)

    # find all the dark objects in the image compared to the median
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
            x, y, area, ecc = fit_ellipse_to_contour(contour)
            #print 'Contour area: ', area
            fly = {'x': x, 'y': y, 'area': area} # this is a dictionary
            if area < maximum_contour_size and area > minimum_contour_size:
                flies.append(fly)
            else:
                not_flies.append(fly)
    return flies, not_flies

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

def show_image_with_circles_drawn_around_putative_flies(color_image, flies_on_trap, flies_in_trap, not_flies):
    font = cv2.FONT_HERSHEY_SIMPLEX

    for fly in flies_on_trap:
        cv2.circle(color_image, (fly['x'], fly['y']), 50, [0,0,0], 5)
        #cv2.putText(color_image, str(fly['area']),(fly['x']+50, fly['y']+50), font, 1, (0,255,0),2, cv2.LINE_AA)
        cv2.putText(color_image, str(int(fly['contrast metric'])),(fly['x']+50, fly['y']+50), font, 1, (0,0,0),2, cv2.LINE_AA)
        #cv2.putText(color_image, str(fly['eccentricity']),(fly['x']+50, fly['y']+25), font, 1, (0,0,0),2, cv2.LINE_AA)

    for fly in flies_in_trap:
        cv2.circle(color_image, (fly['x'], fly['y']), 50, [153,0,153], 5)
        #cv2.putText(color_image, str(fly['area']),(fly['x']+50, fly['y']+50), font, 1, (0,255,255),2, cv2.LINE_AA)
        cv2.putText(color_image, str(int(fly['contrast metric'])),(fly['x']+50, fly['y']+50), font, 1, [153,0,153],2, cv2.LINE_AA)
        #cv2.putText(color_image, str(fly['eccentricity']),(fly['x']+50, fly['y']+25), font, 1, (100,100,100),2, cv2.LINE_AA)

    for not_fly in not_flies:
        cv2.circle(color_image, (not_fly['x'], not_fly['y']), 50, [178,255,102], 5)
        cv2.putText(color_image, str(not_fly['area']),(not_fly['x']+50, not_fly['y']+50), font, 1, [178,255,102],2, cv2.LINE_AA)
        #cv2.putText(color_image, str(not_fly['eccentricity']),(not_fly['x']+50, not_fly['y']+25), font, 1, (255,0,0),2, cv2.LINE_AA)

def fit_data_to_bimodal(data, expected, ax_handle):
    y,x,_=hist(data,100,alpha=.3,color = 'black', label='data')

    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

    def gauss(x,mu,sigma,A):
        return A*exp(-(x-mu)**2/2/sigma**2)

    def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
        return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

    # expected=(1,.2,250,2,.2,125)
    try:
        params,cov=curve_fit(bimodal,x,y,expected) # <---- NEED ERROR HANDLING HERE
    except RuntimeError:
        print ('optimal parameters not found')
        return None

    sigma=sqrt(diag(cov))
    plot(x,bimodal(x,*params),color='black',lw=2,label='bimodal fit')

    #print(params,'\n',sigma)
    fit = bimodal(x,*params)
    dy = np.diff(fit,1)
    dx = np.diff(x,1)
    y_first = dy/dx
    x_first = 0.5*(x[:-1]+x[1:])

    # dy_first = np.diff(y_first,1)
    # dx_first = np.diff(x_first,1)
    # y_second = dy_first/dx_first
    #scatter(x[:-2], [3*i for i in y_second], color = 'blue', lw=2, label='2nd deriv')

    #plot(x_first, y_first, color = 'green', lw=2, label = '1st deriv')
    neg_to_pos_z_crossing_index = np.where(np.sign(np.diff(np.sign(y_first)))>0)[0] +1
    neg_to_pos_z_crossing_metric = x[neg_to_pos_z_crossing_index]
    scatter(neg_to_pos_z_crossing_metric, fit[neg_to_pos_z_crossing_index], color = 'black', s =40, zorder=10)

    plt.legend()
    ax_handle.spines['right'].set_visible(False)
    ax_handle.spines['top'].set_visible(False)
    ax_handle.set_ylim(0,max(y)*1.1)
    ax_handle.set_xlim(0,max(x)*1.1)
    # Only show ticks on the left and bottom spines
    ax_handle.yaxis.set_ticks_position('left')
    ax_handle.xaxis.set_ticks_position('bottom')
    ax_handle.tick_params(direction='out')#, length=6, width=2, colors='r',grid_color='r', grid_alpha=0.5)
    return neg_to_pos_z_crossing_metric

def find_contours_using_pretrained_backsub_MOG2(full_image_stack,
                                                filename_stack,
                                                train_num,
                                                mahalanobis_squared_thresh,
                                                buffer_btw_training_and_test, #I'm adding this because flies often stand still on the trap-top
                                                minimum_contour_size,
                                                maximum_contour_size,
                                                time_of_fly_release,
                                                camera_time_offset, # in seconds, how far advanced is the trap's timestamp relative to real time
                                                ontrap_intrap_threshold):

    fgbg = cv2.createBackgroundSubtractorMOG2(history =train_num, varThreshold = mahalanobis_squared_thresh, detectShadows = False)
    annotated_output_stack = []
    time_since_release_list = []
    analyzed_filename_stack = []
    all_flies_over_time = []
    all_contrast_metrics = []
    print ('time of fly release: ' + str(time_of_fly_release))
    for index, training_image in enumerate(full_image_stack):
        fgmask = fgbg.apply(training_image, None, -1) # TRAINING STEP.
        if index > train_num: # when current index is less than train_num, the model hasn't been trained on the specified number of frames. After this point, the declaration of history = train_num should make the model "forget" earlier frames so it works as a sliding window
            test_index = index+buffer_btw_training_and_test
            try:
                test_image = full_image_stack[test_index]
                test_filename = filename_stack[test_index]
            except:
                break

            time_since_release = get_time_since_release_from_filename(release_time = time_of_fly_release, name = test_filename, offset = camera_time_offset)

            fgmask1 = fgbg.apply(test_image, None, 0) # the 0 specifies that no learning is occurring
            fgmask1 = smooth_image(fgmask1)
            image, contours, hierarchy = cv2.findContours(fgmask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            flies_on_trap = []
            flies_in_trap = []
            not_flies = []
            fly_contours = []
            for contour in contours:
                if len(contour) > 5:
                    x, y, area, ecc = fit_ellipse_to_contour(contour)
                    fly = {'x': x, 'y': y, 'area': area, 'eccentricity': ecc}
                    if area < maximum_contour_size and area > minimum_contour_size:
                        fly_contours.append(contour)
                    else:
                        not_flies.append(fly)
            #now that we have a list of fly contours, let's see if they're inside the trap or outside it
            bgimg = fgbg.getBackgroundImage()
            gray_bgimg = cv2.cvtColor(bgimg, cv2.COLOR_RGB2GRAY)
            gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
            #color = (255, 255, 255)
            color = 255
            thresh = ontrap_intrap_threshold
            for fly_contour in fly_contours: #these are all the contours whose area is > min and < max contour size specified in analysis params
                stencil = np.zeros(gray_bgimg.shape).astype(gray_bgimg.dtype)
                cv2.fillPoly(stencil, [fly_contour], color) # needed to put fly_contour in a list structure, otherwise it treats each point as an individual contour I believe.
                pts = np.where(stencil == 255)
                bg_masked_by_contour = cv2.bitwise_and(gray_bgimg, stencil)
                fg_masked_by_contour = cv2.bitwise_and(gray_test_image, stencil)

                difference_img = cv2.absdiff(bg_masked_by_contour, fg_masked_by_contour)#bg_masked_by_contour - fg_masked_by_contour

                mean_difference_in_this_contour = np.mean(difference_img[pts[0],pts[1]])
                pixel_num_in_this_contour = len(pts)
                contrast_metric = mean_difference_in_this_contour *np.sqrt(pixel_num_in_this_contour) # <---- a way to incorporate both contour area and mean contrast
                contrast_metric = mean_difference_in_this_contour

                x, y, area, ecc = fit_ellipse_to_contour(fly_contour)
                fly = {'x': x, 'y': y, 'area': area, 'eccentricity': ecc, 'contrast metric': contrast_metric}
                if contrast_metric > thresh:
                    flies_on_trap.append(fly)
                if contrast_metric < thresh:
                    flies_in_trap.append(fly)

                all_contrast_metrics.append(contrast_metric) # I use this list to see if there is sufficient bimodality in contrast metrics that the location of its local minimum could be used to automatically select a threshold for classifying in-trap vs. on-trap flies
            ##############################################################################################
            test_image_copy = test_image.copy()
            show_image_with_circles_drawn_around_putative_flies(test_image_copy, flies_on_trap,flies_in_trap, not_flies)
            annotated_output_stack.append(test_image_copy)
            time_since_release_list.append(time_since_release)
            analyzed_filename_stack.append(test_filename)
            all_flies_over_time.append({'seconds since release':time_since_release, 'flies on trap': flies_on_trap, 'flies in trap': flies_in_trap, 'not_flies': not_flies})
    return all_flies_over_time, annotated_output_stack, time_since_release_list, analyzed_filename_stack, all_contrast_metrics


# --------------------------------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 14})

d = analysis_parameters_json
for key in d:
    print ('analyzing ' +key)
    # if key != 'trap_D':  # this is just for the development stage; at the moment I don't want to iterate through the whole folder
    #     continue   # just for development
    analysis_params = d[key]['analysis parameters']
    timelapse_directory = directory +'/trapcam_timelapse/'+key
    square_mask = load_mask(square_mask_path = timelapse_directory+'/mask.jpg')
    masked_image_stack = []
    filename_list = []

    full_filename_list = get_filenames(path = timelapse_directory, contains = "tl", does_not_contain = ['th']) # this is the full list of image filenames in the folder
    for filename in full_filename_list:
        time_since_release = get_time_since_release_from_filename(release_time = field_parameters["time_of_fly_release"], name = filename, offset = analysis_params["camera time advanced by __ sec"])
        if time_since_release < -60* analysis_params['analyze_how_many_minutes_prior_to_release']:
            continue
        if time_since_release > 60* analysis_params["analyze_how_many_minutes_post_release"]:
            break
        img = load_color_image(filename)
        if img is None:
            print ('img is None!')
            continue
        else:
            masked_image_stack.append(cv2.bitwise_and(img,img,mask = square_mask))
            filename_list.append(filename)

    # print len(masked_image_stack)
    # print len(filename_list)

    all_flies_over_time, annotated_output_stack, time_since_release_list, analyzed_filename_stack, all_contrast_metrics = find_contours_using_pretrained_backsub_MOG2(full_image_stack = masked_image_stack,
                                                               filename_stack = filename_list,
                                                               train_num = analysis_params['number of frames for bg model training'],
                                                               mahalanobis_squared_thresh = analysis_params['mahalanobis squared threshold'],
                                                               buffer_btw_training_and_test = analysis_params['frames between training and test'], # if too high, shadows get bad
                                                               minimum_contour_size = analysis_params['minimum_contour_size'],
                                                               maximum_contour_size = analysis_params['maximum_contour_size'],
                                                               time_of_fly_release = field_parameters["time_of_fly_release"],
                                                               camera_time_offset = analysis_params['camera time advanced by __ sec'],
                                                               ontrap_intrap_threshold = analysis_params["threshold to differentiate in- and on- trap"])

############### NOW TO STEP THROUGH FRAMES IN OUTPUTSTACK
    # print len(annotated_output_stack)
    # print len(time_since_release_list)
    # print len(analyzed_filename_stack)
    # print len(all_flies_over_time)

    flies_on_trap_over_time = []
    flies_in_trap_over_time = []
    not_flies_over_time = []
    seconds_since_release_over_time = []
    for i in all_flies_over_time:
        flies_on_trap_over_time.append(len(i['flies on trap']))
        flies_in_trap_over_time.append(len(i['flies in trap']))
        not_flies_over_time.append(len(i['not_flies']))
        seconds_since_release_over_time.append(i['seconds since release'])

    # doing a sort of low-pass filter on the data just for plotting purposes
    low_pass_flies_on_trap = []
    low_pass_flies_in_trap = []
    window_size = 10
    for i in range (window_size, len(flies_on_trap_over_time)):
        low_pass_flies_on_trap.append(int(np.mean(flies_on_trap_over_time[i-window_size:i])))
        low_pass_flies_in_trap.append(int(np.mean(flies_in_trap_over_time[i-window_size:i])))

    font = cv2.FONT_HERSHEY_SIMPLEX

    frame_pos = 0

    while True:
        plt.close('all') # < ---- my attempt to deal with the memory issues of having too many windows open at once

        print('frame_pos: {0}'.format(frame_pos))

        try:
            display_image = annotated_output_stack[frame_pos]
            filename = analyzed_filename_stack[frame_pos]
            time_since_release = time_since_release_list[frame_pos]
            print('')
        except:
            break

        display_image_resized = cv2.resize(display_image, (1296,972)) #halves image dimensions just for display purposes
        display_image_resized = display_image_resized[:,200:-240].copy()
    #### now plotting the graph of flies over time

        fig = plt.figure(figsize=(10,9), facecolor="white")
        ax2 = fig.add_subplot(212)
        #ax2.hist(all_contrast_metrics, bins = 30)
        expected=(15,5,100,32,5,100)
        proposed_contrast_metric = fit_data_to_bimodal(all_contrast_metrics, expected, ax_handle = ax2)
        plt.xlabel('contrast metric (per-pixel fg-bg; avg per contour)')
        plt.ylabel('count')
        ax = fig.add_subplot(211)
        ax.scatter(seconds_since_release_over_time, flies_in_trap_over_time, color = [0.6,0,0.6])
        ax.plot(seconds_since_release_over_time[window_size:], low_pass_flies_in_trap, color = [0.6,0,0.6], lw = 2, label = 'in trap')
        ax.scatter(seconds_since_release_over_time, flies_on_trap_over_time, color = 'black')
        ax.plot(seconds_since_release_over_time[window_size:], low_pass_flies_on_trap, color = 'black', lw =2, label = 'on trap')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out')
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #ax.scatter(seconds_since_release_over_time, not_flies_over_time, color = 'blue')
        legend = ax.legend(loc='upper left', shadow=False) #, fontsize='x-large')
        ax.axvline(x = time_since_release, color = 'black', lw =2)
        plt.xlabel('seconds since release')
        plt.ylabel('flies in frame')
        plt.tight_layout()
        fig.canvas.draw()
        graph = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        graph  = graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        graph = cv2.cvtColor(graph,cv2.COLOR_RGB2BGR)

        h1, w1 = graph.shape[:2]
        h2, w2 = display_image_resized.shape[:2]

        #create empty matrix
        height = max(h1,h2)
        vis = np.zeros((height, 30+w1+w2,3), np.uint8)
        #vis = np.zeros((h1+h2, max(w1, w2),3), np.uint8)
        #combine 2 images
        vis[150:150+h1, 30:30+w1,:3] = graph
        vis[:h2, 30+w1:30+w1+w2,:3] = display_image_resized

        timestr = filename.split('_')[-1].split('.')[0]
        seconds = str(divmod(time_since_release,60)[1])
        if len(seconds) == 1:
            seconds = '0'+seconds
        textstr = 'timestamp: '+timestr[0:2]+':'+timestr[2:4]+':'+timestr[4:6]+'; time since release: '+ str(divmod(time_since_release,60)[0])+':'+ seconds

        cv2.putText(vis, textstr, (30+w1+60,50), font, 1, (255,255,255),2, cv2.LINE_AA)

        idstr = '~%d flies released at %s ' %(field_parameters["estimated_number_of_flies_released"], field_parameters['time_of_fly_release'])
        cv2.putText(vis, idstr, (35,50), font, 1, (255,255,255),2, cv2.LINE_AA)
        idstr2 = key.split('_')[0]+' ' +key.split('_')[1]+'; %d flies caught' %(field_parameters['trap counts'][key])
        cv2.putText(vis,idstr2, (35,90), font, 1, (255,255,255),2, cv2.LINE_AA)

        # if analysis_params["step through frames"]:
        cv2.imshow("test", vis)
        wait_key_val = cv2.waitKey(0) & 0xFF
        if wait_key_val == ord('q'):
            break
        if wait_key_val == ord('f'):
            frame_pos += 1
        if wait_key_val == ord('b'):
            frame_pos -= 1
        if wait_key_val == ord('j'):
            frame_time = int(raw_input("Jump to time: "))
            diff_list = [abs(t - frame_time) for t in time_since_release_list]
            frame_pos = diff_list.index(min(diff_list))
            frame_pos = max(frame_pos,0)
            frame_pos = min(frame_pos,len(annotated_output_stack)-1)
    if proposed_contrast_metric is not None:
        print ('proposed contrast cutoff: '+ str(proposed_contrast_metric))
    print ('')
    # Clean up
    cv2.destroyAllWindows()

    if analysis_params["save frames as video"]:
        timestamp = str(int(time.time()))
        video_dir = timelapse_directory+'/gui_video_'+timestamp
        os.mkdir(video_dir)
        for frame_pos in range(len(annotated_output_stack)):
            try:
                display_image = annotated_output_stack[frame_pos]
                filename = analyzed_filename_stack[frame_pos]
                time_since_release = time_since_release_list[frame_pos]
                print('')
            except:
                break

            plt.close('all')

            display_image_resized = cv2.resize(display_image, (1296,972)) #halves image dimensions just for display purposes
            display_image_resized = display_image_resized[:,200:-240].copy()
            #### now plotting the graph of flies over time

            fig = plt.figure(figsize=(10,9), facecolor="white")
            ax2 = fig.add_subplot(212)
            #ax2.hist(all_contrast_metrics, bins = 30)
            expected=(15,5,100,32,5,100)
            fit_data_to_bimodal(all_contrast_metrics, expected, ax_handle = ax2)
            plt.xlabel('contrast metric (per-pixel fg-bg; avg per contour)')
            plt.ylabel('count')
            ax = fig.add_subplot(211)
            ax.scatter(seconds_since_release_over_time, flies_in_trap_over_time, color = [0.6,0,0.6])
            ax.plot(seconds_since_release_over_time[window_size:], low_pass_flies_in_trap, color = [0.6,0,0.6], lw = 2, label = 'in trap')
            ax.scatter(seconds_since_release_over_time, flies_on_trap_over_time, color = 'black')
            ax.plot(seconds_since_release_over_time[window_size:], low_pass_flies_on_trap, color = 'black', lw =2, label = 'on trap')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out')
            # Only show ticks on the left and bottom spines
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            #ax.scatter(seconds_since_release_over_time, not_flies_over_time, color = 'blue')
            legend = ax.legend(loc='upper left', shadow=False) #, fontsize='x-large')
            ax.axvline(x = time_since_release, color = 'black', lw =2)
            plt.xlabel('seconds since release')
            plt.ylabel('flies in frame')
            plt.tight_layout()
            fig.canvas.draw()
            graph = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            graph  = graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            graph = cv2.cvtColor(graph,cv2.COLOR_RGB2BGR)

            h1, w1 = graph.shape[:2]
            h2, w2 = display_image_resized.shape[:2]

            #create empty matrix
            height = max(h1,h2)
            vis = np.zeros((height, 30+w1+w2,3), np.uint8)
            #vis = np.zeros((h1+h2, max(w1, w2),3), np.uint8)
            #combine 2 images
            vis[150:150+h1, 30:30+w1,:3] = graph
            vis[:h2, 30+w1:30+w1+w2,:3] = display_image_resized

            timestr = filename.split('_')[-1].split('.')[0]
            seconds = str(divmod(time_since_release,60)[1])
            if len(seconds)==1:
                seconds = '0'+seconds
            textstr = 'timestamp: '+timestr[0:2]+':'+timestr[2:4]+':'+timestr[4:6]+'; time since release: '+ str(divmod(time_since_release,60)[0])+':'+ seconds

            cv2.putText(vis, textstr, (30+w1+60,50), font, 1, (255,255,255),2, cv2.LINE_AA)

            idstr = '~%d flies released at %s ' %(field_parameters["estimated_number_of_flies_released"], field_parameters['time_of_fly_release'])
            cv2.putText(vis, idstr, (35,50), font, 1, (255,255,255),2, cv2.LINE_AA)
            idstr2 = key.split('_')[0]+' ' +key.split('_')[1]+'; %d flies caught' %(field_parameters['trap counts'][key])
            cv2.putText(vis,idstr2, (35,90), font, 1, (255,255,255),2, cv2.LINE_AA)
            cv2.imwrite(video_dir + "/%d.jpg" % frame_pos, vis)

        min_post_r = analysis_params["analyze_how_many_minutes_post_release"]
        sample_video_str = directory +'/trapcam_timelapse/'+key+'/'+key + '_analyzed_%d_min_post_release' %(min_post_r)+'.mp4'

        video_dir_jpgs = video_dir+"/%d.jpg"
        import subprocess
        subprocess.call(["ffmpeg", "-framerate", "3", "-i", video_dir_jpgs, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", sample_video_str])
