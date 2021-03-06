#! /usr/bin/python

from __future__ import print_function
import cv2 # opencv
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import time
import json
from pylab import *
from scipy.optimize import curve_fit
# from scipy.stats import gamma
from pympler.tracker import SummaryTracker
import subprocess
tracker = SummaryTracker()

class TrapcamAnalyzer:
    def __init__(self, directory):  # <---- instances of this class will specify the directory, most likely using directory = sys.argv[1]
        self.directory = directory

    def get_filenames(self, path, contains, does_not_contain=['~', '.pyc']):
        cmd = 'ls ' + '"' + path + '"'
        ls = os.popen(cmd).read()
        all_filelist = ls.split('\n')
        try:
            all_filelist.remove('')
        except:
            pass
        #filelist = []
        #filelist = np.chararray(len(all_filelist),itemsize = 70)
        filelist = ['']*(len(all_filelist))
        filename_count = 0
        for i, filename in enumerate(all_filelist):
            if contains in filename:
                fileok = True
                for nc in does_not_contain:
                    if nc in filename:
                        fileok = False
                if fileok:
                    #filelist.append( os.path.join(path, filename) )
                    filelist[filename_count] = str(os.path.join(path,filename))
                    filename_count +=1
        filelist_trimmed = filelist[0:filename_count-1]
        return filelist_trimmed

    def load_color_image(self, filename):
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

    def load_mask(self, square_mask_path):
        mask = cv2.imread(square_mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask = mask/255 #rescales mask to be 0s and 1s
        return mask

    def load_image_file(self, filename):
        img = cv2.imread(filename)
        img[0:100,:] = 0
        return img

    def fit_ellipse_to_contour(self, contour):
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
            eccentricity = round(eccentricity, 3)
            return cx, cy, area, eccentricity
        else:
            return None
        #return cx, cy, area

    def smooth_image(self, thresh_img):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh_img_smooth = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations = 1)
        thresh_img_smooth = cv2.dilate(thresh_img_smooth, kernel, iterations=10) # make the contour bigger to join neighbors
        thresh_img_smooth = cv2.erode(thresh_img_smooth, kernel, iterations=10) # make contour smaller again
        return thresh_img_smooth

    def get_contours_from_frame_and_stack(self, image_stack, frame_to_analyze, threshold=25):
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

    def get_time_since_release_from_filename (self, release_time='', name = '', offset = 0):
        frame_time_string = name.split('.')[-2].split('_')[-1]
        frame_hour = int(frame_time_string[0:2])
        frame_min = int(frame_time_string[2:4])
        frame_sec = int(frame_time_string[4:6])
        frame_seconds_timestamp = (frame_hour)*3600 + (frame_min)*60 + (frame_sec)
        frame_seconds = frame_seconds_timestamp - offset
        release_time_seconds = int(release_time.split(':')[0])*3600 +int(release_time.split(':')[1])*60 + int(release_time.split(':')[2])
        time_elapsed = frame_seconds - release_time_seconds
        return time_elapsed

    def show_image_with_circles_drawn_around_putative_flies(self, color_image, flies_on_trap, flies_in_trap, not_flies):
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
            cv2.putText(color_image, str(not_fly['eccentricity']),(not_fly['x']+50, not_fly['y']+50), font, 1, [178,255,102],2, cv2.LINE_AA)
            #cv2.putText(color_image, str(not_fly['eccentricity']),(not_fly['x']+50, not_fly['y']+25), font, 1, (255,0,0),2, cv2.LINE_AA)

    def fit_data_to_trimodal(self,data,expected,ax_handle,plot_histogram):
        y,x,_=hist(data,100,alpha=.3,color = 'black', label='data')

        x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

        def gauss(x,mu,sigma,A):
            return A*exp(-(x-mu)**2/2/sigma**2)

        def trimodal(x,mu1,sigma1,A1,mu2,sigma2,A2, mu3, sigma3, A3):
            return (gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3, sigma3,A3))

        try:
            params,cov=curve_fit(trimodal,x,y,expected)#, maxfev = 10000) # <---- NEED ERROR HANDLING HERE
        except RuntimeError:
            print ('optimal parameters not found')
            return None

        sigma=sqrt(diag(cov))

        fit = trimodal(x,*params)
        dy = np.diff(fit,1)
        dx = np.diff(x,1)
        y_first = dy/dx
        x_first = 0.5*(x[:-1]+x[1:])

        neg_to_pos_z_crossing_indices = np.where(np.sign(np.diff(np.sign(y_first)))>0)[0] +1
        neg_to_pos_z_crossing_metric = x[neg_to_pos_z_crossing_indices]
        if len(neg_to_pos_z_crossing_metric)>1: # this means more than one local minimum was found in the curve fit
            neg_to_pos_z_crossing_indices = (np.abs(neg_to_pos_z_crossing_metric - 20.)).argmin()
            neg_to_pos_z_crossing_metric = neg_to_pos_z_crossing_metric[neg_to_pos_z_crossing_indices]

        if plot_histogram:
            plot(x,trimodal(x,*params),color='black',lw=2,label='trimodal fit')
            plt.scatter(neg_to_pos_z_crossing_metric, fit[neg_to_pos_z_crossing_indices], color = 'black', s =40, zorder=10)
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

    def testing_step_of_backsub_MOG2(self, index, train_num, fgbg, test_image, ontrap_intrap_threshold, time_since_release, maximum_contour_size, minimum_contour_size):
        fgmask1 = fgbg.apply(test_image, None, 0) # the 0 specifies that no learning is occurring
        fgmask1 = self.smooth_image(fgmask1)
        image, contours, hierarchy = cv2.findContours(fgmask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flies_on_trap = [{} for _ in range(200)]
        flies_in_trap = [{} for _ in range(200)]
        not_flies     = [{} for _ in range(500)]
        fly_contours  = [[] for _ in range(200)] # < -------- bummer that I need to pre-allocate this as such a chunky thing
        frame_contrast_metrics = np.zeros(150)
        contrast_metric_count=0
        flies_on_trap_count=0
        flies_in_trap_count=0
        not_flies_count=0
        fly_contours_count=0
        for contour in contours:
            if len(contour) > 5:
                x, y, area, ecc = self.fit_ellipse_to_contour(contour)
                fly = {'x': x, 'y': y, 'area': area, 'eccentricity': ecc}
                if area < maximum_contour_size and area > minimum_contour_size:
                    if ecc > 0.15:
                        fly_contours[fly_contours_count]=(contour)
                        fly_contours_count +=1
                    else:
                        print ('eccentricity less than 0.15')
                else:
                    not_flies[not_flies_count] = fly
                    not_flies_count +=1
        #now that we have a list of fly contours, let's see if they're inside the trap or outside it
        bgimg = fgbg.getBackgroundImage()
        gray_bgimg = cv2.cvtColor(bgimg, cv2.COLOR_RGB2GRAY)
        gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        color = 255
        thresh = ontrap_intrap_threshold
        not_flies = not_flies[0:not_flies_count]
        fly_contours = fly_contours[0:fly_contours_count]
        for fly_contour in fly_contours: #these are all the contours whose area is > min and < max contour size specified in analysis params
            stencil = np.zeros(gray_bgimg.shape).astype(gray_bgimg.dtype)
            cv2.fillPoly(stencil, [fly_contour], color) # needed to put fly_contour in a list structure, otherwise it treats each point as an individual contour I believe.
            pts = np.where(stencil == 255)
            bg_masked_by_contour = cv2.bitwise_and(gray_bgimg, stencil)
            fg_masked_by_contour = cv2.bitwise_and(gray_test_image, stencil)
            difference_img = cv2.absdiff(bg_masked_by_contour, fg_masked_by_contour)#bg_masked_by_contour - fg_masked_by_contour
            mean_difference_in_this_contour = np.mean(difference_img[pts[0],pts[1]])
            pixel_num_in_this_contour = len(pts)
            contrast_metric = mean_difference_in_this_contour
            x, y, area, ecc = self.fit_ellipse_to_contour(fly_contour)
            fly = {'x': x, 'y': y, 'area': area,'contrast metric': contrast_metric}
            if contrast_metric > thresh:
                flies_on_trap[flies_on_trap_count]= fly
                flies_on_trap_count +=1
            if contrast_metric < thresh:
                flies_in_trap[flies_in_trap_count]= fly
                flies_in_trap_count +=1
            frame_contrast_metrics[int(contrast_metric_count)]=contrast_metric # I use this list to see if there is sufficient bimodality in contrast metrics that the location of its local minimum could be used to automatically select a threshold for classifying in-trap vs. on-trap flies
            contrast_metric_count += 1
        # now trimming trailing empty dictionaries
        flies_on_trap = flies_on_trap[0:flies_on_trap_count]
        flies_in_trap = flies_in_trap[0:flies_in_trap_count]
        ##############################################################################################
        test_image_copy = test_image.copy()
        self.show_image_with_circles_drawn_around_putative_flies(test_image_copy, flies_on_trap,flies_in_trap, not_flies)
        dict_to_add_to_all_flies_over_time = {'seconds since release':time_since_release, 'flies on trap': flies_on_trap, 'flies in trap': flies_in_trap, 'not_flies': not_flies}
        frame_contrast_metrics = frame_contrast_metrics[0:contrast_metric_count]
        return test_image_copy, time_since_release, dict_to_add_to_all_flies_over_time, frame_contrast_metrics, contrast_metric_count

    def find_contours_using_pretrained_backsub_MOG2(self,
                                                    full_image_stack,
                                                    filename_stack,
                                                    train_num,
                                                    mahalanobis_squared_thresh,
                                                    buffer_btw_training_and_test, #I'm adding this because flies often stand still on the trap-top
                                                    minimum_contour_size,
                                                    maximum_contour_size,
                                                    time_of_fly_release,
                                                    camera_time_offset, # in seconds, how far advanced is the trap's timestamp relative to real time
                                                    ontrap_intrap_threshold,
                                                    video_dir):

        fgbg = cv2.createBackgroundSubtractorMOG2(history =train_num, varThreshold = mahalanobis_squared_thresh, detectShadows = False)
        #standard_out_array_length = len(full_image_stack) -train_num -buffer_btw_training_and_test -2# #ugh, not so principled, empirically determined to work for a range of temporal windows tho
        standard_out_array_length = len(full_image_stack) -train_num -buffer_btw_training_and_test
        sample_image_zero =  np.zeros_like(full_image_stack[0]) # <---- just using the first image as an "example" to be sure I preallocate the array with the right data type, dimensions etc
        # annotated_output_stack = np.stack([sample_image_zero for _ in range(standard_out_array_length)], axis = 0)
        time_since_release_list = np.zeros(standard_out_array_length)
        analyzed_filename_stack = ['']*(standard_out_array_length)
        all_flies_over_time = [{} for _ in range(standard_out_array_length)]  # <-----
        all_contrast_metrics = np.zeros(standard_out_array_length*80)# <---- vastly over-allocated assuming 80 flies per analyzed frame; will need to trim zeros using a counter of contrast metrics

        contrast_metric_count = 0
        last_frame_contrast_metric_count =0
        for index, training_image in enumerate(full_image_stack):
            fgbg.apply(training_image, None, -1) # TRAINING STEP.
            if index > train_num-1: # when current index is less than train_num, the model hasn't been trained on the specified number of frames. After this point, the declaration of history = train_num should make the model "forget" earlier frames so it works as a sliding window
                test_index = index+buffer_btw_training_and_test
                try:
                    test_image = full_image_stack[test_index]
                    test_filename = filename_stack[test_index]
                except:
                    break
                #time_since_release = self.get_time_since_release_from_filename(release_time = time_of_fly_release, name = test_filename, offset = camera_time_offset)
                annotated_output_image, time_since_release, dict_to_add_to_all_flies_over_time, frame_contrast_metrics, frame_contrast_metric_count = self.testing_step_of_backsub_MOG2(index, train_num, fgbg, test_image, ontrap_intrap_threshold, self.get_time_since_release_from_filename(release_time = time_of_fly_release, name = test_filename, offset = camera_time_offset),maximum_contour_size, minimum_contour_size)
                time_since_release_list [index -train_num] = time_since_release
                analyzed_filename_stack [index -train_num] = test_filename
                all_flies_over_time     [index -train_num] = dict_to_add_to_all_flies_over_time
                contrast_metric_count += frame_contrast_metric_count

                all_contrast_metrics[last_frame_contrast_metric_count:contrast_metric_count] = frame_contrast_metrics
                last_frame_contrast_metric_count = contrast_metric_count

                #here, save annotated output image!!!!
                cv2.imwrite(video_dir + "%04d.jpg" % index, annotated_output_image)
        all_contrast_metrics = all_contrast_metrics[0:contrast_metric_count-1] #trimming trailing zeros
        return all_flies_over_time, time_since_release_list, analyzed_filename_stack, all_contrast_metrics

    def format_matplotlib_ax_object(self, ax_handle):
        ax_handle.spines['right'].set_visible(False)
        ax_handle.spines['top'].set_visible(False)
        ax_handle.tick_params(direction='out')
        # Only show ticks on the left and bottom spines
        ax_handle.yaxis.set_ticks_position('left')
        ax_handle.xaxis.set_ticks_position('bottom')
        plt.tight_layout()

    def step_through_annotated_output_stack(self,
                                            all_flies_over_time,
                                            all_contrast_metrics,
                                            trimodal_expected,
                                            timestamp,
                                            output_dir,
                                            video_dir,
                                            analyzed_filename_stack,
                                            field_parameters,
                                            key,
                                            analysis_params,
                                            manually_step_through_stack = False,
                                            save_video= True,
                                            calculate_final = False):

        font = cv2.FONT_HERSHEY_SIMPLEX

        number_ive_empirically_determined =2
        all_flies_over_time       =  all_flies_over_time     [0:-1*number_ive_empirically_determined]
        #time_since_release_list   =  time_since_release_list [0:-1*number_ive_empirically_determined]
        analyzed_filename_stack   =  analyzed_filename_stack [0:-1*number_ive_empirically_determined] # these 4 lines are obviously shameful

        flies_on_trap_over_time = np.zeros(len(all_flies_over_time))
        flies_in_trap_over_time = np.zeros(len(all_flies_over_time))
        not_flies_over_time = np.zeros(len(all_flies_over_time))
        seconds_since_release_over_time = np.zeros(len(all_flies_over_time))
        for index, i in enumerate(all_flies_over_time):
            try:
                flies_on_trap_over_time[index]=(len(i['flies on trap']))
                flies_in_trap_over_time[index]=(len(i['flies in trap']))
                not_flies_over_time[index]=(len(i['not_flies']))
                seconds_since_release_over_time[index]=(i['seconds since release'])
            except:
                continue

        window_size = 10
        low_pass_flies_on_trap = np.zeros(len(flies_on_trap_over_time)-window_size)
        low_pass_flies_in_trap = np.zeros(len(flies_on_trap_over_time)-window_size)
        for i in range (window_size, len(flies_on_trap_over_time)):
            low_pass_flies_on_trap[i-window_size] = (np.mean(flies_on_trap_over_time[i-window_size:i]))
            low_pass_flies_in_trap[i-window_size] = (np.mean(flies_in_trap_over_time[i-window_size:i]))

        annotated_frame_filenames = self.get_filenames(path = video_dir, contains = ".jpg", does_not_contain = [])
        print ('now reading in annotated frames and merging them with other graphics')
        for frame_pos, name in enumerate(annotated_frame_filenames):
            display_image = cv2.imread(name)
            plt.close('all') # < ---- dealing with the memory issues of having too many windows open at once
            try:
                filename = analyzed_filename_stack[frame_pos]
                #time_since_release = time_since_release_list[frame_pos]
                time_since_release = seconds_since_release_over_time[frame_pos]
                print('')
            except:
                break

            display_image_resized = cv2.resize(display_image, (1296,972)) #halves image dimensions just for display purposes
            display_image_resized = display_image_resized[:,200:-240].copy()

        #### now plotting the graph of flies over time
            fig = plt.figure(figsize=(10,9), facecolor="white")
            ax2 = fig.add_subplot(212)

            proposed_contrast_metric = self.fit_data_to_trimodal(all_contrast_metrics,
                                                    trimodal_expected,
                                                    ax_handle = ax2,
                                                    plot_histogram = True) # <---- THIS SHOULD REEEALLY ONLY HAPPEN ONCE, NOT IN THIS LOOP

            plt.xlabel('contrast metric (per-pixel fg-bg; avg per contour)')
            plt.ylabel('count')
            ax = fig.add_subplot(211)
            ax.scatter(seconds_since_release_over_time, flies_in_trap_over_time, color = [0.6,0,0.6])
            ax.plot(seconds_since_release_over_time[window_size:], low_pass_flies_in_trap, color = [0.6,0,0.6], lw = 2, label = 'in trap')
            ax.scatter(seconds_since_release_over_time, flies_on_trap_over_time, color = 'black')
            ax.plot(seconds_since_release_over_time[window_size:], low_pass_flies_on_trap, color = 'black', lw =2, label = 'on trap')
            legend = ax.legend(loc='upper left', shadow=False) #, fontsize='x-large')
            ax.axvline(x = time_since_release, color = 'black', lw =2)
            plt.xlabel('seconds since release')
            plt.ylabel('flies in frame')
            self.format_matplotlib_ax_object (ax_handle = ax)
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
            seconds = str(int(divmod(time_since_release,60)[1]))
            if len(seconds) == 1:
                seconds = '0'+seconds
            textstr = 'timestamp: '+timestr[0:2]+':'+timestr[2:4]+':'+timestr[4:6]+'; time since release: '+ str(int(divmod(time_since_release,60)[0]))+':'+ seconds

            cv2.putText(vis, textstr, (30+w1+60,50), font, 1, (255,255,255),2, cv2.LINE_AA)

            idstr = '~%d flies released at %s ' %(field_parameters["estimated_number_of_flies_released"], field_parameters['time_of_fly_release'])
            cv2.putText(vis, idstr, (35,50), font, 1, (255,255,255),2, cv2.LINE_AA)
            idstr2 = key.split('_')[0]+' ' +key.split('_')[1]+'; %d flies caught' %(field_parameters['trap counts'][key])
            cv2.putText(vis,idstr2, (35,90), font, 1, (255,255,255),2, cv2.LINE_AA)

            #now saving into stack
            cv2.imwrite(output_dir + "/%d.jpg" % frame_pos, vis)

        min_post_r = analysis_params["analyze_how_many_minutes_post_release"]
        sample_video_str = output_dir+'/'+key + '_analyzed_%d_min_post_release' %(min_post_r)+'.mp4'
        output_dir_jpgs = output_dir+"/%d.jpg"
        if save_video:
            subprocess.call(["ffmpeg", "-framerate", "3", "-i", output_dir_jpgs, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", sample_video_str])

        if calculate_final == True:
            print ('calculating final')
            current_trap_dictionary = {key:{'flies on trap over time:': flies_on_trap_over_time.tolist(),
                                            'flies in trap over time:': flies_in_trap_over_time.tolist(),
                                            'not flies over time:'    : not_flies_over_time.tolist(),
                                            'seconds since release:'  : seconds_since_release_over_time.tolist()}}
            with open(self.directory+'/all_traps_final_analysis_output.json') as f:
                growing_json = json.load(f)
            #add current trap dictionary to growing_json
            growing_json.update(current_trap_dictionary) #CAREFUL; THIS WILL OVERWRITE ANY KEYS THAT ALREADY EXIST IN THE JSON
            with open(self.directory+'/all_traps_final_analysis_output.json', mode = 'w') as f:
                json.dump(growing_json,f, indent = 1)

            fig = plt.figure(figsize=(10,5), facecolor="white")
            ax = fig.add_subplot(111)
            ax.scatter(seconds_since_release_over_time, flies_in_trap_over_time, color = [0.6,0,0.6])
            ax.plot(seconds_since_release_over_time[window_size:], low_pass_flies_in_trap, color = [0.6,0,0.6], lw = 2, label = 'in trap')
            ax.scatter(seconds_since_release_over_time, flies_on_trap_over_time, color = 'black')
            ax.plot(seconds_since_release_over_time[window_size:], low_pass_flies_on_trap, color = 'black', lw =2, label = 'on trap')
            legend = ax.legend(loc='upper left', shadow=False) #, fontsize='x-large')
            plt.xlabel('seconds since release')
            plt.ylabel('flies in frame')
            self.format_matplotlib_ax_object(ax_handle = ax)
            time_string = str(time.time()).split('.')[0] # this is not very human readable but helps prevent overwriting
            namestr = self.directory+'/arrival_dynamics_figs/'+key+'_flies_over_time_'+time_string+'.svg'
            plt.savefig(namestr, bbox_inches='tight')
            pngnamestr = self.directory+'/arrival_dynamics_figs/'+key+'_flies_over_time_'+time_string+'.png'
            plt.savefig(pngnamestr, bbox_inches='tight')
        # if manually_step_through_stack:
        #     frame_pos = 0
        #     while True:
        #         vis =  cv2.imread(annotated_frame_filenames[frame_pos])
        #         cv2.imshow(" ", vis)
        #         wait_key_val = cv2.waitKey(0) & 0xFF
        #         if wait_key_val == ord('q'):
        #             break
        #         if wait_key_val == ord('f'):
        #             frame_pos += 1
        #         if wait_key_val == ord('b'):
        #             frame_pos -= 1
        #         if wait_key_val == ord('j'):
        #             frame_time = int(raw_input("Jump to time: "))
        #             diff_list = [abs(t - frame_time) for t in time_since_release_list]
        #             frame_pos = diff_list.index(min(diff_list))
        #             frame_pos = max(frame_pos,0)
        #             frame_pos = min(frame_pos,len(annotated_output_stack)-1)

    # --------------------------------------------------------------------------------------------------------
    def run(self):
        with open(self.directory+'/all_traps_gaussian_analysis_params.json') as f:
            analysis_parameters_json = json.load(f)
        with open(self.directory+'/field_parameters.json') as f:
            field_parameters = json.load(f)
        with open(self.directory+'/trap_layout_parameters.json') as f:
            trap_layout_parameters = json.load(f)

        plt.rcParams.update({'font.size': 14}) # <--- there is probably a better place to specify this so it's more flexible, but this'll work for now

################################## getting some user input #####################################################
        print ('')
        while True:
            analyze_trap_list = []
            letter = raw_input("Enter a trap letter to analyze: ")
            analyze_trap_list.append('trap_'+letter)
            while True:
                letter = raw_input("Enter another trap letter to analyze, or enter 'go' to start batch analysis: ")
                if letter == 'go':
                    break
                else:
                    analyze_trap_list.append('trap_'+letter)
            print ('')
            print ('you said you want to analyze: ')
            for an_trap in analyze_trap_list:
                print (an_trap)
            user_go_ahead = raw_input("Are those the traps you'd like to analyze? (y/n) ")
            if user_go_ahead == 'y':
                break
            if user_go_ahead == 'n':
                continue
        print ('')
        calculate_threshold = False
        calculate_final = False
        thresh_or_final = raw_input("Do you want to analyze just a subset of frames to determine the best in-trap/on-trap threshold, or do you want to do the final analysis? (threshold/final) ")
        if thresh_or_final =='threshold':
            calculate_threshold = True
        if thresh_or_final == 'final':
            calculate_final = True
##############################################################################################################

        d = analysis_parameters_json
        for key in d:
            proposed_contrast_metric = None
            if key not in analyze_trap_list:
                print ('skipping '+ key)
                continue
            if key == 'common to all traps':
                continue
            print ('analyzing ' +key)
            analysis_params = d[key]['analysis parameters']
            trimodal_expected = tuple(analysis_params['trimodal expected'])
            timelapse_directory = self.directory +'/trapcam_timelapse/'+key
            square_mask = self.load_mask(square_mask_path = timelapse_directory+'/mask.jpg')
            full_filename_list = self.get_filenames(path = timelapse_directory, contains = "tl", does_not_contain = ['th']) #  full list of image filenames in the folder
            filename_list = ['']*(len(full_filename_list))
            image_count = 0

            for filename in full_filename_list:
                time_since_release = self.get_time_since_release_from_filename(release_time = field_parameters["time_of_fly_release"], name = filename, offset = analysis_params["camera time advanced by __ sec"])
                if time_since_release < -60* analysis_params['analyze_how_many_minutes_prior_to_release']:
                    continue
                if time_since_release > 60* analysis_params["analyze_how_many_minutes_post_release"]:
                    break
                filename_list[image_count]= filename
                image_count += 1
            filename_list = filename_list[0:image_count-1] # <----could be off-by-one

            del(full_filename_list)

            sample_image =  np.zeros_like(self.load_color_image(filename_list[40]))
            masked_image_stack = np.stack([sample_image for _ in range(image_count+1)], axis = 0)

            image_count = 0
            for filename in filename_list:
                img = self.load_color_image(filename)
                if img is None:
                    print ('img is None!')
                    continue
                else:
                    masked_image_stack[image_count] = cv2.bitwise_and(img,img,mask = square_mask)
                    image_count +=1
            del(img)

            if calculate_final == True:
                try:
                    on_in_thresh = analysis_parameters_json[key]["fixed in-trap on-trap threshold"]
                except:
                    use_provisional_value =raw_input('Looks like you have not yet fixed the in-trap/on-trap threshold; use provisional threshold? (y/n)')
                    if use_provisional_value:
                        on_in_thresh = analysis_params["threshold to differentiate in- and on- trap"]
                    else:
                        print ('OK, skipping '+ key+' for now')
                        print ('')
                        continue
            else:
                on_in_thresh = analysis_params["threshold to differentiate in- and on- trap"]
            print ('length of masked image stack: '+str(len(masked_image_stack)))

            timestamp = str(int(time.time()))
            annotated_frame_dir = self.directory+'/'+key+'_videos/'+timestamp+'/annotated_frames/'
            subprocess.call(['mkdir', '-p', annotated_frame_dir])
            #os.mkdir(annotated_frame_dir)

            all_flies_over_time, time_since_release_list, analyzed_filename_stack, all_contrast_metrics =self.find_contours_using_pretrained_backsub_MOG2(
                                                                                full_image_stack = masked_image_stack,
                                                                                filename_stack = filename_list,
                                                                                train_num = analysis_params['number of frames for bg model training'],
                                                                                mahalanobis_squared_thresh = analysis_params['mahalanobis squared threshold'],
                                                                                buffer_btw_training_and_test = analysis_params['frames between training and test'], # if too high, shadows get bad
                                                                                minimum_contour_size = analysis_params['minimum_contour_size'],
                                                                                maximum_contour_size = analysis_params['maximum_contour_size'],
                                                                                time_of_fly_release = field_parameters["time_of_fly_release"],
                                                                                camera_time_offset = analysis_params['camera time advanced by __ sec'],
                                                                                ontrap_intrap_threshold = on_in_thresh,
                                                                                video_dir = annotated_frame_dir)

            ########### NOW TO STEP THROUGH FRAMES IN ANNOTATED_OUTPUTSTACK
            del(masked_image_stack) # <--- if I'm properly managing references to masked_image_stack, this shouldn't really be necessary

            #save all_contrast_metrics so I can play around with curve fitting
            contrast_metric_dictionary = {'all contrast metrics': all_contrast_metrics.tolist()}
            with open(self.directory+'/all_contrast_metrics/'+key+'.json', mode = 'w') as f:
                json.dump(contrast_metric_dictionary,f, indent = 1)

            annotated_frames_plus_graphs_dir = self.directory+'/'+key+'_videos/'+timestamp+'/annotated_frames_plus_graphs'
            subprocess.call(['mkdir', annotated_frames_plus_graphs_dir])

            self.step_through_annotated_output_stack(all_flies_over_time,
                                                    all_contrast_metrics,
                                                    trimodal_expected,
                                                    timestamp,
                                                    annotated_frames_plus_graphs_dir,
                                                    annotated_frame_dir,
                                                    analyzed_filename_stack,
                                                    field_parameters,
                                                    key,
                                                    analysis_params,
                                                    manually_step_through_stack = analysis_params["step through frames"],
                                                    save_video = analysis_params["save frames as video"],
                                                    calculate_final = calculate_final)

            if calculate_threshold:
                proposed_contrast_metric = self.fit_data_to_trimodal(all_contrast_metrics, trimodal_expected, ax_handle = None, plot_histogram = False)

            if proposed_contrast_metric is not None:
                if proposed_contrast_metric.size != 0:
                    print ('proposed contrast cutoff: '+ str(proposed_contrast_metric))
                    if calculate_threshold:
                        with open(self.directory+'/all_traps_gaussian_analysis_params.json') as f:
                            growing_json = json.load(f)
                        growing_json[key]['fixed in-trap on-trap threshold'] = int(round(proposed_contrast_metric))
                        with open(self.directory+'/all_traps_gaussian_analysis_params.json', mode = 'w') as f:
                            json.dump(growing_json,f, indent = 4)

            cv2.destroyAllWindows()
