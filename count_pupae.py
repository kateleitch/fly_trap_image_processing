#! /usr/bin/python

from __future__ import print_function
import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import scipy.stats
import json
import time

directory = sys.argv[1]+'/fly_rearing_records'
output_name = directory+'/pupal_count_results.json'
output_figure_name = directory+'/contour_histogram'

with open(directory+'/pupal_count_analysis_parameters.json') as f:
    analysis_parameters = json.load(f)

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

def load_image_from_filelist (filename):
    fd = open(filename)
    rows = 480
    cols = 640
    f = np.fromfile(fd, dtype=np.uint8,count=rows*cols)
    im = f.reshape((rows, cols))
    fd.close()
    return im

def smooth_image(thresh_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_img_smooth = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations = 0)
    #thresh_img_smooth = cv2.erode(thresh_img, kernel, iterations=2) # make contour smaller again
    #thresh_img_smooth = cv2.dilate(thresh_img_smooth, kernel, iterations=3) # make the contour bigger to join neighbors
    return thresh_img_smooth

def get_contours_from_frame(image, adaptive_thresh_neighborhood, adaptive_thresh_offset):
    adaptive_thresh_img = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV,
                                                adaptive_thresh_neighborhood,
                                                adaptive_thresh_offset)

    #fig = plt.figure(figsize=(48,32))
    #ax = fig.add_subplot(211)
    #ax.imshow(adaptive_thresh_img, cmap ='gray')
    #ax2 = fig.add_subplot(212)
    #ax2.imshow(image, cmap = 'gray')
    thresh_img_smooth = smooth_image(adaptive_thresh_img)
    image, contours, hierarchy = cv2.findContours(thresh_img_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def count_flies(image,
                adaptive_thresh_neighborhood,
                adaptive_thresh_offset):
    contours = get_contours_from_frame(image,
                                       adaptive_thresh_neighborhood = adaptive_thresh_neighborhood,
                                       adaptive_thresh_offset = adaptive_thresh_offset)
    all_contour_areas = []
    image_copy = np.copy(image)
    cv2.drawContours(image, contours, -1, (0,255,0), 0)

    for contour in contours:
        area = cv2.contourArea(contour)
        all_contour_areas.append(area)
    return all_contour_areas

def find_pos_zero_crossings(signal):
    pos_crossings = []
    for index, value in enumerate(signal):
        if value < 0 and signal[index+1] >0:
            pos_crossings.append(index+1)
    return pos_crossings
################################################################################
filelist = get_filenames(directory, contains = 'Basler', does_not_contain=['.pyc'])

pupae_per_sheet = []
all_contours_all_sheets = []

for f in filelist:
    im = load_image_from_filelist(f)
    all_contour_areas = count_flies(image = im,
                               adaptive_thresh_neighborhood = analysis_parameters['adaptive threshold neighborhood'],
                               adaptive_thresh_offset = analysis_parameters['adaptive threshold offset']);
    all_contours_all_sheets.append(all_contour_areas)

print ('counted %d sheets' %(len(all_contours_all_sheets)))

for sublist in all_contours_all_sheets:
    if max(all_contours_all_sheets) == 0:
        print ('problem here; more detailed troubleshooting needed')

flat_list = [item for sublist in all_contours_all_sheets for item in sublist]

fig = plt.figure(figsize=(12,4),facecolor="white")
ax = fig.add_subplot(111)
pupal_histogram = ax.hist(flat_list, bins = analysis_parameters['histogram bin number'] , range = (0,analysis_parameters['histogram binning max']), alpha=.3,color = 'black', label='data')

ax.plot(pupal_histogram[1][:-1],pupal_histogram[0], color ='black', lw = 2)

#now, finding the first instance at which diff(histogram counts) crosses zero; I'll want to exclude everything prior to this index in calculating mode and in integrating all pupal area
histogram_diff = np.diff(pupal_histogram[0])
#ax.plot(pupal_histogram[1][1:-1], histogram_diff, color = 'black', lw =2, label = '')
# ax2 = fig.add_subplot(212)
# ax2.plot(pupal_histogram[1][1:-1], histogram_diff)
# ax2.axhline(y=0)
# ax2.set_xlim(0,50)
# ax2.set_ylim(-500,500)

first_positive_diff_index = 1+ find_pos_zero_crossings(histogram_diff)[0]
print ('will ignore bin counts prior to index %d' %(first_positive_diff_index))

ax.set_xlim(0,50)
indices_to_skip = first_positive_diff_index
peak_index = np.argmax(pupal_histogram[0][indices_to_skip:])+indices_to_skip
pupal_mode_size = pupal_histogram[1][peak_index]
ax.set_ylim(0, pupal_histogram[0][peak_index]*1.1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(direction='out')
plt.xlabel('contour area, pixels**2')
plt.ylabel('count')
plt.tight_layout()
plt.scatter(peak_index, pupal_histogram[0][peak_index], color = 'black', s =40, zorder=10)

print ('single pupa mode size: %0.2f' %(pupal_mode_size))
total_contour_area = np.sum(flat_list)
minimum_size = pupal_histogram[1][indices_to_skip] # should specify this in a more principled/dynamic way
total_contour_area_flecks_excluded = np.sum([x for x in flat_list if x > minimum_size])
total_pupae_counted = round(total_contour_area_flecks_excluded/pupal_mode_size)
print ('counted a total of %d pupae; an average of %0.1f pupae per sheet' %(total_pupae_counted, total_pupae_counted/len(all_contours_all_sheets)))

ax.text(peak_index*1.05, pupal_histogram[0][peak_index]*0.9,  'single pupa mode size: %0.1f \n      total contour area: %d \n      estimated pupae: %d' %(pupal_mode_size, int(total_contour_area_flecks_excluded), int(total_pupae_counted)))


namestr = output_figure_name+'.svg'
plt.savefig(namestr, bbox_inches='tight')
pngnamestr = output_figure_name+'.png'
plt.savefig(pngnamestr, bbox_inches='tight')
plt.show()



total_pupae_estimated = (total_pupae_counted/len(all_contours_all_sheets))*analysis_parameters["total number of sheets in release chamber"]
print ('given that average, and given that I transferred %d sheets, estimating a total of %d pupae in this release chamber' %(analysis_parameters["total number of sheets in release chamber"], total_pupae_estimated))
with open(output_name,'w') as fid:
    to_write = {'analysis parameters': analysis_parameters,
                'ignoring bin counts prior to index':first_positive_diff_index,
                'single pupa mode size' : pupal_mode_size,
                'total pupae counted': total_pupae_counted,
                'number of pupal sheets analyzed': len(all_contours_all_sheets),
                'total estimate for entire release chamber': total_pupae_estimated}

    to_write_json = json.dumps(to_write)
    fid.write('{0}\n'.format(to_write_json))
