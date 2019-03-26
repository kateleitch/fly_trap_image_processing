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
import seaborn as sns
import datetime
from scipy.stats import circmean

# ONLY COMPATIBLE WITH DATA PROCESSED THROUGH TEENSY -- > ROTCAM A LA WILL'S EXPERTISE; ONLY FOR WIND DATA ACQUIRED 2018 AND LATER

# winddirection transformation is based on the anemometer's "lower notch" oriented SOUTH, which is my new convention in the field.
# any time "wind direction" is mentioned, this means "direction the wind is coming FROM"
# the anemometer's output wraps when the wind is coming from the NORTH

directory = sys.argv[1]
with open(directory+'/field_parameters.json') as f:
    field_parameters = json.load(f)
release_time = field_parameters["time_of_fly_release"]
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

def get_time_since_release_from_epoch_timestamp (epoch_timestamp, release_time):
    release_time_seconds = int(release_time.split(':')[0])*3600 +int(release_time.split(':')[1])*60 + int(release_time.split(':')[2])

    epoch_datetime_string = str(datetime.datetime.fromtimestamp(epoch_timestamp)) # comes out formatted like 2019-03-14 21:47:24.071131
    t = epoch_datetime_string.split(' ')[1]
    epoch_seconds = int(t.split(':')[0])*3600 +   int(t.split(':')[1])*60 + float(t.split(':')[2])

    time_elapsed = epoch_seconds - release_time_seconds
    return time_elapsed


wind_filename_list = get_filenames(directory+'/weather_data/metone_anemometer_data/', contains = 'm1_wind_data')

#wind_file_name = directory+ '/weather_data/metone_anemometer_data/m1_wind_data_2018_05_14_12_19_43.txt"

for index, wind_file_name in enumerate(wind_filename_list):
    response = raw_input('Found %d files in this dir; want to analyze file: %s? (y/n) ' %(len(wind_filename_list), wind_file_name))
    if response == 'y':
        print ('OK, analyzing ' +wind_file_name)
        break
    if response == 'n':
        continue

winddirection_list = []
windspeed_list =[]
timestamp_list = []
with open(wind_file_name, 'r') as wind_direction_file:
    for line in wind_direction_file:
        winddirection_list.append(float(line.split(' ')[1]))
        timestamp_list.append(float(line.split(' ')[0]))
        windspeed_list.append(float(line.split(' ')[2]))

release_time = '12:27:54'  #TROUBLESHOOTING ONLY just an arbitrary time for testing; in the future this will come from the field_parameters.json
sec_since_release_list =[]
for timestamp in timestamp_list:
    sec_since_release_list.append(get_time_since_release_from_epoch_timestamp(timestamp, release_time ))

plot_how_many_minutes_pre_release = 10
plot_how_many_minutes_post_release = 20
indices_to_plot=[]
for index, sec_since_release in enumerate(sec_since_release_list):
    if sec_since_release < -1*plot_how_many_minutes_pre_release*60:
        continue
    if sec_since_release > plot_how_many_minutes_post_release*60:
        break
    indices_to_plot.append(int(index))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.argmin((np.abs(array - value)))
    return idx

bin_duration = 60 #in seconds
binned_wind_dir = []
binned_windspeeds = []
direction_sublist = winddirection_list[indices_to_plot[0]:indices_to_plot[-1]]
speed_sublist = windspeed_list[indices_to_plot[0]:indices_to_plot[-1]]
sec_since_release_sublist = sec_since_release_list[indices_to_plot[0]:indices_to_plot[-1]]

start_time = sec_since_release_sublist[0]
bin_count = 0
total_bin_number = np.ceil(len(speed_sublist)/(20*bin_duration))
print (total_bin_number)
while True:
    start_index = find_nearest(sec_since_release_sublist, start_time)
    if bin_count <20:
        print (start_index)
        print (sec_since_release_sublist[start_index])
        print ('')
    end_time = start_time+bin_duration
    end_index = find_nearest(sec_since_release_sublist, end_time)

    if bin_count < total_bin_number:
        direction_slice = direction_sublist[start_index:end_index]
        speed_slice = speed_sublist[start_index:end_index]
    else:
        print ('this last bin is not a full %d minute in length ' %(bin_duration))
        binned_wind_dir.append(circmean(direction_sublist[start_index:-1]))
        binned_windspeeds.append(np.mean(speed_sublist[start_index:-1]))
        break
    binned_wind_dir.append(circmean(direction_slice))
    binned_windspeeds.append(np.mean(speed_slice))
    start_time = end_time
    bin_count += 1

wind_vector_points = [[0,0]]  #in cartesian coordinates, start point of head-to-tail vectors
for i in range (len(binned_wind_dir)):
    x = wind_vector_points[i][0]- (np.cos(binned_wind_dir[i])* binned_windspeeds[i])
    y = wind_vector_points[i][1]- (np.sin(binned_wind_dir[i])* binned_windspeeds[i])
    wind_vector_points.append([x,y])

print (wind_vector_points)
#print [x*180/np.pi for x in binned_wind_dir]

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)

for i in range(len(wind_vector_points)-1):

    ax.scatter(wind_vector_points[i][0], wind_vector_points[i][1], s = 0) #dummy plot; not sure why I need this
    ax.annotate('', xy=(wind_vector_points[i+1][0],wind_vector_points[i+1][1]),
                xytext=(wind_vector_points[i][0],wind_vector_points[i][1]),
                arrowprops=dict(arrowstyle="simple, head_width=1", linewidth = 2, color = 'black'))

wind_vector_points = np.array(wind_vector_points)
x_range = max(wind_vector_points[:,0]) - min(wind_vector_points[:,0])
y_range = max(wind_vector_points[:,1]) - min(wind_vector_points[:,1])
margin = 1
set_range = max(x_range, y_range) + margin
y_center = max(wind_vector_points[:,1]) - y_range/2
x_center=  max(wind_vector_points[:,0]) - x_range/2
ax.set_ylim(y_center -set_range/2, y_center +set_range/2)
ax.set_xlim(x_center- set_range/2 ,x_center +set_range/2)
plt.axis('off')
ax.set_title('2017_11_13 wind direction/speed for %d minutes post release' %(plot_how_many_minutes_post_release))

scalebar_position = [-3,50]
scalebar_mph = 4
ax.annotate(#str(scalebar_mph)+' m.p.h.',
            '',
            xy =scalebar_position,
            xytext=(scalebar_position[0]+scalebar_mph, scalebar_position[1]),
            arrowprops = dict(arrowstyle= '|-|', linewidth =3, color = 'black'))
ax.text(scalebar_position[0], scalebar_position[1] - 3, str(scalebar_mph)+' mph')

note_avg_wind = False

if note_avg_wind:
    avg_speed = np.mean(binned_windspeeds)
    #ax.text(scalebar_position[0], scalebar_position[1] - 6, ("%.1f" % avg_speed)+' mph average speed')
    mean_dx =  wind_vector_points[0][0] - wind_vector_points[-1][0]
    mean_dy = wind_vector_points[0][1] - wind_vector_points[-1][1]
    mean_dir = np.arctan2(mean_dy, mean_dx)
    mean_dir = (2*np.pi + mean_dir)
    mean_dir = divmod(mean_dir, 2*np.pi)[1]

    radians_at_16_compass_points = np.linspace(0,2*np.pi,16, endpoint=False)
    idx = find_nearest_index(radians_at_16_compass_points,mean_dir)
    compass_points = ['E','ENE', 'NE','NNE',
                      'N', 'NNW', 'NW', 'WNW',
                      'W', 'WSW', 'SW', 'SSW',
                      'S', 'SSE', 'SE', 'ESE']
    compass_point = compass_points[idx]

    ax.text(scalebar_position[0], scalebar_position[1] - 10,
            'avg wind ' +('%.1f'%avg_speed)+ ' mph out of ' + compass_point)


savefig = True
if savefig:
    plt.savefig(directory+'/weather_data/metone_anemometer_data/' + 'wind_direction_and_speed.png', transparent = True)

plt.show()
