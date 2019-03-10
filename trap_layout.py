#!/usr/bin/python

from __future__ import print_function
import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import json

directory = sys.argv[1] #specify experiment date directory with respect to current working directory
experiment_date = directory.split('/')[-1]
planned_layout_or_actual_layout = sys.argv[2]
output_name = directory+'/trap_layout_'+ planned_layout_or_actual_layout+'.png'
print (directory+'/trap_layout_parameters.json')
with open(directory+'/trap_layout_parameters.json') as f:
    plotting_parameters = json.load(f)

# plotting_parameters = {'figsize' = (10,10), 'inner_trap_radius' = 250, 'outer_trap_radius' = 1000, 'inner_trap_number' = 10, 'outer_trap_number' = 10,
# 'are_traps_ordered_clockwise'= True, 'inner_ring_first_trap_rad_cw_from_n' = 0, 'outer_ring_first_trap_rad_cw_from_n' = 0}

figure_size = plotting_parameters['figsize']
inner_trap_radius = plotting_parameters['inner_trap_radius']
outer_trap_radius = plotting_parameters['outer_trap_radius']
inner_trap_number = plotting_parameters['inner_trap_number']
outer_trap_number = plotting_parameters['outer_trap_number']
are_traps_ordered_clockwise = plotting_parameters['are_traps_ordered_clockwise']
inner_ring_first_trap_rad_cw_from_n = plotting_parameters['inner_ring_first_trap_deg_cw_from_n']*np.pi/180
outer_ring_first_trap_rad_cw_from_n = plotting_parameters['outer_ring_first_trap_deg_cw_from_n']*np.pi/180

fig = plt.figure(figsize=(figure_size,figure_size))
ax = fig.add_subplot(111, projection='polar')
ax.set_theta_offset(np.pi/2)
if are_traps_ordered_clockwise:
    ax.set_theta_direction(-1)

trap_names = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
trap_name_list = [trap_names[x] for x in range(0, len(trap_names))] #this list most likely ends up longer than the total number of traps, which is fine

theta = np.linspace(0, 2*np.pi, 360)
r1 = np.ones(360) *outer_trap_radius
r2 = np.ones(360) *inner_trap_radius
ax.plot(theta, r1, 'k')
ax.plot(theta, r2, 'k')

inner_trap_angles = np.linspace(0,2*np.pi, inner_trap_number, endpoint = False) + inner_ring_first_trap_rad_cw_from_n
outer_trap_angles = np.linspace(0,2*np.pi, outer_trap_number, endpoint = False) + outer_ring_first_trap_rad_cw_from_n

ax.scatter(inner_trap_angles,np.ones(inner_trap_number)*inner_trap_radius, color = 'grey')
ax.scatter(outer_trap_angles,np.ones(outer_trap_number)*outer_trap_radius, color = 'grey')
for i in range(inner_trap_number+outer_trap_number):
    if i < inner_trap_number:
        ax.text(inner_trap_angles[i], inner_trap_radius*1.2, trap_name_list[i], va = 'center', ha = 'center')
    else:
        ax.text(outer_trap_angles[i-inner_trap_number], outer_trap_radius*1.1, trap_name_list[i], va = 'center', ha = 'center')

ax.scatter(0,0, color = 'black')
ax.text(0, outer_trap_radius*1.3, experiment_date+' ' +planned_layout_or_actual_layout +'\n %d traps at %d m, and %d traps at %d m from release site'%(inner_trap_number, inner_trap_radius, outer_trap_number, outer_trap_radius),va = 'center', ha = 'center')
#ax.text(0,outer_trap_radius*1.35, experiment_date+' ' +planned_layout_or_actual_layout, va = 'center', ha = 'center')

ax.grid(False)
ax.axis("off")
plt.savefig(output_name, transparent = True)
plt.show()
