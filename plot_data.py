from __future__ import print_function
import sys
import json
import trapcam_analysis as t
import matplotlib.pyplot as plt
import numpy as np

dir = sys.argv[1]
analyzer = t.TrapcamAnalyzer(dir)



print ('')
while True:
    analyze_trap_list = []
    letter = raw_input("Enter a trap letter to plot: ")
    analyze_trap_list.append('trap_'+letter)
    while True:
        letter = raw_input("Enter another trap letter to plot, or enter 'go': ")
        if letter == 'go':
            break
        else:
            analyze_trap_list.append('trap_'+letter)
    print ('')
    print ('you said you want to plot: ')
    for an_trap in analyze_trap_list:
        print (an_trap)
    user_go_ahead = raw_input("Are those the traps you'd like to plot? (y/n) ")
    if user_go_ahead == 'y':
        break
    if user_go_ahead == 'n':
        continue
print ('')

with open(dir+'/all_traps_final_analysis_output.json') as f:
    data = json.load(f)

for trap_name in data:
    if trap_name not in analyze_trap_list:
        print ('skipping '+ trap_name)
        continue
    sec_since_release = data[trap_name]['seconds since release:']
    flies_on_trap = data[trap_name]["flies on trap over time:"]
    flies_in_trap = data[trap_name]["flies in trap over time:"]

    window_size = 10
    low_pass_flies_on_trap = np.zeros(len(flies_on_trap)-window_size)
    low_pass_flies_in_trap = np.zeros(len(flies_in_trap)-window_size)
    for i in range (window_size, len(flies_in_trap)):
        low_pass_flies_on_trap[i-window_size] = (np.mean(flies_on_trap[i-window_size:i]))
        low_pass_flies_in_trap[i-window_size] = (np.mean(flies_in_trap[i-window_size:i]))


    fig = plt.figure(figsize=(10,4.5), facecolor="white")
    ax = fig.add_subplot(111)
    ax.scatter(sec_since_release, flies_in_trap, color = [0.6,0,0.6])
    ax.plot(sec_since_release[window_size:], low_pass_flies_in_trap, color = [0.6,0,0.6], lw = 2, label = 'in trap')
    ax.scatter(sec_since_release, flies_on_trap, color = 'black')
    ax.plot(sec_since_release[window_size:], low_pass_flies_on_trap, color = 'black', lw =2, label = 'on trap')
    legend = ax.legend(loc='upper left', shadow=False) #, fontsize='x-large')
    #ax.axvline(x = time_since_release, color = 'black', lw =2)
    plt.xlabel('seconds since release')
    plt.ylabel('flies in frame')
    ax.set_ylim(0,25)
    analyzer.format_matplotlib_ax_object(ax_handle = ax)
    plt.show()
