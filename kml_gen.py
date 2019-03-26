# a nice website for quickly looking at the output kml file is      https://www.doogal.co.uk/KmlViewer.php
# eventually, I'll upload this kml file into my gps/mapping app (topomaps) on my ipad. This uploading process is a bit stressful in the field so it will be good to practice it a few times.


import numpy as np
from datetime import datetime
import json

def getCirclecoordinates(lat, lon, radius, starting_angle, angle, last_iteration, aslist=False):
  # convert coordinates to radians

  d_rad = radius/6378137.
  lat1 = np.deg2rad(lat)
  long1 = np.deg2rad(lon)

  if not aslist:
    coordinatesList = ""
  else:
    coordinatesList = []
  # loop through the array and write path linestrings
  for i in range(starting_angle,360+angle+starting_angle,angle):
    radial = np.deg2rad(i);
    lat_rad = np.arcsin(np.sin(lat1)*np.cos(d_rad) + np.cos(lat1)*np.sin(d_rad)*np.cos(radial))
    dlon_rad = np.arctan2(np.sin(radial)*np.sin(d_rad)*np.cos(lat1), np.cos(d_rad)-np.sin(lat1)*np.sin(lat_rad))
    lon_rad = np.fmod((long1+dlon_rad + np.pi), 2*np.pi) - np.pi
    if not aslist:
      coordinatesList += str(np.rad2deg(lon_rad)) + "," + str(np.rad2deg(lat_rad)) + ",0.0 "
    else:
      if i != 360:
          coordinatesList.append(str(np.rad2deg(lon_rad)) + "," + str(np.rad2deg(lat_rad)) + ",0.0 ")

  if aslist:
      if last_iteration:
          # here, adding the release site to the list of pins
          coordinatesList.append(str(lon) + "," + str(lat) + ",0.0 ")

  return coordinatesList


now = datetime.now() # current date and time
#time = now.strftime("%H:%M:%S")
datestring = now.strftime("%Y_%d_%m_")
print ('')

general_lat_and_lon = [35.05884, -116.74565]
release_site_number = int(raw_input('Enter the number of distinct release sites for this experiment: '))
if release_site_number != 1:
    print('OK, going to ask questions about each of these release sites and its traps...')
print('')

trap_naming_list = ['trap_'+i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz']
coordinatePoints_dictionary={}
coordinatesList_path = ''
coordinatesList_point = []
total_trap_num = 0
for i in range(release_site_number):
    print ('i: '+str(i))
    change_lat_and_lon = raw_input('Is this fly release site at the typical coords of %0.5f N, %0.5f W? (y/n) ' %(general_lat_and_lon[0], general_lat_and_lon[1]))
    if change_lat_and_lon == 'n':
        print ('')
        lat =  float(raw_input('Enter the fly release site latitude: ')) # latitude center
        lon =  float(raw_input('Enter the fly release site longitude: ')) #longitude center
    if change_lat_and_lon == 'y':
        print ('OK, using the typical coordinates')
        print ('')
        lat = general_lat_and_lon[0]
        lon = general_lat_and_lon[1]

    radius_list = []
    radius = raw_input('Enter the radius of the smallest ring of traps, in meters: ')
    radius_list.append(int(radius))
    keep_asking = raw_input('Are there more, concentric, rings of traps? (y/n) ')
    if keep_asking == 'y':
        while True:
            radius = raw_input('Enter the radius of the next ring of traps, also in meters: ')
            radius_list.append(int(radius))
            keep_asking = raw_input('Are there more, concentric, rings of traps? (y/n) ')
            if keep_asking == 'n':
                break
    inter_trap_angle_list = []
    for j in range(len(radius_list)):
        if j == 0:
            traps = raw_input('How many traps are in the smallest ring? ')
            inter_trap_angle_list.append(np.int(np.round(360./int(traps))))
            total_trap_num += int(traps)
        else:
            inter_trap_angle_list.append(np.int(np.round(360./int(traps))))
            total_trap_num += int(traps)

    trap_naming_list.insert(total_trap_num+i, 'release_site_'+str(i+1))
    starting_angle = 0
    if raw_input('Does the clockwise layout of traps start at 0 degrees north? (y/n) ') == 'n':
        starting_angle = int(raw_input('OK, at what angle (clockwise from north) do they start? '))

    trap_prefix = datestring # for name of pin
    filename = datestring+"fly_release.kml"
    color = "red" # options include red, green
    # Get circle coordinates
    point_list_for_json = []
    for index, radius in enumerate(radius_list):
        last_iteration = False
        if index +1 == len(radius_list):
            last_iteration = True
        intertrap_angle = inter_trap_angle_list[index]
        coordinatesList_path += ' '+getCirclecoordinates(lat, lon, radius, starting_angle, intertrap_angle, last_iteration, aslist=False);
        coordinatesList_point += getCirclecoordinates(lat, lon, radius, starting_angle, intertrap_angle, last_iteration, aslist=True);
        point_list_for_json += getCirclecoordinates(lat, lon, radius, starting_angle, intertrap_angle, last_iteration, aslist=True);

    coordinatePoints_dictionary['release_site_'+str(i+1)] = point_list_for_json

kml = ""
kml += "<?xml version='1.0' encoding='UTF-8'?>\n"
kml += "<kml xmlns='http://www.opengis.net/kml/2.2'>\n"
kml += "<Document>\n"
kml += "  <Placemark>\n"
kml += "    <name>Circle</name>\n"
kml += "    <description><![CDATA[radius 100 meters<P>Generated by <a href='http://kml4earth.appspot.com/'>Kml4Earth</a>]]></description>\n"
kml += "    <Style>\n"
kml += "  <IconStyle>\n"
kml += "          <Icon/>\n"
kml += "  </IconStyle>\n"
kml += "  <LineStyle>\n"
if color == "green":
  kml += "    <color>ff00ff00</color>\n"
elif color == "red":
  kml += "    <color>ff0000ff</color>\n"
kml += "    <width>2</width>\n"
kml += "  </LineStyle>\n"
kml += "    </Style>\n"
kml += "    <LineString>\n"
kml += "  <tessellate>1</tessellate>\n"
kml += "  <coordinates>" + coordinatesList_path + "</coordinates>\n"
kml += "    </LineString>\n"
kml += "  </Placemark>\n"

for i, coordinate in enumerate(coordinatesList_point):
  kml += "  <Placemark>\n"
  kml += "    <name>" + datestring +trap_naming_list[i] + "</name>\n"
  kml += "    <visibility>1</visibility>\n"
  kml += "    <Point>\n"
  kml += "      <coordinates>" + coordinate + "</coordinates>\n"
  kml += "    </Point>\n"
  kml += "  </Placemark>\n"
kml += "</Document>\n"
kml += "</kml>\n"


print kml
f = open(filename, 'w')
f.write(kml)
f.close()

with open(filename.split('.')[0]+'_coords.json', mode = 'w') as g:
    json.dump(coordinatePoints_dictionary,g, indent = 4)
