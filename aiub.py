"""
This module is a collection of functions to import, organise and plot
satellite observation data collected by the Astronomy Institute of Bern Universirty
"""
# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: MIT License

from subprocess import Popen,PIPE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

import urllib.request
from bz2 import BZ2File as bzopen

import math, re, os, glob, datetime

import cartopy.crs as ccrs
from pygeodesy import ellipsoidalNvector

from datetime import date, timedelta
import datetime


def cartesian_to_ellipsoidal(x,y,z):
    """Convert cartesian coordinates to ellipsoidal coordinates
    
    Parameters
    ----------
    x,y,z : floats
        cartesian coordinates
        
    Returns
    -------
    ellipsoid_coord: numpy array 
        elipsoidal coordinates
    """
    
    ellipsoid_coord = np.array(ellipsoidalNvector.Cartesian(x,y,z).toLatLon(LatLon = None))
    ellipsoid_coord[0:2] = 2*np.pi*ellipsoid_coord[0:2]/360
    return ellipsoid_coord


def rot_mat(Bcrd, Lcrd):
    
    """Compute rotation matrix needed to go from global cartesian wgs-84 
    eccentricities to local ellipsoidal eccentricities (north,east,up in meters)
    
    Parameters
    ----------
    Bcrd : float
        latitude
    Lcrd: float 
        longitude
        
    Returns
    -------
    drmat : 2d numpy array
        rotation matrix
    """
    
    # SIN and COS functions
    sphi=math.sin(Bcrd)
    cphi=math.cos(Bcrd)
    slmb=math.sin(Lcrd)
    clmb=math.cos(Lcrd)

    # compute rotation matrix
    drmat = np.empty([3, 3])
    drmat[0,0]=-sphi*clmb
    drmat[0,1]=-sphi*slmb
    drmat[0,2]= cphi
    drmat[1,0]=     -slmb
    drmat[1,1]=      clmb
    drmat[1,2]= 0
    drmat[2,0]= cphi*clmb
    drmat[2,1]= cphi*slmb
    drmat[2,2]= sphi
    
    return drmat

def local_ecc(mat, Xdel, Ydel, Zdel):
    """Go from global cartesian wgs-84 eccentricities to 
    local ellipsoidal eccentricities (north,east,up in meters)
    
    Parameters
    ----------
    mat : 2d numpy array
        rotation matrix
    Xdel, Ydel, Zdel: float 
        cartesian coordinates
        
    Returns
    -------
    loc_coord : numpy array
        ellipsoidal eccentricities
    """
    
    loc_coord = np.dot(mat,[Xdel,Ydel,Zdel])
    return loc_coord

def coord_grid(nlats = 30, nlons = 30, radius = 6369915.831429518):
    """Create a grid of cartesian coordinates around the globe and
    compute their elipsoidal coordintes
    
    Parameters
    ----------
    nlats : int
        number of grid latitudes
    nlons: int 
        number of grid longitudes
    radius: Earth radius in m
        
    Returns
    -------
    coord_table : pandas dataframe
        Dataframe with coordinates of all grid points:
            'x_pos', 'y_pos', 'z_pos': float, cartesian coordinates
            'x_geod', 'y_geod', 'z_geod': float, elipsoidal coordinates
            pos_ind: int, index of points
    lons, lats: numpy arrays with list of longitudes and latitudes
    """

    lats = np.linspace(-np.pi / 2, np.pi / 2, nlats)
    lons = np.linspace(0, 2 * np.pi, nlons)
    lons2d, lats2d = np.meshgrid(lons, lats)
    x_pos = np.ravel(np.cos(lons2d)*np.sin(lats2d)*radius)
    y_pos = np.ravel(np.sin(lats2d)*np.sin(lats2d)*radius)
    z_pos = np.ravel(np.cos(lats2d)*radius)
    pos_ind = np.arange(len(x_pos))
    coord_geod = np.array([cartesian_to_ellipsoidal(x_pos[i], y_pos[i],z_pos[i]) for i in range(len(x_pos))])
    
    coord_table = pd.DataFrame([x_pos, y_pos, z_pos,coord_geod[:,0],coord_geod[:,1],coord_geod[:,2], pos_ind]).T
    coord_table.columns = ['x_pos','y_pos','z_pos','x_geod','y_geod','z_geod','pos_ind']
    coord_table['key'] = 1
    
    return coord_table, lons, lats
    
def import_stations(address, coord_file):
    """Read observation station file and get its name and coordinates
    
    Parameters
    ----------
    address : string
        url address of data repository
    coord_file: string aa
        name of file
        
    Returns
    -------
    stations : pandas dataframe
        Dataframe with station information:
            'station': string, station reference
            'statname': string, station name
            'x_pos', 'y_pos', 'z_pos': cartesian coordinates of station
    """
    
    data = urllib.request.urlretrieve(address+coord_file, filename='temp/temp_file.Z')
    foo_proc = Popen(['uncompress', 'temp/temp_file.Z'], stdin=PIPE, stdout=PIPE)
    foo_proc.communicate(input=b'yes')
    
    stations = pd.read_csv('temp/temp_file',skiprows=5, sep = '\s{1,8}', usecols = [1,2,3,4,5,6],names = ['index','station','statname','x_pos','y_pos','z_pos','flag'],engine='python')
    stations= stations.dropna()
    
    stations.x_pos = stations.x_pos.astype(float,copy = False)#/1000;
    stations.y_pos = stations.y_pos.astype(float,copy = False)#/1000;
    stations.z_pos = stations.z_pos.astype(float,copy = False)#/1000;

    return stations


def date_to_gpsweeks(selected_date):
    epoch = date(1980, 1, 6)
    today = selected_date

    epochMonday = epoch - timedelta(epoch.weekday())
    todayMonday = today - timedelta(today.weekday())
    noWeeks = (todayMonday - epochMonday).days / 7

    return today.year, int(noWeeks), (today.weekday()+1)%7

def import_RM_file(address, gps_week):
    """Read RM file with satellite observations
    
    Parameters
    ----------
    address : string
        url address of data repository
    gps_week: tuple
        triplets of values with (year, gps week, week day)
        
    Returns
    -------
    temp_pd : pandas dataframe
        Dataframe with station information:
            'year', month','day','hour','minute': int indicating time
            'satellite': string, satellite type
            'name': string, satellite name
            'c1','c2','c3','c4': float, satellite coordinate
            'time_stamp': float, time-stamp
            'key': int, value = 1, used for table joininig operations
            'datetime': string, time in datetime format
    """
    
    date_address = address+str(gps_week[0])+'/COM'+str(gps_week[1])+str(gps_week[2])+'.EPH.Z'
    
    data = urllib.request.urlretrieve(date_address, filename='temp/temp_file.Z')
    foo_proc = Popen(['uncompress', 'temp/temp_file.Z'], stdin=PIPE, stdout=PIPE)
    foo_proc.communicate(input=b'yes')

    curr_year = ''
    curr_month = ''
    curr_day = ''
    curr_hour = ''
    curr_min = ''
    
    temp_table = []
    with open('temp/temp_file', "r") as bzfin:
        """ Handle lines here """
        lines = []
        for i, line in enumerate(bzfin):
            #if i == 10000: break
            curr_line = line.rstrip()
            #curr_line = str(curr_line,'utf-8')
            curr_line = curr_line.split()
            lines.append(curr_line)

            if curr_line[0]=='*':
                curr_year = curr_line[1]
                curr_month = curr_line[2]
                curr_day = curr_line[3]
                curr_hour = curr_line[4]
                curr_min = curr_line[5]

            if re.match('P[GRECJ][0-9]{2}',curr_line[0]):
                sat = re.findall('P([GRECJ])[0-9]{2}',curr_line[0])[0]
                temp_table.append([curr_year,curr_month,curr_day,curr_hour,curr_min, sat,curr_line[0],
                                   float(curr_line[1]),float(curr_line[2]),float(curr_line[3]),float(curr_line[4])])
    
    temp_pd = pd.DataFrame(np.array(temp_table),columns = ['year','month','day','hour','minute','satellite','name','c1','c2','c3','c4'])
    
    temp_pd.c1 = temp_pd.c1.astype(float,copy = False)*1000;
    temp_pd.c2 = temp_pd.c2.astype(float,copy = False)*1000;
    temp_pd.c3 = temp_pd.c3.astype(float,copy = False)*1000;
    temp_pd.year = temp_pd.year.astype(int,copy = False);
    temp_pd.month = temp_pd.month.astype(int,copy = False);
    temp_pd.day = temp_pd.day.astype(int,copy = False);
    temp_pd.hour = temp_pd.hour.astype(int,copy = False);
    temp_pd.minute = temp_pd.minute.astype(int,copy = False);

    temp_pd['time_stamp'] = temp_pd.apply(lambda row:datetime.datetime(
        row['year'], row['month'], row['day'], row['hour'], row['minute']).timestamp(),axis = 1)
    temp_pd['key'] = 1
    temp_pd['datetime'] = pd.to_datetime(temp_pd[['year','month','day','hour','minute']])


    return temp_pd

def el_al_single_station(temp_pd, curr_stat):
    """Calculate azimuth and elevation for all satellites present in
    temp_pd and a single station curr_stat
    Does the calculation directly in the Dataframe, line by line: slow.
    
    Parameters
    ----------
    temp_pd : Pandas dataframe
        Output of import_RM_file function
    curr_stat: Pandas dataframe
        Output of import_stations function
        
    Returns
    -------
    temp_pd : pandas dataframe
        Copy of input Dataframe with added information:
            'dXgeo', dYgeo','dZgeo': float, station-satellite vector
            'dXloc': array, local ellipsoidal eccentricities
            'a': float, azimuth
            'e': float, elevation
    """
    
    temp_pd['dXgeo'] = temp_pd.c1-curr_stat.x_pos
    temp_pd['dYgeo'] = temp_pd.c2-curr_stat.y_pos
    temp_pd['dZgeo'] = temp_pd.c3-curr_stat.z_pos
    
    curr_rot_mat = rot_mat(curr_stat['rad_stat_coord'][0], curr_stat['rad_stat_coord'][1])
    temp_pd['dXloc'] = temp_pd.apply(lambda row: local_ecc(curr_rot_mat, row['dXgeo'], row['dYgeo'], row['dZgeo']),axis =1)
    
    #temp_pd['dXloc'] = temp_pd.apply(lambda row: staUtil.eccell(curr_stat['rad_stat_coord'][0], curr_stat['rad_stat_coord'][1], curr_stat['rad_stat_coord'][2], row['dXgeo'], row['dYgeo'], row['dZgeo']),axis =1)
    
    temp_pd['a'] = temp_pd.dXloc.apply(lambda x: np.arctan2(x[1],x[0]))
    temp_pd['e'] = temp_pd.dXloc.apply(lambda x: np.arctan(x[2]/np.sqrt(x[0]**2+x[1]**2)))
    
    #grouped = temp_pd.groupby('name')

    return temp_pd


def el_al_single_station_fast(temp_pd, curr_stat):
    """Calculate azimuth and elevation for all satellites present in
    temp_pd and a single station curr_stat. 
    Does the calculation as a single matrix operation: fast.
    
    Parameters
    ----------
    temp_pd : Pandas dataframe
        Output of import_RM_file function
    curr_stat: Pandas dataframe
        Output of import_stations function
        
    Returns
    -------
    temp_pd : pandas dataframe
        Copy of input Dataframe with added information:
            'dXgeo', dYgeo','dZgeo': float, station-satellite vector
            'a': float, azimuth
            'e': float, elevation
    """
    
    curr_stat['key'] = 1
    temp_pd['key'] = 1
    
    curr_rot_mat = rot_mat(curr_stat.rad_stat_coord[0], curr_stat.rad_stat_coord[1])
    
    merged_table = pd.merge(temp_pd, pd.DataFrame(curr_stat).T,on='key')
    
    sub_coord = np.empty((len(merged_table),3,1))
    sub_coord[:,0,0] = merged_table.c1-merged_table.x_pos
    sub_coord[:,1,0] = merged_table.c2-merged_table.y_pos
    sub_coord[:,2,0] = merged_table.c3-merged_table.z_pos
    
    temp_pd['Xgeo'] = pd.Series(sub_coord[:,0,0])
    temp_pd['Ygeo'] = pd.Series(sub_coord[:,1,0])
    temp_pd['Zgeo'] = pd.Series(sub_coord[:,2,0])
    
    new_vect = np.squeeze(np.matmul(curr_rot_mat,sub_coord))
    
    merged_table['dXloc'] = pd.Series(new_vect[:,0])
    merged_table['dYloc'] = pd.Series(new_vect[:,1])
    merged_table['dZloc'] = pd.Series(new_vect[:,2])

    temp_pd['a'] = np.arctan2(merged_table['dYloc'],merged_table['dXloc'])
    temp_pd['e'] = np.arctan(merged_table['dZloc']/np.sqrt(merged_table['dXloc']**2+merged_table['dYloc']**2))
    
    #grouped = temp_pd.groupby('name')

    return temp_pd


def rotation_mat_for_grid(coord_table):
    """Given a datframe with a list of coordinates, calculates for each position
    the rotation matrix necessary to calculate local ellipsoidal eccentricities.
    
    Parameters
    ----------
    coord_table : Pandas dataframe
        Output of coord_grid function
        
    Returns
    -------
    coord_table : pandas dataframe
        Copy of input Dataframe with added information:
            'rot_mat': 2d numpy array, rotation matrix
    """
    
    coord_table['rot_mat'] = coord_table.apply(lambda row: rot_mat(row['x_geod'], row['y_geod']),axis = 1)
    return coord_table


def elevation_azimuth_grid(coord_table, temp_pd):
    """Calculate azimuth and elevation for all satellites present in
    temp_pd and a series of stations present in coord_table. 
    Does the calculation as a single matrix operation: fast.
    
    Parameters
    ----------
    temp_pd : Pandas dataframe
        Output of import_RM_file function
    coord_table: Pandas dataframe
        Output of coord_grid function
        
    Returns
    -------
    merged_table : pandas dataframe
        Dataframe where each of N satellites has been paired with each of M stations,
        to give a dataframe with NxM lines where all fields of both tables have been copied. 
        Additional informatio added is:
            'dXloc': array, local ellipsoidal eccentricities
            'a': float, azimuth
            'e': float, elevation
    """
    
    #merge satellites positions (S) and grid positions (G) to create a dataframe with SxG lines containing
    #information for a given satellite + a given observation position
    merged_table = pd.merge(temp_pd, coord_table,on='key')

        
    #calculate rotation matrix for all grid coordinates
    coord_table['rot_mat'] = coord_table.apply(lambda row: rot_mat(row['x_geod'], row['y_geod']),axis = 1)
    #create an Nx3x3 matrix where N = G*S where G = #of grid points and S = #of satellites positions
    #each block of Sx3x3 matrices is identical and contains all rotation matrices for each specific grid point
    matrot = np.concatenate([np.array(list(coord_table.rot_mat))]*len(temp_pd));
    
    #calculate vectors from satellite to observation
    #merged_table['dXgeo'] = merged_table.c1-merged_table.x_pos
    #merged_table['dYgeo'] = merged_table.c2-merged_table.y_pos
    #merged_table['dZgeo'] = merged_table.c3-merged_table.z_pos
    
    sub_coord = np.empty((len(merged_table),3,1))
    sub_coord[:,0,0] = merged_table.c1-merged_table.x_pos
    sub_coord[:,1,0] = merged_table.c2-merged_table.y_pos
    sub_coord[:,2,0] = merged_table.c3-merged_table.z_pos
    
    #extract Nx3x3 rotation matrix and Nx3x1 vectors to do rotation with numpy
    #matrot = np.array(list(merged_table.rot_mat))
    #matrot = np.stack(merged_table.rot_mat,axis= 0)
    #sub_coord = np.array(merged_table[['dXgeo','dYgeo','dZgeo']])
    #sub_coord = np.swapaxes(np.expand_dims(sub_coord,1),1,2)
    new_vect = np.squeeze(np.matmul(matrot,sub_coord))
    
    merged_table['dXloc'] = pd.Series(new_vect[:,0])
    merged_table['dYloc'] = pd.Series(new_vect[:,1])
    merged_table['dZloc'] = pd.Series(new_vect[:,2])

    merged_table['a'] = np.arctan2(merged_table['dYloc'],merged_table['dXloc'])
    merged_table['e'] = np.arctan(merged_table['dZloc']/np.sqrt(merged_table['dXloc']**2+merged_table['dYloc']**2))
    
    return merged_table


def plot_radial(data, all_dates, all_dates_read, stations, station_sel, sat_type, date_min, date_max,
                col0, col1, col2, col3, col4):
    """Function passed to ipywidget.interactive_output() called in the aiub.interactive_rad() function.
    Selects the data (station, time, satellite type) to be plotted as interactively set by the user.
    
    Parameters
    ----------
    data : Pandas dataframe
        Output of aiub.import_RM_file function
    all_dates: int array, 
        list of all available dates in timestamp format
    stations: pandas dataframe
        station dataframe as output of aiub.import_stations()
    station_sel: string
        selected station
    sat_type: string list
        selected satellite types
    date_min, date_max: int
        minimal/maximal time to plot in timestamp format
    col0, col1, col2, col3, col4: string
        color for each satellite type
        
       
        
    Returns
    -------
    
    """
    
    #calculate e and a for selected data only if selected station has changed 
    if station_sel != data.iloc[0].curr_stat:
        curr_stat = stations[stations.statname == station_sel].iloc[0]
        rad_stat_coord = cartesian_to_ellipsoidal(curr_stat.x_pos, curr_stat.y_pos,curr_stat.z_pos)
        curr_stat['rad_stat_coord'] = rad_stat_coord
        data = el_al_single_station_fast(data, curr_stat)
        data['curr_stat'] = curr_stat.statname
    
    
    all_cols = [col0, col1, col2, col3, col4]
    min_time = all_dates[all_dates_read.index(date_min)]
    max_time = all_dates[all_dates_read.index(date_max)]
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='polar')
    #ax.set_rlim(90, 0,1)
    
    #ubselect satellites, time and elevation > 5° for plotting
    for s in sat_type:
        subselect = data[data.satellite == s]
        subselect = subselect[subselect.e >2*np.pi*5/360]
        subselect = subselect[subselect.time_stamp >= min_time]
        subselect = subselect[subselect.time_stamp <= max_time]
        grouped = subselect.groupby('name')
    
        pos_sat = np.argwhere(np.array(data.satellite.unique()) == s)[0][0]
        for gname, gframe in grouped:
            if np.any(gframe.time_stamp.diff().dropna()!=900):
                split_pos = np.argwhere(gframe.time_stamp.diff().dropna()!=900)[0][0]
                df1 = gframe.iloc[:split_pos+1,:]
                df2 = gframe.iloc[split_pos+1:,:]
                plt.plot(df1.a,-360*df1.e/(2*np.pi), color = all_cols[pos_sat])
                plt.plot(df2.a,-360*df2.e/(2*np.pi), color = all_cols[pos_sat])
            else:
                plt.plot(gframe.a,-360*gframe.e/(2*np.pi), color = all_cols[pos_sat])
    
    #ax.set_ylim(90, 0)
    ax.set_title('From '+ date_min+'   To '+ date_max,pad = 20)
    plt.show()
    
    
def interactive_rad(temp_pd, stations):
    """Interactive plotting of elevation-azimuth plots for a set of satellites and a set
    of stations
    
    Parameters
    ----------
    temp_pd : Pandas dataframe
        Output of aiub.import_RM_file function
    stations: pandas dataframe
        station dataframe as output of aiub.import_stations()
       
        
    Returns
    -------
    
    """
    
    all_dates = temp_pd.time_stamp.unique()#temp_pd.datetime.apply(lambda x: x.timestamp()).unique()
    #all_dates_read = temp_pd.datetime.unique()
    
    time_list = list(map(lambda x: datetime.datetime.fromtimestamp(x), temp_pd.time_stamp.unique()))
    all_dates_read = [str(tl.year)+'-'+str(tl.month)+'-'+str(tl.day)+'  '+str(tl.hour)+':'+str(tl.minute) for tl in time_list]
    
    colors = ['red','blue','green','cyan','orange']
    items = [widgets.ColorPicker(description=temp_pd.satellite.unique()[i],value = colors[i]) for i in range(len(temp_pd.satellite.unique()))]
    if len(items)<5:
        items = items + [widgets.ColorPicker(description='None',value = colors[i]) for i in range(5-len(items))]
    colwidget = widgets.VBox(items)

    #min_date_widget = widgets.IntSlider(min=0, max=len(all_dates)-1, step=1, value = 0, continuous_update = False)
    #max_date_widget = widgets.IntSlider(min=0, max=len(all_dates)-1, step=1, value = 0, continuous_update = False)
    
    style = {'description_width': '20%','readout_width' : '20%'}
    min_date_widget = widgets.SelectionSlider(options = all_dates_read,style=style,description = 'Min Time',
                                       layout={'width': '400px'})
    max_date_widget = widgets.SelectionSlider(options = all_dates_read,style=style,description = 'Max Time',
                                       layout={'width': '400px'})

    sat_type = widgets.SelectMultiple(options = temp_pd.satellite.unique(),
                                     value = [temp_pd.satellite.unique()[0]],disabled=False)
    
    station_widget = widgets.Dropdown(options=stations.statname.unique(),value=stations.iloc[0].statname)

    d = {'data': widgets.fixed(temp_pd), 'all_dates': widgets.fixed(all_dates),'all_dates_read': widgets.fixed(all_dates_read),
          'sat_type': sat_type, 'date_min': min_date_widget,'date_max': max_date_widget, 'stations': widgets.fixed(stations),
         'station_sel': station_widget}
    d2 = {'col'+str(ind): value for (ind, value) in enumerate(items)}
    d.update(d2)


    w1 = widgets.HBox([sat_type])
    w2 = widgets.HBox([station_widget])
    w3 = widgets.HBox([min_date_widget,max_date_widget])
    ui = widgets.VBox([w1,w2,w3, colwidget])
    
    out = widgets.interactive_output(plot_radial, d)
    display(ui,out)
    
    
def plot_map(t, sat, temp_pd, all_dates, coord_table, lons, lats):
    """Plot a density map of visible satellites on a grid
    
    Parameters
    ----------
    t : int
        time point in list of all time points
    sat: str
        satellite type to plot
    temp_pd: pandas dataframe
        dataframe with elevation and azimuth, e.g. output of aiub.elevation_azimuth_grid()
    all_dates: list
        list of all available dates
    coord_table: pandas dataframe
        dataframe with coordinates of a grid as output by aiub.coord_grid()
    lons: numpy array
        list of longitudes as output by aiub.coord_grid()
    lats: numpy array
        list of lattitudes as output by aiub.coord_grid()
       
        
    Returns
    -------
    
    """
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines()
    
    date_time = all_dates[t]
    curr_pd = temp_pd[temp_pd.datetime == date_time]
    curr_pd = curr_pd[curr_pd.satellite == sat]
    el_az = elevation_azimuth_grid(coord_table, curr_pd)
    grouped = el_az[(el_az.e>np.deg2rad(5))].groupby('datetime')
    
    data = np.reshape(list(grouped.get_group(date_time).groupby('pos_ind').size()),(len(lons),len(lats)))
    
    ax.contourf(np.rad2deg(lons), np.rad2deg(lats), data, transform = ccrs.PlateCarree(),vmin=0,vmax = 20)  # didn't use transform, but looks ok...
    ax.set_title(date_time)
    plt.show();
    
    
def plot_map_grouped(t, sat, grouped, all_dates, lons, lats):
    """Plot a density map of visible satellites on a grid
    
    Parameters
    ----------
    t : int
        time point in list of all time points
    sat: str
        satellite type to plot
    grouped: pandas group
        group of dataframe with elevation and azimuth, e.g. output 
        of aiub.elevation_azimuth_grid(), grouped by time
    all_dates: list
        list of all available dates
    lons: numpy array
        list of longitudes as output by aiub.coord_grid()
    lats: numpy array
        list of lattitudes as output by aiub.coord_grid()   
        
    Returns
    -------
    
    """
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines()
    
    date_time = all_dates[t]
    
    subgroup = grouped.get_group(date_time)
    subgroup = subgroup[subgroup.satellite == sat]

    data = np.reshape(list(subgroup.groupby('pos_ind').size()),(len(lons),len(lats)))
    
    ax.contourf(np.rad2deg(lons), np.rad2deg(lats), data, transform = ccrs.PlateCarree(),vmin=0,vmax = 20)  # didn't use transform, but looks ok...
    ax.set_title(date_time)
    plt.show()
    
def date_picker():    
    date1 = widgets.DatePicker(
        description='Pick a start date',
        disabled=False, style = {'description_width': '40%'}
    )
    date2 = widgets.DatePicker(
        description='Pick an end date',
        disabled=False, style = {'description_width': '40%'}
    )
    date1.value = date(2018,9,1)
    date2.value = date(2018,9,3)
    display(widgets.HBox([date1, date2]))
    return date1, date2

    
    
    
    
    
    
    
    
    
    
    
    