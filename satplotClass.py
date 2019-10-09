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
import aiub
from IPython.display import clear_output


class Radplot:
    """Parsing of MicroManager metadata"""
    def __init__(self, temp_pd = None, curr_stat = None, stations = None, address_sat=None):
        """Standard __init__ method.
        Parameters
        ----------
        show_output : bool
            show segmentation images during analysis 
        """
                
        self.temp_pd = temp_pd
        self.curr_stat = curr_stat
        self.stations = stations
        self.address_sat = address_sat
        
        
    def main_date(self):
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
        
        self.date1 = date1.value
        self.date2 = date2.value

        button = widgets.Button(
        description='Update time range',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='check')
        button.on_click(self.interactive_rad)

        d = {'date1': date1, 'date2': date2}


        ui = widgets.VBox([widgets.HBox([date1, date2]), button])
        out = widgets.interactive_output(self.update_date, d)
        
        self.ui = ui
        self.out = out
        display(ui,out)

    def update_date(self, date1, date2):
        self.date1 = date1
        self.date2 = date2
        
    def interactive_rad(self, on_click):
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
        
        clear_output()
        display(self.ui,self.out)
        stations= self.stations
    
        #create a list of dates between start and end date 
        date_list = [self.date1 + timedelta(days=x) for x in range((self.date2-self.date1).days+1)]

        #calculate gps weeks times (year, week, day)
        gps_weeks = [aiub.date_to_gpsweeks(x) for x in date_list]

        #load satellite data
        temp_pd = pd.concat([aiub.import_RM_file(self.address_sat, g) for g in gps_weeks]).reset_index()

        #reanme satellite types
        sat_dict = {'G':'GPS','R':'GLONASS','E':'Galileo','C':'BeiDou','J':'QZSS'}
        temp_pd['satellite'] = temp_pd.satellite.apply(lambda x : sat_dict[x])
        self.temp_pd = temp_pd

        #caluclate elevation and azimuth for all satellites for given station
        temp_pd = aiub.el_al_single_station_fast(temp_pd, self.curr_stat)
        temp_pd['curr_stat'] = self.curr_stat.statname

        #temp_pd = self.temp_pd

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
                                           layout={'width': '400px'}, continuous_update = False)
        max_date_widget = widgets.SelectionSlider(options = all_dates_read,style=style,description = 'Max Time',
                                           layout={'width': '400px'},continuous_update = False)

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

        out = widgets.interactive_output(aiub.plot_radial, d)
        display(ui,out)