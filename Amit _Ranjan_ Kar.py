# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 00:10:32 2022

@author: Amit Ranjan Kar
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import seaborn as sns
from mpl_toolkits.basemap import Basemap
#import warnings
#warnings.filterwarnings("ignore")
#--------------------------------------------------------------------------------
# defiitions for plot size and line thickness
def mm2inch(*tupl):
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
def mm2point(mm):
    return mm/(25.4/72)
font = {'family' : 'Arial',
         'weight' : 'normal',
         'size'   : 15}
mpl.rc('font', **font)
mpl.rcParams['axes.linewidth'] = mm2point(0.4)
mpl.rcParams['ytick.major.width'] = mm2point(0.4)
mpl.rcParams['xtick.major.width'] = mm2point(0.4)
#--------------------------------------------------------------

# land cover analysis
path='D:/BTU FOLDER/land surface and atmosphere interaction/Final/LsAI_EX/'

# land cover analysis(2003)
df_luc_2003=xr.open_dataset(path+'LUC_SAf_2003.nc') 
df_luc2003=df_luc_2003.sel(lat=slice(-20,-24),lon=slice(22,26))

lats1 = df_luc2003.variables['lat'][:]
lons1 = df_luc2003.variables['lon'][:]
lccs1 = df_luc2003.variables['lccs_class'][:]

map1 = Basemap(projection = 'merc',
             llcrnrlon =  22,
             llcrnrlat = -24,
             urcrnrlon = 26,
             urcrnrlat = -20,
             resolution ='i')


lon1,lat1 = np.meshgrid(lons1, lats1)
x,y =map1(lon1, lat1)

c_scheme = map1.pcolor(x, y, np.squeeze(lccs1[0,:,:]), cmap ='jet' )
cbar=map1.colorbar(c_scheme, location="right", pad= "10%")

cbar.ax.tick_params(labelsize=10)
#map1.drawcoastlines()
map1.drawstates()
map1.drawcountries()
map1.drawparallels(np.arange(-24,-20,1),labels=[True,False,False,False], fontsize=10)
map1.drawmeridians(np.arange(22,26,1),labels=[0,0,0,1], fontsize=10)
plt.title('Land cover analysis (2003)', fontsize=15)
src='D:/BTU FOLDER/land surface and atmosphere interaction/Final/Weather_Station_Climate_Diagram/'
plt.savefig(src+ 'Fig1_LUC_03.png',format='png', dpi=300, bbox_inches='tight')
plt.show()

# land cover analysis(2019)
df_luc_2019=xr.open_dataset(path+'LUC_SAf_2019.nc') 
df_luc2019=df_luc_2019.sel(lat=slice(-20,-24),lon=slice(22,26))

lats2 = df_luc2019.variables['lat'][:]
lons2 = df_luc2019.variables['lon'][:]
lccs2 = df_luc2019.variables['lccs_class'][:]

map2 = Basemap(projection = 'merc',
             llcrnrlon =  22,
             llcrnrlat = -24,
             urcrnrlon = 26,
             urcrnrlat = -20,
             resolution ='i')


lon2,lat2 = np.meshgrid(lons2, lats2)
x,y =map2(lon2, lat2)

c_scheme = map2.pcolor(x, y, np.squeeze(lccs1[0,:,:]), cmap ='jet' )
cbar=map2.colorbar(c_scheme, location="right", pad= "10%")

cbar.ax.tick_params(labelsize=10)
#map2.drawcoastlines()
map2.drawstates()
map2.drawcountries()
map2.drawparallels(np.arange(-24,-20,1),labels=[True,False,False,False], fontsize=10)
map2.drawmeridians(np.arange(22,26,1),labels=[0,0,0,1], fontsize=10)
plt.title('Land cover analysis (2019)', fontsize=15)
src='D:/BTU FOLDER/land surface and atmosphere interaction/Final/Weather_Station_Climate_Diagram/'
plt.savefig(src+ 'Fig2_LUC_19.png',format='png', dpi=300, bbox_inches='tight')
plt.show()

# land cover classes changes between  2003 to 2019

lccs1_2 = lccs2-lccs1 
map2_1 = Basemap(projection = 'merc',
             llcrnrlon =  22,
             llcrnrlat = -24,
             urcrnrlon = 26,
             urcrnrlat = -20,
             resolution ='i')

lon2,lat2 = np.meshgrid(lons2, lats2)
x,y =map2_1(lon2, lat2)


c_scheme = map2_1.pcolor(x, y, np.squeeze(lccs1_2[0,:,:]), cmap ='jet' )
cbar=map2_1.colorbar(c_scheme, location="right", pad= "10%")

cbar.ax.tick_params(labelsize=10)
#mp1_2.drawcoastlines()
map2_1.drawstates()
map2_1.drawcountries()
map2_1.drawparallels(np.arange(-24,-20,1),labels=[True,False,False,False], fontsize=10)
map2_1.drawmeridians(np.arange(22,26,1),labels=[0,0,0,1], fontsize=10)
plt.title('Land cover changes 2003 to 2019', fontsize=15)
src='D:/Winter 2021-22/Land Surface- Atmosphere Interaction/Final project/Image/'
plt.savefig(src+ 'Fig3_LUC_changes.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
#-------------------------------------------------------------------------------------------------------

#Analyze atmospheric conditions
path='D:/BTU FOLDER/land surface and atmosphere interaction/Final/LsAI_EX/'
df_atmospheric=xr.open_dataset(path+'Atmo_SAf.nc') 

#precipitation analysis 
weights = np.cos(np.deg2rad(df_atmospheric.tp))
tp_weighted = df_atmospheric.tp.weighted(weights) 
weighted_mean = tp_weighted.mean(("longitude", "latitude")) #weighted area average
df_atmospheric_precipitation=pd. DataFrame(weighted_mean)
df_atmospheric_precipitation["Date"]= df_atmospheric.time

df_atmospheric_precipitation["Date"]= pd.to_datetime(df_atmospheric_precipitation["Date"])
df_atmospheric_precipitation=df_atmospheric_precipitation.set_index(['Date'])
df_atmospheric_precipitation.columns=["Precipitation"] 

#years subsetting
df_atmospheric_precipitation2003= df_atmospheric_precipitation['2003-01-01 00:00:00':'2003-12-01 00:00:00']
df_atmospheric_precipitation2019= df_atmospheric_precipitation['2019-01-01 00:00:00':'2019-12-01 00:00:00']  

#precipitation plotting
df_atmospheric_precipitation2003["Month"]= df_atmospheric_precipitation2003.index.month 
df_atmospheric_precipitation2019["Month"]= df_atmospheric_precipitation2019.index.month 

figure1, ax1 = plt.subplots(1, figsize=(14,10))
ax1.plot(df_atmospheric_precipitation2003.Month, df_atmospheric_precipitation2003.Precipitation, "bo-", label="Precipitation (2003)")
ax1.set_yticks(np.arange(0.00, 0.008, 0.002))
ax1.set_ylabel('Monthly Precipitation (m)')
ax1.set_xlabel('Month')
#ax1.set_xticks(df_atmospheric_precipitation2003.Month)
mon = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
ax1.set_xticklabels(mon)
ax1.legend(loc="upper left", ncol=1)
#ax=ax1.plot
ax1.set_title('Precipitation Variation by Month (2003 and 2019)')

ax2=ax1.twinx()
#ax2=ax1.twiny()
ax2.plot(df_atmospheric_precipitation2019.Month, df_atmospheric_precipitation2019.Precipitation, "go-", label="Precipitation (2019)")
ax2.set_xticks(df_atmospheric_precipitation2003.Month)
mon = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
ax2.set_xticklabels(mon)
ax2.set_yticklabels([]) 
ax2.set_yticks([])
ax2.legend(loc="upper right", ncol=1)
figure1.tight_layout()
figure1.show()

#Analysis of evaporation on a monthly basis
weights = np.cos(np.deg2rad(df_atmospheric.e))
e_weighted = df_atmospheric.e.weighted(weights) 
weighted_mean_e = e_weighted.mean(("longitude", "latitude")) #weighted average over the area
df_atmospheric_evaporation=pd. DataFrame(weighted_mean_e)
df_atmospheric_evaporation["Date"]= df_atmospheric.time #adding column for date 

df_atmospheric_evaporation["Date"]= pd.to_datetime(df_atmospheric_evaporation["Date"])
df_atmospheric_evaporation=df_atmospheric_evaporation.set_index(['Date'])
df_atmospheric_evaporation.columns=["Evaporartion"]  

#Subsetting the periods
df_atmospheric_evaporation2003= df_atmospheric_evaporation['2003-01-01 00:00:00':'2003-12-01 00:00:00']
df_atmospheric_evaporation2019= df_atmospheric_evaporation['2019-01-01 00:00:00':'2019-12-01 00:00:00']        


#Evaporation plotting
df_atmospheric_evaporation2003["Month"]= df_atmospheric_evaporation2019.index.month 
df_atmospheric_evaporation2019["Month"]= df_atmospheric_evaporation2019.index.month 

figure2, ax3 = plt.subplots(1, figsize=(14,10))
ax3.plot(df_atmospheric_evaporation2003.Month, df_atmospheric_evaporation2003.Evaporartion,'o-', color='orange',  label="Evaporation (2003)")
ax3.set_yticks(np.arange( 0.00, -0.005, -0.001))
ax3.set_ylabel('Monthly Evaporation (m of water equivalent)')
ax3.set_xlabel('Month')
ax3.legend(loc="upper left", ncol=1)
ax3.set_title('Evaporartion Variation by Month (2003 and 2019)')
ax3.invert_yaxis()

ax4=ax3.twinx()
ax4.plot(df_atmospheric_evaporation2019.Month, df_atmospheric_evaporation2019.Evaporartion, 'o-', color='brown', label="Evaporation (2019)")
ax4.set_xticks(df_atmospheric_evaporation2019.Month)
mon = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
ax4.set_xticklabels(mon)
ax4.set_yticklabels([]) 
ax4.set_yticks([])
ax4.legend(loc="upper right", ncol=1)
ax4.invert_yaxis()
figure2.tight_layout()
figure2.show()
# Temperature analysis at 2 meters on a monthly basis
weights = np.cos(np.deg2rad(df_atmospheric.e))
t2m_weighted =df_atmospheric.t2m.weighted(weights) 
weighted_mean_t2m = t2m_weighted.mean(("longitude", "latitude")) #weighted average over the area
df_atmospheric_temperature=pd. DataFrame(weighted_mean_t2m)
df_atmospheric_temperature["Date"]= df_atmospheric.time #adding column for date 

df_atmospheric_temperature["Date"]= pd.to_datetime(df_atmospheric_temperature["Date"])
df_atmospheric_temperature=df_atmospheric_temperature.set_index(['Date'])
df_atmospheric_temperature.columns=["Temperature"]  

#Subsetting the years
df_atmospheric_temperature2003=df_atmospheric_temperature['2003-01-01 00:00:00':'2003-12-01 00:00:00']
df_atmospheric_temperature2019= df_atmospheric_temperature['2019-01-01 00:00:00':'2019-12-01 00:00:00']  

#Temperature plotting
df_atmospheric_temperature2003["Month"]= df_atmospheric_temperature2019.index.month 
df_atmospheric_temperature2019["Month"]= df_atmospheric_temperature2019.index.month 

figure3, ax5 = plt.subplots(1, figsize=(14,10))
ax5.plot(df_atmospheric_temperature2003.Month, df_atmospheric_temperature2003.Temperature,'o-', color='red',  label="Temperature (2003)")
ax5.set_yticks(np.arange( 288, 300, 2))
ax5.set_ylabel('Monthly Average Temperature (K)')
ax5.set_xlabel('Month')
ax5.legend(loc="upper left", ncol=1)
ax5.set_title('Temperature analysis at 2 meters on a monthly basis (2003 and 2019)')
#ax5.invert_yaxis()

ax6=ax5.twinx()
ax6.plot(df_atmospheric_temperature2019.Month, df_atmospheric_temperature2019.Temperature, 'o-', color='purple', label="Temperature(2019)")
ax6.set_xticks(df_atmospheric_temperature2019.Month)
mon = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
ax6.set_xticklabels(mon)
ax6.set_yticklabels([]) 
ax6.set_yticks([])
ax6.legend(loc="upper right", ncol=1)
#ax6.invert_yaxis()
figure3.tight_layout()
figure3.show()
#-------------------------------------------------------------------------------

#Analyze soil conditions
path='D:/BTU FOLDER/land surface and atmosphere interaction/Final/LsAI_EX/'
df_soil=xr.open_dataset(path+'Sfc_SAf.nc') 

#Temperature analysis 
weights_sfc_stl = np.cos(np.deg2rad(df_soil.stl1))
stl1_weighted = df_soil.stl1.weighted(weights_sfc_stl) 
weighted_mean_stl1 = stl1_weighted.mean(("longitude", "latitude")) #weighted average over the area
df_soil_temperature=pd. DataFrame(weighted_mean_stl1)
df_soil_temperature["Date"]= df_soil.time

df_soil_temperature["Date"]= pd.to_datetime(df_soil_temperature["Date"])
df_soil_temperature=df_soil_temperature.set_index(['Date'])
df_soil_temperature.columns=["Temperature"]  

#Subsetting the periods
df_soil_temperature2003= df_soil_temperature['2003-01-01 00:00:00':'2003-12-01 00:00:00']
df_soil_temperature2019= df_soil_temperature['2019-01-01 00:00:00':'2019-12-01 00:00:00']   

#Temperature plotting 
df_soil_temperature2003["Month"]= df_soil_temperature2003.index.month 
df_soil_temperature2019["Month"]= df_soil_temperature2003.index.month 

#plotting of the soil temperature
df_soil_temperature2003["Month"]= df_soil_temperature2003.index.month 
df_soil_temperature2019["Month"]= df_soil_temperature2003.index.month 

figure4, ax7 = plt.subplots(1, figsize=(14,10))
ax7.plot(df_soil_temperature2003.Month, df_soil_temperature2003.Temperature, "o-", color="red", label="Temperarture (2003)")
ax7.set_yticks(np.arange(288, 301, 2))
ax7.set_ylabel('Soil Temperature (K)')
ax7.set_xlabel('Month')
#ax7.set_xticks(df_atmospheric_precipitation2003.Month)
mon = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
ax7.set_xticklabels(mon)
ax7.legend(loc="lower left", ncol=1)
#ax=ax7.plot
ax7.set_title(' Soil Temperature Variation by Month (2003 and 2019)')

ax8=ax7.twinx()
#ax8=ax1.twiny()
ax8.plot(df_soil_temperature2019.Month, df_soil_temperature2019.Temperature, "o-", color= "purple", label="Temperature (2019)")
ax8.set_xticks(df_soil_temperature2019.Month)
mon = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
ax8.set_xticklabels(mon)
ax8.set_yticklabels([]) 
ax8.set_yticks([])
ax8.legend(loc="lower right", ncol=1)
figure4.tight_layout()
figure4.show()

#Latent heat flux analysis 
weights_soil_hf = np.cos(np.deg2rad(df_soil.stl1))
slhf_weighted = df_soil.slhf.weighted(weights_soil_hf) 
weighted_mean_slhf = slhf_weighted.mean(("longitude", "latitude")) #weighted average over the area
df_soil_hf =pd. DataFrame(weighted_mean_slhf)
df_soil_hf ["Date"]= df_soil.time

df_soil_hf["Date"]= pd.to_datetime(df_soil_hf["Date"])
df_soil_hf=df_soil_hf.set_index(['Date'])
df_soil_hf.columns=["Heat_Flux"]  

#Subsetting the periods
df_soil_hf2003= df_soil_hf['2003-01-01 00:00:00':'2003-12-01 00:00:00']
df_soil_hf2019= df_soil_hf['2019-01-01 00:00:00':'2019-12-01 00:00:00']

#Latent heat plotting
df_soil_hf2003["Month"]=df_soil_hf2003.index.month 
df_soil_hf2019["Month"]= df_soil_hf2019.index.month 

figure5, ax9 = plt.subplots(1, figsize=(14,10))
ax9.plot(df_soil_hf2003.Month, df_soil_hf2003.Heat_Flux, "o-", color="red", label="Latent heat flux (2003)")
ax9.set_yticks(np.arange(0e6, -10e6, -1e6))
ax9.set_ylabel('heat flux (J/m2)')
ax9.set_xlabel('Month')
ax9.legend(loc="lower left", ncol=1)
#ax=ax7.plot
ax9.set_title('Latent heat flux Variation by Month (2003 and 2019)')
ax9.invert_yaxis()

ax10=ax9.twinx()
ax10.plot(df_soil_hf2019.Month, df_soil_hf2019.Heat_Flux, "o-", color= "purple", label="Latent heat flux (2019)")
ax10.set_xticks(df_soil_hf2019.Month)
mon = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
ax10.set_xticklabels(mon)
ax10.set_yticklabels([]) 
ax10.set_yticks([])
ax10.legend(loc="lower right", ncol=1)
ax10.invert_yaxis()
figure5.tight_layout()
figure5.show()

#Sensible heat flux analysis 
weights_soil_shf = np.cos(np.deg2rad(df_soil.sshf))
sshf_weighted =df_soil.sshf.weighted(weights_soil_shf) 
weighted_mean_sshf = sshf_weighted.mean(("longitude", "latitude")) #weighted average over the area
df_soil_shf=pd. DataFrame(weighted_mean_sshf)
df_soil_shf["Date"]= df_soil.time

df_soil_shf["Date"]= pd.to_datetime(df_soil_shf["Date"])
df_soil_shf=df_soil_shf.set_index(['Date'])
df_soil_shf.columns=["Heat_Flux"] 

#Subsetting the periods
df_soil_shf2003= df_soil_shf['2003-01-01 00:00:00':'2003-12-01 00:00:00']
df_soil_shf2019= df_soil_shf['2019-01-01 00:00:00':'2019-12-01 00:00:00']   

#plotting of the sensible heat 
df_soil_shf2003["Month"]= df_soil_shf2003.index.month 
df_soil_shf2019["Month"]= df_soil_shf2019.index.month 
     
figure6, ax11 = plt.subplots(1, figsize=(14,10))
ax11.plot(df_soil_shf2003.Month, df_soil_shf2003.Heat_Flux, "o-", color="red", label="Sensible heat flux (2003)")
ax11.set_yticks(np.arange(5e6, -11e6, -2e6))
ax11.set_ylabel('heat flux (J/m2)')
ax11.set_xlabel('Month')
ax11.legend(loc="lower left", ncol=1)
#ax=ax7.plot
ax11.set_title('Surface Sensible Heat Flux varition by Month (2003 and 2019)')
ax11.invert_yaxis()

ax12=ax11.twinx()
ax12.plot(df_soil_shf2019.Month, df_soil_shf2019.Heat_Flux, "o-", color= "purple", label="Sensible heat flux (2019)")
ax12.set_xticks(df_soil_shf2019.Month)
mon = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
ax12.set_xticklabels(mon)
ax12.set_yticklabels([]) 
ax12.set_yticks([])
ax12.legend(loc="lower right", ncol=1)
ax12.invert_yaxis()
figure6.tight_layout()
figure6.show()

#Volumetric soil-water layer analysis 
weights_soil_vswl = np.cos(np.deg2rad(df_soil.swvl1))
vswl_weighted = df_soil.swvl1.weighted(weights_soil_vswl) 
weighted_mean_vswl = vswl_weighted.mean(("longitude", "latitude")) #weighted average over the area
df_soil_vswl=pd. DataFrame(weighted_mean_vswl)
df_soil_vswl["Date"]= df_soil.time

df_soil_vswl["Date"]= pd.to_datetime(df_soil_vswl["Date"])
df_soil_vswl=df_soil_vswl.set_index(['Date'])
df_soil_vswl.columns=["SW_layer"]  

#Subsetting the periods
df_soil_vswl2003= df_soil_vswl['2003-01-01 00:00:00':'2003-12-01 00:00:00']
df_soil_vswl2019= df_soil_vswl['2019-01-01 00:00:00':'2019-12-01 00:00:00'] 

#Sensible heat plotting
df_soil_vswl2003["Month"]= df_soil_vswl2003.index.month 
df_soil_vswl2019["Month"]= df_soil_vswl2019.index.month 

figure7, ax13 = plt.subplots(1, figsize=(14,10))
ax13.plot(df_soil_vswl2003.Month, df_soil_vswl2003.SW_layer, "o-", color="red", label="Vol. S-W layer (2003)")
ax13.set_yticks(np.arange(0.1, 0.3, 0.02))
ax13.set_ylabel('Vol. S-W layer (m3/m3)')
ax13.set_xlabel('Month')
ax13.legend(loc="lower left", ncol=1)
#ax=ax7.plot
ax13.set_title('Volumetric Soil-Water Layer by Month (2003 and 2019)')
#ax13.invert_yaxis()

ax14=ax13.twinx()
ax14.plot(df_soil_vswl2019.Month, df_soil_vswl2019.SW_layer, "o-", color= "purple", label="Vol. S-W layer (2019)")
ax14.set_xticks(df_soil_vswl2019.Month)
mon = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
ax14.set_xticklabels(mon)
ax14.set_yticklabels([]) 
ax14.set_yticks([])
ax14.legend(loc="lower right", ncol=1)
#ax14.invert_yaxis()
figure7.tight_layout()
figure7.show()
#----------------------------------------------------------------------------------------------------

#Leaf area index analysis
path='D:/BTU FOLDER/land surface and atmosphere interaction/Final/LsAI_EX/'
#For 2003
df_xrlai2003=xr.open_dataset(path+'LAI_SAf_2003.nc')
df_xrlai2003= df_xrlai2003.fillna(0) #assuming all the missing values as 0 to avoid complication in the calculation

weights_df_xrlai2003 = np.cos(np.deg2rad(df_xrlai2003.LAI))
df_xrlai2003_weighted =df_xrlai2003.LAI.weighted(weights_df_xrlai2003) 

weighted_mean_df_xrlai2003 = df_xrlai2003_weighted.mean(("lon", "lat")) #weighted average over the area
df_lai2003=pd. DataFrame(weighted_mean_df_xrlai2003)
df_lai2003["Date"]= df_xrlai2003.time
df_lai2003["Date"]= pd.to_datetime(df_lai2003["Date"])
df_lai2003=df_lai2003.set_index(['Date'])
df_lai2003.columns=["LA_index"] 

#Subsetting the period
df_lai2003= df_lai2003['2003-01-01 00:00:00':'2003-12-03 00:00:00']

#Extracting the months        
df_lai2003["Month"]= df_lai2003.index.month #Extracting the month as a different column

#For 2019
df_xrlai2019=xr.open_dataset(path+'LAI_SAf_2019.nc')
df_xrlai2019= df_xrlai2019.fillna(0) #assuming all the missing values as 0 to avoid complication in the calculation

weights_df_xrlai2019 = np.cos(np.deg2rad(df_xrlai2019.LAI))
df_xrlai2019_weighted =df_xrlai2019.LAI.weighted(weights_df_xrlai2019) 

weighted_mean_df_xrlai2019 = df_xrlai2019_weighted.mean(("lon", "lat")) #weighted average over the area
df_lai2019=pd. DataFrame(weighted_mean_df_xrlai2019)
df_lai2019["Date"]= df_xrlai2019.time
df_lai2019["Date"]= pd.to_datetime(df_lai2019["Date"])
df_lai2019=df_lai2019.set_index(['Date'])
df_lai2019.columns=["LA_index"]  

#Subsetting the years
df_lai2019= df_lai2019['2019-01-01 00:00:00':'2019-12-19 00:00:00']

#Extracting the months        
df_lai2019["Month"]= df_lai2019.index.month #Extracting the month as a different column

#Comparison of leaf area index

figure8, ax15 = plt.subplots(1, figsize=(14,10))
ax15.plot(df_lai2003.Month, df_lai2003.LA_index, "o-", color="green", label="Leaf Area Index (2003)")
ax15.set_yticks(np.arange(0.4, 2, 0.2))
ax15.set_ylabel('Leaf Area Index')
ax15.set_xlabel('Month')
ax15.legend(loc="lower left", ncol=1)
ax15.set_title('Leaf Area Index by Month (2003 and 2019)')
#ax15.invert_yaxis()

ax16=ax15.twinx()
ax16.plot(df_lai2019.Month, df_lai2019.LA_index, "o-", color= "blue", label="Leaf Area Index (2019)")
ax16.set_xticks(df_lai2019.Month)
mon = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
ax16.set_xticklabels(mon)
ax16.set_yticklabels([]) 
ax16.set_yticks([])
ax16.legend(loc="lower right", ncol=1)
#ax16.invert_yaxis()
figure8.tight_layout()
figure8.show()
#------------------------------------------------------------------------------------------------

# Photosynthetically active radiation analysis 
path='D:/BTU FOLDER/land surface and atmosphere interaction/Final/LsAI_EX/'

#For 2003
df_xrfpr2003=xr.open_dataset(path+'faPAR_SAf_2003.nc')
df_xrfpr2003= df_xrfpr2003.fillna(0) #assuming all the missing values as 0 to avoid complication in the calculation

weights_df_fpr2003 = np.cos(np.deg2rad(df_xrfpr2003.faPAR))
df_fpr2003_weighted =df_xrfpr2003.faPAR.weighted(weights_df_fpr2003) 

weighted_mean_df_fpr2003 = df_fpr2003_weighted.mean(("lon", "lat")) #weighted average over the area
df_fpr2003=pd. DataFrame(weighted_mean_df_fpr2003)
df_fpr2003["Date"]= df_xrfpr2003.time
df_fpr2003["Date"]= pd.to_datetime(df_fpr2003["Date"])
df_fpr2003=df_fpr2003.set_index(['Date'])
df_fpr2003.columns=["PA_Radiation"]  

#Extracting the months        
df_fpr2003["Month"]= df_fpr2003.index.month #Extracting the month as a different column

#For 2019
df_xrfpr2019=xr.open_dataset(path+'faPAR_SAf_2019.nc')
df_xrfpr2019= df_xrfpr2019.fillna(0) #assuming all the missing values as 0 to avoid complication in the calculation

weights_df_fpr2019 = np.cos(np.deg2rad(df_xrfpr2019.faPAR))
df_fpr2019_weighted =df_xrfpr2019.faPAR.weighted(weights_df_fpr2019) 

weighted_mean_df_fpr2019 = df_fpr2019_weighted.mean(("lon", "lat")) #weighted average over the area
df_fpr2019=pd. DataFrame(weighted_mean_df_fpr2019)
df_fpr2019["Date"]= df_xrfpr2019.time
df_fpr2019["Date"]= pd.to_datetime(df_fpr2019["Date"])
df_fpr2019=df_fpr2019.set_index(['Date'])
df_fpr2019.columns=["PA_Radiation"]  

#Extracting the months        
df_fpr2019["Month"]= df_fpr2019.index.month #Extracting the month as a different column

#plotting of monthly comparison of photosynthetically radiation analysis

figure9, ax17 = plt.subplots(1, figsize=(14,10))
ax17.plot(df_fpr2003.Month, df_fpr2003.PA_Radiation, "o-", color="red", label="PA_radiation (2003)")
ax17.set_yticks(np.arange(0.2, 0.60, 0.05))
ax17.set_ylabel('PA_radiation (%)')
ax17.set_xlabel('Month')
ax17.legend(loc="lower left", ncol=1)
ax17.set_title('Fraction of Photosynthetically radiation by Month (2003 and 2019)')
#ax17.invert_yaxis()

ax18=ax17.twinx()
ax18.plot(df_fpr2019.Month, df_fpr2019.PA_Radiation, "o-", color= "blue", label="PA_radiation (2019)")
ax18.set_xticks(df_lai2019.Month)
mon = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
ax18.set_xticklabels(mon)
ax18.set_yticklabels([]) 
ax18.set_yticks([])
ax18.legend(loc="lower right", ncol=1)
#ax18.invert_yaxis()
figure9.tight_layout()
figure9.show()

#------------------------------------------------------------------------------

#Using a weather station from Namibia, Africa, I created a Walter-Leith diagram.
src1 = 'D:/BTU FOLDER/land surface and atmosphere interaction/Final/Station_Data/'
df_weather_station= pd.read_csv(src1+'2902403.csv', sep=",")
df_weather_station ["DATE"]= pd.to_datetime(df_weather_station ["DATE"])
df_weather_station=df_weather_station.set_index(['DATE'])  

station= "HOSEA KUTAKO INTL. A, WA"
altitude= "1700"                   
Lat, Lon= -22.483, 17.467

#Make nenecessary calculations
df_station_monthly = df_weather_station[['TMAX','TMIN']].resample('M').mean() # calculate monthly means
df_station_monthly['PRECIP'] = df_weather_station['PRCP'].resample('M').sum() # calculate monthly sum
df_station_climatology=df_station_monthly.groupby(df_station_monthly.index.month).mean().round(1) #monthly grouping
df_station_climatology['abs_TMAX']=df_station_monthly['TMAX'].groupby(df_station_monthly.index.month).max().round(1)  
df_station_climatology['abs_TMIN']=df_station_monthly['TMIN'].groupby(df_station_monthly.index.month).min().round(1) 
df_station_climatology['TMEAN']=(df_station_climatology['TMIN']+df_station_climatology['TMAX'])/2

#Lacking day counts, generate scalar values for Walter-Lieth diagram
abs_TMIN = df_station_climatology['abs_TMIN'].min() # h,s
Tmin = df_station_climatology['TMIN'].min()         # i,t
abs_TMAX = df_station_climatology['abs_TMAX'].max() # k
Tmax = df_station_climatology['TMAX'].max()    
annual_PRECIP = df_station_climatology['PRECIP'].sum() # d
annual_Tmean = df_station_climatology['TMEAN'].mean() # c
#------------------------------------------------------------------------------

# plotting a climate diagram after Walter-lieth without day counts
# a-d, f-k, m-q from the lecture slides
for_plotDF = df_station_climatology.copy() # not needed

# scaling Precip to Temp, ratio 2/1 for Precip <= 100 mm and 20/1 for > 100
for_plotDF.loc[(for_plotDF['PRECIP'] <= 100), ['PRECIP']] = for_plotDF['PRECIP']/2

for_plotDF.loc[(for_plotDF['PRECIP'] > 100), ['PRECIP']] = 50.+(for_plotDF['PRECIP']-100)*0.05  # ticks,ticklabels, reference to temperature

#plotting
figure10, temp = plt.subplots(1, figsize=(mm2inch(200.0,200.0)))
precip=temp.twinx()             #using two y-axis for temp (left) and for precip (right)
temp.set_ylim([-10,60])          #same limits for both y axis
precip.set_ylim([-10,60])
#set y-ticks and labels
temp.set_yticks([-10,0,10,20,30,40,50])
plt.setp(temp.get_yticklabels()[4:7], visible=False)
precip.set_yticks([0,10,20,30,40,50,60])
precip.set_yticklabels([0,20,40,60,80,100,300])
temp.set_ylabel('°C', labelpad=0, rotation=0.0, ha='left', va='center')
temp.yaxis.set_label_coords(-0.06,0.85)
precip.set_ylabel('mm', labelpad=0, rotation=0.0, ha='right', va='center')
precip.yaxis.set_label_coords(1.08,0.92)
temp.set_xticks(for_plotDF.index)
mon = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec') # nice labels
temp.set_xticklabels(mon)

#plot_Data
temp.plot(for_plotDF.index.values,for_plotDF['TMEAN'], color='red')
precip.plot(for_plotDF.index.values,for_plotDF['PRECIP'], color='blue',) # drawstyle="steps-mid")
#for humid
temp.fill_between(for_plotDF.index,for_plotDF['TMEAN'],for_plotDF['PRECIP'],where=(for_plotDF['PRECIP'] > for_plotDF['TMEAN']),edgecolor='blue', hatch='||', facecolor='none',linewidth=0.0,)# step='mid')
#for dry
temp.fill_between(for_plotDF.index,for_plotDF['TMEAN'],for_plotDF['PRECIP'],where=(for_plotDF['PRECIP'] <= for_plotDF['TMEAN']),edgecolor='red', hatch='.', facecolor='none',linewidth=0.0,)# step='mid')
#for wet
precip.fill_between(for_plotDF.index,for_plotDF['PRECIP'],50.0,where=(for_plotDF['PRECIP']> 50.0),facecolor='blue')# step='mid')
##line at 50° temp / 100 mm precip
##axh = temp.axhline(50,c='black', lw=1.0)
#deleting top spine and shrinking left spine
precip.spines['top'].set_visible(False)
temp.spines['top'].set_visible(False)
temp.spines['left'].set_bounds(-10, 50)
precip.spines['left'].set_bounds(-10, 50)
#set margins to 0
precip.margins(x=0,y=0)
temp.margins(x=0,y=0)

#Plotting text
figure10.text(0.01, 0.95, (station+' ('+str(altitude)+'m)'),transform=figure10.transFigure, fontsize=11)
figure10.text(0.01, 0.87, ('Lat:'+'{:03.2f}'.format(Lat)+' Lon:'+'{:03.2f}'.format(Lat)+'\n1966-2021'),transform=figure10.transFigure, fontsize=9)
figure10.text(0.51, 0.87, ('Average Temperature: '+'{:03.1f}°C'.format(annual_Tmean)+'\nAnnual Precipitation: '+'{:03.0f}mm'.format(annual_PRECIP)),transform=figure10.transFigure, fontsize=9)
figure10.text(0.095, 0.68, ('{:03.1f}'.format(abs_TMAX)+'\n'+'{:03.1f}'.format(Tmax)),transform=figure10.transFigure, fontsize=9,weight='bold', ha='right')
figure10.text(0.095, 0.22, ('{:03.1f}'.format(Tmin)+'\n'+'{:03.1f}'.format(abs_TMIN)),transform=figure10.transFigure, fontsize=9,weight='bold', ha='right')

src2= "D:/BTU FOLDER/land surface and atmosphere interaction/Final/Weather_Station_Climate_Diagram/"
figure10.savefig(src2+ 'Image.png',dpi=300, bbox_inches='tight')










