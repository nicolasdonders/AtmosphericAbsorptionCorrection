#!/usr/bin/env python
# coding: utf-8

# # __Model for Ultraviolet Atmospheric Absorption (Version 9.1)__
# ___

# * Author: Nicolas Donders
# * Revision History:
#     * V1 (Oct 2021): Set up simple model and results
#     * V2 (Nov 2021): Importing UVVIS data from MPI Mainz
#     * V3 (Feb 2022): Utilizing a spherical shells atmospheric model to modulate optical depth calculation. Importing HRTS spectra and folding this into FURST instrument and CCD camera.
#     * V4 (Jun 2022): Fixed missing spectral plate-scale factor. Cleaned up and fixed/added more comments.
#     * V5 (Jul 2022): Focus inversion on absorption spectrum rather than density profile.
#     * V6 (Aug 2022): Impliment raw SR data by simplifying code.
#     * V6.1 (Aug 2022): Rewriting for different spectral range and altitudes.
#     * V7.1 (Nov 2022): Applying to MaGIXS spectral range. 
#     * V7.2 (Jan 2023): Trimmed for Easier use, looked at many launch times to predict effect on MaGIXS missions at various months of the year.
#     * V8 (Apr 2023): Adjusted for 19.3 nm (Hi-C 1) and 17.2 nm (Hi-C 2.1). Produced 2D pictures saved in FITS files. Also played with generating predictions for Hi-C flare (which launched Spring 2024).
#     * V9 (Feb 2025): Modifications for generating publication plots. Moved molar mass data outside of generating functions, part of input now.
#     * V9.1 (Mar 2025): Simplifying code for export to P.S. Athray for use with future mission.

# # 1: Initialize Code with imports and basic user-defined Functions

# ## 1.1 Imports

# In[ ]:


import subprocess

def install_package(package_name):
    try:
        # Attempt to import the package
        __import__(package_name)
    except ImportError:
        # If the package is not installed, install it using pip
        try:
            subprocess.check_call(['python', '-m', 'pip', 'install', package_name])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package_name}: {e}")
            return False
    return True

packages = ['requests','bs4','datetime','matplotlib','nrlmsise00[dataset]','numpy','os','pandas','pysolar','pytz','scipy']
for pack in packages:
    if install_package(pack):
        print('package {} installed successfully'.format(pack))
    else:
        print('Failed to install {} package'.format(pack))

import pytz
import numpy as np
import pandas as pd
import requests as pull
import matplotlib.pyplot as plt

from os.path import join
from datetime import datetime
from bs4 import BeautifulSoup as BS
from scipy.interpolate import interp1d
from nrlmsise00.dataset import msise_4d
from pysolar.solar import get_altitude as zen_func
from scipy.interpolate import pchip_interpolate as spline


# ## 1.2: User Functions

# In[ ]:


def rolling_avg(array, window_size):
    return np.convolve(array, np.ones(window_size)/window_size, mode='valid')


# In[ ]:


def find_continuum(array, window_size):
    # Create an array of minimum values in each window
    continuum = np.array([np.min(array[max(i - window_size // 2, 0):min(i + window_size // 2 + 1, len(array))])
                         for i in range(len(array))])
    # Apply rolling average to the minimum value array
    continuum = np.convolve(continuum, np.ones(window_size) / window_size, mode='valid')
    return continuum


# In[ ]:


def elevation_function(lat, lon):
    
    # Query USGS Elevation Point Service to get elevation in kilometers    
    url = 'https://epqs.nationalmap.gov/v1/json?x={}&y={}&units=Meters&wkid=4326&includeDate=False'\
          .format(lon,lat)
    elevation = pull.get(url).text.split('"')[17]
    
    return float(elevation)


# In[ ]:


def get_earthradius(lat):
    # Earth radius values at Equator and Poles
    r1 = 6378.1370
    r2 = 6356.7523
    
    # Calculation of Earth radius at given latitude
    a = (r1**2 * np.cos(lat))**2 + (r2**2 * np.sin(lat))**2
    b = (r1 * np.cos(lat))**2 + (r2 * np.sin(lat))**2
    
    # Returns radius of Sea Level at given Latitude.
    return np.sqrt(a / b)


# In[ ]:


def calc_absorp_spline(absorp_data_all,waves,map_temp, n_mols, n_waves, n_height):
    
    header = ['Molecule', 'Wavelength (nm)', 'Temperature (K)', 'Absorption Cross Section (cm^2 molec^-1)']
    mols_data = absorp_data_all[header[0]] # Molecules
    absorp_spline = np.zeros((n_mols,n_waves,n_height))
    
    for m in range(n_mols):

        molindex    = np.where(mols_data==mols[m])
        waves_data  = absorp_data_all[header[1]].to_numpy()[molindex]
        temp_data   = absorp_data_all[header[2]].to_numpy()[molindex]
        absorp_data = absorp_data_all[header[3]].to_numpy()[molindex]

        for i in range(len(waves_data)):
            wi = (abs(waves_data[i]-waves)).argmin()
            ti = (abs(temp_data[i]-map_temp)).argmin()
            checkti = np.where(map_temp==map_temp[ti])[0]
            if len(checkti)>1: ti=checkti
            absorp_spline[m,wi,ti] = absorp_data[i]

        for i in range(len(waves)):
            Nnonzero = np.count_nonzero(absorp_spline[m,i,:])
            if Nnonzero > 0:
                index_nonzero = np.array(absorp_spline[m,i,:].nonzero(), dtype=int)[0] 
                if len(index_nonzero)==1:
                    absorp_spline[m,i,:] = absorp_spline[m,i,index_nonzero]
                if len(index_nonzero)>1:
                    ti,tf = index_nonzero[0], index_nonzero[-1]+1
                    f = interp1d(map_temp[index_nonzero], absorp_spline[m,i,index_nonzero])
                    absorp_spline[m,i,ti:tf] = f(map_temp[ti:tf])
                    
        for j in range(len(map_temp)):
            Nnonzero = np.count_nonzero(absorp_spline[m,:,j])
            if Nnonzero > 0:
                index_nonzero = np.array(absorp_spline[m,:,j].nonzero(), dtype=int)[0] 
                if len(index_nonzero)==1:
                    absorp_spline[m,:,j] = absorp_spline[m,index_nonzero,j]
                if len(index_nonzero)>1:
                    wi,wf = index_nonzero[0], index_nonzero[-1]+1
                    f = interp1d(waves[index_nonzero], absorp_spline[m,index_nonzero,j])
                    absorp_spline[m,wi:wf,j] = f(waves[wi:wf])
    return absorp_spline


# In[ ]:


def calc_DiffPathLength(ri_0, r_0, d_height, cosmu, N):
    DiffPathLength = np.zeros(N)
    for z in range(N):
        a = r_0 + d_height*float(z+1)
        b = r_0 + d_height*float(z)
        c = 1.0 # density_all_ratio[m,zz+z]
        d = ri_0[m] / ( c*(ri_0[m]-1) + 1 )
        mu = np.sqrt( 1 - ( (r_0+d_height)**2 / a**2 ) * d**2 * (1-cosmu**2) )
        DiffPathLength[z] = (np.sqrt( a**2 - b**2 * (1-mu**2) ) - b*mu)*1e5
    return DiffPathLength


# In[ ]:


def calc_OptDepth(absorp, ri_0, density_all_ratio, d_height, cosmu):
    n_mols = absorp.shape[0]
    n_waves = absorp.shape[1]
    n_height = absorp.shape[2]
    OptDepth = np.zeros((n_waves, n_height))
    for m in range(n_mols):
        for z in range(n_height):
            OptDepth[:,z] += np.sum( absorp[m,:,z:]*DiffPathLength[z:], axis=1)
    return OptDepth


# In[ ]:


def getdata_xray(label,minmax,molmass):

    # Convert from nm to keV
    minkev = 1.24 / minmax[1] # 1.24 keV*nm
    maxkev = 1.24 / minmax[0]
    minkev = np.floor(minkev*10)/10
    maxkev = np.ceil(maxkev*10)/10

    # Set up Pandas DataFrame for output
    headers = ['Molecule', 'Wavelength (nm)', 'Temperature (K)', 'Absorption Cross Section (cm^2 molec^-1)']
    Data_out = pd.DataFrame(columns=headers)
    
    # Loop over each molecule in the label list
    for m in range(len(label)):
        
        # Query NIST X-ray database and pull data
        url = f'https://physics.nist.gov/cgi-bin/ffast/ffast.pl?Formula={label[m]}&gtype=2&range=S&lower={minkev}&upper={maxkev}&density=&frames=no&htmltable=1'
        grab = pull.get(url)
        soup0 = BS(grab.text, 'html.parser').get_text()
        soup = soup0.split('cm2g-1')[1].split('\n')
        
        # Create array to store data
        Data = np.zeros((2, len(soup)-1), dtype=float)
        for i in range(len(soup)-1):
            Data[0, i] = soup[i].split('\xa0\xa0')[0] # keV
            Data[1, i] = soup[i].split('\xa0\xa0')[1] # cm^2 * g^-1
        Data[0, :] = 1.24 / Data[0, :]
        Data[1, :] = Data[1, :] * molmass[m] # cm^2 * molecule^-1
        Data = Data[:, ::-1]

        # Get range of data to keep based on minmax range
        if minmax != None:
            ji = (abs(minmax[0] - Data[0, :])).argmin()
            jf = (abs(minmax[1] - Data[0, :])).argmin()
        else:
            ji = 0
            jf = n_data - 1

        # Add data to output DataFrame
        for j in range(ji, jf):
            Data_out = Data_out.append({headers[0]: label[m], headers[1]: Data[0, j], headers[2]: 273, headers[3]: Data[1,j] }, ignore_index=True)
        
    return Data_out


# In[ ]:


def getdata_uvvis(label,**kwargs):
    
    # Check to see if text string can be a float 
    def isfloat(string):
        try:
            float(string)
            return True
        except ValueError:
            return False
    
    # Parse through HTML in the URL to pull out hyperlinks
    def htmlparser(urls,subtext):
        # MPI Mainz UVVIS website
        url = 'https://uv-vis-spectral-atlas-mainz.org/uvvis/'\
            + subtext
        
        # Grab all hyperlinks on this website
        grab = pull.get(url)
        soup = BS(grab.text, 'html.parser')
        for link in soup.find_all('a'):
            urls.append(link.get('href'))
            
        # Output the resulting list
        return urls
    
    # Pull together all databases
    def geturl_mols(mols):
    
        # Find all cross-section URLs
        find = 'cross_sections/'
        urls = htmlparser([],find)
        urls = [match for match in urls if find in match][1:]

        # Sort through each cross-section URL and list all molecule filenames
        urls_sub = []
        for u in range(len(urls)):
            urls_sub = htmlparser(urls_sub,urls[u])

        # Sort through list of filenames and grab the ones we care about
        data_urls = [None]*len(mols)
        for m in range(len(mols)):
            data_urls[m] = [match for match in urls_sub if '/'+mols[m]+'.spc' in match][0]

        return data_urls
    
    minmax = kwargs.get('minmax')
    label = np.sort(label)
    # Database URL prefix
    url = 'https://uv-vis-spectral-atlas-mainz.org/uvvis_data/'
    
    # Set up Pandas Dataframe for output
    headers = ['Molecule', \
               'Wavelength (nm)', \
               'Temperature (K)', \
               'Absorption Cross Section (cm^2 molec^-1)']
    Data_out = pd.DataFrame(columns=headers)
    molmass_out = [] #np.zeros(len(mols))
    
    # Finding all URLs associated with desired molecules
    print('Gather Databases for: {}'.format(', '.join(label)))
    data_urls = geturl_mols(label)
    
    for m in range(len(label)):

        # Skipping certain files that aren't formatting yet
        if label[m] == 'O2':
            # skips = [11,23,24,25,26,38,42,48,63,81,87,88,89,119,125,126,141,143,208]
            skips = [12,24,87,126,132,133]
        else:
            if label[m] == 'N2':
                # skips = [4,6]
                skips = [6]
            else:
                skips = []
        
        # Find all text-file URLs under the molecules' specific database
        urls = htmlparser([],data_urls[m])
        urls = [match for match in urls if ".txt" in match]

        # Go through each file with parallel optimization
        Nfiles = len(urls)
        for i in range(Nfiles):
            
            # Figure out data structure from filenames
            
            filename = url+urls[i]
            temp_str = filename.split('K_')[0].split(')_')[1].split('-')
            wave_str = filename.split('K_')[1].split('nm')[0].split('-')
                        
            # If range is specified, find min and max waves 
            # and skip files outside desired range
            if minmax != None:
                if isfloat(wave_str[0]):
                    waves_min = float(wave_str[0])
                    waves_max = float(wave_str[len(wave_str)-1])
                else:
                    waves_min = float(wave_str[0].split(',')[0])
                    waves_max = float(wave_str[0].split(',')[1])
                
                if waves_max<minmax[0] \
                or waves_min>minmax[1]:
                    skips.append(i)
            
            # Skip files as requested
            # Will change to be either automatically detected or not needed            
            if i not in skips:

                # Import data, strip whitespace, and remove headers
                Data_str = pd.read_table(filename,\
                                         header='infer',\
                                         encoding='ISO-8859-1', \
                                         index_col=False,\
                                         keep_default_na=False,\
                                         skip_blank_lines=True)
                if isfloat(list(Data_str.head())[0].split()[0]):
                    Data_str = pd.read_table(filename,\
                                             header=None,\
                                             encoding='ISO-8859-1', \
                                             index_col=False,\
                                             keep_default_na=False,\
                                             skip_blank_lines=True)
                Data_str.dropna(inplace=True)
                n_data = Data_str.shape[0]

                # If there are two rows of headers, drop the leftover row
                if isfloat(list(Data_str.iloc[0].dropna())[0]) == False:
                    Data_str = Data_str.drop(0,axis=0)
                if isfloat(list(Data_str.iloc[0].dropna())[0]) == False:
                    Data_str = Data_str.iloc[1:,:]

                # Split Data to two columns if it isn't already
                n_data = Data_str.shape[0]
                if Data_str.shape[1] == 1:
                    Data_str.insert(1,1,[None]*n_data)
                if isfloat(Data_str.iloc[0,0]) == False:
                    for r in range(n_data):
                        Data_str.iloc[r,1] = Data_str.iloc[r,0].split()[1]
                        Data_str.iloc[r,0] = Data_str.iloc[r,0].split()[0]  
                # print(Data_str)

                n_data = len(Data_str.iloc[:,0])

                # Splitting apart plus/minus symbols
                for r in range(n_data):
                    if isfloat(Data_str.iloc[r,1]) == False:
                        if Data_str.iloc[r,1][0] == '<':
                            # print('Fixing ',Data_str.iloc[r,1])
                            Data_str.iloc[r,1] \
                                = Data_str.iloc[r,1]\
                                  .replace('<','')
                            # print('Fixed ',Data_str.iloc[r,1])
                        else:
                            # print('Fixing ',Data_str.iloc[r,1])
                            Data_str.iloc[r,1] \
                                = Data_str.iloc[r,1]\
                                  .replace(' ','')
                            strfix0 \
                                = Data_str.iloc[r,1]\
                                  .split('(')[1]\
                                  .split(u'\u00B1')[0]
                            strfix1 \
                                = Data_str.iloc[r,1]\
                                  .split(')')[1]           
                            Data_str.iloc[r,1] \
                                = strfix0+strfix1
                            # print('Fixed',Data_str.iloc[r,1])

                # Removing Error Limits / Extra columns
                Data_str = Data_str.iloc[:,0:2]
                
                # Converting to floats
                Data = np.asarray(Data_str,dtype=float)
                
                # Saving Cross-sections
                cross = Data[:,1]
                
                #Sorting out data into dataframe output
                n_data = len(Data)
                if len(temp_str)==1:
                    waves  = Data[:,0]
                    temp   = np.zeros((n_data))+float(temp_str[0])
                else:
                    if len(wave_str)==1:
                        temp  = Data[:,0]
                        waves = np.zeros((n_data))+float(wave_str[0])
                    else:
                        waves  = Data[:,0]
                        temp   = np.zeros((n_data)) \
                               + (float(temp_str[0])+float(temp_str[1]))/2.
                
                # Only append data within desired range
                if minmax != None:
                    ji = (abs(minmax[0]-waves)).argmin()
                    jf = (abs(minmax[1]-waves)).argmin()
                else:
                    ji = 0
                    jf = n_data-1
                    
                for j in range(ji,jf):
                    Data_out = Data_out.append({headers[0]: label[m],      \
                                                headers[1]: waves[j],   \
                                                headers[2]: temp[j],    \
                                                headers[3]: cross[j] }, \
                                                ignore_index=True       )
                
            # Printing run progress
            print("File {:.0f} of {:.0f} for \t {}: \t {:.1f}%" \
                  .format(i+1, Nfiles, label[m], (i+1)/Nfiles*100), end='\r')
            
        # Printing final progress
        # print('\t \t \t \t', end='\r')
        print('\nData pts added for \t {}: \t {:.0f}'\
              .format(label[m],Data_out[Data_out[Data_out.columns[0]]==label[m]].shape[0]))
    
    # Sort output by molecule, then by wavelength, then by temperature, then by cross-section
    Data_out = Data_out.sort_values([Data_out.columns[0],Data_out.columns[1],Data_out.columns[2],Data_out.columns[3]])
    
    # Plot output, if desired
    if kwargs.get('plot') is True:
        groups = Data_out.groupby(Data_out.columns[0])
        aspect,pt = (8,4),12
        plt.style.use('default')
        plt.figure(figsize=aspect)
        plt.rcParams.update({'font.size': pt})
        for mol,group in groups:
            plt.scatter(group[Data_out.columns[1]],group[Data_out.columns[3]],\
                        # c=group[Data_out.columns[2]],cmap='plasma',\
                        lw=1,label=mol)
        plt.yscale('log')
        if minmax != None: plt.xlim(minmax[0],minmax[1])
        else: plt.xlim(Data_out[Data_out.columns[1]].min(),Data_out[Data_out.columns[1]].max())
        plt.xlabel(Data_out.columns[1])
        plt.ylabel(Data_out.columns[3])
        # plt.colorbar(label=Data_out.columns[2])
        plt.legend()
    
    if kwargs.get('save') is True:
        if kwargs.get('minmax') != None:
            savefilename = 'Donders_mols_{}_range_{}_{}nm.csv'\
                            .format('_'.join(label),minmax[0],minmax[1])
        else:
            savefilename = 'Donders_mols_{}_range_{}_{}nm.csv'\
                            .format('_'.join(label),int(Data_out[Data_out.columns[1]].min()),int(Data_out[Data_out.columns[1]].max()))
        Data_out.to_csv(savefilename, index=False)
        print('Saved file as {}'.format(savefilename))
    
    # Return output data
    return Data_out, molmass_out


# ## 1.3 Constants

# In[ ]:


NA = 6.022E23


# # 2: Initial Parameters

# ## 2.1 Flight Profile

# In[ ]:


timezone = pytz.timezone("UTC")   # Set timezone to UTC
launch_date = [2012, 7, 11]       # Date of launch [year, month, day]
launch_time = [18,0,0]            # time of start [hours,minutes,seconds]
lat, lon = 32.417776, -106.321547 # Define latitude and longitude (WSMR = 32.4 N, -106.3 W)

# File with flight profile, can be csv or txt, but should be 
# first column as time after launch (in seconds) and 
# second column as altitude (in km).
# flight_profile_file = 'test.csv'
flight_profile_file = 'test_profile.txt'


# ## 2.2 Science Target

# In[ ]:


# Set minimum and maximum wavelength range and step size
min_wave, max_wave = 19.3-5,19.3+5 # nm
d_wave = 0.01 # nm, could be based on camera pixels if applicable

# List of molecules to consider in atmospheric model
mols = ['O', 'O2', 'N', 'N2', 'Ar', 'He', 'H']
molmass_g = [15.999, 31.999, 14.0067, 28.0134, 39.948, 4.002602, 1.00784] # in g/mol


# # 3: Calculated Parameters

# ## 3.1: Flight Profile

# In[ ]:


# Load Flight Profile from directory
fp_tall = np.loadtxt(flight_profile_file, skiprows=1, usecols=[0], delimiter='\t')
fp_zall = np.loadtxt(flight_profile_file, skiprows=1, usecols=[1], delimiter='\t')

#Define flight date and time with UTC timezone
start_time = launch_time[0] + launch_time[1]/60 + launch_time[2]/60/60
flight_datetime = datetime(launch_date[0], launch_date[1], launch_date[2],
                           launch_time[0], launch_time[1], launch_time[2],
                           tzinfo=timezone)
# Setting up altitude matrix
min_height, max_height = 0, 1000 # Highest NRL MSISE density data is 1000 km
d_height = 0.5 # Delta Z resolution desired (km)
n_height = int(abs(max_height - min_height) / d_height) + 1 #Calculate number of height intervals
alts = np.linspace(start=min_height,stop=max_height,num=n_height) 

# Radius of the Earth at launch site.
# add in elevation function is height is relative instead of absolute.
r_0 = get_earthradius(lat) # + elevation_function(lat,lon)

# Importing Zenith Angle based on Location and DateTime.
cosmu = np.cos(np.radians(90-zen_func(lat,lon,flight_datetime)))


# ## 3.2: Science Target

# In[ ]:


minmax=[min_wave,max_wave]
molmass =  [x/NA for x in molmass_g]
n_mols = len(mols)

# refractive index at ground, not accounted for, setting all to one (1).
ri_0 = np.ones((n_mols))

# Array for wavelengths
n_waves = int((max_wave - min_wave) / d_wave + 1)
waves = np.linspace(min_wave, max_wave, n_waves)

absorp_data_all = getdata_xray(mols,minmax, molmass)
# absorp_data_all, molmass = getdata_uvvis(mols,minmax, molmass)


# ## 3.3: Formatting Data

# In[ ]:


# Get the atmospheric data at the specified lat, lon, time and altitudes
density_data = msise_4d(flight_datetime,alts,lat,180-lon) #g/cm^3

# Get the length of the data for the first molecule
n_data = len(density_data[mols[0]].data.ravel())

# Initialize a 2D array to store density for all molecules
density_all = np.zeros((n_mols,n_data))
# density_all_ratio = np.zeros((n_mols,n_data))

# Loop over all molecules to store the data in the 2D array
for i in range(n_mols):
    density_all[i,:] = density_data[mols[i]].data.ravel()
    # density_all_ratio[i,:] = density_all[i,:] / density_all[i,0]

# Get the temperature and total density data from the atmospheric model and ravel it to a 1D array
map_temp = density_data['Talt'].data.ravel()
map_totalrho = density_data['rho'].data.ravel()

# Formatting Absorption Cross-section into a 2D array based on altitude, temperature, and wavelength maps.
absorp_spline = calc_absorp_spline(absorp_data_all,waves,map_temp, n_mols, n_waves, n_height)


# # 4: Calculating Absorption Model

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Calculate the absorption by multiplying the spline and data of each molecule and height.\nabsorp = np.zeros((n_mols, n_waves, n_height))\nfor m in range(n_mols): # Loop over the number of molecules\n    for z in range(n_height): # Loop over the height\n        absorp[m, :, z] = absorp_spline[m, :, z] * density_all[m, z] # cm^-1\n\n# Calculate Differential Path Length\nDiffPathLength = calc_DiffPathLength(ri_0, r_0, d_height, cosmu, n_height)\n\n# Calculate Optical Depth\nOptDepth = calc_OptDepth(absorp, ri_0, density_all_ratio, d_height, cosmu)\n\n# Calculate Fraction of intensity (0 is fully absorbed, 1 is no absorption).\nIfrac = np.exp(-OptDepth)\n')


# In[ ]:


# Calculate the Transmission over time of the Flight Profile
fp_num = len(fp_zall)
Ifrac_time = np.zeros((n_waves, fp_num))
for z in range(fp_num):
    # Get the index of the height closest to the current altitude
    z_i = np.max(np.where(alts <= fp_zall[z]))
    # Assign the fraction of intensity at the current height
    Ifrac_time[:, z] = Ifrac[:, z_i]


# In[ ]:


pt = 32
aspect = (16, 9)
plt.style.use('default')
plt.rcParams.update({'font.size': pt})
fig, ax1 = plt.subplots(figsize=aspect)
ax1.pcolormesh(fp_tall,waves,Ifrac_time,shading='auto')
ax1.set_xlim(fp_tall[0],fp_tall[-1])
ax1.set_ylim((waves[0],waves[-1]))
ax1.set_ylabel('Wavelength (nm)')
ax1.set_xlabel('T+ (s) after {} on {}'.format(flight_datetime.time(),flight_datetime.date()))
plt.show(); plt.close();


# In[ ]:


# Plot Density
pt = 24
aspect = (16, 9)
plt.style.use('default')
plt.figure(figsize=aspect)
plt.rcParams.update({'font.size': pt})
plt.ylabel('Altitude (km)')
plt.xlabel('Density (kg/m$^3$)')

Y = alts
# ymin,ymax = 0, fp_zall.max()
ymin,ymax = 100, 300
plt.ylim((ymin, ymax))

xmin,xmax = 1e-16, 1e-6
plt.xlim((xmin, xmax))

moleculeweight = np.array(molmass)/1000
X = density_all * moleculeweight[:,None] * 100**3 # (molecules/cm^3)*(kg/molecule) * (100^3 cm^3/m^3) = kg/m^3
X2 = np.sum(X,axis=0)
X3 = map_totalrho* (100**3 / 1000) # Total density converted from g/cm^3 to kg/m^3
percentlabel = np.sum(X[:,101:301],axis=1)/np.sum(X3[101:301])
totallabel = np.sum(X[:,101:301])/np.sum(X3[101:301])

for m in range(n_mols):
    # percentlabel = np.mean(X[m,:]/X3)
    label = mols[m]+' ({:.0%})'.format(percentlabel[m])
    plt.semilogx(X[m,:], Y, '-', lw=2, label=label)

plt.semilogx(X3, Y, '--k', lw=2, label='Total ({:.0%})'.format(totallabel))

plt.title('Atmospheric Density (% Contribution between 100-300 km)')
plt.legend(fontsize=pt,frameon=False)
plt.show(); plt.close();

