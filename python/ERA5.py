"""
Created on Fri Jul  3 13:42:24 2026

@author: fredrik
"""

import numpy as np                          
import xarray as xr                
import matplotlib.pyplot as plt    
import matplotlib as mpl
import cartopy.crs as ccrs         
import cartopy.feature as cfeature 

import xarray as xr

ds = xr.open_dataset("../Data/57a863750aee8b7927f7bf992da2da8b.grib", engine="cfgrib")
