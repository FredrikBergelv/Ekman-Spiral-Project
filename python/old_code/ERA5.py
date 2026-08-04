"""
Created on Fri Jul  3 13:42:24 2026

@author: fredrik
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ds = xr.open_dataset(
    "../Data/57a863750aee8b7927f7bf992da2da8b.grib",
    engine="cfgrib")

# Select a location 
lat, lon = 50, 0

def get_variables(lat, lon, time_idx=2):
    "Extract the closests vlaues"
    
    lat_idx = np.argmin(np.abs(ds.latitude.values - lat))
    lon_idx = np.argmin(np.abs(ds.longitude.values - lon))

    # Extract u and v for the selected location and time
    u = ds.u.isel(time=time_idx, latitude=lat_idx, longitude=lon_idx)
    v = ds.v.isel(time=time_idx, latitude=lat_idx, longitude=lon_idx)

    # Sort everything by pressure level so the line plot is monotonic
    sort_idx = np.argsort(u.isobaricInhPa.values)
    pressure = u.isobaricInhPa.values[sort_idx]
    u_vals = u.values[sort_idx]
    v_vals = v.values[sort_idx]
    
    return u_vals, v_vals, pressure

def angle(u_vals, v_vals):
    "Get the wind angles"
    wind_angle = np.arctan2(v_vals, u_vals) * (180 / np.pi)
    return wind_angle 

u_vals, v_vals, pressure = get_variables(lat, lon)
wind_angle =angle(u_vals, v_vals)

# Create a figure with three subplots — the third needs a cartopy projection
fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())

# Plot u and v vs. pressure (inverted y-axis)
ax1.plot(u_vals, pressure, label='u', color='C0')
ax1.plot(v_vals, pressure, label='v', color='C1')
ax1.set_ylabel('Pressure [hPa]')
ax1.set_xlabel(R'Wind Speed [mS${}^{-1}$]')
ax1.set_title('Wind Components')
ax1.invert_yaxis()
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot wind angle vs. pressure (inverted y-axis)
ax2.plot(wind_angle, pressure, label='Wind Angle', color='C2')
ax2.set_ylabel('Pressure [hPa]')
ax2.set_xlabel('Wind Angle [degrees]')
ax2.set_title('Wind Angle')
ax2.invert_yaxis()
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

# Map showing the selected location as an X
ax3.set_extent([lon - 39, lon + 30, lat - 30, lat + 30], crs=ccrs.PlateCarree())


ax3.add_feature(cfeature.LAND)
ax3.coastlines(resolution='50m')
ax3.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
ax3.plot(lon, lat, marker='x', color='red', markersize=12, markeredgewidth=3,
          transform=ccrs.PlateCarree())
ax3.set_title('The location')

plt.suptitle(f"ERA5 horizontal wind at Lat={lat}°, Lon={lon}°", size=14)

plt.tight_layout()
plt.show()