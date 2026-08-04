"""
Created on Fri Jul  3 13:42:24 2026
@author: fredrik
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import time   as time                 

start = time.time()

ds = xr.open_dataset("../Data/wind_data.nc")  


# Select a location 
lat, lon =  40, 40

def get_variables(lat, lon):
    "Extract the closest values, averaged over the entire time period"
    
    lat_idx = np.argmin(np.abs(ds.latitude.values - lat))
    lon_idx = np.argmin(np.abs(ds.longitude.values - lon))
    # Extract u and v for the selected location, averaged over all time steps
    
    u = ds.u.isel(latitude=lat_idx, longitude=lon_idx).mean(dim='time')
    v = ds.v.isel(latitude=lat_idx, longitude=lon_idx).mean(dim='time')
    
    # Sort everything by pressure level so the line plot is monotonic
    sort_idx = np.argsort(u.isobaricInhPa.values)
    pressure = u.isobaricInhPa.values[sort_idx]
    u_vals = u.values[sort_idx]
    v_vals = v.values[sort_idx]
    
    return u_vals, v_vals, pressure

def angle(u_vals, v_vals, speed_threshold=1.0):
    speed = np.hypot(u_vals, v_vals)

    max_idx = np.argmax(speed)
    u_ref, v_ref = u_vals[max_idx], v_vals[max_idx]

    dot = u_ref * u_vals + v_ref * v_vals
    cross = u_ref * v_vals - v_ref * u_vals

    angle = np.degrees(np.arctan2(cross, dot))

    angle = np.where(speed > speed_threshold, angle, np.nan)
    return angle

u_vals, v_vals, pressure = get_variables(lat, lon)

wind_angle = angle(u_vals, v_vals)

# Create a figure with three subplots — the third needs a cartopy projection
plt.close("all")
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
pressure_ticks = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
ax1.invert_yaxis()
ax1.set_yscale('log')
ax1.set_yticks(pressure_ticks)
ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax1.yaxis.set_minor_formatter(mticker.NullFormatter())
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot wind angle vs. pressure (inverted y-axis)
ax2.plot(wind_angle, pressure, label='Wind Angle', color='C2')

ax2.set_ylabel('Pressure [hPa]')
ax2.set_xlabel('Wind Angle [degrees]')
ax2.set_title('Wind Angle')
ax2.invert_yaxis()
ax2.set_yscale('log')
ax1.set_yticks(pressure_ticks)
ax2.set_yticks(pressure_ticks)
ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax2.yaxis.set_minor_formatter(mticker.NullFormatter())
ax2.legend()
ax2.set_xlim(-2,82)
ax2.grid(True, linestyle='--', alpha=0.6)

# Surface wind angle text box (surface = highest pressure = last index)
surface_angle = wind_angle[-1]
surface_pressure = pressure[-1]
textstr = f'Surface angle: {surface_angle:.1f}°'
ax2.text(0.65, 0.5, textstr, transform=ax2.transAxes,
          fontsize=11, verticalalignment='bottom',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Map showing the selected location as an X
ax3.set_extent([lon - 39, lon + 30, lat - 30, lat + 30], crs=ccrs.PlateCarree())
ax3.add_feature(cfeature.LAND)
ax3.coastlines(resolution='50m')
ax3.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
ax3.plot(lon, lat, marker='x', color='red', markersize=12, markeredgewidth=3,
          transform=ccrs.PlateCarree())

plt.suptitle(f"ERA5 time-mean horizontal wind at Lat={lat}°, Lon={lon}°", size=14)
plt.tight_layout()
plt.show()

end = time.time()
print(end - start, "s")
