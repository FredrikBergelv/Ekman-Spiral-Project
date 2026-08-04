"""
Created on Sat Jul  4 17:46:44 2026
@author: fredrik
"""
import os
import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from global_land_mask import globe
import time

start = time.time()

wind_filename = "../Data/wind_data.nc"
elevation_filename = "../Data/elevation.nc"

EARTH_RADIUS_KM = 6371.0
LAT_MIN, LAT_MAX = 30, 60     # Extent
coast_buffer = 0              # How far away from the coast do we hav to be 
SPEED_THRESHOLD = 1.0         # wind speed cutoff
slope_threshold = 1000        # flat terrain cutoff
MONTH = 0                     # month number,0 means mean

def select_time_period(ds, month='mean'):
    """ Choose month or choose mean for entire year
        1-12 : month number
        0 : mean 
    """
    if month == 0:
        return ds

    ds_month = ds.sel(time=ds['time'].dt.month == month)

    if ds_month.sizes.get('time', 0) == 0:
        raise ValueError(f"No time steps found for month={month} in this dataset")

    return ds_month


def download_elevation(lat_min, lat_max, lon_min, lon_max, filename=elevation_filename):
    """Download ETOPO elevation map from NOAA's ERDDAP server."""

    if os.path.exists(filename):
        return filename

    # Use the right bounds from above
    lat_min, lat_max = max(lat_min, -90), min(lat_max, 90)
    lon_min, lon_max = max(lon_min, -180), min(lon_max, 180)

    url = (f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?"
           f"altitude[({lat_min}):({lat_max})][({lon_min}):({lon_max})]")
    
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename

def get_flat_terrain_mask(lat_grid, lon_grid, slope_threshold=10):
        """Return a mask if terrain is flatter than threshold"""
        
        lat_min, lat_max = lat_grid.min() - 1, lat_grid.max() + 1
        lon_min, lon_max = lon_grid.min() - 1, lon_grid.max() + 1
    
        elevation_data = download_elevation(lat_min, lat_max, lon_min, lon_max)
        elevation_ds = xr.open_dataset(elevation_data)
    
        elevation_at_points = elevation_ds.altitude.interp(         # interpolate the data 
            latitude=xr.DataArray(lat_grid.ravel(), dims="points"),
            longitude=xr.DataArray(lon_grid.ravel(), dims="points"),
            method="linear").values.reshape(lat_grid.shape)
    
        dzdy, dzdx = np.gradient(elevation_at_points)
    
        dlat_km = np.abs(np.diff(lat_grid[:, 0]).mean()) * 111.0 
        dlon_km = np.abs(np.diff(lon_grid[0, :]).mean()) * 111.0 * np.cos(np.deg2rad(lat_grid))
    
        slope_m_per_km = np.hypot(dzdy / dlat_km, dzdx / dlon_km)
        flat_mask = slope_m_per_km <= slope_threshold
    
        return flat_mask, elevation_at_points, slope_m_per_km


def get_land_mask(ds, lat_min=30, lat_max=60, coast_buffer=2, slope_threshold=10):
    """Mask of midlatitude land points without ocean, near-coast, and non-flat terrain."""
    
    lats = ds.latitude.values
    lons = ds.longitude.values

    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    mid_lats = lats[lat_mask]

    lon_grid, lat_grid = np.meshgrid(lons, mid_lats)
    lon_grid = np.where(lon_grid > 180, lon_grid - 360, lon_grid) # convert to -180 to 180 deg

    land_mask = globe.is_land(lat_grid, lon_grid)
    flat_mask, elevation_vals, slope_vals = get_flat_terrain_mask(lat_grid, lon_grid, 
                                                                  slope_threshold)
    land_mask = land_mask & flat_mask

    return land_mask, lat_grid, lon_grid, lat_mask, slope_vals


# Wind angle computation
def compute_mean_surface_angle_grid(ds, lat_mask, speed_threshold=1.0):
    """
    Vectorized: mean surface wind angle (relative to geostrophic wind)
    for every (lat, lon) point in the midlatitude band, averaged over time.
    """
    u = ds.u.isel(latitude=np.where(lat_mask)[0]).values
    v = ds.v.isel(latitude=np.where(lat_mask)[0]).values
    p = ds.isobaricInhPa.values

    sort_idx = np.argsort(p)
    u = u[:, sort_idx, :, :]
    v = v[:, sort_idx, :, :]

    speed = np.hypot(u, v)

    max_idx = np.argmax(speed, axis=1)
    u_ref = np.take_along_axis(u, max_idx[:, None, :, :], axis=1)
    v_ref = np.take_along_axis(v, max_idx[:, None, :, :], axis=1)

    dot = u_ref * u + v_ref * v
    cross = u_ref * v - v_ref * u
    angle = np.degrees(np.arctan2(cross, dot))

    surf_idx = -1
    u_surf, v_surf = u[:, surf_idx, :, :], v[:, surf_idx, :, :]
    speed_surf = np.hypot(u_surf, v_surf)

    wind_angle_surf = angle[:, surf_idx, :, :]
    wind_angle_surf = np.where(speed_surf > speed_threshold, wind_angle_surf, np.nan)

    
    # Here we do a circular mena
    theta = np.deg2rad(wind_angle_surf)

    mean_theta = np.arctan2(
        np.nanmean(np.sin(theta), axis=0),
        np.nanmean(np.cos(theta), axis=0)
    )
    
    mean_angle_grid = np.rad2deg(mean_theta)

    return mean_angle_grid


# ---------------------------------------------------------------------
# Run the code
# ---------------------------------------------------------------------
ds = xr.open_dataset(wind_filename)
ds = select_time_period(ds, MONTH)


land_mask, lat_grid, lon_grid, lat_mask, slope_vals = get_land_mask(ds, LAT_MIN, LAT_MAX, 
                                                          coast_buffer, slope_threshold)

mean_angle_grid = compute_mean_surface_angle_grid(ds, lat_mask, SPEED_THRESHOLD)
mean_angle_plot = np.where(land_mask, mean_angle_grid, np.nan)

print(f"Computed grid in {time.time() - start:.2f} s")
print(f"Valid flat, inland points: {np.sum(~np.isnan(mean_angle_plot))}")

valid_angles = mean_angle_plot[~np.isnan(mean_angle_plot)]


# ---------------------------------------------------------------------
# Figure 1: Map
# ---------------------------------------------------------------------
fig1 = plt.figure(figsize=(8, 5), constrained_layout=True)

ax_americas = fig1.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax_eurasia = fig1.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())

norm_cbar = TwoSlopeNorm(vcenter=45,
                          vmin=np.nanmin(mean_angle_plot),
                          vmax=np.nanmax(mean_angle_plot))

# Americas
ax_americas.set_extent([-170, -30, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
ax_americas.add_feature(cfeature.OCEAN, facecolor='lightgray', zorder=0)
ax_americas.coastlines(resolution='50m')
ax_americas.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
mesh1 = ax_americas.pcolormesh(lon_grid, lat_grid, mean_angle_plot,
                                cmap='RdBu_r', norm=norm_cbar,
                                transform=ccrs.PlateCarree(), shading='auto')
mesh1.set_clim(0, 90)

# Eurasia
ax_eurasia.set_extent([-20, 150, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
ax_eurasia.add_feature(cfeature.OCEAN, facecolor='lightgray', zorder=0)
ax_eurasia.coastlines(resolution='50m')
ax_eurasia.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
mesh2 = ax_eurasia.pcolormesh(lon_grid, lat_grid, mean_angle_plot,
                               cmap='RdBu_r', norm=norm_cbar,
                               transform=ccrs.PlateCarree(), shading='auto')
mesh2.set_clim(0, 90)

gl1 = ax_americas.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl1.top_labels = False
gl1.right_labels = False
gl1.bottom_labels = False
gl2 = ax_eurasia.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl2.top_labels = False
gl2.right_labels = False

cbar = fig1.colorbar(mesh2, ax=[ax_americas, ax_eurasia],
                      pad=0.08, shrink=0.7)
cbar.set_label('Mean surface angle [deg]', fontsize=11)
cbar.ax.tick_params(labelsize=11)

fig1.suptitle('Geographical Spread of Mean Surface Angles', fontsize=14)

save_name1 = "ERA5_angle_map"
plt.savefig(f'plots/{save_name1}.png', dpi=300)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name1}.png", dpi=400)
plt.show()

# ---------------------------------------------------------------------
# Figure 2: The histogram
# ---------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))

n_bins = 50
ax2.hist(valid_angles, bins=n_bins, density=True,
         color='C0', alpha=0.6, edgecolor='black',
         label='ERA5')

ax2.axvline(45, color='black', linestyle='--', alpha=0.7, label='45° reference')
ax2.set_xlabel('Mean surface angle [deg]', fontsize=11)
ax2.set_ylabel('Density [-]', fontsize=11)
plt.suptitle('Distribution of Mean Surface Angles',
              fontsize=14)
ax2.set_title('Midllatitude flat land points',
              fontsize=11)
ax2.legend(loc="center right", fontsize=11)
ax2.tick_params(labelsize=11)
ax2.grid(True, linestyle='--', alpha=0.4)

save_name2  = "ERA5_angle_histogram"
plt.savefig(f'plots/{save_name2}.png', dpi=300)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name2}.png", dpi=400)
plt.show()

print(f"Total time: {time.time() - start:.2f} s")