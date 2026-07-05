"""
Created on Sat Jul  4 17:46:44 2026
@author: fredrik
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from global_land_mask import globe
import time

start = time.time()

ds = xr.open_dataset("../Data/wind_data.nc")

EARTH_RADIUS_KM = 6371.0
LAT_MIN, LAT_MAX = 1, 89
COAST_BUFFER_KM = 2
SPEED_THRESHOLD = 1.0


def is_away_from_coast(lat_grid, lon_grid, buffer_km=2, n_directions=8):
    """Check whether each land point is at least buffer_km from the coast."""
    away_from_coast = np.ones_like(lat_grid, dtype=bool)
    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

    for theta in angles:
        dlat = (buffer_km / EARTH_RADIUS_KM) * (180 / np.pi) * np.cos(theta)
        dlon = (buffer_km / EARTH_RADIUS_KM) * (180 / np.pi) * np.sin(theta) \
               / np.cos(np.deg2rad(lat_grid))

        test_lat = lat_grid + dlat
        test_lon = lon_grid + dlon
        test_lon_adj = np.where(test_lon > 180, test_lon - 360, test_lon)
        test_lon_adj = np.where(test_lon_adj < -180, test_lon_adj + 360, test_lon_adj)

        away_from_coast &= globe.is_land(test_lat, test_lon_adj)

    return away_from_coast


def get_inland_mask(ds, lat_min=30, lat_max=60, coast_buffer_km=2):
    """Boolean mask of midlatitude land points, excluding near-coast points."""
    lats = ds.latitude.values
    lons = ds.longitude.values

    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    mid_lats = lats[lat_mask]

    lon_grid, lat_grid = np.meshgrid(lons, mid_lats)
    lon_grid_adj = np.where(lon_grid > 180, lon_grid - 360, lon_grid)

    land_mask = globe.is_land(lat_grid, lon_grid_adj)
    coast_mask = is_away_from_coast(lat_grid, lon_grid_adj, buffer_km=coast_buffer_km)
    inland_mask = land_mask & coast_mask

    return inland_mask, lat_grid, lon_grid_adj, lat_mask


def compute_mean_surface_angle_grid(ds, lat_mask, speed_threshold=1.0):
    """
    Vectorized: mean surface wind angle (relative to geostrophic wind)
    for every (lat, lon) point in the midlatitude band, averaged over time.
    """
    # Extract full arrays for the midlatitude rows only: shape (time, level, lat, lon)
    u = ds.u.isel(latitude=np.where(lat_mask)[0]).values
    v = ds.v.isel(latitude=np.where(lat_mask)[0]).values
    p = ds.isobaricInhPa.values

    sort_idx = np.argsort(p)
    u = u[:, sort_idx, :, :]
    v = v[:, sort_idx, :, :]

    speed = np.hypot(u, v)

    # Geostrophic reference level per (time, lat, lon)
    max_idx = np.argmax(speed, axis=1)  # (time, lat, lon)
    u_ref = np.take_along_axis(u, max_idx[:, None, :, :], axis=1)  # (time, 1, lat, lon)
    v_ref = np.take_along_axis(v, max_idx[:, None, :, :], axis=1)

    dot = u_ref * u + v_ref * v
    cross = u_ref * v - v_ref * u
    angle = np.degrees(np.arctan2(cross, dot))  # (time, level, lat, lon)

    # Surface = last level after sorting (highest pressure)
    u_surf, v_surf = u[:, -1, :, :], v[:, -1, :, :]
    speed_surf = np.hypot(u_surf, v_surf)

    wind_angle_surf = angle[:, -1, :, :]  # (time, lat, lon)
    wind_angle_surf = np.where(speed_surf > speed_threshold, wind_angle_surf, np.nan)

    mean_angle_grid = np.nanmean(wind_angle_surf, axis=0)  # (lat, lon)
    return mean_angle_grid


# --- Run it ---
inland_mask, lat_grid, lon_grid_adj, lat_mask = get_inland_mask(
    ds, LAT_MIN, LAT_MAX, COAST_BUFFER_KM)

mean_angle_grid = compute_mean_surface_angle_grid(ds, lat_mask, SPEED_THRESHOLD)

# Mask out ocean / near-coast points for plotting
mean_angle_plot = np.where(inland_mask, mean_angle_grid, np.nan)

print(f"Computed grid in {time.time() - start:.2f} s")
print(f"Valid inland points: {np.sum(~np.isnan(mean_angle_plot))}")

# --- Plot ---
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([lon_grid_adj.min(), lon_grid_adj.max(),
               lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.OCEAN, facecolor='lightgray', zorder=0)
ax.coastlines(resolution='50m')
ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)

# Balanced colorbar centered at 45°: red = greater than 45, blue = less than 45
norm = TwoSlopeNorm(vcenter=45,
                     vmin=np.nanmin(mean_angle_plot),
                     vmax=np.nanmax(mean_angle_plot))

mesh = ax.pcolormesh(lon_grid_adj, lat_grid, mean_angle_plot,
                      cmap='RdBu_r', norm=norm,
                      transform=ccrs.PlateCarree(), shading='auto')

cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.05, shrink=0.4)
mesh.set_clim(0, 90)
cbar.set_label('Mean surface wind angle relative to geostrophic [°]')

ax.set_title(f'Mean Surface Wind Angle, Midlatitude Land Points\n'
             f'({LAT_MIN}°–{LAT_MAX}°N)')

plt.tight_layout()
plt.savefig('midlatitude_surface_angle_map.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"Total time: {time.time() - start:.2f} s")