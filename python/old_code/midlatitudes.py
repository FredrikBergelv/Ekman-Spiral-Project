"""
Created on Sat Jul  4 17:46:44 2026
@author: fredrik
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from global_land_mask import globe

ds = xr.open_dataset("../Data/wind_data.nc")  


EARTH_RADIUS_KM = 6371.0


def is_away_from_coast(lat_grid, lon_grid, buffer_km=2, n_directions=8):
    """
    Check whether each land point has all points within `buffer_km`
    (sampled in n_directions around it) also on land — i.e. it is at
    least `buffer_km` away from the coastline.
    """
    away_from_coast = np.ones_like(lat_grid, dtype=bool)

    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

    for theta in angles:
        # Convert buffer distance to degree offsets
        dlat = (buffer_km / EARTH_RADIUS_KM) * (180 / np.pi) * np.cos(theta)
        dlon = (buffer_km / EARTH_RADIUS_KM) * (180 / np.pi) * np.sin(theta) \
               / np.cos(np.deg2rad(lat_grid))

        test_lat = lat_grid + dlat
        test_lon = lon_grid + dlon
        test_lon_adj = np.where(test_lon > 180, test_lon - 360, test_lon)
        test_lon_adj = np.where(test_lon_adj < -180, test_lon_adj + 360, test_lon_adj)

        away_from_coast &= globe.is_land(test_lat, test_lon_adj)

    return away_from_coast


def count_midlatitude_land_points(ds, lat_min=30, lat_max=60, coast_buffer_km=2):
    """
    Count how many land grid points fall within the midlatitude band
    (northern hemisphere), excluding ocean points AND points within
    `coast_buffer_km` of the coastline.
    """
    lats = ds.latitude.values
    lons = ds.longitude.values

    # Latitude mask for midlatitude band
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    mid_lats = lats[lat_mask]

    # Build full 2D grid of (lat, lon) pairs for the midlatitude rows only
    lon_grid, lat_grid = np.meshgrid(lons, mid_lats)

    # global_land_mask expects longitudes in [-180, 180], not [0, 360]
    lon_grid_adj = np.where(lon_grid > 180, lon_grid - 360, lon_grid)

    # Vectorized land/ocean check
    land_mask = globe.is_land(lat_grid, lon_grid_adj)

    # Additionally require points to be at least `coast_buffer_km` from coast
    coast_mask = is_away_from_coast(lat_grid, lon_grid_adj, buffer_km=coast_buffer_km)
    inland_mask = land_mask & coast_mask

    n_land_points = np.sum(land_mask)
    n_inland_points = np.sum(inland_mask)
    n_total_points = land_mask.size

    return n_land_points, n_inland_points, n_total_points, lat_grid, lon_grid_adj, inland_mask


def plot_midlatitude_land_region(ds, lat_min=30, lat_max=60, coast_buffer_km=2):
    """
    Plot the dataset's spatial domain with midlatitude, coast-buffered
    land points shaded/marked, ocean and near-coast points excluded.
    """
    lats = ds.latitude.values
    lons = ds.longitude.values

    n_land_points, n_inland_points, n_total_points, lat_grid, lon_grid_adj, inland_mask = \
        count_midlatitude_land_points(ds, lat_min, lat_max, coast_buffer_km)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                   crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.coastlines(resolution='50m')
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)

    # Plot only the inland points (land, midlatitude, >= buffer from coast)
    ax.scatter(lon_grid_adj[inland_mask], lat_grid[inland_mask],
               color='orange', s=8, alpha=0.7,
               transform=ccrs.PlateCarree(),
               label=f'Inland points ({lat_min}°–{lat_max}°N, ≥{coast_buffer_km} km from coast)')

    ax.legend(loc='lower left')
    ax.set_title(f'Midlatitude Land Points, ≥{coast_buffer_km} km from Coast')
    plt.tight_layout()
    plt.show()


# Run it
n_land_points, n_inland_points, n_total_points, lat_grid, lon_grid_adj, inland_mask = \
    count_midlatitude_land_points(ds)

print(f"Land grid points in midlatitude band (30°N–60°N): {n_land_points}")
print(f"Land points >=2 km from coast: {n_inland_points}")
print(f"Total grid points in that band (land + ocean): {n_total_points}")

plot_midlatitude_land_region(ds)