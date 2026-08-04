import xarray as xr
import numpy as np
import time

start = time.time()

ds = xr.open_dataset("../Data/wind_data.nc")  

lat, lon = 45, 60
speed_threshold = 1.0


def surface_angle_stats(ds, lat, lon, speed_threshold=1.0):
    lat_idx = np.argmin(np.abs(ds.latitude.values - lat))
    lon_idx = np.argmin(np.abs(ds.longitude.values - lon))

    u = ds.u.isel(latitude=lat_idx, longitude=lon_idx).values  # (time, level)
    v = ds.v.isel(latitude=lat_idx, longitude=lon_idx).values
    p = ds.isobaricInhPa.values

    sort_idx = np.argsort(p)
    u = u[:, sort_idx]
    v = v[:, sort_idx]

    speed = np.hypot(u, v)

    # Geostrophic reference level per time step (not a single flattened max)
    max_idx = np.argmax(speed, axis=1)
    time_idx = np.arange(u.shape[0])
    u_ref = u[time_idx, max_idx][:, None]  # reshape for broadcasting against (time, level)
    v_ref = v[time_idx, max_idx][:, None]

    dot = u_ref * u + v_ref * v
    cross = u_ref * v - v_ref * u
    angle = np.degrees(np.arctan2(cross, dot))  # (time, level)

    # Surface = last level after sorting (highest pressure)
    u_surf, v_surf = u[:, -1], v[:, -1]
    speed_surf = np.hypot(u_surf, v_surf)

    wind_angle_surf = angle[:, -1]
    wind_angle_surf = np.where(speed_surf > speed_threshold, wind_angle_surf, np.nan)

    return np.nanmean(wind_angle_surf), np.nanstd(wind_angle_surf)


mean_angle, std_angle = surface_angle_stats(ds, lat, lon, speed_threshold)
print(f"Mean surface angle: {mean_angle:.2f}°")
print(f"Std surface angle:  {std_angle:.2f}°")

print(time.time() - start, "s")