"""
Created on Wed Oct 14 21:14:39 2026

@author: fredrik
"""

import os
import numpy as np
import xarray as xr

IN_FILE = 'SCM/scm_runs/ctrl/scm_in.nc'
OUT_DIR = 'SCM/create_new_in'
os.makedirs(OUT_DIR, exist_ok=True)

ds0 = xr.load_dataset(IN_FILE, decode_times=False)

def transition_ramp(x):
    x = np.maximum(np.minimum(x, 1), 0)
    return x**2 * (3 - 2*x)

# --- Settings ---
WIND_FACTOR = 2.0     # geostrophic wind speed multiplied by this near the ground
Z_FULL = 1.0e3        # full factor below this height (m)
Z_ZERO = 2.5e3        # no change above this height (m), smooth ramp in between

zmid = ds0.height_f.isel(time=0).data
ztrans = transition_ramp((Z_ZERO - zmid) / (Z_ZERO - Z_FULL))   # 1 below 1 km, 0 above 2.5 km
scale = 1 + (WIND_FACTOR - 1) * ztrans                           # shape (nlev,)

ds = ds0.copy(deep=True)

# Roughest surface: all land, all high vegetation (forest)
ds.cl.data[...] = 0.0
ds.lsm.data[...] = 1.0
ds.high_veg_cover.data[...] = 1.0
ds.low_veg_cover.data[...] = 0.0

# Stronger geostrophic wind, same direction (scaling both components keeps the direction)
ds.ug.data[...] = ds0.ug.data * scale[None, :]
ds.vg.data[...] = ds0.vg.data * scale[None, :]

# Scale the initial wind profile too, to reduce the spin-up shock (only time 0 is read)
ds.u.data[0, :] = ds0.u.data[0, :] * scale
ds.v.data[0, :] = ds0.v.data[0, :] * scale

fname = os.path.join(OUT_DIR, 'scm_in_A_rough_strongwind.nc')
ds.to_netcdf(fname)

chk = xr.load_dataset(fname, decode_times=False)
spd0 = np.hypot(ds0.ug.data, ds0.vg.data)[:, -1].mean()
spd1 = np.hypot(chk.ug.data, chk.vg.data)[:, -1].mean()
print(fname)
print({v: float(chk[v].data[0]) for v in ['cl', 'lsm', 'high_veg_cover', 'low_veg_cover']})
print(f'mean |Vg| at lowest level: ctrl = {spd0:.2f} m/s, new = {spd1:.2f} m/s')