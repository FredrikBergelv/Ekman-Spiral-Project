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
MODE        = 'constant'   # 'constant': Vg independent of height in the lowest few km
                           # 'scale'   : ctrl profile multiplied by one factor at all heights
WIND_FACTOR = 2.0          # geostrophic wind speed multiplier
Z_REF       = 1.0e3        # (constant mode) height whose ctrl Vg is used everywhere below Z_CONST (m)
Z_CONST     = 3.0e3        # (constant mode) Vg is constant below this height (m)
Z_BACK      = 5.0e3        # (constant mode) back to the original ctrl Vg above this height (m)
Z_BLEND0    = 0.5e3        # (constant mode) initial wind blends towards Vg between these heights (m)
Z_BLEND1    = 1.5e3

zmid = ds0.height_f.isel(time=0).data          # ground-relative, index 0 = top
ds = ds0.copy(deep=True)

# Roughest surface: all land, all high vegetation (forest)
ds.cl.data[...] = 0.0
ds.lsm.data[...] = 1.0
ds.high_veg_cover.data[...] = 1.0
ds.low_veg_cover.data[...] = 0.0

if MODE == 'scale':
    ds.ug.data[...] = ds0.ug.data * WIND_FACTOR
    ds.vg.data[...] = ds0.vg.data * WIND_FACTOR
    ds.u.data[0, :] = ds0.u.data[0, :] * WIND_FACTOR
    ds.v.data[0, :] = ds0.v.data[0, :] * WIND_FACTOR

elif MODE == 'constant':
    iref = int(np.argmin(np.abs(zmid - Z_REF)))
    print(f'reference level index {iref}, height {zmid[iref]:.0f} m')

    # Vg at the reference level, one vector per timestep, times the factor
    ug_c = ds0.ug.data[:, iref] * WIND_FACTOR       # (NT,)
    vg_c = ds0.vg.data[:, iref] * WIND_FACTOR

    # r = 0 below Z_CONST (constant Vg), 1 above Z_BACK (original Vg), smooth in between
    r = transition_ramp((zmid - Z_CONST) / (Z_BACK - Z_CONST))[None, :]
    ds.ug.data[...] = ug_c[:, None] * (1 - r) + ds0.ug.data * r
    ds.vg.data[...] = vg_c[:, None] * (1 - r) + ds0.vg.data * r

    # Initial wind: ERA5 shape (scaled) near the ground, equal to Vg from Z_BLEND1 upwards,
    # which limits the spin-up shock
    w = transition_ramp((zmid - Z_BLEND0) / (Z_BLEND1 - Z_BLEND0))
    ds.u.data[0, :] = ds0.u.data[0, :] * WIND_FACTOR * (1 - w) + ds.ug.data[0, :] * w
    ds.v.data[0, :] = ds0.v.data[0, :] * WIND_FACTOR * (1 - w) + ds.vg.data[0, :] * w
else:
    raise ValueError("MODE must be 'constant' or 'scale'")

fname = os.path.join(OUT_DIR, f'scm_in_A_rough_strongwind.nc')
ds.to_netcdf(fname)

chk = xr.load_dataset(fname, decode_times=False)
print(fname)
print({v: float(chk[v].data[0]) for v in ['cl', 'lsm', 'high_veg_cover', 'low_veg_cover']})
for label, idx in [('surface', -1), ('~500 m', int(np.argmin(np.abs(zmid - 500)))),
                   ('~1 km', int(np.argmin(np.abs(zmid - 1000)))), ('~2 km', int(np.argmin(np.abs(zmid - 2000))))]:
    s0 = np.hypot(ds0.ug.data, ds0.vg.data)[:, idx].mean()
    s1 = np.hypot(chk.ug.data, chk.vg.data)[:, idx].mean()
    print(f'mean |Vg| at {label:8s}: ctrl = {s0:5.2f} m/s, new = {s1:5.2f} m/s')