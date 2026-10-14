"""
Created on Wed Oct 14 21:16:04 2026

@author: fredrik
"""
import os
import numpy as np
import xarray as xr

IN_FILE = 'SCM/scm_runs/ctrl/scm_in.nc'
OUT_DIR = 'SCM/create_new_in'
os.makedirs(OUT_DIR, exist_ok=True)

ds0 = xr.load_dataset(IN_FILE, decode_times=False)

print('q_soil ctrl (time 0, per soil layer):', ds0.q_soil.data[0, :], '| units:', ds0.q_soil.attrs.get('units'))
print('soil type soty:', ds0.soty.data[0])

# --- Settings ---
Q_DRY = 0.08     # close to wilting point for coarse soil
Q_WET = 0.38     # close to saturation for coarse soil

# Only ~52% of the column is land in ctrl, so soil moisture is diluted by the lakes.
# ALL_LAND removes the lakes and rescales the vegetation covers to keep their land proportions.
ALL_LAND = True

def make(fname, q_soil_value=None):
    ds = ds0.copy(deep=True)

    if ALL_LAND:
        lsm_old = ds0.lsm.data[0]
        hi = ds0.high_veg_cover.data[0] / lsm_old
        lo = ds0.low_veg_cover.data[0] / lsm_old
        tot = hi + lo
        if tot > 1:
            hi, lo = hi / tot, lo / tot
        ds.cl.data[...] = 0.0
        ds.lsm.data[...] = 1.0
        ds.high_veg_cover.data[...] = hi
        ds.low_veg_cover.data[...] = lo

    if q_soil_value is not None:
        ds.q_soil.data[...] = q_soil_value

    path = os.path.join(OUT_DIR, fname)
    ds.to_netcdf(path)
    chk = xr.load_dataset(path, decode_times=False)
    print(path, {v: round(float(chk[v].data[0]), 4) for v in ['cl', 'lsm', 'high_veg_cover', 'low_veg_cover']},
          'q_soil[0] =', chk.q_soil.data[0, :])

make('scm_in_C_soil_dry.nc', Q_DRY)
make('scm_in_C_soil_wet.nc', Q_WET)