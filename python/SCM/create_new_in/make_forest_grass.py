#!/usr/bin/env python3
"""
@author: Fredrik Bergelv
"""

import numpy as np, xarray as xr
dsin = xr.load_dataset('SCM/scm_runs/ctrl/scm_in.nc', decode_times=False)

def make(fname, hi_cover, lo_cover, lo_type=None, water=False):
    ds = dsin.copy(deep=True)
    ds.cl.data[...]  = 1.0 if water else 0.0      # water fraction
    ds.lsm.data[...] = 1.0 - ds.cl.data
    ds.high_veg_cover.data[...] = hi_cover
    ds.low_veg_cover.data[...]  = lo_cover
    if lo_type is not None:
        ds.low_veg_type.data[...] = lo_type
    ds.to_netcdf(fname)
    chk = xr.load_dataset(fname, decode_times=False)
    print(fname, {v: float(chk[v].data[0]) for v in
          ['cl','lsm','high_veg_cover','low_veg_cover','low_veg_type']})

make('SCM/create_new_in/scm_in_forest.nc', hi_cover=1.0, lo_cover=0.0)              
make('SCM/create_new_in/scm_in_grass.nc',  hi_cover=0.0, lo_cover=1.0, lo_type=2)   

