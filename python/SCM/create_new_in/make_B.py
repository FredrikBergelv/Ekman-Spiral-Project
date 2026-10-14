"""
Created on Wed Oct 14 21:15:21 2026

@author: fredrik
"""
import os
import numpy as np
import xarray as xr

IN_FILE = 'SCM/scm_runs/ctrl/scm_in.nc'
OUT_DIR = 'SCM/create_new_in'
os.makedirs(OUT_DIR, exist_ok=True)

ds0 = xr.load_dataset(IN_FILE, decode_times=False)
NT = ds0.sizes['time']
TMID = np.arange(NT)

def transition_ramp(x):
    x = np.maximum(np.minimum(x, 1), 0)
    return x**2 * (3 - 2*x)

def transition_bump(x):
    x = np.maximum(np.minimum(x, 1), -1)
    return (1 - x**2)**2

def es(T):
    """Saturation vapour pressure (hPa), Magnus formula. Only the ratio is used."""
    return 6.112 * np.exp(17.67 * (T - 273.15) / (T - 29.65))

# --- Switches ---
USE_INVERSION  = True    # cold, stable lowest layer in the initial profile (time 0)
USE_COLD_ADV   = True    # continued low-level cooling through the run (tadv)
USE_SUBSIDENCE = True    # extra low-level subsidence (omega / etadot)

# --- Settings ---
DT_COOL    = 5.0         # K of cooling at the surface in the initial profile
DT_SKIN    = 5.0         # K of cooling of the initial skin temperature
Z_FULL_T   = 50.0        # full cooling below this height (m)
Z_ZERO_T   = 500.0       # no cooling above this height (m)
TADV_COOL  = -3e-5       # K/s (about -2.6 K/day) at the surface, tapering to 0 at Z_ZERO_T
T_RAMP_H   = 6.0         # hours over which the advection / subsidence is switched on
ETADOT_ADD = 0.3         # Pa/s added subsidence (the notebook uses 0.6)
P0, DP     = 900.0, 100.0   # subsidence bump centre and half-width (hPa)

zmid = ds0.height_f.isel(time=0).data
ztrans = transition_ramp((Z_ZERO_T - zmid) / (Z_ZERO_T - Z_FULL_T))   # 1 near ground, 0 above 500 m
ttrans = transition_ramp(TMID / T_RAMP_H)                              # 0 -> 1 over first hours

ds = ds0.copy(deep=True)

if USE_INVERSION:
    t_old = ds0.t.data[0, :]
    t_new = t_old - DT_COOL * ztrans
    ds.t.data[0, :] = t_new
    # keep relative humidity unchanged, otherwise the cooling creates fog/cloud
    ds.q.data[0, :] = ds0.q.data[0, :] * es(t_new) / es(t_old)
    # keep the derived field consistent (the model reads t, not this)
    ds.pot_temperature.data[0, :] = t_new * (1e5 / ds0.pressure_f.data[0, :])**0.2857
    ds.t_skin.data[0] = ds0.t_skin.data[0] - DT_SKIN

if USE_COLD_ADV:
    ds.tadv.data[...] = ds0.tadv.data + TADV_COOL * ttrans[:, None] * ztrans[None, :]

if USE_SUBSIDENCE:
    pedge = ds0.pressure_h.isel(time=0).data * 1e-2          # hPa, half levels
    ptrans = transition_bump((pedge - P0) / DP)
    trans_total = ttrans[:, None] * ptrans[None, :]
    etadot_new = ds0.etadotdpdeta.data + ETADOT_ADD * trans_total
    etadot_new[:, [0, -1]] = 0                               # must be 0 at top and bottom
    omega_new = 0.5 * (etadot_new[:, :-1] + etadot_new[:, 1:])
    ds.etadotdpdeta.data[...] = etadot_new
    ds.omega.data[...] = omega_new

fname = os.path.join(OUT_DIR, 'scm_in_B_stable.nc')
ds.to_netcdf(fname)

chk = xr.load_dataset(fname, decode_times=False)
print(fname)
print('t lowest level, time 0: ctrl = %.2f K, new = %.2f K' % (ds0.t.data[0, -1], chk.t.data[0, -1]))
print('t_skin, time 0:         ctrl = %.2f K, new = %.2f K' % (ds0.t_skin.data[0], chk.t_skin.data[0]))
z = chk.height_f.data[0, :]
i300 = np.argmin(np.abs(z - 300))
print('theta(300 m) - theta(lowest), time 0: ctrl = %.2f K, new = %.2f K' %
      (ds0.pot_temperature.data[0, i300] - ds0.pot_temperature.data[0, -1],
       chk.pot_temperature.data[0, i300] - chk.pot_temperature.data[0, -1]))