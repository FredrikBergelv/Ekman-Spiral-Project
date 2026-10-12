import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

scm_run_name = "ctrl"
RUNDIR = f'SCM/scm_runs/{scm_run_name}/'
dsd = xr.load_dataset(RUNDIR + 'diagvar.nc', decode_times=False)
dsx = xr.load_dataset(RUNDIR + 'diagvar2.nc', decode_times=False)
dsin = xr.load_dataset(f"SCM/scm_runs/{scm_run_name}/scm_in.nc", decode_times=False)


# =========================================================
# 0) Time grids: dsd is coarse (hourly), dsin is fine (15-min)
#    -> interpolate dsin fields onto dsd's time grid
# =========================================================
TIME_S_D  = dsd.time.data.astype(float)     # coarse grid, seconds
TIME_S_IN = dsin.time.data.astype(float)    # fine grid, seconds

TIME_H = TIME_S_D / 3600.0
NT = dsd.sizes['time']
TTICKS = np.arange(0, TIME_H[-1] + 1, 12)

ZMAX = 1.5e3

zf = dsin.height_f.data
zfmean = zf.mean(axis=0)
izf0 = np.nonzero(zfmean >= ZMAX)[0][-1]

zh = dsd.height_h.data
zhmean = zh.mean(axis=0)
izh0 = np.nonzero(zhmean >= ZMAX)[0][-1]

# =========================================================
# 1) Interpolate dsin's u, v, height_f onto dsd's time grid
#    (pot_temperature no longer needed - pbl_height replaces the theta method)
# =========================================================
def interp_time(field_in, t_in, t_out):
    # field_in: (time_in, nlev); interpolate along axis 0 (time) onto t_out
    return np.array([np.interp(t_out, t_in, field_in[:, k]) for k in range(field_in.shape[1])]).T

U_full  = interp_time(dsin.u.data,       TIME_S_IN, TIME_S_D)
V_full  = interp_time(dsin.v.data,       TIME_S_IN, TIME_S_D)
ZM_full = interp_time(dsin.height_f.data, TIME_S_IN, TIME_S_D)

U  = U_full[:, izf0:]
V  = V_full[:, izf0:]
ZM = ZM_full[:, izf0:]

TAU_U = dsd.turb_flx_u.data[:, izh0:]
TAU_V = dsd.turb_flx_v.data[:, izh0:]
ZH = zh[:, izh0:]

# =========================================================
# 2) nu_eddy(z, t) - all arrays now on dsd's time grid, NT=72
# =========================================================
OMEGA = 7.2921e-5
F_COR = 2 * OMEGA * np.sin(np.radians(dsin.lat.isel(time=0).data))

dU_dz_full = np.gradient(U, axis=-1) / np.gradient(ZM, axis=-1)
dV_dz_full = np.gradient(V, axis=-1) / np.gradient(ZM, axis=-1)

dU_dz = np.array([np.interp(ZH[it], ZM[it][::-1], dU_dz_full[it][::-1])
                   for it in range(NT)])
dV_dz = np.array([np.interp(ZH[it], ZM[it][::-1], dV_dz_full[it][::-1])
                   for it in range(NT)])

SHEAR2 = dU_dz**2 + dV_dz**2
SHEAR_THRESH = 1e-4

NU_EDDY = (TAU_U * dU_dz + TAU_V * dV_dz) / (SHEAR2)
NU_EDDY = np.where(SHEAR2 > SHEAR_THRESH**2, NU_EDDY, np.nan)

# =========================================================
# 3) H_ABL from the model's own PBL height diagnostic
# =========================================================
H_ABL = dsd.pbl_height.data    # already on dsd's time grid, shape (NT,)

# =========================================================
# 4) h_EK and phi
# =========================================================
NU_MAX = np.nanmax(NU_EDDY, axis=1)
H_EK = np.sqrt(2 * NU_MAX / F_COR)
PHI = H_ABL / H_EK

# =========================================================
# Plot
# =========================================================
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(9, 6))

ax1.plot(TIME_H, H_EK, ms=3, color='C0', label=r'$h_{EK}$')
ax1.plot(TIME_H, H_ABL, ms=3, color='C1', label=r'$H_{ABL}$')
ax1.set_ylabel('Height, $[m]$')
ax1.set_title(r'Ekman height and ABL timeseries')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(0, 2000)

ax2.axhline(1.0, color='black', lw=1)
ax2.plot(TIME_H, PHI, ms=3, color='C2')
ax2.set_xlabel('time, $[h]$')
ax2.set_xticks(TTICKS)
ax2.set_ylabel(r'$\varphi$')
ax2.set_title(r'$\varphi$ timeseries')
ax2.set_yscale("log")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_ylim(10**(-1), 10**1)

plt.tight_layout()
plt.savefig(f"SCM/phi_time.png", dpi=400)
plt.show()


# =========================================================
# Statistics of phi: fraction above/below 1, min and max
# =========================================================
phi_valid = PHI[np.isfinite(PHI) & (PHI > 0)]
n_valid = len(phi_valid)
n_total = len(PHI)

pct_above = 100 * np.sum(phi_valid > 1) / n_valid
pct_below = 100 * np.sum(phi_valid < 1) / n_valid
pct_equal = 100 * np.sum(phi_valid == 1) / n_valid

print(f'Valid phi values: {n_valid} of {n_total} timesteps '
      f'({100 * n_valid / n_total:.1f}% survive the NaN filter)')
print(f'phi > 1: {pct_above:.0f}%')
print(f'phi < 1: {pct_below:.0f}%')
if pct_equal > 0:
    print(f'phi = 1: {pct_equal:.1f}%')
print(f'min phi: {phi_valid.min():.3f}  (at t = {TIME_H[np.nanargmin(np.where(np.isfinite(PHI) & (PHI > 0), PHI, np.nan))]:.0f} h)')
print(f'max phi: {phi_valid.max():.3f}  (at t = {TIME_H[np.nanargmax(np.where(np.isfinite(PHI) & (PHI > 0), PHI, np.nan))]:.0f} h)')
print(f'median phi: {np.median(phi_valid):.3f}')

print(f'\n {scm_run_name}: rough_len_mom mean = {dsd.rough_len_mom.mean().item():.4e} m')