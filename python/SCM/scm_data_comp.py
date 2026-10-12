import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

scm_run_name = "ctrl"
RUNDIR = f'SCM/scm_runs/{scm_run_name}/'


dsp  = xr.load_dataset(RUNDIR + 'progvar.nc', decode_times=False)   # SCM prognostic output
dsd  = xr.load_dataset(RUNDIR + 'diagvar2.nc', decode_times=False)   # SCM prognostic output
dsin = xr.load_dataset(f"SCM/scm_runs/{scm_run_name}/scm_in.nc", decode_times=False)  # ERA5 input/forcing


print(list(dsd.data_vars))

ZMAX = 1.5e3
ZLABEL = r'Height, $z$ [m]'
ZLIM = (0, ZMAX)
ZTICKS = np.arange(0, ZMAX + 1, 500)

# --- Height cutoffs (time-mean full-level heights), separately for each file ---
zf_p = dsp.height_f.data
izp0 = np.nonzero(zf_p.mean(axis=0) >= ZMAX)[0][-1]

zf_i = dsin.height_f.data
izi0 = np.nonzero(zf_i.mean(axis=0) >= ZMAX)[0][-1]

ZMBAR_SCM  = zf_p.mean(axis=0)[izp0:]
ZMBAR_ERA5 = zf_i.mean(axis=0)[izi0:]

# --- Time-mean wind profiles ---
U_SCM  = dsp.u.data[:, izp0:].mean(axis=0)
V_SCM  = dsp.v.data[:, izp0:].mean(axis=0)
U_ERA5 = dsin.u.data[:, izi0:].mean(axis=0)
V_ERA5 = dsin.v.data[:, izi0:].mean(axis=0)

# =========================================================
# Plot: u(z) and v(z), SCM (solid) vs ERA5 input (dashed)
# =========================================================
fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(8, 5))

axs[0].plot(U_SCM,  ZMBAR_SCM,  color='C0', label='SCM')
axs[0].plot(U_ERA5, ZMBAR_ERA5, color='C0', ls='--', label='ERA5')
axs[0].axvline(0, color='k', lw=0.5)
axs[0].set_xlabel(r'Zonal wind, $u$ [m s$^{-1}$]', fontsize=11)
axs[0].set_title('Zonal wind', fontsize=12)
axs[0].legend(fontsize=11)

axs[1].plot(V_SCM,  ZMBAR_SCM,  color='C1', label='SCM')
axs[1].plot(V_ERA5, ZMBAR_ERA5, color='C1', ls='--', label='ERA5')
axs[1].axvline(0, color='k', lw=0.5)
axs[1].set_xlabel(r'Meridional wind, $v$ [m s$^{-1}$]', fontsize=11)
axs[1].set_title('Meridional wind', fontsize=12)
axs[1].legend(fontsize=11)

for ax in axs:
    ax.grid(True, linestyle='--', alpha=0.6)

axs[0].set_ylabel(ZLABEL)
axs[0].set_ylim(ZLIM)
axs[0].set_yticks(ZTICKS)

plt.suptitle('SCM vs ERA5', fontsize=14)
plt.tight_layout()
plt.savefig(f"SCM/scm_vs_era5.png", dpi=400)
plt.show()
