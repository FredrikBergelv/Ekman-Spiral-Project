import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# =========================================================
# Wrap your whole pipeline into a function, one run in / one dict out
# =========================================================
def process_run(scm_run_name, zmax=1.5e3, shear_thresh=1e-4):
    RUNDIR = f'SCM/scm_runs/{scm_run_name}/'
    dsd  = xr.load_dataset(RUNDIR + 'diagvar.nc', decode_times=False)
    dsin = xr.load_dataset(RUNDIR + 'scm_in.nc', decode_times=False)

    TIME_S_D  = dsd.time.data.astype(float)
    TIME_S_IN = dsin.time.data.astype(float)
    NT = dsd.sizes['time']

    zf = dsin.height_f.data
    izf0 = np.nonzero(zf.mean(axis=0) >= zmax)[0][-1]
    zh = dsd.height_h.data
    izh0 = np.nonzero(zh.mean(axis=0) >= zmax)[0][-1]

    def interp_time(field_in, t_in, t_out):
        return np.array([np.interp(t_out, t_in, field_in[:, k])
                          for k in range(field_in.shape[1])]).T

    U  = interp_time(dsin.u.data, TIME_S_IN, TIME_S_D)[:, izf0:]
    V  = interp_time(dsin.v.data, TIME_S_IN, TIME_S_D)[:, izf0:]
    ZM = interp_time(dsin.height_f.data, TIME_S_IN, TIME_S_D)[:, izf0:]

    TAU_U = dsd.turb_flx_u.data[:, izh0:]
    TAU_V = dsd.turb_flx_v.data[:, izh0:]
    ZH = zh[:, izh0:]

    OMEGA = 7.2921e-5
    F_COR = 2 * OMEGA * np.sin(np.radians(dsin.lat.isel(time=0).data))

    dU_dz_full = np.gradient(U, axis=-1) / np.gradient(ZM, axis=-1)
    dV_dz_full = np.gradient(V, axis=-1) / np.gradient(ZM, axis=-1)
    dU_dz = np.array([np.interp(ZH[it], ZM[it][::-1], dU_dz_full[it][::-1]) for it in range(NT)])
    dV_dz = np.array([np.interp(ZH[it], ZM[it][::-1], dV_dz_full[it][::-1]) for it in range(NT)])

    SHEAR2 = dU_dz**2 + dV_dz**2
    NU_EDDY = (TAU_U * dU_dz + TAU_V * dV_dz) / (SHEAR2)
    NU_EDDY = np.where(SHEAR2 > shear_thresh**2, NU_EDDY, np.nan)

    H_ABL = dsd.pbl_height.data
    NU_MAX = np.nanmax(NU_EDDY, axis=1)
    H_EK = np.sqrt(2 * NU_MAX / F_COR)
    PHI = H_ABL / H_EK

    phi_valid = PHI[np.isfinite(PHI) & (PHI > 0)]
    return dict(name=scm_run_name, phi_valid=phi_valid,
                rough_len_mom=dsd.rough_len_mom.mean().item())

# =========================================================
# Run for ctrl only
# =========================================================
CASES = ['ctrl']
COLORS = {'ctrl': 'tab:blue'}

results = {case: process_run(case) for case in CASES}

# =========================================================
# Distribution plot, log-x centered at phi=1
# =========================================================
case = CASES[0]
r = results[case]
max_abs_log = np.nanmax(np.abs(np.log10(r['phi_valid'])))
bins = np.logspace(-max_abs_log, max_abs_log, 41)

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.hist(r['phi_valid'], bins=bins, density=True, color=COLORS[case],
        alpha=0.75, edgecolor='k')
ax.axvline(1.0, ls='--', color='gray', lw=1.5)
ax.set_xscale('log')
ax.set_xlabel(r'$\varphi$')
ax.set_ylabel('Density')
ax.grid(alpha=0.3, which='both')
ax.set_xlim(10**(-1), 10)

plt.suptitle(r'Distribution of $\varphi$', fontsize=14)
plt.tight_layout()
plt.savefig('SCM/phi_distribution.png', dpi=400)
plt.show()