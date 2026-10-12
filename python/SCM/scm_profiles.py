import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

scm_run_name = "ctrl"
RUNDIR = f'SCM/scm_runs/{scm_run_name}/'

dsd  = xr.load_dataset(RUNDIR + 'diagvar.nc', decode_times=False)
dsin = xr.load_dataset(f"SCM/scm_runs/{scm_run_name}/scm_in.nc", decode_times=False)

TIME_S_D  = dsd.time.data.astype(float)
TIME_S_IN = dsin.time.data.astype(float)
NT = dsd.sizes['time']

ZMAX = 1.5e3
ZLABEL = r'Height, $z$ [m]'
ZLIM = (0, ZMAX)
ZTICKS = np.arange(0, ZMAX + 1, 500)

zf = dsin.height_f.data
izf0 = np.nonzero(zf.mean(axis=0) >= ZMAX)[0][-1]

zh = dsd.height_h.data
izh0 = np.nonzero(zh.mean(axis=0) >= ZMAX)[0][-1]

def interp_time(field_in, t_in, t_out):
    return np.array([np.interp(t_out, t_in, field_in[:, k]) for k in range(field_in.shape[1])]).T

U_full  = interp_time(dsin.u.data,        TIME_S_IN, TIME_S_D)
V_full  = interp_time(dsin.v.data,        TIME_S_IN, TIME_S_D)
UG_full = interp_time(dsin.ug.data,       TIME_S_IN, TIME_S_D)
VG_full = interp_time(dsin.vg.data,       TIME_S_IN, TIME_S_D)
ZM_full = interp_time(dsin.height_f.data, TIME_S_IN, TIME_S_D)

U  = U_full[:, izf0:]
V  = V_full[:, izf0:]
UG = UG_full[:, izf0:]
VG = VG_full[:, izf0:]
ZM = ZM_full[:, izf0:]

TAU_U = dsd.turb_flx_u.data[:, izh0:]
TAU_V = dsd.turb_flx_v.data[:, izh0:]
ZH = zh[:, izh0:]

ZMBAR = ZM.mean(axis=0)
ZHBAR = ZH.mean(axis=0)

OMEGA = 7.2921e-5
F_COR = 2 * OMEGA * np.sin(np.radians(dsin.lat.isel(time=0).data))

dU_dz_full = np.gradient(U, axis=-1) / np.gradient(ZM, axis=-1)
dV_dz_full = np.gradient(V, axis=-1) / np.gradient(ZM, axis=-1)

dU_dz = np.array([np.interp(ZH[it], ZM[it][::-1], dU_dz_full[it][::-1]) for it in range(NT)])
dV_dz = np.array([np.interp(ZH[it], ZM[it][::-1], dV_dz_full[it][::-1]) for it in range(NT)])

SHEAR2 = dU_dz**2 + dV_dz**2
SHEAR_THRESH = 1e-4

NU_EDDY = (TAU_U * dU_dz + TAU_V * dV_dz) / (SHEAR2 + 1e-12)
NU_EDDY = np.where(SHEAR2 > SHEAR_THRESH**2, NU_EDDY, np.nan)

HOUR_OF_DAY = (TIME_S_D % 86400) / 3600.0
LOCAL_HOUR = (HOUR_OF_DAY + dsin.lon.isel(time=0).data / 15.0) % 24

DAY_MASK = (LOCAL_HOUR >= 6) & (LOCAL_HOUR < 18)
NIGHT_MASK = ~DAY_MASK

def mean_profiles(mask):
    ug_ref = UG[mask, -1]
    vg_ref = VG[mask, -1]
    theta_g = np.arctan2(vg_ref, ug_ref)
    c = np.cos(theta_g)[:, None]
    s = np.sin(theta_g)[:, None]

    U_rot = U[mask, :] * c + V[mask, :] * s
    V_rot = -U[mask, :] * s + V[mask, :] * c

    UG_rot = UG[mask, :] * c + VG[mask, :] * s
    VG_rot = -UG[mask, :] * s + VG[mask, :] * c

    TAU_U_rot = TAU_U[mask, :] * c + TAU_V[mask, :] * s
    TAU_V_rot = -TAU_U[mask, :] * s + TAU_V[mask, :] * c

    u_mean = np.nanmean(U_rot, axis=0)
    v_mean = np.nanmean(V_rot, axis=0)
    ug_mean = np.nanmean(UG_rot, axis=0)
    vg_mean = np.nanmean(VG_rot, axis=0)
    tau_u_mean = np.nanmean(TAU_U_rot, axis=0)
    tau_v_mean = np.nanmean(TAU_V_rot, axis=0)

    angle_deg = np.degrees(np.arctan2(v_mean[-1], u_mean[-1]))

    return dict(tau_u=tau_u_mean, tau_v=tau_v_mean, u=u_mean, v=v_mean, ug=ug_mean, vg=vg_mean, nu=np.nanmean(NU_EDDY[mask, :], axis=0), angle=angle_deg, vg_rot=ug_mean[-1])

profiles = {'day': mean_profiles(DAY_MASK), 'night': mean_profiles(NIGHT_MASK)}

for regime in ['day', 'night']:
    p = profiles[regime]
    print(f"{regime}: u_g(z=-1) = {p['ug'][-1]:.6f} m/s, v_g(z=-1) = {p['vg'][-1]:.6e} m/s, turning angle = {p['angle']:.2f} degrees")

fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(10, 5))

# Turbulent momentum flux
axs[0].plot(profiles['night']['tau_u'], ZHBAR,  label=r"$\tau_x$ night", color="C1")
axs[0].plot(profiles['night']['tau_v'], ZHBAR,  ls='--',  label=r"$\tau_y$ night", color="C1")
axs[0].plot(profiles['day']['tau_u'], ZHBAR,  label=r"$\tau_x$ day", color="C0")
axs[0].plot(profiles['day']['tau_v'], ZHBAR, ls='--', label=r"$\tau_y$ day", color="C0")
axs[0].set_xlabel(r'Momentum flux, $\tau$ [m²/s²]', fontsize=11)
axs[0].set_title('Turbulent momentum flux', fontsize=12)
axs[0].legend(fontsize=11)

# Wind
axs[1].plot(profiles['night']['u'], ZMBAR,   label=r"$u$ night", color="C1")
axs[1].plot(profiles['night']['v'], ZMBAR,  ls='--',  label=r"$v$ night", color="C1")
axs[1].plot(profiles['day']['u'], ZMBAR,  label=r"$u$ day", color="C0")
axs[1].plot(profiles['day']['v'], ZMBAR,  ls='--', label=r"$v$ day", color="C0")
#axs[1].plot(profiles['day']['ug'], ZMBAR, ls='--',  lw=1.5, label=r"$U_g$ day", color="grey")
#axs[1].plot(profiles['night']['ug'], ZMBAR, ls=':', lw=1.5, label=r"$U_g$ night", color="grey")
axs[1].set_xlabel(r'Wind, $U$ [m/s]', fontsize=11)
axs[1].set_title('Wind', fontsize=12)
axs[1].legend(loc='upper left')

axs[1].text(0.97, 0.03, rf'$\theta_\text{{day}}$ = {profiles["day"]["angle"]:.1f}$^\circ$' + '\n' + rf'$\theta_\text{{night}}$ = {profiles["night"]["angle"]:.1f}$^\circ$', transform=axs[1].transAxes, ha='right', va='bottom', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

# Eddy viscosity
axs[2].plot(profiles['night']['nu'], ZHBAR,  label='Night', color="C1")
axs[2].plot(profiles['day']['nu'], ZHBAR, label='Day', color="C0")
axs[2].set_xlabel(r'Viscoity $\nu_{eddy}$, [m²/s]', fontsize=11)
axs[2].set_title('Eddy kinematic viscosity', fontsize=12)
axs[2].legend(fontsize=11)

for ax in axs:
    ax.grid(True, linestyle='--', alpha=0.6)

axs[0].set_ylabel(ZLABEL)
axs[0].set_ylim(ZLIM)
axs[0].set_yticks(ZTICKS)

plt.suptitle(f'Day and Night Mean Profiles', fontsize=14)
plt.tight_layout()
plt.savefig(f"SCM/scm_profiles.png", dpi=400)
plt.show()
