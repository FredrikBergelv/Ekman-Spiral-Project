import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import interp1d
from scipy.integrate import solve_bvp

scm_run_name = "ctrl"
RUNDIR = f'SCM/scm_runs/{scm_run_name}/'

dsd  = xr.load_dataset(RUNDIR + 'diagvar.nc', decode_times=False)
dsin = xr.load_dataset(f"SCM/scm_runs/{scm_run_name}/scm_in.nc", decode_times=False)

TIME_S_D  = dsd.time.data.astype(float)
TIME_S_IN = dsin.time.data.astype(float)
NT = dsd.sizes['time']

ZMAX = 1.6e3
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

    return dict(tau_u=tau_u_mean, tau_v=tau_v_mean, u=u_mean, v=v_mean, ug=ug_mean, vg=vg_mean,
                nu=np.nanmean(NU_EDDY[mask, :], axis=0), angle=angle_deg, vg_rot=ug_mean[-1])

profiles = {'day': mean_profiles(DAY_MASK), 'night': mean_profiles(NIGHT_MASK)}

for regime in ['day', 'night']:
    p = profiles[regime]
    print(f"{regime}: u_g(z=-1) = {p['ug'][-1]:.6f} m/s, v_g(z=-1) = {p['vg'][-1]:.6e} m/s, turning angle = {p['angle']:.2f} degrees")

# =========================================================
# Build nu(z) interpolator from the SCM-diagnosed profile
# =========================================================
def build_nu_interp(z_arr, nu_arr, nu_z0=3.3e-1):
    """
    Here we assume a lower viscity nu0 at z=0
    """
    order = np.argsort(z_arr)
    z_sorted = z_arr[order]
    nu_sorted = nu_arr[order]
    valid = np.isfinite(nu_sorted)
    z_valid = z_sorted[valid]
    nu_valid = np.clip(nu_sorted[valid], 1e-2, None)

    # Force nu -> nu_z0 (a small, near-zero value) at z=0, i.e. assume a
    # true no-slip-like surface condition for the eddy viscosity itself,
    # and linearly interpolate up to the lowest valid SCM level.
    z_full  = np.concatenate([[0.0], z_valid])
    nu_full = np.concatenate([[nu_z0], nu_valid])

    return interp1d(z_full, nu_full, kind='linear', bounds_error=False,
                     fill_value=(nu_full[0], nu_full[-1]))

def build_nu_interp(z_arr, nu_arr, z_fit_max=100.0):
    """
    Here we extrapolite the data using the z_fit_max length
    """
    order = np.argsort(z_arr)
    z_sorted = z_arr[order]
    nu_sorted = nu_arr[order]

    valid = np.isfinite(nu_sorted) & np.isfinite(z_sorted)
    z_valid = z_sorted[valid]
    nu_valid = np.clip(nu_sorted[valid], 1e-2, None)

    # Use all valid points between the surface and 100 m
    near = z_valid <= z_fit_max

    z_near = z_valid[near]
    nu_near = nu_valid[near]

    # Linear fit near the surface
    slope, intercept = np.polyfit(z_near, nu_near, 1)

    # Extrapolated viscosity at z = 0
    nu_at_z0 = max(intercept, 1e-3)

    # Add the extrapolated surface point
    z_full = np.concatenate([[0.0], z_valid])
    nu_full = np.concatenate([[nu_at_z0], nu_valid])

    return interp1d(
        z_full,
        nu_full,
        kind='linear',
        bounds_error=False,
        fill_value=(nu_full[0], nu_full[-1])
    )


# =========================================================
# Analytical Ekman solution using the SCM-diagnosed nu(z)
# =========================================================
ZTOP_EKMAN = 4000.0
N_MESH = 300

def tau_ode(z, y, nu_func, f):
    nu = nu_func(z)
    kappa2 = f / nu
    taur, taur_p, taui, taui_p = y
    return np.vstack([taur_p, -kappa2 * taui, taui_p, kappa2 * taur])

def tau_bc(ya, yb, f, Ug):
    taur0, taur_p0, taui0, taui_p0 = ya
    taurL, taur_pL, tauiL, taui_pL = yb
    return np.array([
        taur_p0 - f * Ug.imag,
        taui_p0 + f * Ug.real,
        taurL,
        tauiL,
    ])

def velocity_ode(z, y, nu_func, f, Ug):
    nu = nu_func(z)
    Ur, Ui, Pr, Pi = y
    dUr = Pr / nu
    dUi = Pi / nu
    dPr = -f * (Ui - Ug.imag)
    dPi =  f * (Ur - Ug.real)
    return np.vstack([dUr, dUi, dPr, dPi])

def velocity_bc(ya, yb, Ug):
    Ur0, Ui0, Pr0, Pi0 = ya
    UrL, UiL, PrL, PiL = yb
    return np.array([
        Ur0,
        Ui0,
        UrL - Ug.real,
        UiL - Ug.imag,
    ])

def solve_ekman(p, f, z_top=ZTOP_EKMAN, n_mesh=N_MESH):
    nu_func = build_nu_interp(ZHBAR, p['nu'])
    Ug = complex(p['ug'][0], p['vg'][0])

    z_mesh = np.linspace(0, z_top, n_mesh)

    y0_tau = np.zeros((4, z_mesh.size))
    sol_tau = solve_bvp(
        lambda z, y: tau_ode(z, y, nu_func, f),
        lambda ya, yb: tau_bc(ya, yb, f, Ug),
        z_mesh, y0_tau, max_nodes=20000, tol=1e-6)
    if not sol_tau.success:
        print(f'  [warning] stress BVP did not converge: {sol_tau.message}')

    Ur_guess = Ug.real * (z_mesh / z_top)
    Ui_guess = Ug.imag * (z_mesh / z_top)
    y0_vel = np.vstack([Ur_guess, Ui_guess, np.zeros_like(z_mesh), np.zeros_like(z_mesh)])
    sol_vel = solve_bvp(
        lambda z, y: velocity_ode(z, y, nu_func, f, Ug),
        lambda ya, yb: velocity_bc(ya, yb, Ug),
        z_mesh, y0_vel, max_nodes=20000, tol=1e-6)
    if not sol_vel.success:
        print(f'  [warning] velocity BVP did not converge: {sol_vel.message}')

    # Surface stress angle, theta = arg(tau(z=0)), per Eq. 11
    taur0, _, taui0, _ = sol_tau.sol(0.0)
    angle_tau_deg = np.degrees(np.arctan2(taui0, taur0))

    return dict(nu_func=nu_func, Ug=Ug, sol_tau=sol_tau, sol_vel=sol_vel, angle_tau=angle_tau_deg)

ekman = {regime: solve_ekman(profiles[regime], F_COR) for regime in ['day', 'night']}

for regime in ['day', 'night']:
    print(f"{regime}: Ekman surface stress angle theta_tau = {ekman[regime]['angle_tau']:.2f} degrees "
          f"(SCM wind-based angle = {profiles[regime]['angle']:.2f} degrees)")

# =========================================================
# Plot: separate figure for day, separate figure for night,
# each with SCM (solid) vs analytical Ekman (dashed) overlaid
# =========================================================
Z_PLOT = np.linspace(0, ZMAX, 300)

def plot_regime(regime, color):
    p = profiles[regime]
    ek = ekman[regime]

    taur_ek, _, taui_ek, _ = ek['sol_tau'].sol(Z_PLOT)
    Ur_ek, Ui_ek, _, _     = ek['sol_vel'].sol(Z_PLOT)

    fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(10, 5))

    # Turbulent momentum flux: SCM solid, Ekman dashed
    axs[0].plot(p['tau_u'], ZHBAR, color=color, label=r"$\tau_x$ SCM")
    axs[0].plot(p['tau_v'], ZHBAR, color=color, ls='--', label=r"$\tau_y$ SCM")
    axs[0].plot(taur_ek, Z_PLOT, color='k', label=r"$\tau_x$ eq. 8")
    axs[0].plot(taui_ek, Z_PLOT, color='k', ls='--', label=r"$\tau_y$ eq. 8")
    axs[0].set_xlabel(r'Momentum flux, $\tau$ [m²/s²]', fontsize=11)
    axs[0].set_title('Turbulent momentum flux', fontsize=12)
    axs[0].legend(fontsize=11)

    # Wind: SCM solid, Ekman dashed
    axs[1].plot(p['u'], ZMBAR, color=color, label=r"$u$ SCM")
    axs[1].plot(p['v'], ZMBAR, color=color, ls='--', label=r"$v$ SCM")
    axs[1].plot(Ur_ek, Z_PLOT, color='k', label=r"$u$ eq. 8")
    axs[1].plot(Ui_ek, Z_PLOT, color='k', ls='--', label=r"$v$ eq. 8")
    axs[1].set_xlabel(r'Wind, $U$ [m/s]', fontsize=11)
    axs[1].set_title('Wind', fontsize=12)
    axs[1].legend(loc='upper left', fontsize=11)

    axs[1].text(
        0.97, 0.03,
        rf'$\theta_\text{{SCM}}$ = {p["angle"]:.1f}$^\circ$' + '\n'
        rf'$\theta_\text{{eq. 8}}$ = {ek["angle_tau"]:.1f}$^\circ$',
        transform=axs[1].transAxes, ha='right', va='bottom', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
    )

    # Eddy viscosity (SCM only - this is the input used to build the Ekman solution)
    axs[2].plot(p['nu'], ZHBAR, color=color, label='SCM')
    axs[2].plot(ek['nu_func'](Z_PLOT), Z_PLOT, color='k', ls=':', label='Interpolated/extrapolated')
    axs[2].set_xlabel(r'Viscoity $\nu_{eddy}$, [m²/s]', fontsize=11)
    axs[2].set_title('Eddy kinematic viscosity', fontsize=12)
    axs[2].legend(loc="upper right", fontsize=11)

    print(ek['nu_func'](0))
    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.6)

    axs[0].set_ylabel(ZLABEL)
    axs[0].set_ylim(ZLIM)
    axs[0].set_yticks(ZTICKS)

    plt.suptitle(f'{regime.capitalize()} Mean Profiles: SCM vs eq. 8', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"SCM/eq8_comp_{regime}_gradient.png", dpi=400)
    plt.show()

plot_regime('day',   'C0')
plot_regime('night', 'C1')

