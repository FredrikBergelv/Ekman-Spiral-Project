"""
Created on Mon Oct 12 18:53:30 2026

@author: fredrik
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.integrate import solve_bvp

scm_run_name = "ctrl"
ZMAX = 2.0e3
RUNDIR = f'SCM/scm_runs/{scm_run_name}/'

# ============================================================================
# Finding the SCM/ERA5 profiles and turbulent viscosity
# ============================================================================

#This is the ERA5 data saved in 15 min intervalls
dsin = xr.load_dataset(RUNDIR + 'scm_in.nc', decode_times=False)

# Thiis is the SCM data that has new variables (such as tau)
dsd  = xr.load_dataset(RUNDIR + 'diagvar.nc', decode_times=False)
dsp = xr.load_dataset(RUNDIR + 'progvar.nc', decode_times=False)

# Rougness length
z0 = dsd.rough_len_mom.mean().item()

# Get the time scales
TIME_SEC_DIAG  = dsd.time.data.astype(float)
TIME_SEC_IN = dsin.time.data.astype(float)
NT = dsd.sizes['time'] # Number of timesteps

# The data is in both half and full levels
zfull = dsp.height_f.data
Zhalf = dsd.height_h.data

# The idx for the height limit
izfull0 = np.nonzero(zfull.mean(axis=0) >= ZMAX)[0][-1]
iZhalf0 = np.nonzero(Zhalf.mean(axis=0) >= ZMAX)[0][-1]

# Extract the data on the height limit in 
U = dsp.u.data[:, izfull0:]
V = dsp.v.data[:, izfull0:]
Zfull = dsp.height_f.data[:, izfull0:]

UG = np.broadcast_to(U[:, 0:1], U.shape).copy() # use the U at the top
VG = np.broadcast_to(V[:, 0:1], V.shape).copy()

TAU_U = dsd.turb_flx_u.data[:, iZhalf0:]
TAU_V = dsd.turb_flx_v.data[:, iZhalf0:]
Zhalf = Zhalf[:, iZhalf0:]

# Time-mean height of each level, used as the y-axis when plotting
Z_SCM_AVG = Zfull.mean(axis=0)
Z_SCM_AVG_half = Zhalf.mean(axis=0)

F_CORIOLIS = 2*2*np.pi/(24*60**2) * np.sin(np.radians(dsin.lat.isel(time=0).data))

# Extract the shear from the full levels, and interpolate in time
dU_dz_full = np.gradient(U, axis=-1) / np.gradient(Zfull, axis=-1)
dV_dz_full = np.gradient(V, axis=-1) / np.gradient(Zfull, axis=-1)
dU_dz = np.array([np.interp(Zhalf[it], Zfull[it][::-1], dU_dz_full[it][::-1]) for it in range(NT)])
dV_dz = np.array([np.interp(Zhalf[it], Zfull[it][::-1], dV_dz_full[it][::-1]) for it in range(NT)])

# Eddy kinematic viscosity through least-squares
NU_EDDY = (TAU_U * dU_dz + TAU_V * dV_dz) / (dU_dz**2 + dV_dz**2)

#Find local time and use that to find Day and Night masks
HOUR_OF_DAY = (TIME_SEC_DIAG % 86400) / 3600.0
LOCAL_HOUR = (HOUR_OF_DAY + dsin.lon.isel(time=0).data / 15.0) % 24
DAY_MASK = (LOCAL_HOUR >= 6) & (LOCAL_HOUR < 18)
NIGHT_MASK = ~DAY_MASK

def mean_profiles(mask):
    """"
    Extract the profiles for plotting:
    """
    
    # Use geostropic wind of the lowest level(higest up)
    ug_ref = UG[mask, -1]
    vg_ref = VG[mask, -1]
    
    # Rotate the profiles along the x-axis 
    theta_g = np.arctan2(vg_ref, ug_ref)
    c = np.cos(theta_g)[:, None]
    s = np.sin(theta_g)[:, None]

    U_rot = U[mask, :] * c + V[mask, :] * s
    V_rot = -U[mask, :] * s + V[mask, :] * c
    UG_rot = UG[mask, :] * c + VG[mask, :] * s
    VG_rot = -UG[mask, :] * s + VG[mask, :] * c
    TAU_U_rot = TAU_U[mask, :] * c + TAU_V[mask, :] * s
    TAU_V_rot = -TAU_U[mask, :] * s + TAU_V[mask, :] * c

    #Take the time mean 
    u_mean = np.nanmean(U_rot, axis=0)
    v_mean = np.nanmean(V_rot, axis=0)
    ug_mean = np.nanmean(UG_rot, axis=0)
    vg_mean = np.nanmean(VG_rot, axis=0)
    tau_u_mean = np.nanmean(TAU_U_rot, axis=0)
    tau_v_mean = np.nanmean(TAU_V_rot, axis=0)
    
    # Extract the turning angle
    # SCM turning angle at the lowest level 
    i_ref = np.argmin(np.abs(Z_SCM_AVG))
    z_angle_scm = Z_SCM_AVG[i_ref]
    
    angle_deg = np.degrees(np.arctan2(v_mean[i_ref], u_mean[i_ref]))

    return dict(tau_u=tau_u_mean, tau_v=tau_v_mean, u=u_mean, v=v_mean, ug=ug_mean, vg=vg_mean,
                nu=np.nanmean(NU_EDDY[mask, :], axis=0), z_angle=z_angle_scm, angle=angle_deg, vg_rot=ug_mean[-1])

profiles = {'day': mean_profiles(DAY_MASK), 'night': mean_profiles(NIGHT_MASK)}


# ============================================================================
# Interpolate nu(z) from the SCM profile to the surface
# ============================================================================


def interp_nu_surf(z_arr, SCM):
    """
    Interpolate eddy viscosity linearly in log(height)
    For z <= z0, return the surface anchor nu_z0 to avoid log(0).
    """
    kappa_VK = 0.399
    nu_mean = np.asarray(SCM['nu'])
    u_mean = np.asarray(SCM['u'])

    z_ref = 21.0

    # Estimate friction velocity from the neutral logarithmic wind law
    valid_u = np.isfinite(Z_SCM_AVG) & np.isfinite(u_mean)
    z_u = np.asarray(Z_SCM_AVG)[valid_u]
    u_u = u_mean[valid_u]

    order_u = np.argsort(z_u)
    z_u = z_u[order_u]
    u_u = u_u[order_u]

    u_ref = np.interp(z_ref, z_u, u_u)
    u_star = kappa_VK * abs(u_ref) / np.log(z_ref / z0)
    nu_z0 = kappa_VK * z0 * u_star
    
    print("u_*²=",u_star**2)

    # Sort viscosity data and retain positive, finite heights
    z = np.asarray(z_arr)
    valid = np.isfinite(z) & (z > 0) & np.isfinite(nu_mean)

    nu_molecular = 1.506e-5
    z_valid = z[valid]
    nu_valid = np.clip(nu_mean[valid], nu_molecular, None)

    order = np.argsort(z_valid)
    z_valid = z_valid[order]
    nu_valid = nu_valid[order]

    # Anchor the profile at z0, then interpolate in log(height)
    z_data = np.concatenate(([z0], z_valid))
    nu_data = np.concatenate(([nu_z0], nu_valid))

    # Remove duplicate heights (np.interp expects increasing x values)
    z_data, unique_idx = np.unique(z_data, return_index=True)
    nu_data = nu_data[unique_idx]

    log_z_data = np.log(z_data)

    def nu_profile(z_query):
        z_query = np.asarray(z_query)
        z_safe = np.maximum(z_query, z0)

        result = np.interp(
            np.log(z_safe),
            log_z_data,
            nu_data,
            left=nu_data[0],
            right=nu_data[-1]
        )

        return result.item() if result.ndim == 0 else result

    return nu_profile

# ============================================================================
# Analytical Ekman solution using the SCM nu(z)
# ============================================================================
ZTOP_EKMAN = ZMAX
N_MESH = 300

def tau_ode(z, y, nu_profile, f):
    "Define a coupled first order ODE for tau(z)"
    nu = nu_profile(z)
    kappa2 = f / nu
    taur, taur_p, taui, taui_p = y
    return np.vstack([taur_p, -kappa2 * taui, taui_p, kappa2 * taur])

def tau_bc(y_surf, y_inf, f, Ug):
    taur0, taur_p0, taui0, taui_p0 = y_surf
    taurL, taur_pL, tauiL, taui_pL = y_inf
    return np.array([
        taur_p0 - f * Ug.imag,
        taui_p0 + f * Ug.real,
        taurL,
        tauiL,
    ])

def velocity_ode(z, y, nu_profile, f, Ug):
    "Define a coupled first order ODE for U(z)"
    nu = nu_profile(z)
    Ur, Ui, Pr, Pi = y
    dUr = Pr / nu
    dUi = Pi / nu
    dPr = -f * (Ui - Ug.imag)
    dPi =  f * (Ur - Ug.real)
    return np.vstack([dUr, dUi, dPr, dPi])

def velocity_bc(y_surf, y_inf, Ug):
    Ur0, Ui0, Pr0, Pi0 = y_surf
    UrL, UiL, PrL, PiL = y_inf
    return np.array([
        Ur0,
        Ui0,
        UrL - Ug.real,
        UiL - Ug.imag,
    ])

def solve_ekman(scm, f=F_CORIOLIS, z_top=ZTOP_EKMAN, n_mesh=N_MESH):
    "Solve the analytical Ekman profles from the nu(z)"
    
    # extract the 
    nu_profile = interp_nu_surf(Z_SCM_AVG_half, scm)
    Ug = complex(scm['ug'][0], scm['vg'][0])

    z_mesh = np.linspace(0, z_top, n_mesh)

    # Intial guess
    y0_tau = np.zeros((4, z_mesh.size))
    
    # Solve th e
    sol_tau = solve_bvp(
        lambda z, y: tau_ode(z, y, nu_profile, f),
        lambda ya, yb: tau_bc(ya, yb, f, Ug),
        z_mesh, y0_tau, max_nodes=20000, tol=1e-6)
    if not sol_tau.success:
        print(f'BVP (tau) did not converge! {sol_tau.message}')

    # Make linear guess
    Ur_guess = Ug.real * (z_mesh / z_top)
    Ui_guess = Ug.imag * (z_mesh / z_top)
    y0_vel = np.vstack([Ur_guess, Ui_guess, np.zeros_like(z_mesh), np.zeros_like(z_mesh)])
    sol_vel = solve_bvp(
        lambda z, y: velocity_ode(z, y, nu_profile, f, Ug),
        lambda ya, yb: velocity_bc(ya, yb, Ug),
        z_mesh, y0_vel, max_nodes=20000, tol=1e-6)
    if not sol_vel.success:
        print(f'BVP (U) did not converge: {sol_vel.message}')

    # Surface stress angle
    taur0, _, taui0, _ = sol_tau.sol(0.0)
    angle_tau_deg = np.degrees(np.arctan2(taui0, taur0))

    return dict(nu_profile=nu_profile, Ug=Ug, sol_tau=sol_tau, sol_vel=sol_vel, angle_tau=angle_tau_deg)

ekman = {regime: solve_ekman(profiles[regime]) for regime in ['day', 'night']}


# =========================================================
# PLOTTING
# =========================================================
Z_PLOT_EK = np.linspace(0, ZMAX, 300)

def plot_regime(regime, color):
    
    # Extract SCM data
    SCM = profiles[regime]
    # Extract analytical data
    ek = ekman[regime]
    taur_ek, _, taui_ek, _ = ek['sol_tau'].sol(Z_PLOT_EK)
    Ur_ek, Ui_ek, _, _     = ek['sol_vel'].sol(Z_PLOT_EK)
    # Compare SCM and Ekman turning angles at the same height
    z_ref = SCM['z_angle']
    u_ek_ref = np.interp(z_ref, Z_PLOT_EK, Ur_ek)
    v_ek_ref = np.interp(z_ref, Z_PLOT_EK, Ui_ek)
    angle_ek_ref = np.degrees(np.arctan2(v_ek_ref, u_ek_ref))

    fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(10, 5))
    print("|tau(0)|=",np.sqrt(np.hypot(SCM['tau_u'][-1], SCM['tau_v'][-1])))   # u* in m/s   
    
    # Turbulent momentum flux: SCM solid, Ekman dashed
    axs[0].plot(SCM['tau_u'], Z_SCM_AVG_half, color=color, label=r"$\tau_x$ SCM")
    axs[0].plot(SCM['tau_v'], Z_SCM_AVG_half, color=color, ls='--', label=r"$\tau_y$ SCM")
    axs[0].plot(taur_ek, Z_PLOT_EK, color='k', label=r"$\tau_x$ eq. 8")
    axs[0].plot(taui_ek, Z_PLOT_EK, color='k', ls='--', label=r"$\tau_y$ eq. 8")
    axs[0].set_xlabel(r'Momentum flux, $\tau$ [m²/s²]', fontsize=11)
    axs[0].set_title('Turbulent momentum flux', fontsize=12)
    axs[0].legend(fontsize=11)

    # Wind: SCM solid, Ekman dashed
    axs[1].plot(SCM['u'], Z_SCM_AVG, color=color, label=r"$u$ SCM")
    axs[1].plot(SCM['v'], Z_SCM_AVG, color=color, ls='--', label=r"$v$ SCM")
    axs[1].plot(Ur_ek, Z_PLOT_EK, color='k', label=r"$u$ eq. 8")
    axs[1].plot(Ui_ek, Z_PLOT_EK, color='k', ls='--', label=r"$v$ eq. 8")
    axs[1].set_xlabel(r'Wind, $U$ [m/s]', fontsize=11)
    axs[1].set_title('Wind', fontsize=12)
    axs[1].legend(loc='upper left', fontsize=11)

    axs[1].text(
        0.97, 0.03,
        rf'$\theta_\text{{SCM}}$ = {SCM["angle"]:.1f}$^\circ$' + '\n'
        rf'$\theta_\text{{eq. 8}}$ = {angle_ek_ref:.1f}$^\circ$',
        transform=axs[1].transAxes, ha='right', va='bottom', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
        )

    # Eddy viscosity (SCM only - this is the input used to build the Ekman solution)
    axs[2].plot(SCM['nu'], Z_SCM_AVG_half, color=color, label='SCM')
    axs[2].plot(ek['nu_profile'](Z_PLOT_EK), Z_PLOT_EK, color='k', ls=':', label='Interpolated/extrapolated')
    axs[2].set_xlabel(r'Viscoity $\nu_{eddy}$, [m²/s]', fontsize=11)
    axs[2].set_title('Eddy kinematic viscosity', fontsize=12)
    axs[2].legend(loc="upper right", fontsize=11)

    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.6)

    axs[0].set_ylabel(r'Height, $z$ [m]')
    axs[0].set_ylim(0, ZMAX)
    axs[0].set_yticks(np.arange(0, ZMAX+1, 500))

    plt.suptitle(f'{regime.capitalize()} Mean Profiles: SCM vs eq. 8', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"SCM/SCM_vs_eq8_{regime}.png", dpi=400)
    plt.show()

plot_regime('day',   'C0')
plot_regime('night', 'C1')

