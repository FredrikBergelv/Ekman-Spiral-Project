"""
Created on Tue Feb 24 10:49:15 2026

@author: fredrik
"""

import numpy as np
import matplotlib.pyplot as plt

def subfig (z_heights, nu_profile, u, v, 
            savename=None, 
            title=r'2-layer eddy-viscosity: $\nu(z)$ step-function',
            nu_version=r"$\nu_{2layer}$",
            z_surf=0.5) :
    
    idx = np.argmin(np.abs(z_heights - z_surf))
    theta0 = np.arctan2(np.abs(v[idx]-v[0]), np.abs(u[idx]-u[0]))
    theta_deg = np.degrees(theta0)


    plt.close("all")
    fig, axs = plt.subplots(1,3, figsize=(12,5))
    fig.suptitle(title, fontsize=15)
    
    # (a) Viscosity profile
    ax = axs[0]
    ax.plot(nu_profile, z_heights, color='green', label = nu_version)
    ax.set_xlabel(r"$\nu(z)$ [m²/s]")
    ax.set_ylabel("Height [m]")
    ax.set_title("Kinematic Viscosity Profile")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="center right")
    ax.set_xlim(0-max(nu_profile)*0.05, max(nu_profile)*1.1)
    ax.text(0.97, 0.95, "(a)", transform=ax.transAxes,
            ha='right', va='top')
    
    # (b) Velocity vs height
    ax = axs[1]
    ax.plot(u, z_heights, label=r"$u(z)$")
    ax.plot(v, z_heights, label=r"$v(z)$")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Height [m]")
    ax.set_title("Ekman Spiral vs Height")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.text(0.97, 0.95, "(b)", transform=ax.transAxes,
            ha='right', va='top')
    
    # (c) Spiral top view
    ax = axs[2]
    ax.plot(u, v, '-k', label=r"$\vec u (z) $")
    ax.scatter(u[0], v[0], color='red', label='surface')
    ax.scatter(u[-1], v[-1], color='green', label='top')
    ax.set_xlabel("u [m/s]")
    ax.set_ylabel("v [m/s]")
    ax.set_title("Ekman Spiral (Top View)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.axis('equal')
    ax.text(0.97, 0.95, "(c)", transform=ax.transAxes,
            ha='right', va='top')
    
    angle_text = f"Surface angle: {theta_deg:.1f}°"
    ax.text(0.05, 0.3, angle_text, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    if savename:
        plt.savefig(f"{savename}.png",dpi=400)