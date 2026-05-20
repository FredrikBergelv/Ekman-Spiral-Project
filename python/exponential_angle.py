import numpy as np
from scipy.special import kv
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from kelvinfunctions_ker_kei import ker0, kei0, ker1, kei1

f = 1e-4 
U0 = 1
phi0 = 0.5
k = 10 # do not need in reality 

def old_sol(z, phi=phi0):
      alpha = phi*2*np.sqrt(1j*f)
      num = kv(0,alpha*np.exp(k*z/2))
      denum = 2*kv(1,alpha)
      ans = U0*k*alpha*np.exp(k*z)* num / denum
      return ans 


def dUdz(z, phi=phi0):

      arg1 = 2*phi*np.exp(k*z/2)
      arg2 = 2*phi
      num = ker0(arg1)+ 1j*kei0(arg1)
      denum = ker1(arg2) + 1j*kei1(arg2)
      ans = np.sqrt(1j)*U0*k*phi*np.exp(k*z)*(num /(denum) )
      return ans

def calculate_angle(dUdz):
    # Compute the angle of dU/dz
    theta = np.angle(dUdz, deg=True)
    return theta

# ===============================
# Plotting
# ===============================

phis = np.logspace(-5,0.6,1000)


plt.figure(figsize=(8, 5))
plt.suptitle("Surface Angle for Exponetial Model", fontsize=14)
plt.title("Surface angle vs decayrate")

angles = np.angle(dUdz(0, phis), deg=True)
plt.plot(phis, angles)

plt.xlabel(r"Height ratio, $\varphi$ [-]")
plt.ylabel(r"Surface angle, $\theta$ [deg]")
plt.grid(True, linestyle='--', alpha=0.6)

plt.hlines(45, min(phis), max(phis), color="black", linestyle='--', label="45° reference")
plt.legend(fontsize=11)
#plt.xscale("log")
plt.gca().xaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"{x:.5f}".rstrip('0').rstrip('.'))
)
plt.ylim(37,93)
plt.savefig("exponential_angle.png", dpi=400)
plt.show()

