import numpy as np
import matplotlib.pyplot as plt    
from scipy.special import kv   # modified Bessel K
from kelvinfunctions_ker_kei import ker0, kei0, ker1, kei1
from plot import subfig 


# parametrar
f = 1e-4
nu0 = 1
k = 0.03
U0 = 10
phi = np.sqrt(f)/(k*np.sqrt(nu0))
z = np.linspace(0,400,500)


def old_sol(z):     
        # komplex parameter
        alpha = 2*np.sqrt(1j*f/(k**2*nu0))
        
        # variabel
        t = alpha*np.exp(k*z/2)
        ans = U0*(1 - (t*kv(1,t))/(alpha*kv(1,alpha)))
        return ans 

def new_sol(z):
      arg1 = 2*phi*np.exp(k*z/2)
      arg2 = 2*phi
      num = ker1(arg1) + 1j*kei1(arg1)
      denum = ker1(arg2) + 1j*kei1(arg2)
      ans = U0*(1-np.exp(k*z/2)*(num / denum))
      return ans


U = new_sol(z)

plt.plot(old_sol(z),z,label="old")
plt.plot(new_sol(z),z,label="new")

plt.title("U(z)")
plt.xlabel("U(z)")
plt.ylabel("z")
plt.legend()
plt.show()


# komponenter
u = np.real(U)
v = np.imag(U)

def nu(z, k=k, A=nu0, B=0):
    z = np.asarray(z)
    nu_array = A * np.exp(-k*z) + B
    return nu_array

nu_profile = nu(z)
subfig(z, nu_profile, u, v, 
       savename="exponential_analytic_solution",
       title="exponetial eddy-viscosity: Analytic solution",
       nu_version = r"$\nu=A e^{-kz}$")