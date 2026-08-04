# Ekman Spiral with Variable Eddy Viscosity

This project investigates how vertically varying eddy viscosity affects the structure of the Ekman spiral, Ekman transport, and surface turning angle.

Both theoretical approaches, analytical and numerical, and statistical approaches with ERA5 was done to analyse the Ekman spiral with variable eddy viscosity. 

Analytical solutions: 
- Constant viscosity
- Two-layer step-function viscosity
- Exponentially decaying viscosity

Numerical solutions: 
- A simplified upper boundary condition model with:
- Linearly varying viscosity
- Parabolic viscosity

The numerical solutions are obtained using boundary value problem solvers in Python (`scipy.integrate.solve_bvp`).

---

## Features

- Numerical Ekman spiral solver
- Variable viscosity profiles
- Surface angle calculations
- Comparison with classical Ekman theory
- Dimensionless parameter studies
- Analytical derivations for selected cases
- Visualization of velocity profiles and spiral geometry
- Statistical analysis

---

## Governing Equation

The governing equation is

$$
\frac{d}{dz}\left(\nu(z)\frac{dU}{dz}\right)=if(U-U_g)
$$

where

- $U = u + iv$ is the complex velocity
- $f$ is the Coriolis parameter
- $\nu(z)$ is the eddy viscosity profile
- $U_g$ is the geostrophic wind

Boundary conditions:

$$
U(0)=0
$$

$$
U(z\rightarrow \infty)=0
$$

---

## Author 
Fredrik Bergelv, Master’s student in Meteorology at Stockholm University. 





