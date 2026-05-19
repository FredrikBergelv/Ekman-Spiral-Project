# Ekman Spiral with Variable Eddy Viscosity

This project investigates how vertically varying eddy viscosity affects the structure of the Ekman spiral, Ekman transport, and surface turning angle.

Both analytical and numerical approaches are used to study several viscosity profiles, including:

- Constant viscosity
- Two-layer step-function viscosity
- "1.5-layer"" step-function viscosity
- Exponentially decaying viscosity
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

---

## Governing Equation

The governing equation is

```math
\frac{d}{dz}\left(\nu(z)\frac{dU}{dz}\right)
=
if(U-U_g)
```

where

- \(U = u + iv\) is the complex velocity
- \(f\) is the Coriolis parameter
- \(\nu(z)\) is the eddy viscosity profile
- \(U_g\) is the geostrophic wind

Boundary conditions:

```math
U(0)=0
```

```math
\frac{dU}{dz}(H)=0
```

---

## Author
Fredrik Bergelv, Master’s student in Meteorology at Stockholm University
