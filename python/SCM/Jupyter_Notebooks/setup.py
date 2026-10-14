from setuptools import setup

setup(
    name="Lab2_BLT_Course_MISU",
    version="1.0.0",
    description="Python environment for the 2nd lab of the Boundary Layer and Turbulence course at MISU",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "xarray",
        "netCDF4",
        "h5netcdf",
        "jupyterlab",
        "ipykernel",
    ],
)