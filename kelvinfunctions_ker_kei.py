"""
Created on Sat Apr 18 12:35:12 2026

@author: fredrik
"""

from scipy.special import ker, kei
import numpy as np               

def ker0(x):
    return ker(x)

def kei0(x):
    return kei(x)

def kei1(x, h=1e-6):
    # finite differences
    ker_p = ker(x + h)
    ker_m = ker(x - h)
    kei_p = kei(x + h)
    kei_m = kei(x - h)

    dker = (ker_p - ker_m)/(2*h)
    dkei = (kei_p - kei_m)/(2*h)

    factor = -1/np.sqrt(2)

    kei1 = factor * (dkei - dker)

    return kei1

def ker1(x, h=1e-6):
    # finite differences
    ker_p = ker(x + h)
    ker_m = ker(x - h)
    kei_p = kei(x + h)
    kei_m = kei(x - h)

    dker = (ker_p - ker_m)/(2*h)
    dkei = (kei_p - kei_m)/(2*h)

    factor = -1/np.sqrt(2)

    ker1 = factor * (dker + dkei)

    return ker1                      

