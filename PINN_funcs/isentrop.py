# -*- coding: utf-8 -*-
"""
Isentropic flow relations and Sutherland's law for air.
"""

def T0_T(Ma, k = 1.4):
    t = 1 + (k-1)/2 * Ma**2
    return t

def p0_p(Ma, k = 1.4):
    t = T0_T(Ma)**(k/(k-1))
    return t

def r0_r(Ma, k = 1.4):
    t = T0_T(Ma)**(1/(k-1))
    return t

def calc_statics(Ma, p0, T0):
    R = 287.15
    r0 = p0 / (R * T0)
    p = p0 / p0_p(Ma)
    T = T0 / T0_T(Ma)
    r = r0 / r0_r(Ma)
    return p, T, r, r0

def sutherland(T, T_0 = 273, mu_0 = 1.716*10**-5):
    C = 111

    mu = mu_0 * (T/T_0)**1.5 * (T_0 + C) / (C + T)
    return mu
