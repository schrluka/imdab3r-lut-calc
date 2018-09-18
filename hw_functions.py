"""Circuit models for calculating resulting waveforms, currents, etc.

Switching times t are an array with 4 entries that represent the relative switching times. Entries 0 & 1 refer to the
primary side and 2 & 3 to the secondary side.

All equations consider mains voltages in sector 1, i.e. u_a > 0 > u_b > u_c
"""


import numpy as np
from scipy.interpolate import interp1d


__author__ = "Lukas Schrittwieser"
__copyright__ = "Copyright 2018, ETH Zurich"
__credits__ = ["Michael Leibl"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Lukas Schrittwieser"
__email__ = "schrittwieser@lem.ee.ethz.ch"
__status__ = "Production"


# rect_power calculates the power transferred between two 50% duty cycle
# rectangular voltage (+/-u_x,+/-u_y) sources connected with a (stray)
# inductance*frq fL normalized to 1. t_x and t_y represent their absolute normalized phase angles,
# i.e. with s_x = 0 (0deg) and s_y = -0.25 (90deg) a phase shift of 90deg results.
def rect_power(u_x, u_y, t_x, t_y):
    # make sure we wrap correctly
    g = (0.5 + (t_x - t_y)) % 1 - 0.5
    return u_x * u_y * (g - np.sign(g) * 2 * g**2)


# function for the first N odd harmonics of a triangular wave (will return numpy array with n values)
def coef_tri(phi, n):
    harm_num = np.arange(1, 2 * n, 2)
    return (8/np.pi**2) * np.exp(1j*phi*harm_num) / (harm_num**2)


# create symmetric triangular function mapping x in 0..1 to y in -1..1..-1 using interpolation
tri_fun = interp1d([0, 0.5, 1], [-1, 1, -1], kind='linear', bounds_error=True)


# similar as above but this is rectangular, can be used to derive voltage waveforms
def rect_fun(x):
    return np.sign(0.5-x)


# calculate the first n odd harmonics of the dab current (w.r.t to the primary) for the given
# mains & dc voltages u and switching times s,
# Note: divide result by f*L to get a current value in A
# checked this with Matlab, has 180 deg phase shift with Gecko (and our model) but that does not really matter
def rms_current_harm(u, t, n=10):
    assert len(t) == 4  # only support aligned modulation, hence 4 entries

    # we do this by calculating the current produced by all individual 50% duty cycle rectangular
    # voltage sources which we know to be triangular. We can then sum all fourier coefficients with
    # the correct phase angles

    u_ab = u[0] # mains voltages in sector 1 (u_a > 0 > u_b > u_c)
    u_bc = u[1]
    u_ca = -u_ab - u_bc
    u_pn = u[2]

    # calculate Fourier coefficients of the individual sources
    c_ab = u_ab / 8 * coef_tri(2 * np.pi * t[0], n)
    c_bc = u_bc / 8 * coef_tri(2 * np.pi * t[1], n)
    c_ca = u_ca / 8 * coef_tri(0, n)	# sign needs to be inverted later on
    c_dc_lead = -u_pn / 8 * coef_tri(2 * np.pi * t[2], n)  # voltage source opposes inductor current
    c_dc_lag = -u_pn / 8 * coef_tri(2 * np.pi * t[3], n)

    c = c_ab + c_bc - c_ca + c_dc_lead + c_dc_lag

    # get the rms value of all harmonics
    harm = np.real(c * np.conj(c) / 2)   # /2 to get rms from the amplitude values in c, real to remove 0j
    rms_sqr = np.sum(harm)

    return [rms_sqr, harm, c]


# calculates the gradient of the rms current squared, with respect to the switching times vector t, considering n odd
# harmonics
def rms_current_grad(u, t, n=10):
    harm_num = np.arange(1, 2*n, 2)   # which harmonics we consider

    # derivative of the real part of the first n harmonics with respect the phase angle phi
    def d_re_d_phi(phi):
        return 8/(np.pi**2) * (-np.sin(phi*harm_num)) / harm_num

    # derivative of the imag part of the first n harmonics with respect the phase angle phi
    def d_im_d_phi(phi):
        return 8/(np.pi**2) * np.cos(phi*harm_num) / harm_num

    # This is somewhat uggly, we need the complex amplitudes of the current's Fourier series to calculate the
    # derivative, so we calculate it again here
    _, _, c = rms_current_harm(u, t, n)

    u_ab = u[0]  # renaming for better readability
    u_bc = u[1]
    u_pn = u[2]

    # For each harmonic we calculate:
    #  s = [s1, s2, s3, s4]
    #  c(s) = a(s) + jb(s), c* = a - jb
    # the derivative of IrmsSqr w.r.t. s1 can be found as:
    # d / d_s1(c * c*) = d / d_s1(a(s)^2 + b(s)^2) =
    #  = 2 a(s) d/d_s1(a(s)) + 2 b(s) d/d_s1(b(s))

    a = np.real(c)
    b = np.imag(c)

    # inner derivative: 2 * pi !
    d_i_d_t0 = 2*np.pi * u_ab / 8 * np.sum(
        a * d_re_d_phi(2 * np.pi * t[0]) + b * d_im_d_phi(2 * np.pi * t[0]))

    d_i_d_t1 = 2*np.pi * u_bc / 8 * np.sum(
        a * d_re_d_phi(2 * np.pi * t[1]) + b * d_im_d_phi(2 * np.pi * t[1]))

    d_i_d_t2 = 2*np.pi * -u_pn / 8 * np.sum(
        a * d_re_d_phi(2 * np.pi * t[2]) + b * d_im_d_phi(2 * np.pi * t[2]))

    d_i_d_t3 = 2*np.pi * -u_pn / 8 * np.sum(
        a * d_re_d_phi(2 * np.pi * t[3]) + b * d_im_d_phi(2 * np.pi * t[3]))

    return np.real(np.array([d_i_d_t0, d_i_d_t1, d_i_d_t2, d_i_d_t3]))


def sw_instants(t):
    # create a vector with switching instants (this not the same as the vector t of switching times)
    sw3 = -t[2]
    sw4 = -t[3]
    if sw3 < 0:
        sw3 += 1
    if sw4 < 0:
        sw4 += 1
    # note: to have all currents from the same half-wave we take the sw instant at 0.5 instead the one at 0
    return np.array([0.5, 0.5 - t[0], 0.5 - t[1], sw3, sw4])


def switched_current_fourier(u, t, n=50):
    # calculate switched current value using a Fourier series approximation of the current waveform which allows
    #  continuously differentiable constraints
    sw_inst = sw_instants(t)

    # get the fourier coefficients of our current function
    _, _, coefficients = rms_current_harm(u, t, n)

    # go back to time domain and evaluate at switching times (we could probably implement this without a for loop)
    i = np.zeros(5)
    for n, c in enumerate(coefficients):
        i = i - c * np.exp(2 * np.pi * 1j * (2 * n + 1) * sw_inst)
    i = np.real(i)

    return i, sw_inst


# calculate the switched current values and the switching instants for stair case type modulation (aligned rising
# edges) for ZVS to obtain current values in Ampere divide result by f*L (this function assumes a normalization to
# fL=1) positive values mean ZVS, the first 3 return values correspond to the primary side, remaining 2 are on sec
# (dc) side
def switched_current(u, t):
    # create a vector with switching times
    sw_inst = sw_instants(t)

    # use superposition: calculate contributions of the individual components
    i_ab = u[0] / 8 * tri_fun((sw_inst + t[0]) % 1)   # take phase shift of u_ab into account
    i_bc = u[1] / 8 * tri_fun((sw_inst + t[1]) % 1)   # take phase shift of u_bc into account
    i_ac = (u[0]+u[1]) / 8 * tri_fun(sw_inst%1)   # times are w.r.t. u_ac, so no phase shift here
    i_dc_lead = u[2] / 8 * tri_fun((sw_inst + t[2]) % 1)
    i_dc_lag = u[2] / 8 * tri_fun((sw_inst + t[3]) % 1)

    i_switched = i_ab + i_bc + i_ac - i_dc_lead - i_dc_lag   # dc voltage sign is opposite to pri side!
    return [i_switched, sw_inst]


def dab_io_currents(u, t):
    # calculates mains input and dc output currents given the mains and dc voltages in u and the switching times s
    # Note: it assumed that f*L=1Ohm (normalization) holds, if not divide result by f*L)
    u_ab = u[0]  # corresponding line-to-line voltages in sector 1
    u_bc = u[1]
    u_ac = u_ab + u_bc
    u_dc = u[2]

    # Note: use a voltage of '1' instead of u_ab to impl. divide by u_ab to get a current instead of a power
    i_ab = (- rect_power(1, u_bc, t[0], t[1])       # u_ab -> u_bc
            - rect_power(1, u_ac, t[0], 0)          # u_ab -> u_ac
            + rect_power(1, u_dc, t[0], t[2])       # u_ab -> u_dc
            + rect_power(1, u_dc, t[0], t[3])) / 4  # u_ab -> u_dc, /4 as both rect voltages have an amplitude of 1/2

    i_bc = (- rect_power(1, u_ab, t[1], t[0])       # u_bc -> u_ab
            - rect_power(1, u_ac, t[1], 0)          # u_bc -> u_ac
            + rect_power(1, u_dc, t[1], t[2])       # u_bc -> u_dc
            + rect_power(1, u_dc, t[1], t[3])) / 4  # u_bc -> u_dc

    # Note: the u_ac 50% rectangle switches at time 0 by definition (first rising edge)
    i_ac = (- rect_power(1, u_bc, 0, t[1])          # u_ac -> u_bc
            - rect_power(1, u_ab, 0, t[0])          # u_ac -> u_ab
            + rect_power(1, u_dc, 0, t[2])          # u_ac -> u_dc
            + rect_power(1, u_dc, 0, t[3])) / 4     # u_ac -> u_dc
    i_ca = -i_ac    # go to conventional 3-phase delta system

    # do a delta -> star transformation using Kirchoff's current law
    i_a = (i_ab - i_ca)
    i_b = (i_bc - i_ab)
    i_c = (i_ca - i_bc)

    i_dc = (+ rect_power(u_ac, 1, 0, t[2])      # u_ac -> u_dc lead
            + rect_power(u_ac, 1, 0, t[3])      # u_ac -> u_dc lag
            + rect_power(u_ab, 1, t[0], t[2])   # u_ac -> u_dc lead
            + rect_power(u_ab, 1, t[0], t[3])   # u_ac -> u_dc lag
            + rect_power(u_bc, 1, t[1], t[2])   # u_bc -> u_dc lead
            + rect_power(u_bc, 1, t[1], t[3])   # u_bc -> u_dc lag
            ) / 4

    # calculate instantaneous reactive power
    q = (u_ab * i_c + u_bc * i_a - u_ac * i_b) / np.sqrt(3)
    return [i_a, i_b, i_c, i_dc, q]


# create a sampled, time domain waveforms of transformer voltages and current, e.g. for plotting
def sampled_waveform(u, t, sampling_time):
    # use superposition: calculate contributions of the individual components
    i_ab = u[0] / 8 * tri_fun((sampling_time + t[0]) % 1)  # take phase shift s[0] of u_ab into account
    i_bc = u[1] / 8 * tri_fun((sampling_time + t[1]) % 1)  # take phase shift s[0] of u_bc into account
    i_ac = (u[0] + u[1]) / 8 * tri_fun(sampling_time % 1)  # time is w.r.t. u_ac, so no phase shift here
    i_dc_lead = u[2] / 8 * tri_fun((sampling_time + t[2]) % 1)
    i_dc_lag = u[2] / 8 * tri_fun((sampling_time + t[3]) % 1)

    i = i_ab + i_bc + i_ac - i_dc_lead - i_dc_lag  # dc voltage sign is opposite to pri side!

    u_p = (u[0] * rect_fun((sampling_time + t[0]) % 1) + u[1] * rect_fun((sampling_time + t[1]) % 1) +
           (u[0] + u[1]) * rect_fun(sampling_time % 1)) / 2
    u_s = u[2] * (rect_fun((sampling_time + t[2]) % 1) + rect_fun((sampling_time + t[3]) % 1)) / 2

    return i, u_p, u_s


def waveform(u, t):
    i_p, time = switched_current(u, t)
    # second half switches one half cycle later, wrap points that go beyond 1
    sw_times = np.mod(np.concatenate((time, time+0.5)), 1)
    i_p = np.concatenate((i_p, i_p*(-1)))
    # repeat the first point at (t == 0) at t == 1
    assert (np.abs(sw_times[5]) < 1e-3)
    sw_times = np.append(sw_times, 1)
    i_p = np.append(i_p, i_p[5])
    # our time axis is scattered -> sort things
    ind = np.argsort(sw_times)
    sw_times = sw_times[ind]
    i_p = i_p[ind]
    # print('sw_times: ', sw_times)
    # print('i_p:      ', i_p)
    return [sw_times, i_p]


# recreate waveform from harmonics c at time points in vector t
def harmonics_to_time_domain(t, coefficients):
    i = 0
    for n, c in enumerate(coefficients):
        i = i - c * np.exp(2 * np.pi * 1j * (2 * n + 1) * t)
    i = np.real(i)  # remove imaginary component resulting from rounding errors
    return i

