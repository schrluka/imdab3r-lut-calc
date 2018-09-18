#!/usr/bin/env python
""" Calculates a lookup table with optimal switching times for an isolated matrix-type DAB three-phase rectifier.

This file calculates a 3D lookup table of relative switching times for an IMDAB3R, which are optimized for minimal
conduction losses. In discontinuous conduction mode (DCM) analytical equations for the optimal
operating conditions are used and numerical optimization is used in continuous conduction mode (CCM).
"""


import sys
import argparse
import time
import numpy as np
from scipy.optimize import fmin_slsqp
import hw_functions as hw
from csv_io import export_csv


__author__ = "Lukas Schrittwieser"
__copyright__ = "Copyright 2018, ETH Zurich"
__credits__ = ["Michael Leibl"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Lukas Schrittwieser"
__email__ = "schrittwieser@lem.ee.ethz.ch"
__status__ = "Production"


def solver_to_sw_times(x):
    d1 = x[0]   # space used by the numeric solver
    d2 = x[1]
    d_dc = x[2]
    shift = x[3]

    shift = np.clip(shift, -0.25, 0.25)  # -0.25 ... 0.25
    d_dc = np.clip(d_dc, 0, 0.5)  # duty cycle of dc-side transformer voltage, 0 ... 0.5

    t = [0.5-d1, 0.5-d2, 0, 0]
    t[2] = 0.5 - (d1/2 + shift + d_dc/2)
    t[3] = -(d1/2 + shift - d_dc/2)
    return t


def solver_to_sw_times_jac(x, u=None):
    # jacobian of solver_to_sw_times which maps d1, d2, d_dc and shift to sw_times
    J1 = np.array(
          [[ -1,  0,   0,   0], # derivative of s[0] w.r.t to x[0]...x[3]
          [  0, -1,   0,   0],
          [-1/2, 0, -1/2, -1],
          [-1/2, 0,  1/2, -1]])
    return J1


def sw_times_to_solver(s):
    # transform from switching times to solver coordinate system
    d1 = 0.5 - s[0]
    d2 = 0.5 - s[1]
    shift = -0.5 * (s[2] + s[3] - 0.5 + d1)
    shift = np.clip(shift,-0.25,0.25)
    d_dc = 2 * (s[3] + d1 / 2 + shift)
    d_dc = np.clip(d_dc, 0, 0.5)

    x = np.zeros(4)
    x[0] = d1
    x[1] = d2
    x[2] = d_dc
    x[3] = shift
    return x


# helper functions for DCM to create switching time vectors for given duty cycles
def align_fe(d_1, d_2, d_dc):
    t_3 = 0.5 - d_1
    t = [0.5 - d_1, 0.5 - d_2, t_3, d_dc - 0.5 + t_3]
    return t


def align_re(d_1, d_2, d_dc):
    t = np.array([0.5 - d_1, 0.5 - d_2, 0.5 - d_dc, 0])
    return t


# check that obtained switching times achieved the required output current
def check_solution(u, t, i_dc_ref):
    _, _, _, i_dc, q = hw.dab_io_currents(u, t)
    i_dc_err = (i_dc - i_dc_ref)
    q_err = q
    # if possible: normalize
    if (i_dc_err > 1e-6):
        i_dc_err = i_dc_err / i_dc_ref
        q_err = q_err / i_dc_ref
    ret = 0
    if (np.abs(i_dc_err) > 1e-3):
        print('invalid solution, i_dc_err: ', i_dc_err)
        ret = ret + 100
    if (np.abs(q_err) > 1e-3):
        print('invalid solution, q_err: ', q_err)
        ret = ret + 10
    return ret


# in max output current DCM, there is a u_pn for which both d_1 and d_dc become 0.5 for a given ac voltage (aka grid
# voltage angle wt) and achieve no reactive power at the mains input (q=0)
# this is the boundary case where the solution switches from aligned rising edges (u_pn lower than equivalent AC
# voltage) to falling edge aligned
def dcm_max_u_pn_boundary(u):
    u_ab = u[0]  # in sector 1
    u_bc = u[1]
    u_pn = 2 * (u_ab**2 + u_ab*u_bc + u_bc**2) / (2*u_ab + u_bc)
    return u_pn


# calculates max dc output current possible with DCM for given voltages u
def dcm_i_dc_max(u):
    u_ab = u[0]  # in sector 1
    u_bc = u[1]
    u_pn = u[2]

    if u_pn < dcm_max_u_pn_boundary(u):
        if u_bc < 1e-4:
            # wt = 0, calculate d_2 required to draw the same charge from phase a and phase b (to achieve q=0)
            uac = u_ab + u_bc  # by definition of the three-phase voltages
            d_dc = 0.5
            d_1 = d_dc * u_pn / uac  # this is 1/2 - t1
            # this case is simple: only during 0 < t < d_1 we are connected to the mains and the transformer
            # current rises linearly (due to d_dc = 0.5 the dc voltage must be on all the time)
            # therefore we just have to make d_2 so long that the area of the current triangle is split in half:
            d_2 = np.sqrt(0.5) * d_1
            t = align_re(d_1, d_2, d_dc)
        else:
            # analytic solution found by mathematica, works fine from 0 < wt <= 30deg
            det = (u_bc**2)*(u_ab + 2*u_bc)*(u_ab + u_bc - u_pn)*(u_pn**2)*(2*(u_ab**2 + u_ab*u_bc + u_bc**2) -
                                                                            (2*u_ab + u_bc)*u_pn)
            t1 = (u_ab*(u_ab+u_bc-u_pn)*(2*(u_ab**2 + u_ab*u_bc + u_bc**2) - (2*u_ab + u_bc)*u_pn) +
                  np.sqrt(det)) / (4*u_ab*(u_ab+u_bc)*(u_ab**2+u_ab*u_bc+u_bc**2) -
                                   2*(u_ab-u_bc)*(2*u_ab**2+3*u_ab*u_bc+2*u_bc**2)*u_pn)

            x = (u_pn/(1-2*t1) - u_ab) / u_bc  # that's d_2 / d_1, this ensures volt-sec balance
            t2 = 0.5 - x*(0.5-t1)
            t = [t1, t2, 0, 0]
    else:
        # analytic solution found by mathematica
        det =(u_ab**2 - u_bc**2)*(u_ab - u_pn)*u_pn*(2*(u_ab**2 + u_ab*u_bc + u_bc**2) - (2*u_ab + u_bc)*u_pn)
        t2 = (-(u_ab**2)*u_bc+(u_bc**3)-np.sqrt(det))/(2*(u_bc**2)*(-u_ab+u_bc) + 2*(2*(u_ab**2)+u_bc**2)*u_pn -
                                                    2*(2*u_ab+u_bc)*(u_pn**2))
        t4 = -1/2 + (u_ab/2 + u_bc*(1/2-t2)) / u_pn    # ensure volt sec balance
        t = [0, t2, 0, t4]

    _, _, _, i_dc, _ = hw.dab_io_currents(u, t)
    return [t, i_dc]


# check if DCM can be used to the achieve an output current of i_dc_ref at operating point u
def check_dcm(u, i_dc_ref, do_print=False):
    # calc max output current for discontinuous conduction mode (DCM) for the given voltages
    t_opt, i_dc_dcm_max = dcm_i_dc_max(u)
    if do_print:
        print('i_dc_dcm_max: ', i_dc_dcm_max)

    # check if this output current should be realized by DCM
    if i_dc_ref > i_dc_dcm_max:
        return None

    if do_print:
        print('using DCM solution')
    # the requested current is achievable by TCM, so we use this solution as it is the one with the lowest rms
    # transformer current
    k = np.sqrt(i_dc_ref / i_dc_dcm_max)  # scaling factor for the three duty cycles of u_p and u_s
    # extract duty cycles from switching times calculated for max output current
    d_1 = 0.5 - t_opt[0]
    d_2 = 0.5 - t_opt[1]
    d_dc = 0.5 - t_opt[2] + t_opt[3]
    is_re_aligned = (t_opt[3] == 0)  # If t_4 (t[3]) is 0 the rising edges of pri and sec voltage are aligned

    # apply scaling factor and re-create switching times
    d_1 = d_1 * k
    d_2 = d_2 * k
    d_dc = d_dc * k
    if is_re_aligned:
        t_opt = align_re(d_1, d_2, d_dc)
    else:
        t_opt = align_fe(d_1, d_2, d_dc)

    return t_opt


# derive optimal switching times in CCM for given voltages u with an output current i_dc_ref
def calc_ccm(u, i_dc_ref, i_dc_nom, t0, do_print=False):
    # objective function
    def obj(x):
        s = solver_to_sw_times(x)
        # note: Imperfections result from the fact that we can only consider an finite amount of harmonics. To avoid
        # problems with the numeric solver we select a rather high n here as the computational burden is low.
        i_rms_sqr, _, _ = hw.rms_current_harm(u, s, n=200)
        f = (i_rms_sqr / (i_dc_nom ** 2))
        return f

    # gradient of the objective
    def gradient(x):
        t = solver_to_sw_times(x)
        J = solver_to_sw_times_jac(x)
        di_dt = hw.rms_current_grad(u, t, n=200)
        res = np.dot(di_dt, J) / (i_dc_nom ** 2)
        return res

    # equality constraint functions:  demanded dc output current (ie active power) and reactive power
    def eqcons(x):
        t = solver_to_sw_times(x)
        _, _, _, i_dc, q = hw.dab_io_currents(u, t)
        i_dc_err = (i_dc - i_dc_ref) / i_dc_nom
        q_err = q / i_dc_nom
        return [i_dc_err, q_err]

    # inequality constraints: ensure ZVS
    def ieqcons(x):
        s = solver_to_sw_times(x)
        i, _ = hw.switched_current(u, s)
        return np.array(i) / i_dc_nom # pos values are ZVS, which is what the inequality constraints ensure

    x0 = sw_times_to_solver(t0)
    b = [(0, 0.5), (0, 0.5), (0, 0.5), (-0.24, 0.24)]   # bounds

    if do_print:
        iprint = 1
    else:
        iprint = 0

    # call the solver
    opt_x, fx, its, i_mode, s_mode = fmin_slsqp(obj, x0, f_eqcons=eqcons, fprime=gradient, f_ieqcons=ieqcons,
                                                bounds=b, full_output=True, iter=1000,
                                                iprint=iprint)
    opt_s = solver_to_sw_times(opt_x)

    if do_print or i_mode != 0:
        print('opt terminated in {0:} iterations with {1:}: {2:} '.format(its, i_mode, s_mode))
    eqc = eqcons(opt_x)
    ieqc = ieqcons(opt_x)
    if (np.max(np.abs((eqc))) > 1e-3) or (np.min(ieqc) < -1e-6):
        i_mode = 100
        print('Constraint violation detected: eq cons={0:}  ieq cons = {1:}'.format(eqc, ieqc))

    return [opt_s, i_mode]  # i_mode is zero on success or positive otherwise


# Maximum output current in Triangular Current Mode (TCM) of a conventional dc/dc DAB according to
# KRISMER AND KOLAR: CLOSED FORM SOLUTION FOR MINIMUM CONDUCTION LOSS MODULATION OF DAB CONVERTERS
# in IEEE Transactions on Power Electronics, vol. 27, no. 1, pp. 174-188, Jan. 2012
# https://doi.org/10.1109/TPEL.2011.2157976
# Note: TCM in a conventional DAB is like DCM in the IMDAB3R
def krismer_i_dc_tcm_max(u):
    # check that we are either at wt=0 or wt=30deg, otherwise we can't operate like a conventional DAB
    assert ((np.abs(u[0] - u[1]) < 1e-6) or (np.abs(u[1]) <= 1e-6))

    # abbreviation, assuming mains voltage in sector 1
    u_ac = u[0] + u[1]
    u_pn = u[2]

    # corner case: with 0 output voltage we cannot operate in TCM (there is no way to control the current with the
    # secondary side voltage)
    if u_pn < 1e-6:
        return 0

    # normalized quantities in Krimer's notation
    v_ref = u_ac
    v_A = np.min([u_ac, u_pn]) / v_ref
    v_B = np.max([u_ac, u_pn]) / v_ref

    # calc max power for which we use triangular current mode (ZCS)
    p_tcm_max = np.pi / 2 * v_A ** 2 * (v_B - v_A) / v_B
    # rescale back to the dc output current in our notation
    i_dc_tcm_max = p_tcm_max * v_ref**2 / (2 * np.pi * u_pn)
    return i_dc_tcm_max


# calc optimal switching times according to
# KRISMER AND KOLAR: CLOSED FORM SOLUTION FOR MINIMUM CONDUCTION LOSS MODULATION OF DAB CONVERTERS
# in IEEE Transactions on Power Electronics, vol. 27, no. 1, pp. 174-188, Jan. 2012
# https://doi.org/10.1109/TPEL.2011.2157976
def krismer(u, i_dc_ref):
    # abbreviation, assuming mains voltage in sector 1
    u_ac = u[0] + u[1]
    u_pn = u[2]

    # check that we are at wt=30deg, otherwise we can't operate like a conventional DAB
    # Note: For wt=0 the transformer current looks the same, however, we need to determine an additional
    # duty cycle d_2 (switching time t_2) for phase b. Even though this does not change the shape of the current (as
    # u_bc is zero) changing d_2 will result in different reactive power and a function solver is required)
    assert (np.abs(u[0] - u[1]) < 1e-6)

    # normalized quantities according to Krimer's notation
    v_ref = u_ac
    v_A = np.min([u_ac, u_pn]) / v_ref
    v_B = np.max([u_ac, u_pn]) / v_ref
    p = u_pn * i_dc_ref * 2 * np.pi / (v_ref**2)

    # calc max power for which we use triangular current mode (discontinuous conduction mode)
    p_tcm_max = np.pi/2 * v_A**2 * (v_B-v_A) / v_B

    if (p <= p_tcm_max) and (v_A != v_B):
        if v_A < 0.001:
            # we have no output voltage hence output power is 0 for any output current so we cannot use krismers eq.
            # however this is trivial, we always operate at max phase shift and create the required transformer current
            # amplitude
            d_a = 0.5   # get as much current to secondary as we can
            phi = 0.25  # i.e. 90deg
            d_b = 0.5 - np.sqrt(0.25 - 2*i_dc_ref)
            assert (d_b <= 0.5) # if this fails we demanded too much current
            # print('I0, d_b: ', d_b)
        else:
            # standard case as considered by krismer
            phi = np.pi * np.sqrt((v_B - v_A) * p/np.pi / (2 * v_A**2 * v_B))
            d_a = phi / np.pi * v_B / (v_B - v_A)
            d_b = phi / np.pi * v_A / (v_B - v_A)
            phi = phi / (2*np.pi)   # we use 0..1 for 0..2pi
            # print('TCM, phi: ',phi)
    else:
        # try OTM, equations are copy/paste from the paper mentioned above
        e1 = -(2*v_A**2 + v_B**2)/(v_A**2 + v_B**2)
        e2 = (v_A**3*v_B + p/np.pi * (v_A**2 + v_B**2)) / (v_A**3 * v_B + v_A * v_B**3)
        e3 = ((8 * v_A**7 * v_B**5)
                - (64 * (p/np.pi)**3 * (v_A**2 + v_B**2)**3)
                - (p/np.pi * v_A**4 * v_B**2 * (4*v_A**2 + v_B**2) * (4*v_A**2 + 13*v_B**2))
                + (16 * (p/np.pi)**2 * v_A * (v_A**2 + v_B**2)**2 * (4*v_A**2*v_B + v_B**3)) )
        e4 = ((8 * v_A**9 * v_B**3)
                - ( 8 * (p/np.pi)**3 * (8*v_A**2 - v_B**2) * (v_A**2 + v_B**2)**2 )
                - (12 * p/np.pi * v_A**6 * v_B**2 * (4*v_A**2 + v_B**2) )
                + ( 3 * (p/np.pi)**2 * v_A**3 * v_B * (4*v_A**2 + v_B**2) * (8*v_A**2 + 5*v_B**2) )
                + ((3*p/np.pi)**1.5 * v_A * v_B**2 * np.sqrt(e3)) )
        e5 = ((2 * v_A**6 * v_B**2
               + (2 * p/np.pi * (4*v_A**2 + v_B**2) * (p/np.pi * (v_A**2 + v_B**2) - v_A**3 * v_B) )) /
              (3 * v_A * v_B * (v_A**2 + v_B**2) * e4**(1/3.0)) )
        e6 = ((4 * (v_A**3*v_B**2 + 2*v_A**5) + 4 * p/np.pi * (v_A**2*v_B + v_B**3))
              / (v_A * (v_A**2 + v_B**2)**2))
        e7 = ( (e4**(1/3.0) / (6 * v_A**3 * v_B + 6 * v_A * v_B**3) )
            + (e1**2 / 4) - (2*e2 / 3) + e5)
        e8 = 0.25 * ( (-e1**3 - e6)/np.sqrt(e7) + 3*e1**2 - 8*e2 - 4*e7 )

        d_a = 0.5
        d_b = 0.25 * (2*np.sqrt(e7) - 2*np.sqrt(e8) - e1)

        if (d_b <= 0.5):
            # print('OTM, d_b: ', d_b)
            # unlike krismer's our phi is 0..1 for 0..360deg, he uses 0..2pi
            phi = 0.5 * (0.5 - np.sqrt(d_b*(1-d_b) - p/(np.pi*v_A*v_B) ))
            # print('OTM, phi: ', phi)
        else:
            # OTM did not yield a valid solution, so use phase shift modulation
            d_a = 0.5
            d_b = 0.5
            phi = 0.5 * (0.5 - np.sqrt(0.25 - p/(np.pi*v_A*v_B)))
            # print('CPM, phi: ', phi)

    # now transform the duty cycles and phase shifts back to our switching times
    if u_pn < u_ac:
        t_opt = [0.5 - d_b, 0.5 - d_b, 0, 0]  # by def. u1 and u2 switch at the same time
        t_opt[3] = -0.5 * (2 * phi - t_opt[0] - d_a + 0.5)
        t_opt[2] = -d_a + 0.5 + t_opt[3]
    else :
        t_opt = [0.5 - d_a, 0.5 - d_a, 0, 0]  # by def. u1 and u2 switch at the same time
        t_opt[3] = -0.5 * (2 * phi - t_opt[0] - d_b + 0.5)
        t_opt[2] = -d_b + 0.5 + t_opt[3]

    return t_opt


#
# i_dc_ref
def calc_t_opt(u, i_dc_ref, i_dc_nom, t0, do_print=True):
    """Calculate optimal (min. rms current) switching times t_opt for given operating conditions

    :param u: AC and DC voltages
    :param i_dc_ref: requested dc output current, f*L = 1 is assumed
    :param i_dc_nom: normalization for i_dc_ref (to improve convergence of numerical solver)
    :param t0: initial conditions for numerical solver
    :param do_print: set to true for debug output
    :return: [t_opt, mode]: t_opt - array with rel switching times, mode: error code (0 = success)
    """
    if i_dc_ref < 0.001:
        # 0 output current required -> trivial solution is to produce no transformer voltages
        t_opt = [0.5, 0.5, 0.5, 0]
        return [t_opt, 0]

    if u[2] <= 0.01:
        # no ouput voltage, trivial solution: use max duty cycle (0.5) and phase shift (0.25) for secondary
        # and the same duty cycles d_1 and d_2 on the primary, which will lead to mains input currents
        # with 0 amplitude
        d_1 = 0.5 - np.sqrt(0.25 - 2 * i_dc_ref) # select duty cycle to create correct transformer current
        d_2 = d_1  # switch both u_ab and u_bc at the same time, this leads to 0 power transfer between them
        d_dc = 0.5
        shift = 0.25
        t_opt = solver_to_sw_times([d_1, d_2, d_dc, shift])
        return [t_opt, 0]

    if np.abs(u[0] - u[1]) < 1e-6:
        # u_ab and u_bc are equal, i.e. we can use the analytic solution for a conventional DAB
        t_opt = krismer(u, i_dc_ref)
        return [t_opt, 0]

    # if possible, try to use DCM
    t_opt = check_dcm(u, i_dc_ref, do_print)
    if t_opt is not None:
        return [t_opt, 0]

    # i_dc_ref is too high for DCM, so use the numeric optimizer for CCM
    return calc_ccm(u, i_dc_ref, i_dc_nom, t0, do_print)


def calc_table(resolution, i_dc_max, u_pn_max, lut_fn, log_fn=None):
    """ Calculate 3D lookup table (LUT)
    @params:
        resolution   - Required : Number of sampling points in each dimension (int)
        i_dc_max     - Required : Highest normalized dc current in final LUT (float)
        u_pn_max     - Required : Highest normalized output voltage in final LUT (float)
        lut_fn       - Required : File name were LUT will be stored
        log_fn       - Optional : Log file name, stdout if this is None
    """
    grid_res = [resolution, resolution, resolution]

    if log_fn is not None:
        log_file = open(log_fn, mode='w')
    else:
        log_file = sys.stderr

    i_dc_range = np.linspace(0, i_dc_max, num=grid_res[0])
    u_pn_range = np.linspace(0, u_pn_max, num=grid_res[1])
    u_bc_range = np.linspace(0, 0.5, num=grid_res[2])

    opt_mode = np.zeros(grid_res)   # optimizer return code (error code, 0 means success)
    grid_res.append(4)
    sw_times = np.zeros(grid_res)
    n_not_solved = 0

    log_file.write('resolution: {}\n'.format(resolution))
    log_file.write('i_dc_max: {}\n'.format(i_dc_max))
    log_file.write('u_pn_max: {}\n'.format(u_pn_max))

    time.clock()
    total_pts = len(i_dc_range) * len(u_pn_range) * len(u_bc_range)
    pts_done = 0

    # sweep the 3D grid, u_bc must be the inner most loop for convergence reasons
    for (k1, i_dc) in enumerate(i_dc_range):
        log_file.write('---------------------\n')
        for (k2, u_pn) in enumerate(u_pn_range):
            log_file.write('--------\n')
            log_file.write('k1={0:}  k2={1:}\n'.format(k1,k2))

            last_t_opt = []

            # traverse starting with u2=05 for which we operate like a conventional DAB were we have a closed
            # analytic solution. This is then used as starting point for the next point
            for (k3, u_bc) in reversed(list(enumerate(u_bc_range))):
                u_ac = 1 # this is our normalization ref voltage
                u_ab = u_ac - u_bc
                u = [u_ab, u_bc, u_pn]
                log_file.write('u={0:}  i_dc={1:.7f}\n'.format(u, i_dc))

                t_opt, m = calc_t_opt(u, i_dc, i_dc, last_t_opt, do_print=False)

                if m == 0:
                    # double check the validity of the obtained solution
                    m = check_solution(u, t_opt, i_dc)

                opt_mode[k1, k2, k3] = m
                sw_times[k1, k2, k3, 0:4] = t_opt

                if m != 0:
                    n_not_solved += 1
                    log_file.write('^ not solved\n')
                    # mark point in table so the user can investigate the problem
                else :
                    last_t_opt = t_opt  # keep a copy of our initial conditions
                # show a progress bar in the terminal
                pts_done = pts_done + 1
                suffix = 'elapsed: {}s'.format(int(time.clock()))
                print_progress(pts_done, total_pts, prefix='Progress', suffix=suffix, decimals=1, bar_length=80)

    log_file.write('\nnumber of points not solved: {}\n'.format(n_not_solved))
    if log_fn is not None:
        log_file.close()
        sys.stderr.write('\nnumber of points not solved: {}\n'.format(n_not_solved))
    # write LUT data to file
    export_csv(lut_fn, grid_res, i_dc_range, u_pn_range, u_bc_range, sw_times)


# Snippet taken from: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    # send output to stderr instead of stdout as stdout is can be used as log file
    #sys.stderr.write('\x1b[2K')    # should clear the last display line but does not work for some reason
    sys.stderr.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LUT calculation for IMDAB3R converters')
    parser.add_argument('-o', '--output', type=str, help='output file name', default='recent.csv')
    parser.add_argument('-l', '--log', type=str, help='log file name, goes to stdout if no file is given')
    parser.add_argument('-n', type=int, help='LUT resolution (no of sampling points per dimension).', default=30)
    parser.add_argument('-i-dc', type=float, help='Max. normalized output current', default=0.07)
    parser.add_argument('-u-pn', type=float, help='Max. normalized output voltage w.r.t. primary', default=1.33)

    args = parser.parse_args()

    resolution = int(args.n)
    i_dc_max = args.i_dc
    u_pn_max = args.u_pn
    lut_fn = args.output
    log_fn = args.log

    if i_dc_max > 0.12:
        print('i_dc values above 0.12 are not feasible, limiting range of LUT')
        i_dc_max = 0.12

    calc_table(resolution, i_dc_max, u_pn_max, lut_fn, log_fn)


