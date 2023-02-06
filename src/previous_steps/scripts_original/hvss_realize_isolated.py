import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
import subprocess
import parmap
import multiprocessing as mp
from tqdm import tqdm

import hvss_utils

################################################################################


def bin_log_lin(mn, mx, nb_log, nb_lin):
    """Takes an interval and splits it into layered bins.
    First layer is split logarithmically, second is linearly.
    Arguments:
        mn = lower bound of interval
        mx = upper bound of interval
        nb_log = number of logarithmic bins
        nb_lin = numer of linear bins
    Returns:
        bins = returns list of bin intervals
    """

    # create logarithmic bins
    log_bins = np.logspace(np.log10(mn), np.log10(mx), nb_log)

    # create linear bins
    bins = np.linspace(log_bins[:-1], log_bins[1:], nb_lin)

    # transpose and flatten
    bins = bins.transpose().flatten()

    return bins


def simple_fb_input(N, a, ecc, hb_C, hb_D, seed_max, m_list, v_list):
    """Generates data for high-velocity star analysis with fewbody.
    Arguments:
        N = number of interactions to be integrated
        a = semi-major axis of initial binary (in AU)
        ecc = eccentricity of initial binary
        hb_C = C parameter from Hut & Bahcall 1983
        hb_D = D parameter from Hut & Bahcall 1983
        seed_max = maximum seed for binsingle (loops if bigger)
        mass_list = list of masses
        v_list = list of velocities
    Returns:
        fb_input = array of N*N_v*N_masses inputs for fewbody
    """

    parms_list = [[mass, v] for mass in m_list for v in v_list]
    fb_input_list = []

    for parms in parms_list:

        # generate impact parameter using velocity (in units of semi-major axis)
        b_max = hb_C / parms[1] + hb_D

        parms_input = np.zeros((N, 17))

        parms_input[:, 1] = b_max * np.random.rand(N)
        parms_input[:, 2] = parms[1]
        parms_input[:, 3] = a
        parms_input[:, 4] = ecc
        parms_input[:, 6] = parms[0][1]
        parms_input[:, 7] = parms[0][2]
        parms_input[:, 8] = parms[0][0]
        parms_input[:, 9] = 1
        parms_input[:, 10] = 1
        parms_input[:, 11] = 1
        # and now for some jank linear mappings
        parms_input[:, 12] = 14.0 / 9.0 * (parms[0][1] - 1.0)
        parms_input[:, 13] = 14.0 / 9.0 * (parms[0][2] - 1.0)
        parms_input[:, 14] = 14.0 / 9.0 * (parms[0][0] - 1.0)
        parms_input[:, 15] = 1.0 / 9.0 * (parms[0][0] + 8.0)
        parms_input[:, 16] = np.random.randint(seed_max, size=N)

        fb_input_list.append(parms_input)

    fb_input = np.vstack(fb_input_list)

    return fb_input


def run_fb(
    time,
    b,
    vinf,
    a1,
    e1,
    vesc,
    m10,
    m11,
    m0,
    r10,
    r11,
    r0,
    k10,
    k11,
    k0,
    type_i,
    s,
    print_out=False,
):
    """Generates data for high-velocity star analysis with fewbody.
    Arguments:
        masses = list of masses [m0, m10, m11]
        v = velocity of single object at infinity before collision
        b = impack parameter
        s = seed
        print_out = toggles printing the output string from fewbody
    Returns:
        List of parameters for hvs analysis
    """

    # run fewbody; '2> /dev/null' directs (>) stderr output (2) to the trash vent of linux (/dev/null)
    fb_stdout = subprocess.check_output(
        hvss_utils.FB_PATH
        + hvss_utils.FB_PROCESS
        + " --m0 {}".format(m0)
        + " --m10 {}".format(m10)
        + " --m11 {}".format(m11)
        + " --r0 {}".format(r0)
        + " --r10 {}".format(r10)
        + " --r11 {}".format(r11)
        + " --k0 {}".format(k0)
        + " --k10 {}".format(k10)
        + " --k11 {}".format(k11)
        + " --a1 {}".format(a1)
        + " --e1 {}".format(e1)
        + " --vinf {}".format(vinf)
        + " --b {}".format(b)
        + " --seed {}".format(s)
        + " --tcpustop 60 --PN1 0 --PN2 0 --PN25 0"
        + " 2> /dev/null",
        shell=True,
    )
    out = fb_stdout.decode("utf-8")
    if print_out:
        print(out)
    [
        type_f,
        v_crit,
        a_fin,
        e_fin,
        Lx,
        Ly,
        Lz,
        Lbinx,
        Lbiny,
        Lbinz,
        Ei,
        DeltaEfrac,
        vfin0,
        kf0,
        Rmin0,
        Rmin_j0,
        vfin1,
        kf1,
        Rmin1,
        Rmin_j1,
        vfin2,
        kf2,
        Rmin2,
        Rmin_j2,
    ] = [float(i) for i in out.split()]

    return [
        time,
        b,
        vinf,
        a1,
        e1,
        vesc,
        m10,
        m11,
        m0,
        r10,
        r11,
        r0,
        k10,
        k11,
        k0,
        s,
        type_i,
        type_f,
        v_crit,
        a_fin,
        e_fin,
        Lx,
        Ly,
        Lz,
        Lbinx,
        Lbiny,
        Lbinz,
        Ei,
        DeltaEfrac,
        vfin0,
        kf0,
        Rmin0,
        Rmin_j0,
        vfin1,
        kf1,
        Rmin1,
        Rmin_j1,
        vfin2,
        kf2,
        Rmin2,
        Rmin_j2,
    ]


################################################################################

a = 10.0  # semi-major axis of initial binary (in AU)
ecc = 0.0  # eccentricity of initial binary
hb_C = 4.0  # C parameter from Hut & Bahcall 1983
hb_D = 0.6 * (1 + ecc)  # D parameter from Hut & Bahcall 1983
seed_max = 2 ** 32  # maximum seed for binsingle (loops if bigger)
mass_list = [
    [1.0, 10.0, 10.0],
    [10.0, 1.0, 10.0],
]  # list of mass groupings [m0, m10, m11]
vel_list = [0.5, 5.0]  # list of v_infs ""
seed_max = 2 ** 32  # maximum seed for binsingle (loops if bigger)
nprocs = 128  # number of subprocesses to use
N = 1000000

# make fewbody
subprocess.run("make -C {}".format(hvss_utils.FB_PATH), shell=True)

# code for integrating simple encounters
data_file_name = "{}isolated/{}_N-{}.txt".format(
    hvss_utils.DATA_PATH, hvss_utils.FB_OUTPUT_FILENAME, N
)
os.makedirs("{}isolated".format(hvss_utils.DATA_PATH), exist_ok=True)

# generate list of inputs
fb_input = simple_fb_input(N, a, ecc, hb_C, hb_D, seed_max, mass_list, vel_list)
if nprocs == 1:
    parmap.starmap(run_fb, fb_input, pm_parallel=False, pm_pbar=True)
else:
    p = mp.Pool(nprocs)
    dat = parmap.starmap(run_fb, fb_input, pm_pool=p, pm_pbar=True)
pd.DataFrame(dat).to_csv(data_file_name, header=False, index=False)
