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


def gen_fb_input(
    path_to_cluster, N, seed_max, cmc_input_filename="full_initial.encounters.dat"
):
    """
    Generates data for high-velocity star analysis with fewbody.
        Arguments:
            path_to_cluster = path specifying one CMC cluster
            N = number of realizations per scattering
            seed_max = maximum seed for fb_randorient
            cmc_input_filename = file in zipped cluster to use
        Returns:
            fb_input = array of N*N_scatterings inputs for fewbody
    """

    # load cmc scatterings
    scats = np.loadtxt(path_to_cluster + "/" + cmc_input_filename)

    # check if there are no encounters
    if not scats.any():
        return scats

    # filter out any with e1>=1.0
    scats = scats[scats[:, 4] < 1.0]

    # initialize fb_input array by repeating scattering array N times
    fb_input = np.zeros((N * scats.shape[0], scats.shape[1] + 1))
    fb_input[:, :-1] = np.repeat(scats, N, axis=0)

    # fill in last column with seeds
    fb_input[:, -1] = seed_max * np.random.rand(fb_input.shape[0])

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
N = 10

# Get cluster list
cmc_cluster_list = hvss_utils.CMC_CLUSTER_LIST_V2

# Make fewbody
subprocess.run("make -C {}".format(hvss_utils.FB_PATH), shell=True)

# Realize cluster encounters
print("Starting iteration...")
for cmc_cluster in tqdm(cmc_cluster_list):
    print("\n{}, N={}".format(cmc_cluster, N))

    # Particular shorthands
    data_file_name = (
        hvss_utils.DATA_PATH
        + cmc_cluster
        + "/{}_N-{}.txt".format(hvss_utils.FB_OUTPUT_FILENAME, N)
    )
    os.makedirs(hvss_utils.DATA_PATH + "/" + cmc_cluster, exist_ok=True)

    # Generate list of inputs
    print("\tGenerating fb_input...")
    try:
        fb_input = gen_fb_input(
            hvss_utils.CMC_PATH + cmc_cluster.replace("rv", "v2_rv") + ".tar.gz",
            N,
            seed_max,
        )
    except Exception as e:
        print("\tError in getting encounter information:")
        print("\t\t", e)
        print("\tContinuing...")
        continue
    print("\tfb_input: {}".format(fb_input))
    if not fb_input.any():
        print(
            "\tNo encounters found for {}, N={}; continuing...".format(cmc_cluster, N)
        )
        continue

    print("\tInput generated, shape={}".format(fb_input.shape))
    print("\tIntegrating...")

    try:
        if nprocs == 1:
            parmap.starmap(run_fb, fb_input, pm_parallel=False)
        else:
            p = mp.Pool(nprocs)
            dat = parmap.starmap(run_fb, fb_input, pm_pool=p)
    except Exception as e:
        print("\tError in integrating encounters:")
        print("\t\t", e)
        print("\tContinuing...")
        continue

    pd.DataFrame(dat).to_csv(data_file_name, header=False, index=False)

    print("\tdone")

print("done")
