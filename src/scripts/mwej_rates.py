import numpy as np
from numba import njit
import pandas as pd
import multiprocessing as mp
import parmap
import matplotlib.pyplot as plt

import rustics
import paths

###############################################################################

# Tested this code with and without njit, and with and without multiprocessing
# Only ran once each so no averaging, but there's enough to see nonetheless
# window_step = 1, window_width = 10
#      nprocs     |   1   |   4
# ---------------------------------
#  time w/o @njit |  85s  |  37s
#  time w/ @njit  |  50s  |  24s

plt.style.use("./src/scripts/matplotlibrc")

###############################################################################


@njit
def _rolling_window_count(data, window_centers, window_width):
    """! A function to count the number of instances in a rolling window.
    Should all be numpy to take full advantage of numba.njit.
    Atm. automatically trims the domain s.t. only full windows are included.

    @param  data                Data to count, as a 1D numpy array
    @param  window_centers      Centers of windows
    @param  window_width        Width of rolling window
    """
    # Loop through windows
    window_counts = []
    for wc in window_centers:
        wmin = wc - window_width / 2.0
        wmax = wc + window_width / 2.0
        mask = (data >= wmin) & (data < wmax)
        window_counts.append(mask.sum())
    return window_counts


def calc_rolling_rate(
    mwrow, window_centers, window_width=10, vlim=(0.0, np.inf), stellartype=None
):
    """! Function for parallelizing rolling rate calculation.

    @param  mwrow               mwdf row for a certain GC
    @param  vlim                Limits of velocities to include
    @param  window_step         Step size of rolling window, in Myr
    @param  window_width        Width of rolling window, in Myr
    """

    # Load CMC ejections
    gcdf = pd.read_csv(
        paths.data_mwgcs / mwrow.Cluster / "output_N-10_ejections.txt",
        usecols=["time", "kf", "U", "V", "W"],
    )

    # Filter -1's (post-present-day ejections)
    if stellartype == None or stellartype == "S":
        mask = (gcdf.U != -1) & (gcdf.V != -1) & (gcdf.W != -1) & (gcdf.kf < 10)
    elif stellartype == "WD":
        mask = (
            (gcdf.U != -1)
            & (gcdf.V != -1)
            & (gcdf.W != -1)
            & (gcdf.kf >= 10)
            & (gcdf.kf < 13)
        )  # white dwarfs
    elif stellartype == "NS":
        mask = (
            (gcdf.U != -1) & (gcdf.V != -1) & (gcdf.W != -1) & (gcdf.kf == 13)
        )  # neutron stars
    elif stellartype == "BH":
        mask = (
            (gcdf.U != -1) & (gcdf.V != -1) & (gcdf.W != -1) & (gcdf.kf == 14)
        )  # black holes
    gcdf = gcdf[mask]

    # Calculate galactocentric velocity & apply vlim
    gcdf["vgc"] = (gcdf.U**2 + gcdf.V**2 + gcdf.W**2) ** 0.5
    mask = (gcdf.vgc >= vlim[0]) & (gcdf.vgc < vlim[1])
    gcdf = gcdf[mask]

    # Calculate rolling rate
    # Note: gcdf.time - mwrow.t sets the time 0 to the present day
    times = gcdf.time.to_numpy() - mwrow.t
    return times, _rolling_window_count(times, window_centers, window_width)


###############################################################################

# Settings
nprocs = None

# Load MW catalog
mwdf = pd.read_csv(paths.data / "mwgcs-cmc.dat")
# print(mwdf.columns)
print("mwdf.shape:", mwdf.shape)

# Initialize figure
mosaic = [["S"], ["WD"]]
mosaic = [["S"]]
fscale = 2
fig, axd = plt.subplot_mosaic(
    mosaic, figsize=(fscale * len(mosaic[0]) * 1.618, fscale * len(mosaic) * 1.25)
)

for st in np.array(mosaic).flatten():

    # Parallel rolling rate calculation
    window_width = 200
    window_step = 20
    tlim = (-14000.0, 0.0)
    # Set the minimum and maximum window centers to the values where their window entirely lies in the data domain
    tmin = tlim[0] + window_width / 2.0
    tmax = tlim[1] - window_width / 2.0
    # Generate the window centers
    window_centers = np.arange(tmin, tmax, window_step)
    # Calculate rate for all [stars]
    if nprocs == 1:
        # Don't run in parallel
        mp_input = [x for _, x in mwdf.iterrows()]
        output = parmap.map(
            calc_rolling_rate,
            mp_input,
            window_centers,
            window_width=window_width,
            stellartype=st,
            pm_parallel=False,
            pm_pbar=True,
        )
    else:
        # Run in parallel
        mp_input = [x for _, x in mwdf.iterrows()]
        pool = mp.Pool(nprocs)
        output = parmap.map(
            calc_rolling_rate,
            mp_input,
            window_centers,
            window_width=window_width,
            stellartype=st,
            pm_pool=pool,
            pm_pbar=True,
        )
    # Extract outputs
    times = np.concatenate([o[0] for o in output])
    full_counts = np.array([o[1] for o in output]).sum(axis=0)

    ##############################
    ###  Calc. ej. quantiles   ###
    ##############################

    # Cut out "future" ejections, and move present day from 0 to 14000
    times /= 1000
    times_forward = times[times < 0]
    times_forward += 14000

    # Calculate quantiles
    quants5090 = np.quantile(times, [0.5, 0.9])
    quants5090_forward = np.quantile(times_forward, [0.5, 0.9])
    print("quants5090 (Myr):", quants5090)

    # Print to output file
    with open(
        __file__.replace("scripts", "tex/output").replace(".py", ".txt"), "w"
    ) as f:
        f.write("%.2f(%.2f) Gyr" % (14 + quants5090[0], 14 + quants5090[1]))

    ##############################
    ###          Plot          ###
    ##############################

    # Calculate rate for HV[Ss]
    if nprocs == 1:
        # Don't run in parallel
        mp_input = [x for _, x in mwdf.iterrows()]
        output = parmap.map(
            calc_rolling_rate,
            mp_input,
            window_centers,
            window_width=window_width,
            stellartype=st,
            vlim=(500, np.inf),
            pm_parallel=False,
            pm_pbar=True,
        )
    else:
        # Run in parallel
        mp_input = [x for _, x in mwdf.iterrows()]
        pool = mp.Pool(nprocs)
        output = parmap.map(
            calc_rolling_rate,
            mp_input,
            window_centers,
            window_width=window_width,
            stellartype=st,
            vlim=(500, np.inf),
            pm_pool=pool,
            pm_pbar=True,
        )
    hvs_counts = np.array([o[1] for o in output]).sum(axis=0)
    # # Calc WD rate, if one plot
    # if np.array(mosaic).flatten().shape[0] == 1:
    #     if nprocs == 1:
    #         # Don't run in parallel
    #         mp_input = [x for _, x in mwdf.iterrows()]
    #         output = parmap.map(
    #             calc_rolling_rate,
    #             mp_input,
    #             window_centers,
    #             window_width=window_width,
    #             stellartype="WD",
    #             pm_parallel=False,
    #             pm_pbar=True,
    #         )
    #     else:
    #         # Run in parallel
    #         mp_input = [x for _, x in mwdf.iterrows()]
    #         pool = mp.Pool(nprocs)
    #         output = parmap.map(
    #             calc_rolling_rate,
    #             mp_input,
    #             window_centers,
    #             window_width=window_width,
    #             stellartype="WD",
    #             pm_pool=pool,
    #             pm_pbar=True,
    #         )
    #     wd_counts = np.array([o[1] for o in output]).sum(axis=0)
    #     if nprocs == 1:
    #         # Don't run in parallel
    #         mp_input = [x for _, x in mwdf.iterrows()]
    #         output = parmap.map(
    #             calc_rolling_rate,
    #             mp_input,
    #             window_centers,
    #             window_width=window_width,
    #             stellartype="WD",
    #             vlim=(500, np.inf),
    #             pm_parallel=False,
    #             pm_pbar=True,
    #         )
    #     else:
    #         # Run in parallel
    #         mp_input = [x for _, x in mwdf.iterrows()]
    #         pool = mp.Pool(nprocs)
    #         output = parmap.map(
    #             calc_rolling_rate,
    #             mp_input,
    #             window_centers,
    #             window_width=window_width,
    #             stellartype="WD",
    #             vlim=(500, np.inf),
    #             pm_pool=pool,
    #             pm_pbar=True,
    #         )
    #     hvwd_counts = np.array([o[1] for o in output]).sum(axis=0)

    # Plot
    ax = axd[st]
    ax.plot(
        14 + (window_centers / 1000.0),
        full_counts / window_width / 1e6,
        color="xkcd:azure",
        label="All stars",
        rasterized=True,
    )
    # if np.array(mosaic).flatten().shape[0] == 1:
    #     ax.plot(
    #         window_centers / 1000.0,
    #         wd_counts / window_width / 1e6,
    #         color="xkcd:goldenrod",
    #         label="All WDs",
    #     )
    #     ax.plot(
    #         window_centers / 1000.0,
    #         hvwd_counts / window_width / 1e6,
    #         color="xkcd:violet",
    #         label=r"WDs, $v > 500~{\rm km~s^{-1}}$",
    #     )
    ax.plot(
        14 + (window_centers / 1000.0),
        hvs_counts / window_width / 1e6,
        color="xkcd:orangered",
        label=r"Stars, $v > 500~{\rm km~s^{-1}}$",
        rasterized=True,
    )
    if st == "S":
        ax.axhline(2e-3, c="xkcd:azure", ls="--", alpha=0.7, rasterized=True)
        ax.axhline(1e-4, c="xkcd:orangered", ls="--", alpha=0.7, rasterized=True)
        ax.annotate(
            "Disc runaways",
            xy=[14, 1.5e-3],
            color="xkcd:azure",
            ha="right",
            va="top",
            rasterized=True,
        )
        ax.annotate(
            "Galactic center HVSs",
            xy=[14, 1.25e-4],
            color="xkcd:orangered",
            ha="right",
            va="bottom",
            rasterized=True,
        )

    # Plot 50 and 90% ejection thresholds
    qheight = 5e-9
    ax.vlines(
        14 + quants5090,
        ymin=0,
        ymax=qheight,
        color="xkcd:azure",
    )
    for s, q in zip(["50", "90"], 14 + quants5090):
        ax.annotate(
            r"%s\%%$\textbf{-}$" % s,
            (q, qheight),
            color="xkcd:azure",
            ha="right",
            va="center",
        )

    # Axes
    ax.set_xlabel(rustics.HEADERS_TO_LABELS["time"])
    ax.set_ylabel(r"Rate [yr$^{-1}$]")
    ax.set_ylim((10**-8.9, 10**-2.5))
    ax.set_yscale("log")
    ax.legend(loc="center right")

# Make neat and save
plt.tight_layout()
plt.savefig(paths.figures / __file__.split("/")[-1].replace(".py", ".pdf"))
plt.close()
