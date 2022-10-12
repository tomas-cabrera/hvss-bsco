import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as mplticker

import rustics
import paths

################################################################################

#plt.style.use(rustics.PATH_TO_MPLRC)

# Get cluster names
cmc_cluster_list = [
    "N8e5_rv0.5_rg8_Z0.0002",
    "N4e5_rv0.5_rg8_Z0.0002",
    "N8e5_rv2_rg8_Z0.0002",
    "N8e5_rv0.5_rg8_Z0.02",
]

mosaic = np.reshape(cmc_cluster_list, (4,1))
figheight = 1.5 
figwidth = 4
fig, axd = plt.subplot_mosaic(
    mosaic,
    figsize=(mosaic.shape[1]*figwidth, mosaic.shape[0]*figheight),
    sharex=True,
    gridspec_kw={
        "hspace": 0,
        "wspace": 0,
    },
)
for ci, cmc_cluster in enumerate(cmc_cluster_list):
    print("{}, N_REALZ={}".format(cmc_cluster, rustics.N_REALZ))

    # File names and directory creation
    path_to_ejections = paths.data_cmc / cmc_cluster / rustics.FILE_EJECTIONS

    # Read saved data file
    ejdf = rustics.EjectionDf(path_to_ejections)
    ejdf.df = ejdf.df

    # Convert to physical units
    ejdf.convert_from_fewbody()

    # Filter out mergers
    ejdf.df = ejdf.df[ejdf.df["type_f"] > 0]

    # Load times and core radii from initial.dyn.dat
    df_dyn = pd.read_csv(
        paths.data_cmc / cmc_cluster / "initial.dyn.dat",
        # names=["t", "rc_nb"],
        # usecols=[0, 24],  # 0 for t, 24 for rc_nb
        names=["t", "Dt", "rho_0", "rc_nb"],
        usecols=[0, 1, 21, 24],  # 0 for t, 1 for Dt, 24 for rho_0
        skiprows=2,
        delim_whitespace=True,
    )

    # Load intial.conv.sh (for converting TotalTime to Myr and core radius to pcs)
    # Copied from cmctools/cmctoolkit.py
    conv_path = paths.data_cmc / cmc_cluster / "initial.conv.sh"
    f = open(conv_path, "r")
    convfile = f.read().split("\n")
    f.close()
    timeunitsmyr = float(convfile[19][13:])
    lengthunitparsec = float(convfile[15][17:])
    df_dyn["t"] *= timeunitsmyr
    df_dyn["rc_nb"] *= lengthunitparsec

    # Generate ordered list of type_i's, descending by number of occurences
    # type_is = [x[0] for x in ejdf.df.value_counts("type_i").index]
    type_is = ejdf.df.value_counts("type_i").index
    type_is = rustics.INFO_TYPES_I.iloc[type_is]

    # Some parameters
    ms = 0.25
    minimum = 50

    # Select x-/y-axis labels; stars only (k<10)
    x = "time"
    y = "vesc"
    kgroup = rustics.INFO_BSE_K_GROUPS.iloc[0]

    # Set up axes
    ax = axd[cmc_cluster]
    ax.set_xscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_yscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])

    for ti, ti_row in type_is.iterrows():
        # for ti, ti_row in rustics.INFO_TYPES_I.iterrows():
        print(ti, ti_row)
        filtered = ejdf.df[
            (ejdf.df.kf >= kgroup.lo)
            & (ejdf.df.kf < kgroup.hi)
            & (ejdf.df.type_i == ti)
        ]
        print("\tfiltered.shape: ", filtered.shape)
        ax.plot(
            filtered[x] / 1000,
            filtered[y],
            color=ti_row.color,
            linestyle="None",
            label=ti_row.label,
            marker="+",
            markersize=ms,  # irrelevant for marker="," (pixels)
            alpha=0.25,
            rasterized=True,
        )

    # Add rho_0 axis
    colorr = "xkcd:crimson"
    axr = ax.twinx()
    axr.set_xlim(ax.get_xlim())
    axr.set_yscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])
    tsample = [
        int(i)
        for i in np.logspace(
            0,
            np.log10(df_dyn.shape[0] - 1),
            1000,
        )
    ]
    tsample = list(set(tsample))
    tsample.sort()
    axr.plot(
        df_dyn.loc[tsample, "t"] / 1000.0,
        df_dyn.loc[tsample, "rho_0"] / df_dyn.loc[0, "rho_0"],
        color=colorr,
        alpha=0.7,
        lw=0.5,
    )
    axr.tick_params(
        axis="y",
        colors=colorr,
    )
    # axr.set_ylim(
    #    (0.0011, axr.get_ylim()[1])
    # )  # hacky way of getting 0.0 label off the adjacent axes

    # Expand ylim if not large enough
    ylim = ax.get_ylim()
    if np.floor(np.log10(ylim[1]))==np.floor(np.log10(ylim[0])):
        ax.set_ylim((0.5*ylim[0], 2*ylim[1]))

    # Add cluster label
    axr.annotate(cmc_cluster, (0.025,0.0525), xycoords="axes fraction")

    # Add minor ticks
    xlim = ax.get_xlim()
    numticks = np.ceil(np.log10(xlim[1] / xlim[0])) + 2
    locmin = mplticker.LogLocator(base=10, subs=np.arange(0, 1, 0.1), numticks=numticks)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mplticker.NullFormatter())
    ylim = axr.get_ylim()
    numticks = np.ceil(np.log10(ylim[1] / ylim[0])) + 2
    locmin = mplticker.LogLocator(base=10, subs=np.arange(0, 1, 0.1), numticks=numticks)
    axr.yaxis.set_minor_locator(locmin)
    axr.yaxis.set_minor_formatter(mplticker.NullFormatter())

# Add superlabels
fig.supxlabel(rustics.HEADERS_TO_LABELS[x], x=0.55, y=0.05, ha="center")
fig.supylabel(rustics.HEADERS_TO_LABELS[y], x=0.05, y=0.55, va="center")
fig.text(0.95, 0.55, r"$\rho_{c} / \rho_{c,0}$", va="center", color=colorr, rotation="vertical")

# Adjust spacing
plt.tight_layout()

# hvss_utils.grid_selfmade(ax)
plt.savefig(paths.figures / "cmc_single_clusters_vesc.pdf")
plt.close()
