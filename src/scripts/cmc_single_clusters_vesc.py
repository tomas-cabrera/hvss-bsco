import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import numpy as np
import pandas as pd
import paths
import rustics
import scipy.interpolate
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

################################################################################

# plt.style.use(rustics.PATH_TO_MPLRC)

# Get cluster names
cmc_cluster_list = list(rustics.SAMPLE_MODELS.keys())
mosaic = np.reshape(cmc_cluster_list, (4, 1))
# Redefine for single-cluster plots
cmc_cluster_list = cmc_cluster_list[:1]
mosaic = np.array([[c + "_standard", c + "_flattened"] for c in cmc_cluster_list]).T
figwidth = rustics.textwidth
figheight = rustics.textwidth / 4 * 1.75
fig, axd = plt.subplot_mosaic(
    mosaic,
    figsize=(mosaic.shape[1] * figwidth, mosaic.shape[0] * figheight),
    sharex=True,
    # sharey=True,
    gridspec_kw={
        "hspace": 0,
        "wspace": 0,
    },
)
ms = 0.25
kw_scatter = {
    "linestyle": "None",
    # label=ti_row.label,
    "marker": "+",
    "markersize": 0.25,  # irrelevant for marker="," (pixels)
    "alpha": ms,
    "rasterized": True,
}
kw_vesccore = {
    "color": "xkcd:dark blue",
    "lw": 1,
    "rasterized": True,
}
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
    df_esc = pd.read_csv(
        paths.data_cmc / cmc_cluster / "initial.esc.dat",
        names=["t", "phi_rtidal", "phi_zero"],
        usecols=[1, 9, 10],  # 1 for t, 9 for phi_rtidal, 10 for phi_zero
        skiprows=1,
        delim_whitespace=True,
    )

    # Load intial.conv.sh (for converting TotalTime to Myr and core vesc to km/s)
    # Copied from cmctools/cmctoolkit.py
    conv_path = paths.data_cmc / cmc_cluster / "initial.conv.sh"
    f = open(conv_path, "r")
    convfile = f.read().split("\n")
    f.close()
    lengthunitcgs = float(convfile[13][14:])
    timeunitsmyr = float(convfile[19][13:])
    nbtimeunitcgs = float(convfile[21][14:])
    nb_kms = 1e-5 * lengthunitcgs / nbtimeunitcgs
    df_esc["t"] *= timeunitsmyr
    df_esc["phi_rtidal"] *= nb_kms**2
    df_esc["phi_zero"] *= nb_kms**2

    # Generate ordered list of type_i's, descending by number of occurences
    type_is = ejdf.df.value_counts("type_i").index
    type_is = rustics.INFO_TYPES_I.iloc[type_is]

    # Select x-/y-axis labels; stars only (k<10)
    x = "time"
    y = "vesc"
    kgroup = rustics.INFO_BSE_K_GROUPS.iloc[0]

    ##############################
    ###     Standard plot      ###
    ##############################

    # Set up axes
    ax = axd[cmc_cluster + "_standard"]
    ax.set_xscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])
    # ax.set_yscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])

    scatter_artists = []
    for ti, ti_row in type_is.iterrows():
        filtered = ejdf.df[
            (ejdf.df.kf >= kgroup.lo)
            & (ejdf.df.kf < kgroup.hi)
            & (ejdf.df.type_i == ti)
        ]
        (s,) = ax.plot(
            filtered[x] / 1000,
            filtered[y],
            color=ti_row.color,
            **kw_scatter,
        )
        scatter_artists.append(s)

    # Get xlim
    if ci == 0:
        xlims = ax.get_xlim()

    # Add vesc_core
    tsample = [
        int(i)
        for i in np.logspace(
            0,
            np.log10(df_esc.shape[0] - 1),
            1000,
        )
    ]
    tsample = list(set(tsample))
    tsample.sort()
    (s_center,) = ax.plot(
        df_esc.loc[tsample, "t"] / 1000.0,
        (2 * (df_esc.loc[tsample, "phi_rtidal"] - df_esc.loc[tsample, "phi_zero"]))
        ** 0.5,
        **kw_vesccore,
    )

    # Create legend
    ax.legend(
        # handles=[tuple(scatter_artists), s_center],
        handles=[
            Line2D(
                (0, 1),
                (0, 0),
                color="k",
                linestyle="None",
                markersize=0.25,
                marker="+",
            ),
            s_center,
        ],
        labels=[r"Stellar ejections", "Core"],
        markerscale=8.0 / ms,
    )

    # Add cluster label
    ax.annotate(
        rustics.SAMPLE_MODELS[cmc_cluster], (0.025, 0.1), xycoords="axes fraction"
    )

    # Axes limits
    ax.set_xlim(xlims)

    # # yticks
    # ylim = ax.get_ylim()
    # yts = []
    # ytls = []
    # level = 1
    # while 10**level < ylim[1] or len(yts) == 0:
    #     locs = list(
    #         np.arange(
    #             2 * 10**level,
    #             min(1.1 * 10 ** (level + 1), ylim[1]),
    #             2 * 10**level,
    #         )
    #     )
    #     yts += locs
    #     ytls += ["%d" % l for l in locs]
    #     level += 1
    # yts = ax.set_yticks(yts)
    # ax.set_yticklabels(ytls)

    # Axes labels
    ax.set_ylabel(rustics.HEADERS_TO_LABELS[y])

    ##############################
    ###     Flattened plot     ###
    ##############################

    # Set up axes
    ax = axd[cmc_cluster + "_flattened"]
    ax.set_xscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])
    # ax.set_yscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])

    # Generate interpolation
    y_interp = scipy.interpolate.interp1d(
        df_esc["t"],
        (2 * (df_esc["phi_rtidal"] - df_esc["phi_zero"])) ** 0.5,
    )

    for ti, ti_row in type_is.iterrows():
        filtered = ejdf.df[
            (ejdf.df.kf >= kgroup.lo)
            & (ejdf.df.kf < kgroup.hi)
            & (ejdf.df.type_i == ti)
        ]
        print("\tfiltered.shape: ", filtered.shape)
        fynorm = filtered[y] / y_interp(filtered[x].to_numpy())
        ax.plot(
            filtered[x] / 1000,
            # filtered[y],
            fynorm,
            color=ti_row.color,
            **kw_scatter,
        )

    # Get xlim
    if ci == 0:
        xlims = ax.get_xlim()

    # Add vesc_core
    ax.axhline(
        1,
        xmin=0.0375,
        xmax=0.9625,
        **kw_vesccore,
    )

    # # Add cluster label
    # ax.annotate(
    #     rustics.SAMPLE_MODELS[cmc_cluster], (0.025, 0.1), xycoords="axes fraction"
    # )

    # Axes limits
    ax.set_xlim(xlims)

    # # yticks
    # ylim = ax.get_ylim()
    # yts = []
    # ytls = []
    # level = 1
    # while 10**level < ylim[1] or len(yts) == 0:
    #     locs = list(
    #         np.arange(
    #             2 * 10**level,
    #             min(1.1 * 10 ** (level + 1), ylim[1]),
    #             2 * 10**level,
    #         )
    #     )
    #     yts += locs
    #     ytls += ["%d" % l for l in locs]
    #     level += 1
    # yts = ax.set_yticks(yts)
    # ax.set_yticklabels(ytls)

    # Axes labels
    if cmc_cluster == cmc_cluster_list[-1]:
        ax.set_xlabel(rustics.HEADERS_TO_LABELS[x])
    ax.set_ylabel(r"$v_{\rm esc} / v_{\rm esc, c}$")
# Adjust spacing and save
plt.tight_layout()
plt.savefig(paths.figures / "cmc_single_clusters_vesc.pdf")
plt.close()
