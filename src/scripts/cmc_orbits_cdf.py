import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
from matplotlib.patches import Patch

import rustics
import paths

################################################################################

# Notes on velocity definitions
# Galactic: position from earth, velocity from galactic center
# Galactocentric: position+velocity from galactic center
# Brown+18:
#   Brown+14 gives SDSS Catalog identifiers for stars
#   v_\odot (v_helio in eq. 1?): heliocentric radial velocity
#   v_rf: "Heliocentric radial velocity transformed to the Galactic frame"
# Hattori+18:
#   Gives Gaia source_ids for stars
#   v_los: Gaia radial_velocity = "Spectroscopic radial velocity in Solar system barycentric rest frame"
#       Reminder: "barycentric" is about CoM of system (i.e. ~CoM of Sun+earth), so ICRS is the relevant frame
#   v_tan: Tangetial motion measured from Earth in Galactic rest frame (see eq. 1)
#   v_r: Galactocentric radial velocity (radial velocity from galactic center)
#   v_total: Presumably (v_tan**2 + v_rf(v_los)**2)**0.5, where v_rf is the radial velocity transformed to the Galactic rest frame
# This work:
#   compare Brown v_rf with Hattori v_rf(l, b, v_los) (after Brown+18 eq.1, with additional things for barycentrism) and radial_velocity in Galactic frame in this work

################################################################################

# plt.style.use(rustics.PATH_TO_MPLRC)

# Get model info
path_to_data = paths.data_mwgcs
df_mwgcs = pd.DataFrame(os.listdir(path_to_data), columns=["fname"])

# Iterables
columns = ["vlos", "mf"]
columns = ["vlos"]

# Specify kgroup
kgroup = None
kgroup = rustics.INFO_BSE_K_GROUPS.iloc[0]

# Make subplot mosiac
plot_size = rustics.textwidth
fig, axd = plt.subplot_mosaic(
    [columns],
    sharey=True,
    figsize=(plot_size * len(columns), plot_size * 0.7),
    gridspec_kw={
        "wspace": 0,
        "hspace": 0,
    },
)

# Iterate over columns (each should have an entry in rustics.BINS)
for ci, c in enumerate(columns):
    print("column=%s" % c)

    # Initialize hist. generator
    bins = rustics.BINS_CDF[c]
    if c == "mf":
        # Bin by pre-/post-core collapse for mass histograms
        hg = rustics.HistogramGenerator(
            c, bins, nprocs=None, kgroup=kgroup
        )  # , cc=True)
    else:
        hg = rustics.HistogramGenerator(c, bins, nprocs=None, kgroup=kgroup)

    # Generate histograms
    df_mwgcs["hists"] = hg.generate_catalog_histograms(
        df_mwgcs, pm_pbar=True, path_to_data=path_to_data
    )

    # Choose axis
    ax = axd[c]

    # Plot
    if 0:  # c == "mf":
        for cci, cc_row in rustics.INFO_CC_GROUPS.iterrows():
            # Plot histogram
            weights = df_mwgcs["hists"].sum()[cci]
            ax.hist(
                bins[:-1],
                bins=bins,
                weights=np.cumsum(weights) / np.sum(weights),
                label="CMC",
                histtype="step",
                lw=2,
                ls=cc_row.ls,
                alpha=0.7,
                rasterized=True,
            )
    else:
        # Plot histogram
        weights = df_mwgcs["hists"].sum()
        ax.hist(
            bins[:-1],
            bins=bins,
            weights=np.cumsum(weights) / np.sum(weights),
            label="CMC",
            histtype="step",
            lw=2,
            alpha=0.7,
            rasterized=True,
        )
    if (c == "vout") or (c == "v_radial") or (c == "vlos"):

        # Actually using radial velocities from Brown+18
        ebins = bins
        eweights, _ = np.histogram(
            [
                275.2,
                277.8,
                279.9,
                282.8,
                283.9,
                286.8,
                288.3,
                288.9,
                289.1,
                289.1,
                289.6,
                294.1,
                302.1,
                306.2,
                308.6,
                311.4,
                319.6,
                328.5,
                344.1,
                344.6,
                358.8,
                363.9,
                391.9,
                392.1,
                397.7,
                413.3,
                417.0,
                417.4,
                418.5,
                439.5,
                440.3,
                449.0,
                458.8,
                487.4,
                496.2,
                501.1,
                551.7,
                644.0,
                669.8,
            ],
            bins=ebins,
        )

        ax.hist(
            ebins[:-1],
            bins=ebins,
            weights=np.cumsum(eweights) / np.sum(eweights),
            label="Brown+18",
            histtype="step",
            lw=1.0,
            alpha=0.7,
            color="k",
            rasterized=True,
        )

        # And the velocities from Hattori+18
        ebins = bins
        ls = np.array(
            [
                193.87,
                278.09,
                61.28,
                332.40,
                287.70,
                302.68,
                321.80,
                289.93,
                87.67,
                88.96,
                153.60,
                319.08,
                102.44,
                295.99,
                269.29,
                301.17,
                299.83,
                255.66,
                67.47,
                137.29,
                307.46,
                75.16,
                267.94,
                289.09,
                152.87,
                100.58,
                72.19,
                108.61,
                316.14,
                229.68,
            ]
        )
        bs = np.array(
            [
                -36.61,
                -6.83,
                -46.88,
                -53.84,
                -25.27,
                67.81,
                -42.67,
                -28.26,
                49.03,
                13.49,
                36.20,
                -44.99,
                67.05,
                -23.96,
                -28.85,
                -22.50,
                -21.25,
                16.50,
                -31.88,
                -24.37,
                1.96,
                52.41,
                -54.95,
                11.25,
                -45.72,
                29.08,
                37.90,
                -36.13,
                -23.51,
                -52.12,
            ]
        )
        vloss = np.array(
            [
                1.7,
                160.2,
                -219.7,
                -38.2,
                159.9,
                88.7,
                -8.2,
                298.2,
                -168.7,
                -343.9,
                73.9,
                -11.5,
                -83.6,
                191.8,
                434.1,
                171.1,
                319.1,
                252.5,
                -271.8,
                -120.6,
                380.5,
                48.2,
                25.3,
                333.8,
                -166.1,
                14.6,
                -34.2,
                -303.5,
                27.8,
                -174.3,
            ]
        )
        Usun = rustics.Usun_gc
        eweights, _ = np.histogram(
            vloss
            + Usun[0] * np.cos(ls) * np.cos(bs)
            + Usun[1] * np.sin(ls) * np.cos(bs)
            + Usun[2] * np.sin(bs),
            bins=ebins,
        )

        ax.hist(
            ebins[:-1],
            bins=ebins,
            weights=np.cumsum(eweights) / np.sum(eweights),
            label="Hattori+18",
            histtype="step",
            lw=1.0,
            alpha=0.7,
            color="r",
            rasterized=True,
        )

    # Add x-axis label
    ax.set_xlabel(rustics.HEADERS_TO_LABELS[c])

    # Adjust y-axis labels
    if c == columns[0]:
        ax.set_ylabel("CDF")
        ax.legend(
            frameon=False,
        )
    elif c == columns[-1]:
        ax.tick_params(labelright=True)
    else:
        ax.set_yticklabels([])

    # Add supplemental legend
    if 0:  # c == "mf":
        # Add legend
        kwargs = {
            "color": "gray",
            "fill": False,
            "lw": 2,
        }
        hands = [Patch(**kwargs), Patch(**kwargs, ls="--")]
        labels = ["Pre-CC", "Post-CC"]
        ax.legend(
            handles=hands,
            labels=labels,
        )

# Make it neat
plt.tight_layout()

# Save and close
fname = __file__.split("/")[-1].replace(".py", "")
if type(kgroup) != type(None):
    fname = "_".join((fname, kgroup.initials))
fname = ".".join((fname, "pdf"))
plt.savefig(paths.figures / fname)
plt.close()
