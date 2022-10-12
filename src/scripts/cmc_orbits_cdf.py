import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
from matplotlib.patches import Patch

import rustics
import paths

################################################################################

# TODO: add functionality to calculate vlos for object
#       Currently fails at creating SkyCoord object in rustics.py

################################################################################

#plt.style.use(rustics.PATH_TO_MPLRC)

# Get model info
path_to_data = paths.data_mwgcs
df_mwgcs = pd.DataFrame(os.listdir(path_to_data), columns=["fname"]) 

# Iterables
columns = ["vlos", "mf"]

# Specify kgroup
kgroup = None
kgroup = rustics.INFO_BSE_K_GROUPS.iloc[0]

# Make subplot mosiac
plot_size = 2.3
fig, axd = plt.subplot_mosaic(
    [columns],
    sharey=True,
    figsize=(plot_size * len(columns), plot_size),
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
        hg = rustics.HistogramGenerator(c, bins, nprocs=4, kgroup=kgroup)#, cc=True)
    else:
        hg = rustics.HistogramGenerator(c, bins, nprocs=4, kgroup=kgroup)

    # Generate histograms
    df_mwgcs["hists"] = hg.generate_catalog_histograms(df_mwgcs, pm_pbar=True, path_to_data=path_to_data)

    # Choose axis
    ax = axd[c]

    # Plot
    if 0:#c == "mf":
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

        # Crude figure-measurement
        ebins = np.arange(275,2000,53)
        eweights = [0] * (len(ebins)-1)
        eweights[0] = 10
        eweights[1] = 4
        eweights[2] = 6
        eweights[3] = 2
        eweights[4] = 3
        eweights[5] = 1
        eweights[6] = 0
        eweights[7] = 2

        # Actually using radial velocities from Brown+18
        ebins=bins
        eweights, _ = np.histogram(
            [275.2, 277.8, 279.9, 282.8, 283.9, 286.8, 288.3, 288.9, 289.1, 289.1, 289.6, 294.1, 302.1, 306.2, 308.6, 311.4, 319.6, 328.5, 344.1, 344.6, 358.8, 363.9, 391.9, 392.1, 397.7, 413.3, 417.0, 417.4, 418.5, 439.5, 440.3, 449.0, 458.8, 487.4, 496.2, 501.1, 551.7, 644.0, 669.8,],
            bins=ebins,
        )

        ax.hist(
            ebins[:-1],
            bins=ebins,
            weights=np.cumsum(eweights) / np.sum(eweights),
            label="Brown+18",
            histtype="step",
            lw=1.,
            alpha=0.7,
            color="k",
            rasterized=True,
        )

        # And the velocities from Hattori+18
        ebins=bins
        np.random.RandomState(123456)
        # v_r
        vels = [337.0, 539.3, 552.6, 568.9, 456.3, 524.7, 572.7, 553.7, 323.6, 78.9, 403.2, 32.1, 490.3, 506.3, 479.8, 513.5, 516.6, 236.2, 376.3, 150.9, 142.0, 138.8, 276.2, 501.1, 206.9, 362.5, 430.3, 291.8, 526.8, 464.1,]
        # v_total
        vels = [598.3, 577.1, 581.1, 583.0, 574.7, 566.7, 578.1, 582.9, 541.6, 539.2, 523.4, 547.0, 530.4, 536.3, 502.9, 536.9, 528.9, 502.6, 523.6, 504.4, 537.3, 496.5, 508.9, 511.5, 501.9, 494.5, 505.3, 500.9, 527.0, 481.4,]
        eweights, _ = np.histogram(
            #vels,
            vels * np.sin(np.pi * np.random.rand(len(vels))),
            bins=ebins,
        )

        ax.hist(
            ebins[:-1],
            bins=ebins,
            weights=np.cumsum(eweights) / np.sum(eweights),
            label="Hattori+18",
            histtype="step",
            lw=1.,
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
    if 0:#c == "mf":
        # Add legend
        kwargs = {
            "color": "gray",
            "fill": False,
            "lw": 2,
        }
        hands = [Patch(**kwargs), Patch(**kwargs, ls="--")]
        labels = ["Pre-CC", "Post-CC"]
        ax.legend(
            handles = hands,
            labels = labels,
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
