import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rustics
import paths

###############################################################################

# Plot settings
kw_mwgcs = {
    "c": "k",
    "s": 10,
    "lw": 0,
    "marker": "^",
    "label": "MW GCs",
}
kw_cmcs = {
    "c": "xkcd:azure",
    "s": 10,
    "lw": 0,
    "label": r"\texttt{CMC} models",
}

# Load data
cmcs = pd.read_csv(paths.data / "cmcs.dat")
cmcs["R_GC"] = cmcs.rg
mwgcs = pd.read_csv(paths.data / "mwgcs.dat")

# Setup figure
mosaic = np.array([["logM-rc/rh"], ["R_GC-[Fe/H]"]])
fs = 2.25
fig, axd = plt.subplot_mosaic(
    mosaic,
    figsize=(fs * 1.618 * mosaic.shape[1], fs * 1 * mosaic.shape[0]),
)

# Plot data
for ai in axd:
    ax = axd[ai]
    y,x = ai.split("-")

    # cmcs
    ax.scatter(
        cmcs[x],
        cmcs[y],
        **kw_cmcs,
    )

    # mwgcs
    ax.scatter(
        mwgcs[x],
        mwgcs[y],
        **kw_mwgcs,
    )

    # Extra things
    ax.set_xlabel(rustics.HEADERS_TO_LABELS[x])
    ax.set_ylabel(rustics.HEADERS_TO_LABELS[y])
    ax.legend(
        frameon=True,
        shadow=False,
        scatterpoints=1,
        facecolor="w",
    )

# Cleanup and save
plt.tight_layout()
plt.savefig(paths.figures / __file__.split("/")[-1].replace(".py",".pdf"))
plt.close()
