import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import os

import rustics

###############################################################################


def nn_in_cmc(df_cmc, df_mw, cols, translator=None):
    """
    Given a df of CMC models and a df of observed GCs, finds the nearest CMC model for each observed GC in the 2D space of the CMC headers cols.
    Note that the specified columns are expected to be normalized to the same space.
    Arguments:
        df_cmc : pd.DataFrame of parameters for CMC models
        df_mw : pd.DataFrame of parameters for observed clusters
        cols : CMC column names to use as dimensions to calculate distance in
        translator : dictionary to translate CMC column names to observation column names
    Returns:
        nnis : pd.Series of indices to df_cmc, ordered to prescribe the closest CMC model for each observed GC.
    """
    # Initialize distance array
    d = np.zeros((df_mw.shape[0], df_cmc.shape[0]))
    print(df_cmc)
    print(df_mw)

    # Add squared differences
    for col in cols:
        # If there is a translator, use it
        if translator != None:
            col_mw = translator[col]
        else:
            col_mw = col
        # Add the squared distance to the df
        cx = np.array(df_cmc[col])
        mx = np.reshape(df_mw[col_mw].to_numpy(), (df_mw.shape[0], 1))
        d += (cx - mx) ** 2

    # Sqrt to get distance; get array of minima
    d = np.sqrt(d)
    nnis = np.argmin(d, axis=1)
    return nnis


###############################################################################

# Use styling
plt.style.use(rustics.PATH_TO_MPLRC)

## Generate observational catalog, from Baumgardt+Hilker and Harris
# Cleaned from original file (ID->Cluster, names edited to match, lack of metallicity measurement denoted with Z=-100)
fname_mw_bh = "holger_baumgardt_clean.txt"
fname_mw_h = "harris2010_II_clean.txt"
df_mw_bh = pd.read_csv(
    "/".join((rustics.PATH_TO_DATA, fname_mw_bh)), delim_whitespace=True
)
df_mw_h = pd.read_csv(
    "/".join((rustics.PATH_TO_DATA, fname_mw_h)), delim_whitespace=True
)

# Drop Harris clusters missing a metallicity measurement(Z=-100)
df_mw_h = df_mw_h[df_mw_h["[Fe/H]"] != -100]

# Merge the catalogs, keeping only the clusters that appear in both
df_mw = df_mw_bh.merge(df_mw_h, how="inner", on="Cluster")

# Calculate radius ratios, log10(Mass)
df_mw["rc/rh,l"] = df_mw["rc"] / df_mw["rh,l"]
df_mw["rc/rh,m"] = df_mw["rc"] / df_mw["rh,m"]
df_mw["logMass"] = np.log10(df_mw["Mass"])

## Load df of final dyn.dat info for each cluster
# N = initial size (/1e5), N.1 = ending size, N_ej = total number of BSCO ejections for this model
# TODO: update to include only stellar ejections?
df_cmc = pd.read_csv("/".join((rustics.PATH_TO_DATA, "cmc-catalog.final_dyn.dat")))

# Calculate CMC names, radius ratios, log10(Mass), and [Fe/H] = log10(Z/Zsun), with Zsun = 0.02
df_cmc["Cluster"] = [
    "N{}_".format(rustics._ns_to_str(df_cmc.loc[i, "N"] * 100000))
    + "rv{:n}_".format(df_cmc.loc[i, "rv"])
    + "rg{:n}_".format(df_cmc.loc[i, "rg"])
    + "Z{}".format(df_cmc.loc[i, "Z"])
    for i in df_cmc.index
]
df_cmc["rc_spitzer/r_h"] = df_cmc["rc_spitzer"] / df_cmc["r_h"]
df_cmc["logM"] = np.log10(df_cmc["M"])
df_cmc["[Fe/H]"] = np.log10(df_cmc["Z"] / 0.02)
print(df_cmc)

## Calculate nearest neighbors
# Dictionary to convert headers
cmc_to_mw = {
    "M": "Mass",
    "logM": "logMass",
    "rc_spitzer/r_h": "rc/rh,m",
    "logM_norm": "logMass_norm",
    "rc_spitzer/r_h_norm": "rc/rh,m_norm",
    "[Fe/H]": "[Fe/H]",
    "[Fe/H]_norm": "[Fe/H]_norm",
}

# Data normalization, by CMC space
means = {}
stds = {}
nn_cols = [
    "logM",
    "rc_spitzer/r_h",
    "[Fe/H]",
]
for col in nn_cols:
    means[col] = df_cmc[col].mean()
    stds[col] = df_cmc[col].std()
    df_cmc["%s_norm" % col] = (df_cmc[col] - means[col]) / stds[col]
    df_mw["%s_norm" % cmc_to_mw[col]] = (df_mw[cmc_to_mw[col]] - means[col]) / stds[col]

# Find the nearest neighbor Euclidean-wise in the normalized space
df_mw["nn_cmc"] = nn_in_cmc(
    df_cmc,
    df_mw,
    ["%s_norm" % c for c in nn_cols],
    translator=cmc_to_mw,
)

# Add column of weights to df_cmc, equal to the number of MW GCs that have the model as its nearest CMC neighbor
# Old version, definitely works
# df_cmc["weights_mw"] = np.histogram(
#    df_mw["nn_cmc"],
#    bins=np.arange(-0.5, df_cmc.shape[0] + 0.5),
# )[0]
# This is cleaner and appears to work the same, but introduces NaN's wherever there are no MW GCs for a CMC model
df_cmc["weights_mw"] = df_mw.value_counts("nn_cmc")
print("# of models used:", (df_cmc.weights_mw > 0).sum())

# Calculate the total number of ejections for the catalog!!!
N_ej = (df_cmc["weights_mw"] * df_cmc["N_ej"]).sum()
print("Expected number of ejections N_ej = %f" % N_ej)
print("N_ej/13.8Gyr = %g/Myr = 1/(%g yr)" % (N_ej / 13800, 13800000000 / N_ej))

# Calculate the number of core-collapsed clusters 
df_cmc_cc = pd.read_csv("/home/tomas/Documents/cmu/research/hvss/Catalog_models_ccw.dat")
df_cmc = pd.merge(df_cmc, df_cmc_cc, how="inner", on=["N","rv","rg","Z"], validate="1:1", suffixes=[None, "_cc"])
N_cc = (df_cmc["weights_mw"] * df_cmc["cc_flag"]).sum()
print("# of core-collapsed clusters = %d" % N_cc)

## Plot the populations over one another, for visualization of matchups
# Parameters
rows = ["M"]
columns = ["rc_spitzer/r_h", "[Fe/H]"]
mosaic = [["_".join((r, c)) for c in columns] for r in rows]
plot_scale = 3.5
markersize = 20
mask_mw = df_mw.Cluster != "NGC_5272"
cmc_samples = [
    "N8e5_rv0.5_rg8_Z0.0002",
    "N4e5_rv0.5_rg8_Z0.0002",
    "N8e5_rv2_rg8_Z0.0002",
    "N8e5_rv0.5_rg8_Z0.02",
]
cmc_M3like = ["N1.6e6_rv1_rg8_Z0.0002"]
mask_samples = [c in cmc_samples for c in df_cmc.Cluster]
mask_M3like = [c in cmc_M3like for c in df_cmc.Cluster]

# Subplot mosaic
fig, axd = plt.subplot_mosaic(
    mosaic,
    gridspec_kw={
        "hspace": 0,
        "wspace": 0,
    },
    figsize=(plot_scale * len(columns), 1.25 * plot_scale * len(rows)),
)
print(df_cmc.columns)
print(df_cmc)
for r in rows:
    for c in columns:
        # Get the right axes
        ax = axd["_".join((r, c))]

        # Plot MW data, separating M3/NGC 5272
        ax.scatter(
            df_mw[mask_mw][cmc_to_mw[c]],
            df_mw[mask_mw][cmc_to_mw[r]],
            marker="s",
            color="k",
            label="MW GCs",
            s=np.pi / 4 * markersize,
            lw=0,
            alpha=0.7,
            rasterized=True,
        )

        # Plot CMC data
        mask = [not x and not y for x, y in zip(mask_samples, mask_M3like)]
        ax.scatter(
            df_cmc[mask][c],
            df_cmc[mask][r],
            marker="o",
            color="xkcd:orangered",
            label="CMC models",
            s=markersize,
            lw=0,
            alpha=0.7,
            rasterized=True,
        )
        ax.scatter(
            df_cmc[mask_samples][c],
            df_cmc[mask_samples][r],
            marker="*",
            color="xkcd:azure",
            label="Sample models",
            s=10 * markersize,
            lw=0,
            alpha=0.9,
            rasterized=True,
        )

        # Plot the M3 data and matching CMC model
        ax.scatter(
            df_mw[[not x for x in mask_mw]][cmc_to_mw[c]],
            df_mw[[not x for x in mask_mw]][cmc_to_mw[r]],
            marker="^",
            color="xkcd:violet",
            label="M3",
            s=5 * markersize,
            lw=0,
            alpha=0.9,
            rasterized=True,
        )
        ax.scatter(
            df_cmc[mask_M3like][c],
            df_cmc[mask_M3like][r],
            marker="v",
            color="xkcd:violet",
            label="M3-like model",
            s=5 * markersize,
            lw=0,
            alpha=0.9,
            rasterized=True,
        )

        ax.set_yscale("log")

        ## Format plot
        # Make opposite axes the normed data
        axn = ax.twiny()
        axn.set_xlim([(x - means[c]) / stds[c] for x in ax.get_xlim()])
        ayn = ax.twinx()
        ayn.set_ylim(
            [
                (np.log10(x) - means["log%s" % r]) / stds["log%s" % r]
                for x in ax.get_ylim()
            ]
        )

        # Make sure y-ticks are all there
        ylim = ax.get_ylim()
        numticks = np.ceil(np.log10(ylim[1] / ylim[0])) + 2
        # locmaj = mplticker.LogLocator(base=10, numticks=numticks)
        # ax.xaxis.set_major_locator(locmaj)
        locmin = mplticker.LogLocator(
            base=10, subs=np.arange(0, 1, 0.1), numticks=numticks
        )
        axn.yaxis.set_minor_locator(locmin)
        axn.yaxis.set_minor_formatter(mplticker.NullFormatter())

        # Adjust axis labels and ticks
        ax.set_xlabel(rustics.HEADERS_TO_LABELS[c])
        axn.set_xlabel(rustics.HEADERS_TO_LABELS["%s_norm" % c], labelpad=6.0)
        if c == columns[0]:
            ax.set_ylabel(rustics.HEADERS_TO_LABELS[r])
            ayn.tick_params(labelright=False)
            ax.legend(
                frameon=True,
                shadow=False,
                scatterpoints=1,
            )
        elif c == columns[-1]:
            ayn.set_ylabel(rustics.HEADERS_TO_LABELS["%s_norm" % r])
            ax.tick_params(labelleft=False)
        else:
            ax.set_yticklabels([])

plt.tight_layout()
plt.savefig("/".join(("figures", "mw-cmc.pdf")))
