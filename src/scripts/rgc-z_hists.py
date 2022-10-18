import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import multiprocessing as mp
import parmap

import rustics
import paths

###############################################################################

#helio_baumgardt = {
#    "galcen_distance": 8.1 * u.kpc,
#    "galcen_v_sun": np.array([11.1, 12.24 + 240, 7.25]) * u.km / u.s,
#    "z_sun": 0.0 * u.pc,
#}

###############################################################################

def generate_rgc_hist(
    gc_fname,
    path=paths.data_mwgcs,
    bins=np.linspace(0,1e5,50),
    ej_fname="output_N-10_ejections.txt",
    #kgroup=None,
    kgroup=rustics.INFO_BSE_K_GROUPS.loc[0],
):
    """ Function to make a single rgc histogram, for parallelization. """
    # Define path, load ejections with relevant columns
    path = path / gc_fname / ej_fname
    ejdf = rustics.EjectionDf(path, usecols=["X","Y","kf"])
    # Filter with BSE k's if specified
    if type(kgroup) != type(None):
        ejdf.df = ejdf.df[
            (ejdf.df.kf >= kgroup.lo) & (ejdf.df.kf < kgroup.hi)
        ]
    # Throw out -1's
    ejdf.df = ejdf.df[
        (ejdf.df.X != -1)
        & (ejdf.df.Y != -1)
    ]
    # Calculate rgc and make histogram
    ejdf.df["rgc"] = (ejdf.df.X**2 + ejdf.df.Y**2)**0.5
    hist, _ = np.histogram(ejdf.df.rgc, bins=bins)
    return hist

def generate_Z_hist(
    gc_fname,
    path=paths.data_mwgcs,
    bins=np.linspace(0,1e5,50),
    ej_fname="output_N-10_ejections.txt",
    #kgroup=None,
    kgroup=rustics.INFO_BSE_K_GROUPS.loc[0],
):
    """ Function to make a single rgc histogram, for parallelization. """
    # Define path, load ejections with relevant columns
    path = path / gc_fname / ej_fname
    ejdf = rustics.EjectionDf(path, usecols=["Z","kf"])
    # Filter with BSE k's if specified
    if type(kgroup) != type(None):
        ejdf.df = ejdf.df[
            (ejdf.df.kf >= kgroup.lo) & (ejdf.df.kf < kgroup.hi)
        ]
    # Throw out -1's
    ejdf.df = ejdf.df[
        (ejdf.df.Z != -1)
    ]
    # Make histogram
    hist, _ = np.histogram(ejdf.df.Z, bins=bins)
    return hist

def add_minor_labels(axis):
    """ Utility to add minor labels to axis."""
    lim = axis.get_view_interval()
    if lim[0] > 0:
        numticks = np.ceil(np.log10(lim[1] / lim[0])) + 2
        locmin = mplticker.LogLocator(
            base=10, subs=np.arange(0, 1, 0.1), numticks=numticks
        )
    else:
        locmin = mplticker.SymmetricalLogLocator(
            base=10, linthresh=2, subs=np.arange(0, 1, 0.1),
        )
    axis.set_minor_locator(locmin)
    return 0

###############################################################################

##############################
###     Initial things     ###
##############################

#plt.style.use("./matplotlibrc")
nprocs=4
model_fnames = os.listdir(paths.data_mwgcs)

# Setup figure
fig, axd = plt.subplot_mosaic(
    [["rgc"],["Z"]],
    figsize=(3.236 * 1, 2 * 2),
)

##############################
###        rgc hist        ###
##############################

# Set bins, axes
bins = np.logspace(-3, 4, 50)
ax = axd["rgc"]

# Read data + generate histograms
if nprocs == 1:
    # Don't run in parallel
    hists = parmap.map(
        generate_rgc_hist, model_fnames, bins=bins, pm_parallel=False, pm_pbar=True,
    )
else:
    # Run in parallel
    pool = mp.Pool(nprocs)
    hists = parmap.map(
        generate_rgc_hist, model_fnames, bins=bins, pm_parallel=True, pm_pbar=True,
    )
hists = np.array(hists)

# Plot figure
n_power = 4
ax.hist(
    bins[:-1],
    bins=bins,
    weights = hists.sum(axis=0) / 10. / 10.**n_power,
    histtype="step",
    lw=2,
)
# Extras
ax.set_xlabel(r"$r_{\rm GC}~[{\rm kpc}]$")
ax.set_ylabel(r"Counts/$N_{\rm models}~[\times 10^{%d}]$" % n_power)
ax.set_xscale("log")
#ax.set_yscale("log")
add_minor_labels(ax.xaxis)

##############################
###          Z hist        ###
##############################

# Set bins, axes
bins = np.logspace(-3, 4, 50)
bins = np.array([-1 * bins[::-1], bins]).flatten()
ax = axd["Z"]

# Read data + generate histograms
if nprocs == 1:
    # Don't run in parallel
    hists = parmap.map(
        generate_Z_hist, model_fnames, bins=bins, pm_parallel=False, pm_pbar=True,
    )
else:
    # Run in parallel
    pool = mp.Pool(nprocs)
    hists = parmap.map(
        generate_Z_hist, model_fnames, bins=bins, pm_parallel=True, pm_pbar=True,
    )
hists = np.array(hists)

# Plot figure
n_power = 4
ax.hist(
    bins[:-1],
    bins=bins,
    weights = hists.sum(axis=0) / 10. / 10.**n_power,
    histtype="step",
    lw=2,
)
# Extras
ax.set_xlabel(r"$Z~[{\rm kpc}]$")
ax.set_ylabel(r"Counts/$N_{\rm models}~[\times 10^{%d}]$" % n_power)
ax.set_xscale("symlog")
#ax.set_yscale("log")
add_minor_labels(ax.xaxis)

##############################
###     Cleanup + save     ###
##############################

plt.tight_layout()
plt.savefig(paths.figures / __file__.split("/")[-1].replace(".py", ".pdf"))
plt.close()
