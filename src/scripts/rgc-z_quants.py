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

def fetch_vtodays_rgc(
    gc_fname,
    path=paths.data_mwgcs,
    ej_fname="output_N-10_ejections.txt",
    kgroup=None,
):
    """ Function to make a single rgc histogram, for parallelization. """
    # Define path, load ejections with relevant columns
    path = path / gc_fname / ej_fname
    ejdf = rustics.EjectionDf(path, usecols=["X","Y","U","V","W","kf"])
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
    ejdf.df["vtoday"] = (ejdf.df.U**2 + ejdf.df.V**2 + ejdf.df.W**2)**0.5
    return ejdf.df[["rgc", "vtoday"]] 

def fetch_vtodays_Z(
    gc_fname,
    path=paths.data_mwgcs,
    ej_fname="output_N-10_ejections.txt",
    kgroup=None,
):
    """ Function to make a single rgc histogram, for parallelization. """
    # Define path, load ejections with relevant columns
    path = path / gc_fname / ej_fname
    ejdf = rustics.EjectionDf(path, usecols=["Z","U","V","W","kf"])
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
    ejdf.df["vtoday"] = (ejdf.df.U**2 + ejdf.df.V**2 + ejdf.df.W**2)**0.5
    return ejdf.df[["Z", "vtoday"]] 

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
    axis.set_minor_formatter(mplticker.NullFormatter())
    return 0

###############################################################################

##############################
###     Initial things     ###
##############################

# Settings
#plt.style.use("./matplotlibrc")
nprocs=4
model_fnames = os.listdir(paths.data_mwgcs)
intervals = [0.9, 0.99]
icolors = dict(zip(intervals, ["xkcd:azure", "xkcd:azure"]))
quantiles = [0.5]
for i in intervals:
    quantiles += [(1-i)/2, (1+i)/2]
kw_function = {
    "kgroup": rustics.INFO_BSE_K_GROUPS.loc[1],
}

# Setup figure
fig, axd = plt.subplot_mosaic(
    [["rgc"],["Z"]],
    figsize=(3.286 * 1, 2 * 2),
)

##############################
###        rgc hist        ###
##############################

# Set bins, axes
bins = np.logspace(-3, 4, 50)
ax = axd["rgc"]

# Read data
if nprocs == 1:
    # Don't run in parallel
    data = parmap.map(
        fetch_vtodays_rgc, model_fnames, **kw_function, pm_parallel=False, pm_pbar=True,
    )
else:
    # Run in parallel
    pool = mp.Pool(nprocs)
    data = parmap.map(
        fetch_vtodays_rgc, model_fnames, **kw_function, pm_parallel=True, pm_pbar=True,
    )
data = pd.concat(data, ignore_index=True)

# Calculate quantiles
vquants = []
bin_centers = []
for i in range(bins.shape[0]-1):
    # Filter data to bin
    blo = bins[i]
    bhi = bins[i+1]
    bin_centers.append((blo+bhi)/2)
    d = data[(data.rgc >= blo) & (data.rgc < bhi)]
    # Calculate quantiles
    s = d.vtoday.quantile(quantiles)
    vquants.append(s)
vquants = pd.concat(vquants, axis=1)
vquants.columns = bin_centers

# Scale to make axes nice
n_power = 3
vquants /= 10**n_power
    
# Plot median
ax.plot(
    vquants.columns,
    vquants.loc[0.5, :],
    c="k",
    label="Median",
)

# Plot intervals
# Note the if/else to prevent colors from overlapping.  This assumes the intervals are ordered from smallest to largest.
for ii, i in enumerate(intervals):
    ilo = (1-i)/2
    ihi = (1+i)/2
    vlo = vquants.loc[ilo,:]
    vhi = vquants.loc[ihi,:]
    alpha = -1/np.log10(1-i)
    if ii == 0:
        ax.fill_between(
            bin_centers,
            vlo,
            vhi,
            label="%d\%%" % (i*100),
            color=icolors[i],
            alpha=alpha,
            lw=0,
        )
    else:
        ax.fill_between(
            bin_centers,
            vlo,
            vlo_prev,
            label="%d\%%" % (i*100),
            color=icolors[i],
            alpha=alpha,
            lw=0,
        )
        ax.fill_between(
            bin_centers,
            vhi_prev,
            vhi,
            color=icolors[i],
            alpha=alpha,
            lw=0,
        )
    vlo_prev = vlo
    vhi_prev = vhi
        
# Extras
ax.set_xlabel(r"$r_{\rm GC}~[{\rm kpc}]$")
ax.set_ylabel(r"$v_{\rm GC}~[\times 10^%d~{\rm km~s^{-1}}]$" % n_power)
ax.set_xscale("log")
#ax.set_yscale("log")
add_minor_labels(ax.xaxis)
ax.legend()

##############################
###          Z hist        ###
##############################

# Set bins, axes
bins = np.logspace(-3, 4, 50)
bins = np.array([-1 * bins[::-1], bins]).flatten()
ax = axd["Z"]

# Read data
if nprocs == 1:
    # Don't run in parallel
    data = parmap.map(
        fetch_vtodays_Z, model_fnames, **kw_function, pm_parallel=False, pm_pbar=True,
    )
else:
    # Run in parallel
    pool = mp.Pool(nprocs)
    data = parmap.map(
        fetch_vtodays_Z, model_fnames, **kw_function, pm_parallel=True, pm_pbar=True,
    )
data = pd.concat(data, ignore_index=True)

# Calculate quantiles
vquants = []
bin_centers = []
for i in range(bins.shape[0]-1):
    # Filter data to bin
    blo = bins[i]
    bhi = bins[i+1]
    bin_centers.append((blo+bhi)/2)
    d = data[(data.Z >= blo) & (data.Z < bhi)]
    # Calculate quantiles
    s = d.vtoday.quantile(quantiles)
    vquants.append(s)
vquants = pd.concat(vquants, axis=1)
vquants.columns = bin_centers

# Scale to make axes nice
n_power = 3 
vquants /= 10**n_power
    
# Plot median
ax.plot(
    vquants.columns,
    vquants.loc[0.5, :],
    c="k",
    label="Median",
)

# Plot intervals
# Note the if/else to prevent colors from overlapping.  This assumes the intervals are ordered from smallest to largest.
for ii, i in enumerate(intervals):
    ilo = (1-i)/2
    ihi = (1+i)/2
    vlo = vquants.loc[ilo,:]
    vhi = vquants.loc[ihi,:]
    alpha = -1/np.log10(1-i)
    if ii == 0:
        ax.fill_between(
            bin_centers,
            vlo,
            vhi,
            label="%d\%%" % (i*100),
            color=icolors[i],
            alpha=alpha,
            lw=0,
        )
    else:
        ax.fill_between(
            bin_centers,
            vlo,
            vlo_prev,
            label="%d\%%" % (i*100),
            color=icolors[i],
            alpha=alpha,
            lw=0,
        )
        ax.fill_between(
            bin_centers,
            vhi_prev,
            vhi,
            color=icolors[i],
            alpha=alpha,
            lw=0,
        )
    vlo_prev = vlo
    vhi_prev = vhi
        
# Extras
ax.set_xlabel(r"$Z_{\rm GC}~[{\rm kpc}]$")
ax.set_ylabel(r"$v_{\rm GC}~[\times 10^%d~{\rm km~s^{-1}}]$" % n_power)
ax.set_xscale("symlog")
#ax.set_yscale("log")
add_minor_labels(ax.xaxis)
ax.legend()

##############################
###     Cleanup + save     ###
##############################

plt.tight_layout()
plt.savefig(paths.figures / __file__.split("/")[-1].replace(".py", "_%s.pdf" % kw_function["kgroup"]["initials"]))
plt.close()
