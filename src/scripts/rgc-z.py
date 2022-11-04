import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import matplotlib.scale as mplscale
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

def fetch_rgc_Zgc_vtodays(
    gc_fname,
    path=paths.data_mwgcs,
    ej_fname="output_N-10_ejections.txt",
    kgroup=None,
):
    """ Function to make a single rgc histogram, for parallelization. """
    # Define path, load ejections with relevant columns
    path = path / gc_fname / ej_fname
    ejdf = rustics.EjectionDf(path, usecols=["X","Y","Z","U","V","W","kf"])
    # Filter with BSE k's if specified
    if type(kgroup) != type(None):
        ejdf.df = ejdf.df[
            (ejdf.df.kf >= kgroup.lo) & (ejdf.df.kf < kgroup.hi)
        ]
    # Throw out -1's (after present day)
    ejdf.df = ejdf.df[
        (ejdf.df.X != -1)
        & (ejdf.df.Y != -1)
    ]
    # Calculate rgcs and vtodays; rename Z to Zgc to avoid overlap with metallicity settings
    ejdf.df["rgc"] = (ejdf.df.X**2 + ejdf.df.Y**2)**0.5
    ejdf.df["vtoday"] = (ejdf.df.U**2 + ejdf.df.V**2 + ejdf.df.W**2)**0.5
    ejdf.df.rename(columns={"Z": "Zgc"}, inplace=True)
    return ejdf.df[["rgc", "Zgc", "vtoday"]]

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
            base=10, linthresh=0.01, subs=np.arange(0, 1, 0.1),
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
# Set bins, Zgc by reflecting rgc across 0
bins = {"rgc": np.logspace(-3, 4.6, 50)}
bins["Zgc"] = np.array([-1 * bins["rgc"][::-1], bins["rgc"]]).flatten()

# Load data
kw_function = {
    "path": paths.data_mwgcs,
    "ej_fname": "output_N-10_ejections.txt",
    "kgroup": rustics.INFO_BSE_K_GROUPS.loc[0],
}
if nprocs == 1:
    # Don't run in parallel
    data = parmap.map(
        fetch_rgc_Zgc_vtodays, model_fnames, **kw_function, pm_parallel=False, pm_pbar=True,
    )
else:
    # Run in parallel
    pool = mp.Pool(nprocs)
    data = parmap.map(
        fetch_rgc_Zgc_vtodays, model_fnames, **kw_function, pm_parallel=True, pm_pbar=True,
    )
data = pd.concat(data, ignore_index=True)

# Setup figure
params = ["rgc", "Zgc"]
plots = ["hist", "quants"]
mosaic = np.array([["%s-%s" % (param, plot) for param in params] for plot in plots])
fig, axd = plt.subplot_mosaic(
    mosaic,
    figsize=(3.286 * mosaic.shape[1], 2 * mosaic.shape[0]),
    gridspec_kw = {
        "hspace": 0.,
    }
)

##############################
###         hist           ###
##############################

# Setup
n_power = 4 # Divide counts by 10**n_power, for ticklabel formatting
kw_hist = {
    "histtype": "step",
    "lw": 2,
}

# Iterate over params
for p in params:
    # Set bins, axes
    ax = axd["%s-hist" % p]

    # Plot
    ax.hist(
        bins[p][:-1],
        bins=bins[p],
        weights = np.histogram(data[p], bins=bins[p])[0] / 10. / 10.**n_power,
        **kw_hist,
    )

    # Extras
    ax.tick_params(labeltop=True)
    if p == "rgc":
        ax.set_xscale("log")
    elif p == "Zgc":
        ax.set_xscale("symlog")
        ax.set_xscale(mplscale.SymmetricalLogScale(ax, linthresh=0.01))
        # Hide every other xlabel:
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
    add_minor_labels(ax.xaxis)
    add_minor_labels(ax.yaxis)
    ax.set_yscale("log")
    ax.set_ylabel(r"Counts~[$\times 10^{%d}$]" % n_power)

##############################
###        quants          ###
##############################

# Setup
n_power = 3
intervals = [0.9, 0.99] # These should be in inreasing order (e.g. [0.9, 0.99]), to make sure they are plotted correctly
quantiles = [0.5]
for i in intervals:
    quantiles += [(1-i)/2, (1+i)/2]
kw_quants = {
    "color": "xkcd:azure",
    "lw": 0,
}

# Iterate over params
for p in params:
    # Calculate quantiles
    vquants = []
    for blo, bhi in zip(bins[p][:-1], bins[p][1:]):
        # Filter data to bin
        d = data[(data[p] >= blo) & (data[p] < bhi)]
        # Calculate quantiles
        s = d.vtoday.quantile(quantiles)
        vquants.append(s)
    vquants = pd.concat(vquants, axis=1)

    # Save bin centers as column names of vquants df
    vquants.columns = (bins[p][:-1] + bins[p][1:]) / 2.

    # Remove center bin to prevent issues with log scaling
    if p == "Zgc":
        del vquants[0.0]

    # Scale to make ticklabels nice
    vquants /= 10.**n_power

    ### Plotting
    # Choose axes
    ax = axd["%s-quants" % p]

    # Plot median
    ax.plot(
        vquants.columns,
        vquants.loc[0.5, :],
        c="k",
        label="Median",
    )

    # Plot intervals
    vlo_prev = vquants.loc[0.5, :]
    vhi_prev = vquants.loc[0.5, :]
    for ii, i in enumerate(intervals):
        # Get the correct data
        ilo = (1-i)/2.
        ihi = (1+i)/2.
        vlo = vquants.loc[ilo,:]
        vhi = vquants.loc[ihi,:]
        kw_quants["alpha"] = -1/np.log10(1-i)

        # Plot, in two halves so color in legend matches color in plot
        ax.fill_between(
            vquants.columns,
            vlo,
            vlo_prev,
            label="%d\%%" % (i*100),
            **kw_quants,
        )
        ax.fill_between(
            vquants.columns,
            vhi_prev,
            vhi,
            **kw_quants,
        )

        # Update limits of shaded region
        vlo_prev = vlo
        vhi_prev = vhi

    # Extras
    axh = axd["%s-hist" % p]
    ax.set_xlim(axh.get_xlim())
    if p == "rgc":
        ax.legend(loc="upper left")
        ax.set_xscale(axh.get_xscale())
        axrgc = ax
    elif p == "Zgc":
        ax.set_xscale(mplscale.SymmetricalLogScale(ax, linthresh=0.01))
        ax.set_ylim(axrgc.get_ylim())
        # Hide every other xlabel:
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
    add_minor_labels(ax.xaxis)
    ax.set_xlabel(rustics.HEADERS_TO_LABELS[p])
    ax.set_ylabel(rustics.HEADERS_TO_LABELS["vtoday"])

##############################
###     Cleanup + save     ###
##############################

plt.tight_layout()
plt.savefig(paths.figures / __file__.split("/")[-1].replace(".py", "_%s.pdf" % kw_function["kgroup"]["initials"]))
plt.close()
