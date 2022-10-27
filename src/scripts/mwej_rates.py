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
        wmin = wc - window_width / 2.
        wmax = wc + window_width / 2.
        mask = (data >= wmin) & (data < wmax)
        window_counts.append(mask.sum())
    return window_counts
        

def calc_rolling_rate(mwrow, window_centers, window_width=10, vlim=(0.,np.inf)):
    """! Function for parallelizing rolling rate calculation.

    @param  mwrow               mwdf row for a certain GC
    @param  vlim                Limits of velocities to include
    @param  window_step         Step size of rolling window, in Myr
    @param  window_width        Width of rolling window, in Myr
    """

    # Load CMC ejections
    gcdf = pd.read_csv(
        paths.data_mwgcs / mwrow.Cluster / "output_N-10_ejections.txt",
        usecols=["time","U","V","W"],
    )

    # Filter -1's (post-present-day ejections)
    mask = (gcdf.U != -1) & (gcdf.V != -1) & (gcdf.W != -1)
    gcdf = gcdf[mask]

    # Calculate galactocentric velocity & apply vlim
    gcdf["vgc"] = (gcdf.U**2 + gcdf.V**2 + gcdf.W**2)**0.5
    mask = (gcdf.vgc >= vlim[0]) & (gcdf.vgc < vlim[1])
    gcdf = gcdf[mask]

    # Calculate rolling rate
    # Note: gcdf.time - mwrow.t sets the time 0 to the present day
    return _rolling_window_count(gcdf.time.to_numpy() - mwrow.t, window_centers, window_width)
     
    

###############################################################################

# Settings
nprocs = 4 

# Load MW catalog
mwdf = pd.read_csv(paths.data / "mwgcs-cmc.dat")
print(mwdf.columns)
print(mwdf)

# Parallel rolling rate calculation
window_width = 200
window_step = 20 
tlim = (-14000., 0.)
# Set the minimum and maximum window centers to the values where their window entirely lies in the data domain
tmin = tlim[0] + window_width / 2.
tmax = tlim[1] - window_width / 2.
# Generate the window centers
window_centers = np.arange(tmin, tmax, window_step)
if nprocs == 1:
    # Don't run in parallel
    mp_input = [x for _,x in mwdf.iterrows()]
    output = parmap.map(
        calc_rolling_rate, mp_input, window_centers, window_width=window_width, pm_parallel=False, pm_pbar=True,
    )
else:
    # Run in parallel
    mp_input = [x for _,x in mwdf.iterrows()]
    pool = mp.Pool(nprocs)
    output = parmap.map(
        calc_rolling_rate, mp_input, window_centers, window_width=window_width, pm_pool=pool, pm_pbar=True,
    )
full_counts = np.array(output).sum(axis=0)
if nprocs == 1:
    # Don't run in parallel
    mp_input = [x for _,x in mwdf.iterrows()]
    output = parmap.map(
        calc_rolling_rate, mp_input, window_centers, window_width=window_width, vlim=(500,np.inf), pm_parallel=False, pm_pbar=True,
    )
else:
    # Run in parallel
    mp_input = [x for _,x in mwdf.iterrows()]
    pool = mp.Pool(nprocs)
    output = parmap.map(
        calc_rolling_rate, mp_input, window_centers, window_width=window_width, vlim=(500,np.inf), pm_pool=pool, pm_pbar=True,
    )
hvs_counts = np.array(output).sum(axis=0)

# Initialize figure
fscale = 2
fig, ax = plt.subplots(figsize=(2 * 1.618, 2 * 1.))

# Plot
ax.plot(
    window_centers / 1000.,
    full_counts / window_width / 1e6,
    label="All ejections",
)
ax.plot(
    window_centers / 1000.,
    hvs_counts / window_width / 1e6,
    label=r"$v > 500~{\rm km~s^{-1}}$",
)
ax.axhline(2e-3, c="xkcd:azure", ls="--", alpha=0.7)
ax.axhline(1e-4, c="xkcd:orangered", ls="--", alpha=0.7)

# Axes
ax.set_xlabel(rustics.HEADERS_TO_LABELS["time"])
ax.set_ylabel(r"Rate [yr$^{-1}$]")
ax.set_yscale("log")
ax.legend()

# Make neat and save
plt.tight_layout()
plt.savefig(paths.figures / __file__.replace(".py", ".pdf"))
plt.close()
