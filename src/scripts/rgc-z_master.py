import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import parmap
from tqdm import tqdm
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from astropy.coordinates import SkyCoord, Galactocentric
from astropy import units as u

import rustics

###############################################################################

helio_baumgardt = {
    "galcen_distance": 8.1 * u.kpc,
    "galcen_v_sun": np.array([11.1, 12.24 + 240, 7.25]) * u.km / u.s,
    "z_sun": 0.0 * u.pc,
}

plt.style.use(rustics.PATH_TO_MPLRC)

###############################################################################

df = pd.read_csv("/home/tomas/Documents/cmu/research/hvss/data/mwgcs/NGC_104/output_N-10_ejections.txt", index_col=0)
df["rgc"] = (df.X**2 + df.Y**2)**0.5
df["vtoday"] = (df.U**2 + df.V**2 + df.W**2)**0.5
print(df.columns)

# Histograms
for p in ["Z", "rgc"]:
    d = df[p]
    if p == "Z":
        bins = np.logspace(np.log10(d[d>0].min()), np.log10(d[d>0].max()), 50)
    else:
        bins = np.logspace(np.log10(d.min()), np.log10(d.max()), 50)
    plt.hist(d, bins=bins, histtype="step", label=r"$%s > 0$" % p)
    plt.hist(-1 * d, bins=bins, histtype="step", label=r"$%s < 0$" % p)
    if p == "Z":
        plt.xscale("symlog")
    else:
        plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$|%s| {\rm [kpc]}$" % p)
    plt.ylabel(r"$N$")
    plt.legend(title="NGC_104")
    plt.tight_layout()
    plt.savefig("hist_%s.pdf" % p)
    plt.close()

# Velocity quartiles
for p in ["Z", "rgc"]:
    dx = df[p]
    if p == "Z":
        bins = np.logspace(np.log10(dx[dx>0].min()), np.log10(dx[dx>0].max()), 30)
    else:
        bins = np.logspace(np.log10(dx.min()), np.log10(dx.max()), 20)
    quants = [0.01, 0.1, 0.5, 0.9, 0.99][::-1]
    qc = ["xkcd:azure", "xkcd:violet", "xkcd:orangered", "xkcd:violet", "xkcd:azure"]
    for bli, binl in enumerate([bins, (-1 * bins)[::-1]]):
        bmids = []
        dy = []
        for bi, bmin in enumerate(binl[:-1]):
            bmids.append((bmin + binl[bi+1])/2.)
            d = df[(df[p] >= bmin) & (df[p] < binl[bi+1])]
            if d.shape[0] == 0:
                dy.append([np.nan] * len(quants))
            else:
                dy.append([np.quantile(d.vtoday, q) for q in quants])
        dy = np.array(dy)
        for qi, q in enumerate(quants):
            if bli == 0:
                plt.plot(
                    bmids,
                    dy[:,qi],
                    c=qc[qi],
                    label="%s" % q,
                )
            else:
                plt.plot(
                    bmids,
                    dy[:,qi],
                    c=qc[qi],
                )
    if p == "Z":
        plt.xscale("symlog")
    else:
        plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("%s [kpc]" % p)
    plt.ylabel(r"$v_{\rm out}\ {\rm [km\ s^{-1}]}$")
    plt.legend(title="NGC_104")
    plt.tight_layout()
    plt.savefig("vquants_%s.pdf" % p)
    plt.close()
