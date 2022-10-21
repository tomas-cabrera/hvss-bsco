#! For generating ra-dec and mu-mu plots of GCs and their ejecta
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Galactocentric, Galactic, ICRS
from astropy import units as u
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

import rustics
import paths

###############################################################################

# Load GC data
cat = rustics.GCCatalog(
    str(paths.data / "mwgcs-cmc.dat"),
    pd_kwargs={"index_col": "Cluster"},
)
cat.df.fname = [n.replace("_v2", "").replace(".tar.gz", "") for n in cat.df.fname]

# Load orbital parameters
cat_orbit = rustics.GCCatalog(
    str(paths.data / "baumgardt_orbits_table_clean.txt"),
    pd_kwargs={
        "index_col": "Cluster",
        "delim_whitespace": True,
        "usecols": [
            "Cluster",
            "X",
            "Y",
            "Z",
            "U",
            "V",
            "W",
        ],
    },
)

# Join the two dfs
gcdf = cat.df.join(cat_orbit.df)

# Iterate over sample GCs
for gc in rustics.SAMPLE_GCS:
    # Get GC data
    gcrow = gcdf.loc[gc]
    scgc = SkyCoord(
        frame=Galactocentric,
        x=gcrow.X * u.kpc,
        y=gcrow.Y * u.kpc,
        z=gcrow.Z * u.kpc,
        v_x=gcrow.U * u.km / u.s,
        v_y=gcrow.V * u.km / u.s,
        v_z=gcrow.W * u.km / u.s,
    )
    scgc = scgc.transform_to(Galactic)
    ogc = Orbit(scgc)
    ts = np.linspace(0,-14000,100000) * u.Myr
    ogc.integrate(ts, MWPotential2014, method="dop853_c", progressbar=False)
    scgc_orbit = ogc.SkyCoord(ts)
    scgc_orbit = scgc_orbit.transform_to(Galactic)

    # Get ejections data
    ejdf = rustics.EjectionDf(
        paths.data_mwgcs / gc / "output_N-10_ejections.txt",
        usecols=["time", "X", "Y", "Z", "U", "V", "W"],
    )
    ejdf.df = ejdf.df[
        (ejdf.df.X != -1)
        & (ejdf.df.Y != -1)
    ]
    # Downsample by 1/10
    ejdf.df = ejdf.df.sample(frac=0.1, random_state=int(gcrow.X * 1e3))

    # Convert units
    scej = SkyCoord(
        frame=Galactocentric,
        x=ejdf.df.X.to_numpy() * u.kpc,
        y=ejdf.df.Y.to_numpy() * u.kpc,
        z=ejdf.df.Z.to_numpy() * u.kpc,
        v_x=ejdf.df.U.to_numpy() * u.km / u.s,
        v_y=ejdf.df.V.to_numpy() * u.km / u.s,
        v_z=ejdf.df.W.to_numpy() * u.km / u.s,
    )
    scej = scej.transform_to(Galactic)

    # Set up figure
    nrows = 2
    ncols = 1
    fig = plt.figure(figsize=(2.5*1.618 * ncols, 2.5*1 * nrows))
    axx = fig.add_subplot(*(nrows,ncols,1), projection="aitoff")
    axx.grid(True, c="gray", alpha=0.5, zorder=0, lw=0.5)
    axv = fig.add_subplot(*(nrows,ncols,2))
    # Color scale
    # kwargs for ejecta scatterpoints
    kw_ej = {
        "c": (ejdf.df.time + 14000. - gcrow.t) / 14000.,
        "cmap": "inferno",
        "marker": ".",
        "s": 4,
        "lw": 0,
        "rasterized": True,
    }
    # kwargs for GC point
    kw_gc = {
        "c": "xkcd:azure",
        "ec": "k",
        "s": 20,
        "lw": 0.5,
        "marker": "X",
    }
    # kwargs for GC orbit
    kw_o = {
        "c": "gray",
        "alpha": 0.5,
        "lw": 0.25,
        "zorder": 0,
    }

    # Plot locations (matplotlib aitoff wants coordinates in radians)
    sm=axx.scatter(
        scej.l.wrap_at("180d").radian,
        scej.b.radian,
        **kw_ej,
    )
    axx.scatter(
        scgc.l.wrap_at("180d").radian,
        scgc.b.radian,
        **kw_gc,
    )
    # Add colorbar; 221019: the colorbar spans the time of the ejections, which revealed that I was determining pre- and post- present-day ejections incorrectly.  Rerunning
    ## TODO: These two lines were copied from StackOverflow; implement later with right limits
    ##       Note that the scale will have to be applied to the scatter function as well
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-14, vmax=0))
    axxcb = plt.colorbar(mappable=sm, ax=axx, orientation="horizontal")
    axxcb.set_label(r"$t_{\rm ej}~[{\rm Gyr}]$")
    # Plot GC orbit
    axx.plot(
        scgc_orbit.l.wrap_at("180d").radian,
        scgc_orbit.b.radian,
        **kw_o,
    )

    ## Plot mu_ra_cosdec-mu_dec
    #scej = scej.transform_to(ICRS)
    #scgc = scgc.transform_to(ICRS)
    #scgc_orbit = scgc_orbit.transform_to(ICRS)
    #print(scgc)
    ## Ejected objects
    #axv.scatter(
    #    scej.pm_ra_cosdec,
    #    scej.pm_dec,
    #    c=ejdf.df.time,
    #    cmap="inferno",
    #    marker=".",
    #    s=1,
    #    lw=0,
    #    rasterized=True,
    #)
    ## GC
    #axv.scatter(
    #    scgc.pm_ra_cosdec,
    #    scgc.pm_dec,
    #    c="k",
    #    marker="x",
    #)
    ## GC orbit
    #axv.plot(
    #    scgc_orbit.pm_ra_cosdec,
    #    scgc_orbit.pm_dec,
    #    c="gray",
    #    alpha=0.5,
    #    lw=0.25,
    #    zorder=0,
    #)
    ## Extra things
    #axv.set_xlabel(r"$\mu_{\alpha \cos \delta}~[{\rm mas/yr}]$")
    #axv.set_ylabel(r"$\mu_{\delta}~[{\rm mas/yr}]$")
    # Plot mu_l_cosb-mu_b
    axv.scatter(
        scej.pm_l_cosb,
        scej.pm_b,
        **kw_ej,
    )
    # GC
    axv.scatter(
        scgc.pm_l_cosb,
        scgc.pm_b,
        **kw_gc,
    )
    # GC orbit
    axv.plot(
        scgc_orbit.pm_l_cosb,
        scgc_orbit.pm_b,
        **kw_o,
    )
    # Extra things
    axv.set_xlabel(r"$\mu_{l \cos b}~[{\rm mas/yr}]$")
    axv.set_ylabel(r"$\mu_b~[{\rm mas/yr}]$")

    # Clean up and save
    plt.tight_layout()
    plt.savefig(paths.figures / __file__.split("/")[-1].replace(".py", "_%s.pdf" % gc))
    plt.close()