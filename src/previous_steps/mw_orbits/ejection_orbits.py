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
import readoutput as ro
import aggregate as ag

###############################################################################

helio_baumgardt = {
    "galcen_distance": 8.1 * u.kpc,
    "galcen_v_sun": np.array([11.1, 12.24 + 240, 7.25]) * u.km / u.s,
    "z_sun": 0.0 * u.pc,
}

###############################################################################


def integrate_cmc_ejections(
    cmc_fnames,
    mwgc_catalog,
    cmc_path=rustics.PATH_TO_PROJECT + "/data",
    ejections_fname="output_N-10_ejections.txt",
    nprocs=128,
):
    """! A function to parallelize the integration of encounters."""

    # Iterate over the cmc models first, so their data only have to be loaded once each
    for ci, cmc_fname in enumerate(cmc_fnames):
        print("[%d/%d] %s" % (ci+1, len(cmc_fnames), cmc_fname))

        ##############################
        ###   Load CMC ejections   ###
        ##############################
        # Make path to and load CMC ejections
        path = "/".join((cmc_path, cmc_fname, ejections_fname))
        cmc = rustics.EjectionDf(path)
        cmc.convert_from_fewbody()
        cmc.df["vout"] = cmc.calc_vout()
        #cmc.df = cmc.df[-100:]

        ##############################
        ###    Loop through GCs    ###
        ##############################
        # Preliminary things
        # NOTE: There are three time coordiate systems:
        #       ts: ranges from present (t=0) to beginning of universe (t=-t_int)
        #       cmc.df.time: ranges from beginning of universe (time=0) to present (time=t_int); t_ejs is in this convention
        #       tfes: time from ejection, ranges from time of ejection (tfe=0) to t_int past that (tfe=t_int)
        t0 = 0.0
        t_int = 14000.0
        Nt = 10000
        ts = np.linspace(t0, -t_int, Nt) * u.Myr
        t_ejs = (cmc.df.time.to_numpy() - t_int) * u.Myr

        # Loop.  galpy has native parallelization (i.e. can integrate many GC orbits at once), but there aren't too many GCs/model, so the parallelization is only used when integrating the ejection orbits
        for cluster, gc in mwgc_catalog.df[mwgc_catalog.df.fname == cmc_fname].iterrows():

            # Extract the orbital parameters, adding units
            print("\t%s" % cluster)
            sci_gc = {
                "ra": gc.RA * u.deg,
                "dec": gc.DEC * u.deg,
                "distance": gc.Rsun * u.kpc,
                "pm_ra_cosdec": gc.mualpha * u.mas / u.yr,
                "pm_dec": gc.mudelta * u.mas / u.yr,
                "radial_velocity": gc["<RV>"] * u.km / u.s,
            }
            sci_gc = {
                "x": gc.X * u.kpc,
                "y": gc.Y * u.kpc,
                "z": gc.Z * u.kpc,
                "v_x": gc.U * u.km / u.s,
                "v_y": gc.V * u.km / u.s,
                "v_z": gc.W * u.km / u.s,
            }
            # NOTE: All SkyCoord objects must be initialized with this gc_frame
            gc_frame = Galactocentric
            sci_gc = SkyCoord(frame=gc_frame, **sci_gc)
            o_gc = Orbit(sci_gc)

            # Integrate GC orbit, and save
            o_gc.integrate(ts, MWPotential2014, method="dop853_c", progressbar=False)
            path = "/".join((cmc_path, "mwgcs", cluster, "gc_orbit.dat"))
            os.makedirs("/".join((path.split("/")[:-1])), exist_ok=True)
            pd.DataFrame(
                {
                    "t": ts,
                    "X": o_gc.x(ts),
                    "Y": o_gc.y(ts),
                    "Z": o_gc.z(ts),
                    "U": o_gc.vx(ts),
                    "V": o_gc.vy(ts),
                    "W": o_gc.vz(ts),
                },
            ).to_csv(path, index=False)

            # Parallelize over ejected objects
            if nprocs == 1:
                output = parmap.starmap(
                    _integrate_orbit,
                    [item for item in cmc.df.iterrows()],
                    o_gc,
                    float(gc.t),
                    pm_parallel=False,
                    pm_pbar=True,
                )
            else:
                pool = mp.Pool(nprocs)
                output = parmap.starmap(
                    _integrate_orbit,
                    [item for item in cmc.df.iterrows()],
                    o_gc,
                    float(gc.t),
                    pm_pool=pool,
                    pm_pbar=True,
                )
            # Save final coordinates by adding to ejections df and saving in folder for GC
            output = np.array(output)
            cmc.df["X"] = output[:,0]
            cmc.df["Y"] = output[:,1]
            cmc.df["Z"] = output[:,2]
            cmc.df["U"] = output[:,3]
            cmc.df["V"] = output[:,4]
            cmc.df["W"] = output[:,5]
            path = "/".join((cmc_path, "mwgcs", cluster, ejections_fname))
            cmc.df.to_csv(path)

def _integrate_orbit(eri, ej_row, o_gc, gc_age, gc_frame=Galactocentric, potential=MWPotential2014, Nt=10000):
    """! Function used to parallelize orbit integrations.
    
    @param  eri         Index of ejections row (from df.iterrows())
    @param  ej_row      Row of ejections df
    @param  o_gc        Glob. cluster orbit
    @param  gc_age      GC age.  Here, it is calculated by matching up CMC timesteps to the current shape of the GC.
    @param  potential   Potential to use for integration.
    """

    # Convert times
    tf_ej = (gc_age - ej_row.time) * u.Myr # Time from ejection to present day (if gc_age > ej_row.time, then the ejection occurs before the cluster evolves to the present state, and it is a relevant object
    ti_ej_gc = -tf_ej # Time of ejection in (-14000,0) scale used when integrating gc orbit
    # Return -1s if the ejection occured after gc_age
    if tf_ej < 0:
        return [-1] * 6

    # Generate random angles; note that the ejection velocity is used to seed the rng
    rng = np.random.RandomState(int(ej_row.vout * 1e4))
    theta = np.pi * (1.0 - 2.0 * rng.uniform())
    phi = 2.0 * np.pi * rng.uniform()

    # Add ejection velocities (transforming to galactocentric frame)
    sci_ej = SkyCoord(
        frame=gc_frame,
        x=o_gc.x(ti_ej_gc) * u.kpc,
        y=o_gc.y(ti_ej_gc) * u.kpc,
        z=o_gc.z(ti_ej_gc) * u.kpc,
        v_x=(o_gc.vx(ti_ej_gc) + ej_row.vout * np.cos(phi) * np.sin(theta)) * u.km / u.s,
        v_y=(o_gc.vy(ti_ej_gc) + ej_row.vout * np.sin(phi) * np.sin(theta)) * u.km / u.s,
        v_z=(o_gc.vz(ti_ej_gc) + ej_row.vout * np.cos(theta)) * u.km / u.s,
    )

    # Initialize orbit
    o = Orbit(sci_ej)

    # Define integration times, and integrate 
    # This might be buggy, as Nt is the same regardless of rf_ej
    tfes = np.linspace(0. * u.Myr, tf_ej, Nt)
    o.integrate(tfes, potential, method="dop853_c")

    # Return final values as list 
    return [
        o.x(tf_ej),
        o.y(tf_ej),
        o.z(tf_ej),
        o.vx(tf_ej),
        o.vy(tf_ej),
        o.vz(tf_ej),
    ]


###############################################################################

# Load matched MW GCs
cat = ag.GCCatalog(
    rustics.PATH_TO_PROJECT + "/matching_with_mywheels/gcs-cmc.dat",
    pd_kwargs={"index_col": "Cluster"},
)
cat.df.fname = [n.replace("_v2", "").replace(".tar.gz", "") for n in cat.df.fname]

# Load orbital parameters
cat_orbit = ag.GCCatalog(
    rustics.PATH_TO_PROJECT + "/baumgardt_orbits_table_clean.txt",
    pd_kwargs={
        "index_col": "Cluster",
        "delim_whitespace": True,
        "usecols": [
            "Cluster",
            "RA",
            "DEC",
            "Rsun",
            "mualpha",
            "mudelta",
            "<RV>",
            "RPERI",
            "RAPO",
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
cat.df = cat.df.join(cat_orbit.df)

# Integrate the orbits
integrate_cmc_ejections(cat.df.fname.unique(), cat, nprocs=56)
