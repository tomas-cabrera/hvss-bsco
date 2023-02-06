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
    cmc_fname,
    mwgc_catalog,
    cmc_path="/hildafs/projects/phy200025p/tcabrera/hvss/data",
    ejections_fname="output_N-10_ejections.txt",
):
    """! A function to parallelize the integration of encounters, s.t. the CMC data only have to be loaded once."""

    print(cmc_fname)

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

        ## Generate orbits for ejected objects
        # Get the SkyCoords at the necessary times
        # "tejs" is the list of times of ejections, with tej=0 as the present
        sci_ejs = o_gc.SkyCoord(t_ejs)

        # Generate random angles; note that the %.6f Baumgardt RA is used to seed the rng
        rng = np.random.RandomState(int(gc.RA * 1e6))
        theta = np.pi * (1.0 - 2.0 * rng.uniform(size=cmc.df.shape[0]))
        phi = 2.0 * np.pi * rng.uniform(size=cmc.df.shape[0])

        # Add ejection velocities (transforming to galactocentric frame)
        sci_ejs = sci_ejs.transform_to(gc_frame)
        sci_ejs = SkyCoord(
            frame=gc_frame,
            x=sci_ejs.x,
            y=sci_ejs.y,
            z=sci_ejs.z,
            v_x=sci_ejs.v_x
            + (cmc.df.vout * np.cos(phi) * np.sin(theta)).to_numpy() * u.km / u.s,
            v_y=sci_ejs.v_y
            + (cmc.df.vout * np.sin(phi) * np.sin(theta)).to_numpy() * u.km / u.s,
            v_z=sci_ejs.v_z + (cmc.df.vout * np.cos(theta)).to_numpy() * u.km / u.s,
        )

        # The first option here is faster by >2x, despite not using galpy parallelization
        ####################################### 
        # Integrate orbits, in loop with coordinate calling 
        xs = []
        ys = []
        zs = []
        vxs = []
        vys = []
        vzs = []
        # Note that tfes goes from 0 to 14000, i.e. "tfe" is the time from ejection
        for ti, t in tqdm(enumerate((t_int - cmc.df.time).to_numpy())):
            # Initialize orbit
            o = Orbit(sci_ejs[ti])

            # Define integration times, and integrate 
            tfes = np.linspace(t0, t, Nt) * u.Myr
            o.integrate(tfes, MWPotential2014, method="dop853_c")

            # Save final values to lists
            xs.append(o.x(t * u.Myr))
            ys.append(o.y(t * u.Myr))
            zs.append(o.z(t * u.Myr))
            vxs.append(o.vx(t * u.Myr))
            vys.append(o.vy(t * u.Myr))
            vzs.append(o.vz(t * u.Myr))
        ####################################### 
        ### Integrate orbits with galpy parallelization, then call coordinates individually
        ## Initialize ejection orbits
        #o_ejs = Orbit(sci_ejs)

        ## Integrate orbits
        ## Note that tfes goes from 0 to 14000, i.e. "tfe" is the time from ejection
        #tfes = np.linspace(t0, t_int, Nt) * u.Myr
        #o_ejs.integrate(tfes, MWPotential2014, method="dop853_c")
        #o_ejs.plot(d1="x", d2="y", lw=0.5, alpha=0.5)
        #o_gc.plot(d1="x", d2="y", lw=0.5, c="k", overplot=True)
        #plt.axis("square")
        #plt.show()

        ## Get present-time coordinates (i.e. when tfe=t_int-cmc.df.time)
        #from time import time
        #ttest = time()
        #xs = []
        #ys = []
        #zs = []
        #vxs = []
        #vys = []
        #vzs = []
        #for ti, t in enumerate((t_int - cmc.df.time).to_numpy() * u.Myr):
        #    xs.append(o_ejs.x(t)[ti])
        #    ys.append(o_ejs.y(t)[ti])
        #    zs.append(o_ejs.z(t)[ti])
        #    vxs.append(o_ejs.vx(t)[ti])
        #    vys.append(o_ejs.vy(t)[ti])
        #    vzs.append(o_ejs.vz(t)[ti])
        ####################################### 

        # Save final coordinates by adding to ejections df and saving in folder for GC
        print(cmc.df.columns)
        cmc.df["X"] = xs
        cmc.df["Y"] = ys
        cmc.df["Z"] = zs
        cmc.df["U"] = vxs
        cmc.df["V"] = vys
        cmc.df["W"] = vzs
        path = "/".join((cmc_path, "mwgcs", cluster, ejections_fname))
        cmc.df.to_csv(path)
        
        ## This converts the coordinates into a SkyCoord object, if needed
        #scf_ejs = SkyCoord(
        #    frame=gc_frame,
        #    x=xs * u.kpc,
        #    y=ys * u.kpc,
        #    z=zs * u.kpc,
        #    v_x=vxs * u.km / u.s,
        #    v_y=vys * u.km / u.s,
        #    v_z=vzs * u.km / u.s,
        #)


###############################################################################

# Load matched MW GCs
cat = ag.GCCatalog(
    "/hildafs/projects/phy200025p/tcabrera/hvss/matching_with_mywheels/gcs-cmc.dat",
    pd_kwargs={"index_col": "Cluster"},
)
cat.df.fname = [n.replace("_v2", "").replace(".tar.gz", "") for n in cat.df.fname]

# Load orbital parameters
cat_orbit = ag.GCCatalog(
    "/hildafs/projects/phy200025p/tcabrera/hvss/baumgardt_orbits_table_clean.txt",
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

# Iterate over CMC models
nprocs = 128
if nprocs == 1:
    parmap.map(
        integrate_cmc_ejections,
        cat.df.fname.unique(),
        cat,
        pm_parallel=False,
        pm_pbar=True,
    )
else:
    pool = mp.Pool(nprocs)
    parmap.map(
        integrate_cmc_ejections,
        cat.df.fname.unique(),
        cat,
        pm_pool=pool,
        pm_pbar=True,
    )
