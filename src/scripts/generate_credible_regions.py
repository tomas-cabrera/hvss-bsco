import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import parmap
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
from astropy import units as u
import healpy as hp

import paths
import rustics

###############################################################################


def calc_credible_indices(probs, intervals):
    """! Return an array of indices that compose the specified credible intervals.

    @type   probs:      np.array
    @param  probs:      Array of probabilities

    @type   intervals:  list
    @param  intervals:  List of credible regions to calculate.

    @rtype:             np.array
    @return:            Boolean array that can be used to mask probabilities to credible region.

    """

    # Get sort indices, sort probabilities
    argsort_inds = np.argsort(probs)[::-1]
    probs_sorted = probs[argsort_inds]
    print("probs_sorted:", probs_sorted)

    # Get cumulative probabilities, and find indices of intervals
    probs_cumsum = np.cumsum(probs_sorted)
    ss_inds = np.searchsorted(probs_cumsum, intervals, side="right")
    print("ss_inds:", ss_inds)

    # Generate boolean masks for credible regions
    masks = np.full((len(intervals), probs.shape[0]), False)
    for ii, i in enumerate(ss_inds):
        masks[ii, argsort_inds[:i]] = True
    print("masks:", masks)
    print("probs[masks[0,:]]:", probs[masks[0, :]])

    return masks


def calc_ej_credible_regions(
    gc_name,
    intervals=[0.5, 0.9],
    dir_path=paths.data_mwgcs,
    ej_fname="output_N-10_ejections.txt",
    hp_level=5,
    hp_fname="hp_probs.fits",
    pm_fname="pm_probs.fits",
    overwrite=True,
):
    """! Calculate the credible regions for the ejections from one GC.

    @type   gc_name:    string
    @param  gc_name:    Name of globular cluster

    @type   intervals:  list
    @param  intervals:  List of floats of credible regions to calculate.

    @type   dir_path:   string
    @param  dir_path:   path/to/directory containing GC folders

    @type   ej_fname:   string
    @param  ej_fname:   Filename of ejections data.

    @type   hp_level:   int
    @param  hp_level:   Level/order of healpix map to use for spatial coordinates.

    @type   hp_fname:   string
    @param  hp_fname:   Filename to use for HEALPix counts

    @type   pm_fname:   string
    @param  pm_fname:   Filename to use for proper motion counts

    @rtype:             None
    @return:            No return, just saves the file.

    """

    # Load ejection data; cut out problematic points
    path = dir_path / gc_name / ej_fname
    ejdf = rustics.EjectionDf(
        path,
        usecols=[
            "X",
            "Y",
            "Z",
            "U",
            "V",
            "W",
        ],
    )
    ejdf.df = ejdf.df[(ejdf.df.X != -1) & (ejdf.df.Y != -1)]

    # Convert data to SkyCoord object
    scej = SkyCoord(
        frame=Galactocentric,
        x=ejdf.df.X.to_numpy() * u.kpc,
        y=ejdf.df.Y.to_numpy() * u.kpc,
        z=ejdf.df.Z.to_numpy() * u.kpc,
        v_x=ejdf.df.U.to_numpy() * u.km / u.s,
        v_y=ejdf.df.V.to_numpy() * u.km / u.s,
        v_z=ejdf.df.W.to_numpy() * u.km / u.s,
    )
    scej = scej.transform_to(ICRS)

    #####################
    ###   Positions   ###
    #####################
    # HEALPix numbers
    nside = hp.pixelfunc.order2nside(hp_level)
    npix = hp.pixelfunc.nside2npix(nside)

    # Calculate HEALPixels for objects and count number of objects/hp
    hps = hp.pixelfunc.ang2pix(
        nside,
        scej.ra.deg,
        scej.dec.deg,
        nest=True,
        lonlat=True,
    )
    hp_probs = np.bincount(hps, minlength=npix)
    hp_probs = hp_probs / hp_probs.sum()

    # Get lists of credible indices
    credible_indices = calc_credible_indices(hp_probs, intervals)

    # Stack data and save to HEALPix fits file
    output = (hp_probs, *credible_indices)
    print(output)
    print("hp.maptype:", hp.pixelfunc.maptype(output))
    print("extra_header:")
    hp.fitsfunc.write_map(
        dir_path / gc_name / hp_fname,
        output,
        column_names=["PROB"] + ["CR%d" % (i * 100) for i in intervals],
        dtype=[float] + [bool] * len(intervals),
        nest=True,
        overwrite=overwrite,
        extra_header=[
            ("NEJECT", ejdf.df.shape[0], "Total number of ejected objects"),
        ],
    )

    #####################
    ### Prop. motions ###
    #####################

    ### TODO: Do this next, after checking if extra_header can be used at all


###############################################################################


calc_ej_credible_regions("Liller_1")
path = paths.data_mwgcs / "Liller_1/hp_probs.fits"
# This works, but returns it as a numpy array
m, h = hp.fitsfunc.read_map(path, h=True, field=None)
print(h)
print(m.shape)
plt.savefig(paths.figures / __file__.split("/")[-1].replace(".py", ".pdf"))
