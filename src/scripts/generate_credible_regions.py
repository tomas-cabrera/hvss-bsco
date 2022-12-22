import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import parmap
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
from astropy.io import fits
import healpy as hp

import paths
import rustics

###############################################################################


def calc_credible_indices(probs, intervals):
    """! Return boolean masks that compose the specified intervals.

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

    # Get cumulative probabilities, and find indices of intervals
    probs_cumsum = np.cumsum(probs_sorted)
    ss_inds = np.searchsorted(probs_cumsum, intervals, side="right")

    # Generate boolean masks for credible regions
    masks = np.full((len(intervals), probs.shape[0]), 0)
    for ii, i in enumerate(ss_inds):
        masks[ii, argsort_inds[:i]] = 1

    return masks


def calc_credible_indices_2d(probs, intervals):
    """! Return boolean masks that compose the specified intervals, for a 2D array.

    @type   probs:      np.array
    @param  probs:      Array of probabilities

    @type   intervals:  list
    @param  intervals:  List of credible regions to calculate.

    @rtype:             np.array
    @return:            Boolean array that can be used to mask probabilities to credible region.

    """

    # Get sort indices, sort probabilities
    argsort_inds = np.unravel_index(np.argsort(probs, axis=None)[::-1], probs.shape)
    probs_sorted = probs[argsort_inds]

    # Get cumulative probabilities, and find indices of intervals
    probs_cumsum = np.cumsum(probs_sorted)
    ss_inds = np.searchsorted(probs_cumsum, intervals, side="right")

    # Generate boolean masks for credible regions
    masks = np.full((len(intervals), *probs.shape), 0)
    for ii, i in enumerate(ss_inds):
        masks[ii, (argsort_inds[0][:i], argsort_inds[1][:i])] = 1

    return masks


def calc_ej_credible_regions(
    gc_path,
    intervals=[0.5, 0.9],
    ej_fname="output_N-10_ejections.txt",
    hp_level=5,
    hp_fname="hp_probs.fits",
    pm_fname="pm_probs.fits",
    overwrite=True,
    kw_writemap={
        "nest": True,
    },
):
    """! Calculate the credible regions for the ejections from one GC.

    @type   gc_path:    PosixPath
    @param  gc_path:    Path to globular cluster directory.

    @type   intervals:  list
    @param  intervals:  List of floats of credible regions to calculate.

    @type   ej_fname:   string
    @param  ej_fname:   Filename of ejections data.

    @type   hp_level:   int
    @param  hp_level:   Level/order of healpix map to use for spatial coordinates.

    @type   hp_fname:   string
    @param  hp_fname:   Filename to use for HEALPix counts.  If None, then skips generating the HEALPix histograms.

    @type   pm_fname:   string
    @param  pm_fname:   Filename to use for proper motion countsi.  If None, then skips generating the histograms.

    @type   kw_writemap:   dict
    @param  kw_writemap:   Keyword arguments for hp.fitsfunc.write_map.

    @rtype:             None
    @return:            No return, just saves the file.

    """

    # Check if at least 1 fname has been specified,
    #   and also if desired files exist and should not be overwritten
    want = False
    missing = False
    if hp_fname != None:
        want = True
        hp_path = gc_path / hp_fname
        if not os.path.exists(hp_path):
            missing = True
    if pm_fname != None:
        want = True
        pm_path = gc_path / pm_fname
        if not os.path.exists(pm_path):
            missing = True
    if not want:
        print("No fnames specified.")
        return 1
    if (not missing) & (not overwrite):
        print("Files already exist; use 'overwrite=True' to generate anyway.")
        return 1

    # Choose columns to load
    usecols = ["X", "Y", "Z"]
    if pm_fname != None:
        usecols += ["U", "V", "W"]

    # Load ejection data; cut out problematic points
    ejdf = rustics.EjectionDf(
        gc_path / ej_fname,
        usecols=usecols,
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
    if hp_fname != None:
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
        hp.fitsfunc.write_map(
            hp_path,
            output,
            column_names=["PROB"] + ["CR%d" % (i * 100) for i in intervals],
            dtype=[float] + [bool] * len(intervals),
            extra_header=[
                ("NEJECT", ejdf.df.shape[0], "Total number of ejected objects"),
            ],
            overwrite=overwrite,
            **kw_writemap,
        )

    #####################
    ### Prop. motions ###
    #####################
    # pm_dec and pm_ra_cosdec extrema (mas / yr):
    #   -610.551 <=    pm_dec    <= 154.390
    #   -430.575 <= pm_ra_cosdec <= 220.048
    # Generally, >99.9% of the ejecta have pm_dec in [-45, 15] and pm_ra_cosdec in [-30, 30]
    #   In the worst cases, >98% are within this cut
    if pm_fname != None:
        # Generate 2D histogram
        pmdmin = -45
        pmdmax = 15
        pmrcdmin = -30
        pmrcdmax = 30
        pmnum = 121
        xbinedges = np.linspace(pmrcdmin, pmrcdmax, pmnum)
        ybinedges = np.linspace(pmdmin, pmdmax, pmnum)
        pm_counts, _, _ = np.histogram2d(
            scej.pm_ra_cosdec.value,
            scej.pm_dec.value,
            bins=[xbinedges, ybinedges],
        )
        probs = pm_counts / ejdf.df.shape[0]

        # Calculate additional header info
        coverage = probs.sum()

        # Calculate credible regions
        credible_indices = calc_credible_indices_2d(probs, intervals)

        # Compose HDUList and write
        probs_header = fits.Header(
            [
                ("NEJECT", ejdf.df.shape[0], "Total number of ejected objects"),
                (
                    "COVERAGE",
                    coverage,
                    "Fraction of objects within the histogram domain",
                ),
                ("PMDMIN", pmdmin, "Lower bound of pm_dec cut [mas/yr]"),
                ("PMDMAX", pmdmax, "Upper bound of pm_dec cut [mas/yr]"),
                ("PMRCDMIN", pmrcdmin, "Lower bound of pm_ra_cosdec cut [mas/yr]"),
                ("PMRCDMAX", pmrcdmax, "Upper bound of pm_ra_cosdec cut [mas/yr]"),
                (
                    "PMNUM",
                    pmnum,
                    "Number of bins used along each dimension + 1",
                ),
            ]
        )
        probs_hdu = fits.PrimaryHDU(probs, header=probs_header)
        ci_hdus = [fits.ImageHDU(ci) for ci in credible_indices]
        hdul = fits.HDUList([probs_hdu, *ci_hdus])
        hdul.writeto(pm_path, overwrite=overwrite)

    return 0


def get_ej_credible_regions(
    gc_path,
    hp_fname="hp_probs.fits",
    pm_fname="pm_probs.fits",
    generate=False,
    kw_cecr={},
    kw_readmap={
        "h": True,
        "field": None,
    },
):
    """! Get the credible regions for the ejections from one GC.
    If the regions are not pre-calculated, then generate them.

    @type   gc_path:    PosixPath
    @param  gc_path:    Path to globular cluster directory.

    @type   hp_fname:   string
    @param  hp_fname:   Filename to use for HEALPix counts.  If None, then doesn't look for the HEALPix histograms.

    @type   pm_fname:   string
    @param  pm_fname:   Filename to use for proper motion counts.  If None, then doesn't look for the histograms.

    @type   generate:   bool
    @param  generate:   If true, then generate files regarless if they are present or not.

    @type   kw_cecr:    dict
    @param  kw_cecr:    Keyword arguments to pass to calc_ej_credible_regions.

    @rtype:             list
    @return:            List of results.
                        Each item in the list corresponds to one of the histograms,
                        and is a tuple of the form (header(list), data(ndarray)),
                        with the appropriate fields not included if specified.

    """

    # Generate any missing histograms
    if hp_fname != None:
        hp_path = gc_path / hp_fname
        if not os.path.exists(hp_path):
            kw_cecr["hp_fname"] = hp_fname
    if pm_fname != None:
        pm_path = gc_path / pm_fname
        if not os.path.exists(pm_path):
            kw_cecr["pm_fname"] = pm_fname
    if generate | ((hp_fname != None) | (pm_fname != None)):
        calc_ej_credible_regions(gc_path, **kw_cecr)

    # Initialize output
    output = []

    # Load HEALPix histogram if named
    if hp_fname != None:
        temp = hp.fitsfunc.read_map(hp_path, **kw_readmap)
        output.append(temp)

    # Load proper motion histogram if named
    # TODO: figure out what to do about fits.open remaining open.
    if pm_fname != None:
        temp = fits.open(pm_path)
        output.append(temp)

    return output


###############################################################################


# Calculate credible regions for all GCs
mp_input = [paths.data_mwgcs / gc_name for gc_name in os.listdir(paths.data_mwgcs)]
nprocs = None
if nprocs == 1:
    parmap.map(
        calc_ej_credible_regions,
        mp_input,
        pm_parallel=False,
        pm_pbar=True,
    )
else:
    pool = mp.Pool(nprocs)
    parmap.map(
        calc_ej_credible_regions,
        mp_input,
        pm_pool=pool,
        pm_pbar=True,
    )
