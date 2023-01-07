import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic, Galactocentric, ICRS
from astropy.io import fits
import healpy as hp

import paths
import rustics

###############################################################################


def hist2d_hpx_to_mollweide(
    axes,
    m,
    nest=False,
    xsize=1000,
    longitude_grid_spacing=60,
    latitude_grid_spacing=30,
    graticule_labels=True,
    kw_rotator={},
    kw_pcolormesh={},
):
    """! Add a HEALPix map as a 2D histogram to an existing Mollweide axes.
    Largely inspired by hp.newvisufunc.projview.

    axes: axes to add histogram to

    m: HEALPix map

    xsize: number of pixels to use in longitude space

    kw_rotator: dict, keyword args to build rotator for HEALPix map

    kw_pcolormesh: keywords args for pcolormesh call
    """

    # Make the theta, phi meshgrid
    ysize = xsize // 2
    phi = np.linspace(-np.pi, np.pi, xsize)
    theta = np.linspace(np.pi, 0, ysize)
    phi, theta = np.meshgrid(phi, theta)

    # Rotate meshgrid if needed
    if kw_rotator:
        r = hp.rotator.Rotator(**kw_rotator)
        theta, phi = r(theta.flatten(), phi.flatten())
        theta = theta.reshape(ysize, xsize)
        phi = phi.reshape(ysize, xsize)

    # Get map values at grid points
    nside = hp.pixelfunc.npix2nside(len(m))
    grid_pix = hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)
    grid_map = m[grid_pix]

    # Get longitude and latitude
    longitude = np.linspace(-np.pi, np.pi, xsize)
    latitude = np.linspace(-np.pi / 2, np.pi / 2, ysize)

    # Plot map
    ret = axes.pcolormesh(
        # phi,
        # theta,
        longitude,
        latitude,
        grid_map,
        **kw_pcolormesh,
    )

    # Add longitude/latitude grids
    if (longitude_grid_spacing != None) | (latitude_grid_spacing != None):
        axes.grid()
    if longitude_grid_spacing != None:
        axes.set_longitude_grid(longitude_grid_spacing)
        axes.set_longitude_grid_ends(90)
    if latitude_grid_spacing != None:
        axes.set_latitude_grid(latitude_grid_spacing)

    # Remove grid labels
    if not graticule_labels:
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])

    return ret


###############################################################################


# Get ejections data
gc = "NGC_104"
gc = "NGC_5139"
gc = "NGC_6205"
gc = "NGC_7089"
gc = "Pyxis"
gc = "E_3"
gc_path = paths.data_mwgcs / gc
ejdf = rustics.EjectionDf(
    gc_path / "output_N-10_ejections.txt",
    usecols=["time", "mf", "X", "Y", "Z", "U", "V", "W"],
)
ejdf.df = ejdf.df[(ejdf.df.X != -1) & (ejdf.df.Y != -1) & (ejdf.df.mf != -100)]
# Downsample by 1/10
# ejdf.df = ejdf.df.sample(frac=0.1)

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

# kwargs for ejecta scatterpoints
mmax = ejdf.df.mf.max()
kw_ej = {
    "c": ejdf.df.mf / mmax,
    "cmap": "inferno",
    "marker": ".",
    "s": 4,
    "lw": 0,
    "c": "r",
    "rasterized": True,
}

##############################
###       Positions        ###
##############################

#  Get data/header
ml, h = hp.fitsfunc.read_map(gc_path / "hp_probs.fits", h=True, field=None)
# visufunc.mollview and newvisufunc.projview do the same thing;
#   the latter just allows the graticule and labels to be plotted (~ticklabels)
# The plot appears reflected across ra=0 when compared to the matplotlib mollweide projection,
#   but the graticule labels reveal that the two functions use different axes conventions
#   i.e. the two are the same.
# The default coordinates are celestial (equatorial),
#   so to properly plot a transformation from celestial to galactic is needed

# Set up figure
fig, axd = plt.subplot_mosaic(
    [["hist", "cred"]],
    subplot_kw={
        "projection": "mollweide",
    },
    figsize=(6, 2),
)
for ax in axd.values():
    ax.grid(True, c="gray", alpha=0.5, zorder=0, lw=0.5)

# Plot map as 2D hist
ax = axd["hist"]
kw_rotator = {"coord": "GC"}
hist2d_hpx_to_mollweide(
    ax,
    np.log(ml[0]),
    kw_pcolormesh={
        "rasterized": True,
        "cmap": "inferno_r",
    },
    kw_rotator=kw_rotator,
)

# Plot credible regions as 2D "tophat" hists
ax = axd["cred"]
kw_pcolormesh = {
    "cmap": "Blues",
    "alpha": 0.8,
    "rasterized": True,
    "norm": "log",
    "vmin": 0.1,
}
hist2d_hpx_to_mollweide(
    ax,
    ml[1],
    kw_pcolormesh=kw_pcolormesh,
    kw_rotator=kw_rotator,
)
kw_pcolormesh["alpha"] = 0.5
hist2d_hpx_to_mollweide(
    ax,
    (ml[1] + ml[2]) % 2,
    kw_pcolormesh=kw_pcolormesh,
    kw_rotator=kw_rotator,
)
# Here's the code for adding the ejecta as a scatter plot
# The plot looks a bit messy, but is the best way to verify the the maps are oriented correctly
# ax.scatter(
#     scej.l.wrap_at("180d").radian,
#     scej.b.radian,
#     **kw_ej,
# )
# sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=mmax))
# axcb = plt.colorbar(mappable=sm, ax=ax, orientation="horizontal")
# axcb.set_label(r"$m~(M_\odot)$")

# Clean and show
fig.suptitle(gc)
plt.tight_layout()
plt.savefig(paths.figures / __file__.split("/")[-1].replace(".py", "_x.pdf"))
plt.close()

##############################
###     Proper Motions     ###
##############################

# Set up figure
fig, axd = plt.subplot_mosaic(
    [["hist", "cred"]],
    sharex=True,
    sharey=True,
    gridspec_kw={
        "wspace": 0,
    },
    figsize=(6, 4),
)

# Transform to mu_ra_cosdec-mu_dec
scej = scej.transform_to(ICRS)

# Get histogram map
hdul = fits.open(gc_path / "pm_probs.fits")
# Generate meshgrid with binedges
h = hdul[0].header
xbinedges = np.linspace(h["PMRCDMIN"], h["PMRCDMAX"], h["PMNUM"])
ybinedges = np.linspace(h["PMDMIN"], h["PMDMAX"], h["PMNUM"])
y, x = np.meshgrid(ybinedges[:-1], xbinedges[:-1])
x = x.flatten()
y = y.flatten()

# Plot map as 2D hist
ax = axd["hist"]
d = hdul[0].data
ax.hist2d(
    x,
    y,
    bins=[xbinedges, ybinedges],
    weights=np.log(d.flatten()),
    cmap="inferno_r",
    rasterized=True,
)
ax.set_xlabel(r"$\mu_\alpha \cos \delta~[{\rm mas/yr}]$")
ax.set_ylabel(r"$\mu_{\delta}~[{\rm mas/yr}]$")
ax.set_aspect("equal")

# Plot credible regions
ax = axd["cred"]
kw_hist2d = {
    "cmap": "Blues_r",
    "alpha": 0.8,
    "rasterized": True,
}
d = hdul[1].data
ax.hist2d(
    x,
    y,
    bins=[xbinedges, ybinedges],
    weights=np.log(d.flatten()),
    **kw_hist2d,
)
kw_hist2d["alpha"] = 0.5
d = (d + hdul[2].data) % 2
ax.hist2d(
    x,
    y,
    bins=[xbinedges, ybinedges],
    weights=np.log(d.flatten()),
    **kw_hist2d,
)
# Ejected objects, if desired
# ax.scatter(
#     scej.pm_ra_cosdec,
#     scej.pm_dec,
#     **kw_ej,
# )
ax.set_xlim((-30, 30))
ax.set_ylim((-45, 15))
ax.set_aspect("equal")
ax.set_xlabel(r"$\mu_\alpha \cos \delta~[{\rm mas/yr}]$")

# Clean and show
fig.suptitle(gc)
plt.tight_layout()
plt.savefig(paths.figures / __file__.split("/")[-1].replace(".py", "_v.pdf"))
plt.close()
