# RUnaway STars In Cluster Simulations
###############################################################################

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, ICRS, Galactic, Galactocentric
from astropy import units as u
import multiprocessing as mp
import parmap
from tqdm import tqdm

import paths

###############################################################################


# Physical constants
Usun_gc = np.array([11.1, 247.24, 7.25])  # From Brown+18


# Unit conversions
YR_TO_S = 365.2425 * 24 * 3600
PC_TO_M = 3.0857e16


# Number of realizations per CMC encounter
N_REALZ = 10


# Filename of realization data
FILE_REALZ = "output_N-%d.txt" % N_REALZ


# File suffix for ejection data
SUFFIX_EJECTIONS = "_ejections"


# Filename of ejection data
FILE_EJECTIONS = (SUFFIX_EJECTIONS + ".").join(FILE_REALZ.rsplit(".", 1))


## Path to project
# PATH_TO_PROJECT = "/home/tomas/Documents/cmu/research/hvss"
#
#
## Path to data
# PATH_TO_DATA = "/".join((PATH_TO_PROJECT, "data"))
#
#
## Path to figures
# PATH_TO_FIGURES = "/".join((PATH_TO_PROJECT, "rustics/figures"))
#
#
## Path to CMC data
# PATH_TO_CMC_DATA = PATH_TO_DATA
#
#
## Path to matplotlib_style_file
# PATH_TO_MPLRC = "/home/tomas/Documents/cmu/research/matplotlib_style_file"


# Table of CMC catalog models, received from Kyle Kremer via. e-mail 07/27/2021 and supplemented with cmc_core_collapsed.py and gcs_mw-cmc.ipynb
# Also obtainable from Table A1 in Kremer et al. 2020 (catalog paper)
DF_MODELS = pd.read_csv(
    paths.data / "Catalog_models_ccw.dat",
)
# Old version, reads directly from Kyle's Catalog_models.dat
# DF_MODELS = pd.read_table(
#    "/".join((PATH_TO_PROJECT, "Catalog_models_ccw.dat")),
#    names=["N", "rv", "rg", "Z", "N_BH", "status"],
#    skiprows=2,
#    delimiter=" ",
# )
def _ns_to_str(N):
    """
    Converts N from 10**5 value to string format.
    """
    return "{:.1e}".format(N).replace(".0", "").replace("+0", "")


DF_MODELS["fname"] = [
    "N{}_".format(_ns_to_str(DF_MODELS.loc[i, "N"] * 100000))
    + "rv{:n}_".format(DF_MODELS.loc[i, "rv"])
    + "rg{:n}_".format(DF_MODELS.loc[i, "rg"])
    + "Z{}".format(DF_MODELS.loc[i, "Z"])
    for i in DF_MODELS.index
]
# List of model names that are a second version in the CMC catalog
MODEL_NAMES_V2 = [
    "N1.6e6_rv0.5_rg2_Z0.0002",
    "N4e5_rv0.5_rg2_Z0.02",
    "N2e5_rv0.5_rg8_Z0.0002",
    "N4e5_rv0.5_rg8_Z0.0002",
    "N1.6e6_rv0.5_rg8_Z0.0002",
    "N2e5_rv0.5_rg8_Z0.002",
    "N2e5_rv0.5_rg8_Z0.02",
    "N4e5_rv0.5_rg8_Z0.02",
    "N1.6e6_rv0.5_rg20_Z0.0002",
    "N2e5_rv0.5_rg20_Z0.002",
]

# dict of sample CMC models, with values as labels
SAMPLE_MODELS = {
    "N8e5_rv0.5_rg8_Z0.0002": r"$(N, r_{\rm vir,pc}, Z_{Z_\odot}) = (8{\rm e}5, 0.5, 0.01)$",
    "N4e5_rv0.5_rg8_Z0.0002": r"$N = 4{\rm e}5$",
    "N8e5_rv2_rg8_Z0.0002": r"$r_{\rm vir,pc} = 2$",
    "N8e5_rv0.5_rg8_Z0.02": r"$Z_{\rm Z_\odot} = 1$",
}

# List of sample GCs to use for orbit, mu-mu plots
SAMPLE_GCS = [
    "NGC_104",
    "NGC_5139",
    "NGC_6205",
    "NGC_7089",
]

# Headers for realization dataframes
HEADERS_REALZ = [
    "time",
    "b",
    "vinf",
    "a1",
    "e1",
    "vesc",
    "m10",
    "m11",
    "m0",
    "r10",
    "r11",
    "r0",
    "k10",
    "k11",
    "k0",
    "s",
    "type_i",
    "type_f",
    "v_crit",
    "a_fin",
    "e_fin",
    "Lx",
    "Ly",
    "Lz",
    "Lbinx",
    "Lbiny",
    "Lbinz",
    "Ei",
    "DeltaEfrac",
    "vfin0",
    "kf0",
    "Rmin0",
    "Rmin_j0",
    "vfin1",
    "kf1",
    "Rmin1",
    "Rmin_j1",
    "vfin2",
    "kf2",
    "Rmin2",
    "Rmin_j2",
]


# Headers for ejection dataframes
HEADERS_EJECTIONS = [
    "time",
    "b",
    "vinf",
    "a1",
    "e1",
    "vesc",
    "m10",
    "m11",
    "m0",
    "r10",
    "r11",
    "r0",
    "k10",
    "k11",
    "k0",
    "s",
    "type_i",
    "type_f",
    "v_crit",
    "a_fin",
    "e_fin",
    "Ei",
    "DeltaEfrac",
    "vfin",
    "Rmin",
    "Rmin_j",
    "vout",
    "mf",
    "rf",
    "kf",
]


# Dictionary for turning header names into plot labels
HEADERS_TO_LABELS = {
    "time": "t [Gyr]",
    "b": r"$b$",
    "vinf": r"$v_\infty$",
    "a1": r"$a_i$",
    "e1": r"$e_i$",
    "vesc": r"$v_{\rm esc}~[{\rm km~s}^{-1}]$",
    "m10": r"$m$",
    "m11": r"$m$",
    "m0": r"$m$",
    "r10": r"$r$",
    "r11": r"$r$",
    "r0": r"$r$",
    "k10": r"$k$",
    "k11": r"$k$",
    "k0": r"$k$",
    "s": "seed",
    "type_i": r"BST$_i$",
    "type_f": r"BST$_f$",
    "v_crit": r"$v_{\rm crit}$",
    "a_fin": r"$a_f$",
    "e_fin": r"$e_f$",
    "Lx": r"$L_x$",
    "Ly": r"$L_y$",
    "Lz": r"$L_z$",
    "Lbinx": r"$L_{{\rm bin},x}$",
    "Lbiny": r"$L_{{\rm bin},y}$",
    "Lbinz": r"$L_{{\rm bin},z}$",
    "Ei": r"$E_i$",
    "DeltaEfrac": r"$\Delta E_{\rm frac}$",
    "vfin0": r"$$",
    "kf0": r"$$",
    "Rmin0": r"$$",
    "Rmin_j0": r"$$",
    "vfin1": r"$$",
    "kf1": r"$$",
    "Rmin1": r"$$",
    "Rmin_j1": r"$$",
    "vfin2": r"$$",
    "kf2": r"$$",
    "Rmin2": r"$$",
    "Rmin_j2": r"$$",
    "L": r"$L$",
    "Lbin": r"$L_{\rm bin}$",
    "cos_theta": r"$\cos(\theta)$",
    "mbhx": r"$M_{\rm BH,max}$",
    "vout0": r"$$",
    "vout1": r"$$",
    "vout2": r"$$",
    "vfin": r"$v_{\rm fin}$",
    "kf": r"$k_f$",
    "mf": r"$m (M_\odot)$",
    "Rmin": r"$R_{\rm min} [{\rm AU}]$",
    "Rmin_j": r"$j_{\rm rmin}$",
    "vout": r"$v_{\rm out} [{\rm km~s}^{-1}]$",
    "N": r"$N [10^5]$",
    "rv": r"$r_{\rm vir} [{\rm pc}]$",
    "Z": r"$Z$",
    "dist": r"$d [{\rm kpc}]$",
    "vesc/rho_0": r"$v_{\rm esc}/\rho_0 [{\rm km~s^{-1}/something}]$",
    "v_radial": r"$v_r [{\rm km~s}^{-1}]$",
    "vlos": r"$v_{\rm los} [{\rm km~s}^{-1}]$",
    "mx": r"$m_{\rm max} (M_\odot)$",
    "M": r"$M~[M_\odot]$",
    "rc_spitzer/r_h": r"$r_{\rm core} / r_{h,m}$",
    "[Fe/H]": "[Fe/H]",
    "M_norm": r"$\big[\log_{10} M\big]_{\rm norm}$",
    "rc_spitzer/r_h_norm": r"$\big[r_{\rm core} / r_{h,m}\big]_{\rm norm}$",
    "[Fe/H]_norm": r"$\big[[{\rm Fe/H}]\big]_{\rm norm}$",
    "rgc": r"$r_{\rm gc}~[{\rm kpc}]$",
    "Zgc": r"$Z~[{\rm kpc}]$",
    "R_GC": r"$r_{\rm gc}~[{\rm kpc}]$",
    "rc/rh": r"$r_c / r_h$",
    "logM": r"$\log(M~[M_\odot])$",
    "vtoday": r"$v_{\rm today} [{\rm km~s}^{-1}]$",
}


# Dataframe of type_i info
INFO_TYPES_I = pd.DataFrame(
    [
        ["Placeholder", "k"],
        ["(C,C)+S", "xkcd:violet"],
        ["(S,C)+C", "xkcd:azure"],
        ["(S,C)+S", "xkcd:goldenrod"],
        ["(S,S)+C", "xkcd:orangered"],
    ],
    columns=["label", "color"],
)


# Dataframe of type_f info
INFO_TYPES_F = pd.DataFrame(
    [
        ["Merger", "xkcd:violet"],
        ["Binary+S", "xkcd:azure"],
        ["Binary+CO", "xkcd:orangered"],
        ["Ionization", "xkcd:goldenrod"],
    ],
    columns=["label", "color"],
)


# Dataframe of BSE k-groups info
INFO_BSE_K_GROUPS = pd.DataFrame(
    [
        ["Star", "S", "Stars", 0, 10, "xkcd:goldenrod"],
        ["White dwarf", "WD", "White dwarfs", 10, 13, "xkcd:orangered"],
        ["Neutron star", "NS", "Neutron stars", 13, 14, "xkcd:azure"],
        ["Black hole", "BH", "Black holes", 14, 15, "xkcd:violet"],
    ],
    columns=["label", "initials", "plural", "lo", "hi", "color"],
)


# Dataframe of age groups info
INFO_CC_GROUPS = pd.DataFrame(
    [
        ["Pre-core collapse", "Y", "Pre-CC", 0.0, 1.0, "xkcd:azure", "-"],
        ["Post-core collapse", "O", "Post-CC", 1.0, np.inf, "xkcd:orangered", "--"],
    ],
    columns=["label", "initials", "plural", "lo", "hi", "color", "ls"],
)


# Dictionary of standard limits
AXES_LIMS = {
    "vout": [6e-3, 1e4],
    "mf": [5e-2, 55],
    "dist": [1e-3, 1e5],
    "v_radial": [-1e3, 1e3],
    "vlos": [275, 2e3],
}


# Dictionary of standard histogram limits (y-axis)
HIST_LIMS = {
    "vout": [5e-4, 1e3],
    "mf": [1e-3, 5e2],
    "Nej": [0, 14],
    "dist": [5e-4, 1e3],
    "v_radial": [5e-4, 1e3],
    "vlos": [5e-4, 1e3],
}


# Dictionary of standard bins
BINS = {
    "vout": np.logspace(*np.log10(AXES_LIMS["vout"]), 60),
    "mf": np.logspace(*np.log10(AXES_LIMS["mf"]), 60),
    "Nej": np.linspace(0, 3e4, 15),
    "dist": np.logspace(*np.log10(AXES_LIMS["dist"]), 60),
    # "v_radial": np.linspace(*AXES_LIMS["v_radial"], 120),
    "v_radial": np.arange(*AXES_LIMS["v_radial"], 10),
    "vlos": np.logspace(*np.log10(AXES_LIMS["vlos"]), 20),
}

# Dictionary of standard bins for CDF
BINS_CDF = {
    "vout": np.linspace(275, 2600, 30),
    "mf": np.linspace(0, 10, 30),
    "Nej": np.linspace(0, 3e4, 15),
    "dist": np.logspace(*np.log10(AXES_LIMS["dist"]), 60),
    # "v_radial": np.linspace(*AXES_LIMS["v_radial"], 120),
    "v_radial": np.linspace(275, 2600, 30),
    "vlos": np.linspace(275, 1200, 30),
}

# Dictionary of limits for CC plots
AXES_LIMS_CC = {
    "vout": [6e-3, 1e4],
    "mf": [5e-2, 3e1],
    "dist": [1e-3, 1e5],
}


# Dictionary of histogram limits (y-axis) for CC plots
HIST_LIMS_CC = {
    "vout": [5e-2, 5e4],
    "mf": [5e-2, 5e4],
    "Nej": [0, 14],
    "dist": [5e-2, 5e4],
}


# Dictionary of bins for CC plots
BINS_CC = {
    "vout": np.logspace(*np.log10(AXES_LIMS["vout"]), 60),
    "mf": np.logspace(*np.log10(AXES_LIMS["mf"]), 60),
    "Nej": np.linspace(0, 3e4, 15),
    "dist": np.logspace(*np.log10(AXES_LIMS["dist"]), 60),
}


# Dictionary of legend locations
LOC_LEGEND = {
    "vout": "upper left",
    "mf": "upper right",
    "dist": "upper left",
}


# AAStex twocolumn textwidth
textwidth = 8.5 / 46 * 18
textwidth = 3.5

###############################################################################


class EjectionExtractor:

    HEADERS_012 = [
        "vfin",
        "kf",
        "Rmin",
        "Rmin_j",
    ]
    HEADERS_01011 = [
        "mf",
        "rf",
    ]
    BINSINGLE_INDICES = [
        "0",
        "10",
        "11",
    ]

    def __init__(self, save_suffix=None, chunksize=None):
        self.save_suffix = save_suffix
        self.chunksize = chunksize

    def extract_ejections_from_df(self, df_realz):
        """! Returns a dataframe of ejections from a dataframe of realizations.

        @param df_realz     Realization dataframe

        @return df_ejs      Ejection dataframe
        """

        # Extract ejections of each object index
        df_ejs = []
        for i in range(3):
            # Make the dictionary of indexed headers
            iheaders = dict(
                zip(
                    [s + str(i) for s in self.HEADERS_012],
                    self.HEADERS_012,
                )
            )

            # Make the list of headers
            headers = HEADERS_EJECTIONS[:-7] + list(iheaders.keys())

            # Get the data
            df_temp = df_realz.loc[:, headers]
            # Rename columns
            df_temp.rename(columns=iheaders, inplace=True)
            # Get the mass and radius for each object (note: this fetches the initial values, so merger products will not have the correct mass)
            for h in self.HEADERS_01011:
                df_temp[h] = df_temp[h.replace("f", self.BINSINGLE_INDICES[i])]
            # Filter out anything that's not a binary nor an escaper (note unit conversion with v_crit, but also that velocities have not been converted in the dataframe)
            df_temp = df_temp[
                (df_temp.kf >= 0) & (df_temp.vfin * df_temp.v_crit >= df_temp.vesc)
            ]

            # Append to the holding list
            df_ejs.append(df_temp)

        df_ejs = pd.concat(df_ejs, ignore_index=True)

        return df_ejs

    def extract_ejections(self, path_realz):
        """! Given a file of encounter realizations, creates a file of the ejected objects, using the escape velocities in the encounter data.

        @param path_realz       Path to the realization file.
        """

        # Make output path if needed, raising an error if chunksize is given
        if self.save_suffix:
            path_ejs = (self.save_suffix + ".").join(path_realz.rsplit(".", 1))
        elif chunksize:
            raise Exception(
                "Processing in chunks, but save_suffix is not specified; data return not currently supported"
            )

        # Load realizations
        df_realz = pd.read_csv(
            path_realz, names=HEADERS_REALZ, chunksize=self.chunksize
        )

        if self.chunksize:
            # Process in chunks
            for ci, chunk in enumerate(df_realz):
                # Extract the ejected object data
                df_ejs = self.extract_ejections_from_df(chunk)

                # Save appropriately
                if ci == 0:
                    df_ejs.to_csv(path_ejs, index=False, mode="w")
                else:
                    df_ejs.to_csv(path_ejs, header=False, index=False, mode="a")

        else:
            # Don't process in chunks
            df_ejs = self.extract_ejections_from_df(df_realz)
            if self.save_suffix:
                df_ejs.to_csv(path_ejs, index=False, mode="w")
            return df_ejs


class EjectionDf:
    """! Class for working with the files produced by EjectionExtractor.extract_ejections()."""

    def __init__(self, path_to_df, **kwargs):
        # NOTE: **kwargs included so additional read_csv parameters may be passed.
        # The data
        self.df = pd.read_csv(path_to_df, **kwargs)
        # Tracker for whether physical units need to be computed
        self._units_not_physical = True

    def _check_if_physical(self):
        """! Simply raises an error if the data are not in physical units."""
        if self._units_not_physical:
            raise Exception("Units are not physical.")
        return 0

    def convert_type_f(self, row):
        """! Outputs post-encounter configuration, given type_f index and k-types.
        The configuration indices are:

        - 0: Some kind of merger, which may or may not have ejections
        - 1: Star ejection
        - 2: Compact object ejection
        - 3: Ionization

        @param row      Realization-like row with the four necessary data.
        """
        type_f = row["type_f"]
        k10 = row["k10"]
        k11 = row["k11"]
        k0 = row["k0"]

        if type_f == 0:
            return 3
        elif type_f >= 4:
            return 0
        else:
            if type_f == 1:
                ks = k11
                kb1 = k0
                kb2 = k10
            elif type_f == 2:
                ks = k10
                kb1 = k0
                kb2 = k11
            elif type_f == 3:
                ks = k0
                kb1 = k10
                kb2 = k11
            else:
                # print("\tconvert_type_f: strange input type_f (< 0?); (type_f, k0, k10, k11)=(%d,%d,%d,%d)" % (type_f, k0, k10, k11))
                return -1
            if ks < 10:
                return 1
            elif ks >= 10:
                return 2
            ## This section is the old version, where encounters were classified by the resulting binary, which wasn't the most useful when looking at ejections
            # if kb1 < 10 and kb2 < 10:
            #    return 1
            # elif (kb1 < 10 and kb2 >= 10) or (kb1 >= 10 and kb2 < 10):
            #    return 2
            # elif kb1 >= 10 and kb2 >= 10:
            #    return 3
            else:
                print("\tconvert_type_f: strange BSE k-types")
                return -2

    def convert_from_fewbody(self):
        """! Converts the columns from their fewbody output to useful quantities.
        For several columns this means a simple unit conversion, but this function also converts the fewbody type_f (which says exactly which object goes where) to a type_f that says what kinds of singles were left.
        """
        if not self._units_not_physical:
            raise Exception("Units are already physical.")

        # Convert units
        for r in (
            "a_fin",
            "Rmin",
        ):
            if r in self.df.columns:
                self.df[r] *= self.df["a1"]
        for v in (
            "vinf",
            "vfin",
        ):
            if v in self.df.columns:
                self.df[v] *= self.df["v_crit"]
        # Somehow forgot to include these when making _ejections files.  Currently unsupported.
        # for L in ("Lx", "Ly", "Lz", "Lbinx", "Lbiny", "Lbinz"):
        #    if L in self.df.columns:
        #        self.df[L] *= self.df["a1"]**3 * self.df["v_crit"]**2

        # Convert final encounter index
        if "type_f" in self.df.columns:
            self.df["type_f"] = self.df.apply(self.convert_type_f, axis=1)

        # Note that the units are physical, i.e. we can compare encounters now
        self._units_not_physical = False

        return 0

    def calc_vout(self):
        """! Calculates the velocities at which the objects escape the cluster."""
        return (self.df["vfin"] ** 2 - self.df["vesc"] ** 2) ** 0.5

    def calc_vescrho0(self, df_dyn, tbuf=0.001):
        """! Divides vesc by rho0.
        The idea is that this takes out the continuum of the core evolution.
        The dyn.dat file should have the same times as the ejection file; we just need to match up timesteps to encounter groups.

        @param df_dyn       initial.dyn.dat from CMC output, formatted as a pandas df.
        @param tbuf         Buffer to apply to time filtering of dyn.dat.  Pick something larger than a CMC timestep.
        """
        # Initialize temp column (to hold vesc/rho_0 data)
        self.df["temp"] = 0

        # Set minimum and maximum times, previous time, and buffer
        tmin = self.df["time"].min()
        tmax = self.df["time"].max()
        print("\t\tself.df: tmin, tmax:", tmin, tmax)
        print("\t\tdf_dyn: tmin, tmax:", df_dyn.t.min(), df_dyn.t.max())
        print("\t\tdf_dyn: Dtmin, Dtmax:", df_dyn.Dt.min(), df_dyn.Dt.max())
        tprev = tmin

        # Iterate over dyn.dat (simulation) timesteps
        for _, drow in tqdm(
            df_dyn[(df_dyn.t >= tmin - tbuf) & (df_dyn.index % 100 == 0)].iterrows()
        ):
            # If we haven't gotten to the first ejection yet, continue
            if drow.t < tmin:
                continue
            # Make the mask of times between the previous time and the current time
            mask = (self.df["time"] >= tprev) & (self.df["time"] < drow.t)
            # If there's at least one ejection in this interval
            if mask.sum() != 0:
                # Set vesc/rho_0 to the appropriate value
                self.df.loc[mask, "temp"] = self.df[mask]["vesc"] / drow.rho_0**0.175
                # self.df.loc[mask, "temp"] = self.df[mask]["vesc"] * drow.rc_nb
            # If we've reached the latest time for the ejections, stop iteration
            if drow.t > tmax:
                break
            # Update tprev
            tprev = drow.t

        return self.df["temp"]


class HistogramGenerator:
    def __init__(
        self,
        column,
        bins,
        nprocs=1,
        kgroup=None,
        cc=False,
        t_term=14000,
    ):
        """!

        @param column       Data column from dataframe to use (e.g. "vout", "mf")
        @param bins         Histogram bins to apply to data.
        @param nprocs       Number of processes (for parallelization).
        @param kgroup       BSE kgroup to use (e.g. stars, white dwarfs; see above)
        @param cc           Whether to distingish the ejections that occured before and after the core collapse of the cluster.
            If True, will return two histograms (one for each era).
        @param t_term       Time of integration termination, in Myr (for calculating distance traveled).
        """
        self.column = column
        self.bins = bins
        self.nprocs = nprocs
        self.kgroup = kgroup
        self.cc = cc
        self.t_term = t_term

    def generate_histogram(self, mi, model, path_to_data=paths.data_cmc):
        """! Generates the histogram for the specified model.

        @param mi       Index of model
        @param model    Model info, as a row of rustics.DF_MODELS
        @param path_to_data     Path to directory of clusters.

        @return hist    Values of the histogram (same as the first return of np.histogram).
        """

        # Load ejections data, while only loading the column that matters
        # Column loading is problematic with unit conversions
        # Special cases for columns that are functions of the ejections columns
        path = path_to_data / model["fname"] / FILE_EJECTIONS
        # Specify columns to load, and add "kf" if kgroup filtering is enabled
        if self.column == "vout":
            usecols = ["vfin", "vesc", "v_crit"]
        elif self.column == "dist":
            usecols = ["vfin", "vesc", "v_crit", "time"]
        elif self.column == "v_radial":
            usecols = ["vfin", "vesc", "v_crit"]
        elif self.column == "vlos":
            usecols = ["X", "Y", "Z", "U", "V", "W"]
        elif self.column == "rgc":
            usecols = ["X", "Y"]
        else:
            usecols = [self.column]
        if type(self.kgroup) != type(None):
            usecols += ["kf"]
        if self.cc and "time" not in usecols:
            usecols += ["time"]
        # Load, and convert units
        ejdf = EjectionDf(path, usecols=usecols)
        ejdf.convert_from_fewbody()
        if ("X" in ejdf.df.columns) & ("Y" in ejdf.df.columns):
            # Throw out -1's
            ejdf.df = ejdf.df[(ejdf.df.X != -1) & (ejdf.df.Y != -1)]
        # Calculate additional things if needed
        if self.column == "vout":
            ejdf.df["vout"] = ejdf.calc_vout()
            print("%24s %g" % (model["fname"], ejdf.df["vout"].max()))
        elif self.column == "dist":
            ejdf.df["vout"] = ejdf.calc_vout()
            ejdf.df["dist"] = (
                ejdf.df.vout
                * (self.t_term - ejdf.df.time)
                * (1e6 * YR_TO_S)
                / (PC_TO_M)
            )
        elif self.column == "v_radial":
            # Calculate vout as normal
            ejdf.df["vout"] = ejdf.calc_vout()
            # Generate random projection factors (by ejection directions)
            np.random.RandomState(123456)
            ejdf.df["proj"] = np.sin(np.pi * np.random.rand(ejdf.df.shape[0]))
            # Calculate random v_radials (+- rv of cluster, e.g. -147.20 for M3)
            ejdf.df["v_radial"] = ejdf.df["vout"] * ejdf.df["proj"]
            print("%24s %g" % (model["fname"], ejdf.df["v_radial"].max()))
        elif self.column == "vlos":
            # Create coordinate object; convert to ICRS
            sc_ejs = {
                "x": ejdf.df.X.to_numpy() * u.kpc,
                "y": ejdf.df.Y.to_numpy() * u.kpc,
                "z": ejdf.df.Z.to_numpy() * u.kpc,
                "v_x": ejdf.df.U.to_numpy() * u.km / u.s,
                "v_y": ejdf.df.V.to_numpy() * u.km / u.s,
                "v_z": ejdf.df.W.to_numpy() * u.km / u.s,
            }
            sc_ejs = SkyCoord(frame=Galactocentric, **sc_ejs)
            sc_ejs = sc_ejs.transform_to(Galactic)
            ls = sc_ejs.l.value
            bs = sc_ejs.b.value
            rvs = sc_ejs.radial_velocity.value
            ejdf.df["vlos"] = (
                rvs
                + Usun_gc[0] * np.cos(ls) * np.cos(bs)
                + Usun_gc[1] * np.sin(ls) * np.cos(bs)
                + Usun_gc[2] * np.sin(bs)
            )
        elif self.column == "rgc":
            ejdf.df["rgc"] = (ejdf.df.X**2 + ejdf.df.Y**2) ** 0.5

        # If kgroup is given, filter
        if type(self.kgroup) != type(None):
            ejdf.df = ejdf.df[
                (ejdf.df.kf >= self.kgroup.lo) & (ejdf.df.kf < self.kgroup.hi)
            ]

        # Generate histogram(s)
        # If cc, split by core collapse
        if self.cc:
            ha, _ = np.histogram(
                ejdf.df[ejdf.df.time < model["cc_time"]][self.column], self.bins
            )
            hp, _ = np.histogram(
                ejdf.df[ejdf.df.time >= model["cc_time"]][self.column], self.bins
            )
            hist = np.array([ha, hp])
        # Otherwise, do it simply
        else:
            hist, _ = np.histogram(ejdf.df[self.column], self.bins)

        return hist

    def generate_catalog_histograms(self, df_models, **kwargs):
        """! Generates list of catalog histograms, for those whose data appear in df_models.
        Passes kwargs to starmap.

        @param df_models        rustics.DF_MODELS-like dataframe.

        @result hists           List of histograms, ordered in coherence with df_models.
        """

        if self.nprocs == 1:
            # Don't run in parallel
            mp_input = [x for x in df_models.iterrows()]
            hists = parmap.starmap(
                self.generate_histogram, mp_input, pm_parallel=False, **kwargs
            )
        else:
            # Run in parallel
            mp_input = [x for x in df_models.iterrows()]
            pool = mp.Pool(self.nprocs)
            hists = parmap.starmap(
                self.generate_histogram, mp_input, pm_pool=pool, **kwargs
            )

        return hists


class StackedBarGenerator:
    def __init__(self, nprocs=1, chunksize=None, discrim_ejections=False):
        """! Class for generating stacked bar plot info.

        @param nprocs               Number of processes to use (parallelization).
        @param chunksize            Chunksize to use to process realization data.
        @param discrim_ejections    Boolean, to specify if the encounters should be discriminated by whether they produce ejections or not.
        """
        self.nprocs = nprocs
        self.chunksize = chunksize
        self.discrim_ejections = discrim_ejections

    def generate_stackedbar(self, mi, model):
        """! Generates the stackedbar info for the specified model.

        @param mi       Index of model
        @param model    Model info, as a row of rustics.DF_MODELS

        @return sb      stackedbar info.
        """

        # Load ejections data, while only loading the column that matters
        # Column loading is problematic with unit conversions
        # Special cases for columns that are functions of the ejections columns
        path = "/".join((paths.data_cmc, model["fname"], FILE_REALZ))
        if self.discrim_ejections:
            usecols = [
                "type_i",
                "type_f",
                "k0",
                "k10",
                "k11",
                "vesc",
                "vfin0",
                "vfin1",
                "vfin2",
                "v_crit",
            ]
        else:
            usecols = [
                "type_i",
                "type_f",
                "k0",
                "k10",
                "k11",
            ]
        ejdf = EjectionDf(
            path,
            names=HEADERS_REALZ,
            chunksize=self.chunksize,
            usecols=usecols,
        )

        # Iterate over types, getting bar sizes
        # Without chunking
        if self.chunksize == None:
            # If discriminating by ejection...
            if self.discrim_ejections:
                sbs = np.zeros((INFO_TYPES_I.shape[0], INFO_TYPES_F.shape[0], 2))
                # If there're encounters...
                if ejdf.df.shape[0] != 0:
                    # Convert type_f indices
                    ejdf.df["type_f"] = ejdf.df.apply(ejdf.convert_type_f, axis=1)
                    # Calculate ejection flag (If there's > 1 ejection)
                    ejdf.df["vfinx"] = ejdf.df[["vfin0", "vfin1", "vfin2"]].max(axis=1)
                    ejdf.df["ej_flag"] = (
                        ejdf.df.vfinx * ejdf.df.v_crit >= ejdf.df.vesc
                    ).astype(int)
                    vc = ejdf.df.value_counts(["type_i", "type_f", "ej_flag"])
                    for ti, ti_row in INFO_TYPES_I.iterrows():
                        for tf, tf_row in INFO_TYPES_F.iterrows():
                            for ei in [0, 1]:
                                if (ti, tf, ei) in vc.index:
                                    sbs[ti, tf, ei] += vc[ti, tf, ei]
            else:
                sbs = np.zeros((INFO_TYPES_I.shape[0], INFO_TYPES_F.shape[0]))
                # If there're encounters...
                if ejdf.df.shape[0] != 0:
                    # Convert type_f indices
                    ejdf.df["type_f"] = ejdf.df.apply(ejdf.convert_type_f, axis=1)
                    vc = ejdf.df.value_counts(["type_i", "type_f"])
                    for ti, ti_row in INFO_TYPES_I.iterrows():
                        for tf, tf_row in INFO_TYPES_F.iterrows():
                            if (ti, tf) in vc.index:
                                sbs[ti, tf] += vc[ti, tf]
        # With chunking
        else:
            # If discriminating by ejection...
            if self.discrim_ejections:
                sbs = np.zeros((INFO_TYPES_I.shape[0], INFO_TYPES_F.shape[0], 2))
                for ci, c in enumerate(ejdf.df):
                    # If there're encounters...
                    if c.shape[0] != 0:
                        # Convert type_f indices
                        c["type_f"] = c.apply(ejdf.convert_type_f, axis=1)
                        # Calculate ejection flag
                        c["vfinx"] = c[["vfin0", "vfin1", "vfin2"]].max(axis=1)
                        c["ej_flag"] = (c.vfinx * c.v_crit >= c.vesc).astype(int)
                        vc = ejdf.df.value_counts(["type_i", "type_f", "ej_flag"])
                        for ti, ti_row in INFO_TYPES_I.iterrows():
                            for tf, tf_row in INFO_TYPES_F.iterrows():
                                for ei in [0, 1]:
                                    if (ti, tf, ei) in vc.index:
                                        sbs[ti, tf, ei] += vc[ti, tf, ei]
            else:
                sbs = np.zeros((INFO_TYPES_I.shape[0], INFO_TYPES_F.shape[0]))
                for ci, c in enumerate(ejdf.df):
                    # If there're encounters...
                    if c.shape[0] != 0:
                        # Convert type_f indices
                        c["type_f"] = c.apply(ejdf.convert_type_f, axis=1)
                        vc = ejdf.df.value_counts(["type_i", "type_f"])
                        for ti, ti_row in INFO_TYPES_I.iterrows():
                            for tf, tf_row in INFO_TYPES_F.iterrows():
                                if (ti, tf) in vc.index:
                                    sbs[ti, tf] += vc[ti, tf]

        return sbs

    def generate_catalog_stackedbars(self, df_models, **kwargs):
        """! Generates list of stackedbar info, for those whose data appear in df_models.
        Passes kwargs to starmap.

        @param df_models        rustics.DF_MODELS-like dataframe.

        @result sbs             List of stackedbar info, ordered in coherence with df_models.
        """

        if self.nprocs == 1:
            # Don't run in parallel
            mp_input = [x for x in df_models.iterrows()]
            sbs = parmap.starmap(
                self.generate_stackedbar, mp_input, pm_parallel=False, **kwargs
            )
        else:
            # Run in parallel
            mp_input = [x for x in df_models.iterrows()]
            pool = mp.Pool(self.nprocs)
            sbs = parmap.starmap(
                self.generate_stackedbar, mp_input, pm_pool=pool, **kwargs
            )

        return sbs


###############################################################################


class GCCatalog:
    def __init__(self, paths, pd_kwargs={}):
        """! Basic things for now."""

        # If a single path is specified, assume it is a loadable csv
        if type(paths) == str:
            self.df = pd.read_csv(paths, **pd_kwargs)

        # If a list of paths is specified, assume they are baumgardt and harris catalogs
        elif type(paths) == list:
            # Set paths
            baumgardt_path, harris_path = paths

            # Load cleaned Baumgardt data
            df_baumgardt = pd.read_csv(
                baumgardt_path,
                usecols=["Cluster", "Mass", "rc", "rh,m", "R_GC"],
                delim_whitespace=True,
            )

            # Load cleaned Harris data (for metallicities)
            # "cleaned" = ID renamed to Cluster, names edited to match Baumgardt & Holger, Z=-100 if there's no metallicity measurement (these clusters are also the ones with wt=0 i.e. no stellar metallicities have been measured)
            df_harris = pd.read_csv(
                harris_path, usecols=["Cluster", "[Fe/H]"], delim_whitespace=True
            )
            # drop rows with [Fe/H]=-100
            df_harris = df_harris[df_harris["[Fe/H]"] != -100.0]

            # Merge the two datasets
            self.df = pd.merge(df_baumgardt, df_harris, how="inner", on="Cluster")

            # Set cluster name as index
            self.df.set_index("Cluster", inplace=True)
        else:
            raise Exception("path type %s not implemented" % type(paths))

    def match_to_cmc_models(
        self,
        cmc_path,
        cmc_kwargs={},
        dyn_kwargs={},
    ):
        """! Given a path to a CMC directory, finds the matching CMC models for the GCs."""

        # Save cmc_path
        self.cmc_path = cmc_path

        # Load cmc catalog, and dyn.dat data
        import mywheels.cmcutils.readcatalog as cmccat

        cmc_models = cmccat.CMCCatalog(
            cmc_path, extension=".tar.gz", mp_nprocs=128, **cmc_kwargs
        )
        cmc_models.add_dat_timesteps(
            "initial.dyn.dat",
            tmin=10000.0,
            tmax=13500.0,
            tnum=10,
            dat_kwargs={
                "pd_kwargs": {"usecols": ["t", "M", "rc_spitzer", "r_h"]},
                "convert_units": {
                    "t": "myr",
                    "M": "msun",
                    "rc_spitzer": "pc",
                    "r_h": "pc",
                },
            },
            **dyn_kwargs,
        )
        cmc_models.parse_names(replace={"_v2": "", ".tar.gz": ""})

        # Calculate columns to match
        cmc_models.df["[Fe/H]"] = np.log10(cmc_models.df["Z"] / 0.02)
        self.df["logM"] = np.log10(self.df["Mass"])
        cmc_models.df["logM"] = np.log10(cmc_models.df["M"])
        self.df["rc/rh"] = self.df["rc"] / self.df["rh,m"]
        cmc_models.df["rc/rh"] = cmc_models.df["rc_spitzer"] / cmc_models.df["r_h"]

        self.df.to_csv("gcs.dat")
        cmc_models.df.to_csv("cmcs.dat")

        # Iterate through rg, met bins (bin edges are average of adjacent CMC rg values)
        rgs_cmc = np.sort(cmc_models.df.rg.unique())
        mets_cmc = np.sort(cmc_models.df["[Fe/H]"].unique())
        params_compare = ["logM", "rc/rh"]
        dfs = []
        for rgi, rg in enumerate(rgs_cmc):
            # Set rg bin boundaries
            if rgi == 0:
                rglo = -np.inf
                rghi = (rg + rgs_cmc[rgi + 1]) / 2.0
            elif rgi == len(rgs_cmc) - 1:
                rglo = rghi
                rghi = np.inf
            else:
                rglo = rghi
                rghi = (rg + rgs_cmc[rgi + 1]) / 2.0
            for mi, met in enumerate(mets_cmc):
                # Set metallicity bin boundaries
                if mi == 0:
                    metlo = -np.inf
                    methi = (met + mets_cmc[mi + 1]) / 2.0
                elif mi == len(mets_cmc) - 1:
                    metlo = methi
                    methi = np.inf
                else:
                    metlo = methi
                    methi = (met + mets_cmc[mi + 1]) / 2.0

                # Select MW GCs in range, and CMC models
                clusters_rm = self.df[
                    (self.df.R_GC >= rglo)
                    & (self.df.R_GC < rghi)
                    & (self.df["[Fe/H]"] >= metlo)
                    & (self.df["[Fe/H]"] < methi)
                ]
                models_rm = cmc_models.df[
                    (cmc_models.df.rg == rg) & (cmc_models.df["[Fe/H]"] == met)
                ]
                print(clusters_rm)
                print(models_rm)

                # Iterate over comparison parameters, calculating distances for each
                distances = pd.DataFrame(
                    0.0, index=clusters_rm.index, columns=models_rm.index
                )
                for p in params_compare:
                    distances += (
                        np.subtract.outer(
                            clusters_rm[p].to_numpy(), models_rm[p].to_numpy()
                        )
                        ** 2
                    )
                print(distances)

                try:
                    matching = distances.idxmin(axis=1)
                except:
                    continue
                print(matching)
                clusters_rm["fname"] = [x[0] for x in matching]
                clusters_rm["tcount"] = [x[1] for x in matching]
                print("test", [cmc_models.df.at[i, "t"] for i in matching])
                clusters_rm["t"] = [cmc_models.df.at[i, "t"] for i in matching]
                print(clusters_rm)

                dfs.append(clusters_rm)

        self.df = pd.concat(dfs)
        self.df.to_csv("gcs-cmc.dat")
        print(self.df)
