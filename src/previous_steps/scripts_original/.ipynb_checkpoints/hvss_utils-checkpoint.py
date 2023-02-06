import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


################################################################################


class binsingle_type:
    def __init__(self, id, id_string, color):
        self.id = id
        self.id_string = id_string
        self.color = color

    def print_self(self):
        print(
            (
                f"threebody_type; id={self.id}; "
                f"id_string={self.id_string}; color={self.color}"
            )
        )


# Class for consolidating data groupings
class data_group:
    def __init__(self, name, short_name, label, lo, hi, color):
        self.name = name
        self.short_name = short_name
        self.label = label
        self.lo = lo
        self.hi = hi
        self.color = color


################################################################################


def ns_to_str(N):
    """
    Converts N from 10**5 value to string format.
    """
    return "{:.1e}".format(N).replace(".0", "").replace("+0", "")


def sort_binsingle_types(bs_types, data):
    """
    Given a set of BSE k-type groups,
    returns list of groups sorted by size in given dataset.
    """
    counts = []
    for bs_type in bs_types:
        counts.append(data[data.type_i == bs_type.id].shape[0])
    bs_types_sorted = []
    for ci in np.argsort(counts)[::-1]:
        if counts[ci] > 0:
            bs_types_sorted.append(bs_types[ci])
    return bs_types_sorted


def load_flattened(file_path, shape):
    """
    Loads flattened data files.
    For hvss_bin_cmc.py data:
        Binned data are stored as flattened 4D arrays:
            (n_binsingle_types x n_bse_k_groups x n_agebins x n_vbins)
        Output is a 4D numpy array:
            n_binsingle_types x n_bse_k_groups x n_agebins x n_vbins
    """
    data = np.loadtxt(
        file_path,
        dtype=np.int32,
    ).reshape(shape)
    return data


def convert_type_f(type_f, k10, k11, k0):
    """
    Outputs post-encounter configuration, given type_f index and k-types.
    """
    if type_f == 0:
        return 0
    elif type_f >= 4:
        return 4
    else:
        if type_f == 1:
            kb1 = k0
            kb2 = k10
        elif type_f == 2:
            kb1 = k0
            kb2 = k11
        elif type_f == 3:
            kb1 = k10
            kb2 = k11
        else:
            print("\tconvert_type_f: strange input type_f (< 0?)")
            return -1
        if kb1 < 10 and kb2 < 10:
            return 1
        elif (kb1 < 10 and kb2 >= 10) or (kb1 >= 10 and kb2 < 10):
            return 2
        elif kb1 >= 10 and kb2 >= 10:
            return 3
        else:
            print("\tconvert_type_f: strange BSE k-types")
            return -2


def anchored_coords(c, axes_lims, xscale=None, yscale=None):
    """
    Appropriates matplotlib functions to get left/center/right coords for a given set of axes.
    Also works for logarithmically-scaled axes.
    """
    return


def grid_selfmade(ax, minor=True, color="white", linewidth=0.85):
    """
    Carl's function for making plot grids
    """
    ax.grid(False)
    starting_x = ax.get_xticks()
    starting_y = ax.get_yticks()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    y_axis = ax.yaxis.get_view_interval()
    for xx in ax.get_xticks():
        ax.plot(
            [xx, xx], y_axis, linestyle="-", color=color, linewidth=linewidth, zorder=0
        )
    if minor == True:
        for xx in ax.get_xticks(minor=True):
            ax.plot(
                [xx, xx],
                y_axis,
                linestyle="-",
                color="#FAFAFA",
                linewidth=0.38,
                zorder=-1,
            )
    x_axis = ax.xaxis.get_view_interval()
    for yy in ax.get_yticks():
        ax.plot(
            x_axis, [yy, yy], linestyle="-", color=color, linewidth=linewidth, zorder=0
        )
    if minor == True:
        for yy in ax.get_yticks(minor=True):
            ax.plot(
                x_axis,
                [yy, yy],
                linestyle="-",
                color="#FAFAFA",
                linewidth=0.38,
                zorder=-1,
            )
    ax.set_xticks(starting_x)
    ax.set_yticks(starting_y)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


################################################################################


# Physical constants (in cgs)
CONST_G = 6.67259 * 10 ** -8


# Path/to/fewbody
FB_PATH = "/hildafs/projects/phy200025p/tcabrera/hvss/fewbody-pn/"


# Path/to/hvss[fewbody]/output
DATA_PATH = "/hildafs/projects/phy200025p/tcabrera/hvss/data/"


# Path/to/figures
FIGS_PATH = "/hildafs/projects/phy200025p/tcabrera/hvss/figs/"


# Path/to/cmc/catalog
CMC_PATH = "/hildafs/projects/phy200025p/share/catalog_files/"


# Path/to/matplotlib/style/file
MPL_STYLE_FILE_PATH = "/hildafs/projects/phy200025p/tcabrera/matplotlib_style_file"


# Fewbody process to use
FB_PROCESS = "binsingle_hvss"


# Table of CMC catalog models, received from Kyle Kremer via. e-mail 07/27/2021
# Also obtainable from Table A1 in Kremer et al. 2020 (catalog paper)
CATALOG_MODELS = pd.read_table(
    "Catalog_models.dat",
    names=["N", "rv", "rg", "Z", "N_BH", "status"],
    skiprows=2,
    delimiter=" ",
)
# and lists of unique values for N and rv
N_LIST = CATALOG_MODELS.N.unique()
RV_LIST = CATALOG_MODELS.rv.unique()
Z_LIST = CATALOG_MODELS.Z.unique()


# List of CMC cluster model filenames, composed from Catalog_models.dat,
# or define shorter list
CMC_CLUSTER_LIST = [
    "N{}_".format(ns_to_str(CATALOG_MODELS.loc[i, "N"] * 100000))
    + "rv{:n}_".format(CATALOG_MODELS.loc[i, "rv"])
    + "rg{:n}_".format(CATALOG_MODELS.loc[i, "rg"])
    + "Z{}".format(CATALOG_MODELS.loc[i, "Z"])
    for i in CATALOG_MODELS.index
]
# CMC_CLUSTER_LIST = ["N2e5_rv0.5_rg2_Z0.0002"]
CMC_CLUSTER_LIST_V2 = [
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


# Fewbody version/prefix
FB_VERSION = FB_PATH.split("-", 1)[-1]


# Fewbody realization output filename
FB_OUTPUT_FILENAME = "output"


# Column names for pro data (for each encounter)
PRO_HEADERS = (
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
    "L",
    "Lbin",
    "cos_theta",
    "mbhx",
    "vout0",
    "vout1",
    "vout2",
)


# Column names for singles data (for each single)
SINGLES_HEADERS = (
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
    "L",
    "Lbin",
    "cos_theta",
    "mbhx",
    "vfin",
    "kf",
    "Rmin",
    "Rmin_j",
    "vout",
)


# Standard histogram bins for singles
BINS_SINGLES = {
    "vout" : np.logspace(-2.5, 4, 60),
}


# Standard velocity histogram bins
VBINS = np.logspace(-2.5, 4, 60)


# Velocity histogram limits for binned
HIST_VLIM = (0.06, 1000.0)


# Velocity histogram limits for catalog
CATALOG_HIST_VLIM = (0.06, 100000.0)


# Headers to use for x-axes of scatterplots
SCATTER_HEADERS_X = (
    "b",
    "cos_theta",
    "Lbin",
    "L",
    "Rmin",
    "a_fin",
    "Ei",
    "DeltaEfrac",
    "vesc",
)


# Headers to use for y-axes of scatterplots
SCATTER_HEADERS_Y = (
    "vfin",
    "vout",
    "Rmin",
    "Ei",
    "DeltaEfrac",
)


# Data to use log scale for
HEADERS_LOG_SCALE = (
    "vfin",
    "Rmin",
    "L",
    "Lbin",
    "a_fin",
    "b",
    "vout",
    "Ei",
    "DeltaEfrac",
)


# List of binsingle types/configurations, with properties for plotting
BINSINGLE_TYPES_ISOLATED = [
    binsingle_type(1, "(BH,BH)+S", "xkcd:azure"),
    binsingle_type(2, "(S,BH)+BH", "xkcd:orangered"),
]


# List of binsingle types/configurations, with properties for plotting
BINSINGLE_TYPES_CMC_I = [
    binsingle_type(1, "(C,C)+S", "xkcd:violet"),
    binsingle_type(2, "(S,C)+C", "xkcd:azure"),
    binsingle_type(3, "(S,C)+S", "#07d5bd"),
    binsingle_type(4, "(S,S)+C", "xkcd:orangered"),
]


# List of binsingle types/configurations, with properties for plotting
BINSINGLE_TYPES_CMC_F = [
    binsingle_type(0, "Ionization", "#07d5bd"),
    binsingle_type(1, "(S,S)+X", "xkcd:orangered"),
    binsingle_type(2, "(S,C)+X", "xkcd:azure"),
    binsingle_type(3, "(C,C)+X", "xkcd:violet"),
    binsingle_type(4, "Merger", "xkcd:crimson"),
]


# List of BSE k-type groupings into more accessible categories
BSE_K_GROUPS = [
    data_group("Star", "S", "Stars", 0, 10, "xkcd:orangered"),
    data_group("White dwarf", "WD", "White dwarfs", 10, 13, "xkcd:azure"),
    data_group("Neutron star", "NS", "Neutron stars", 13, 14, "xkcd:violet"),
    data_group("Black hole", "BH", "Black holes", 14, 15, "xkcd:black"),
]


## List of age groupings into more accesible categories (old version)
#AGE_GROUPS = [
#    data_group("Young", "Y", "t<1 Gyr", 0., 1000., "xkcd:azure"),
#    data_group("Middle-aged", "M", "1<t<10 Gyr", 1000., 10000., "#07d5bd"),
#    data_group("Old", "O", "t>10 Gyr", 10000., np.Infinity, "xkcd:orangered"),
#]
# List of age groupings into more accesible categories
AGE_GROUPS = [
    data_group("Pre-core collapse", "Y", "Pre-CC", 0.0, 1.0, "xkcd:azure"),
    data_group("Post-core collapse", "O", "Post-CC", 1.0, np.inf, "xkcd:orangered"),
]


# Shape of binned isolated data (tuple)
BINNED_SHAPE_ISOLATED = (
    len(BINSINGLE_TYPES_ISOLATED),
    len(BSE_K_GROUPS),
    len(AGE_GROUPS),
    len(VBINS) - 1,
)


# Shape of binned data (tuple)
BINNED_SHAPE = (
    len(BINSINGLE_TYPES_CMC_I),
    len(BSE_K_GROUPS),
    len(AGE_GROUPS),
    len(VBINS) - 1,
)


# Shape of full catalog data (tuple)
CATALOG_SHAPE = (
    len(N_LIST),
    len(RV_LIST),
    len(BINSINGLE_TYPES_CMC_I),
    len(BSE_K_GROUPS),
    len(AGE_GROUPS),
    len(VBINS) - 1,
)


# Shape of full catalog data (tuple)
CATALOG_SHAPE_Z = (
    len(N_LIST),
    len(RV_LIST),
    len(Z_LIST),
    len(BINSINGLE_TYPES_CMC_I),
    len(BSE_K_GROUPS),
    len(AGE_GROUPS),
    len(VBINS) - 1,
)


# Dictionary for turning header names into plot labels
HEADERS_TO_LABELS = {
    "time": "Time",
    "b": r"$b$",
    "vinf": r"$v_\infty$",
    "a1": r"$a_i$",
    "e1": r"$e_i$",
    "vesc": r"$v_{\rm esc}$",
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
    "Rmin": r"$R_{\rm min}$",
    "Rmin_j": r"$j_{\rm rmin}$",
    "vout": r"$v_{\rm out}$",
}


################################################################################
# Old things

# Column names for pro data (for each encounter)
OLD_PRO_HEADERS = (
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
    "L",
    "Lbin",
    "cos_theta",
    "mbhx",
    "vout0",
    "vout1",
    "vout2",
)


# Column names for singles data (for each single)
OLD_SINGLES_HEADERS = (
    "time",
    "b",
    "vinf",
    "a1",
    "e1",
    "k0",
    "k10",
    "k11",
    "vesc",
    "a_fin",
    "e_fin",
    "type_i",
    "type_f",
    "L",
    "Lbin",
    "cos_theta",
    "mbhx",
    "vfin",
    "kf",
    "Rmin",
    "Rmin_j",
    "vout",
)


# "Special" because I messed up and mixed the columns around for the old data
SPECIAL_SINGLES_HEADERS = (
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
    "vfin",
    "kf",
    "Rmin",
    "Rmin_j",
)
