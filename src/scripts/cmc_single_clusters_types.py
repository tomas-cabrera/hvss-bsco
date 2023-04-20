import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as mplticker

import rustics
import paths

################################################################################

# plt.style.use(rustics.PATH_TO_MPLRC)

# Get cluster names
cmc_cluster_list = list(rustics.SAMPLE_MODELS.keys())
for ci, cmc_cluster in enumerate(cmc_cluster_list):
    print("{}, N_REALZ={}".format(cmc_cluster, rustics.N_REALZ))

    # File names and directory creation
    path_to_ejections = paths.data_cmc / cmc_cluster / rustics.FILE_EJECTIONS

    # Read saved data file
    ejdf = rustics.EjectionDf(path_to_ejections)
    ejdf.df = ejdf.df

    # Convert to physical units
    ejdf.convert_from_fewbody()

    # Filter out mergers
    ejdf.df = ejdf.df[ejdf.df["type_f"] > 0]

    # Load times and core radii from initial.dyn.dat
    df_esc = pd.read_csv(
        paths.data_cmc / cmc_cluster / "initial.esc.dat",
        names=["t", "phi_rtidal", "phi_zero"],
        usecols=[1, 9, 10],  # 1 for t, 9 for phi_rtidal, 10 for phi_zero
        skiprows=1,
        delim_whitespace=True,
    )

    # Load initial.conv.sh (for converting TotalTime to Myr and core vesc to km/s)
    # Copied from cmctools/cmctoolkit.py
    conv_path = paths.data_cmc / cmc_cluster / "initial.conv.sh"
    f = open(conv_path, "r")
    convfile = f.read().split("\n")
    f.close()
    lengthunitcgs = float(convfile[13][14:])
    timeunitsmyr = float(convfile[19][13:])
    nbtimeunitcgs = float(convfile[21][14:])
    nb_kms = 1e-5 * lengthunitcgs / nbtimeunitcgs
    df_esc["t"] *= timeunitsmyr
    df_esc["phi_rtidal"] *= nb_kms**2
    df_esc["phi_zero"] *= nb_kms**2

    # Generate ordered list of type_i's, descending by number of occurences
    type_is = ejdf.df.value_counts("type_i").index
    type_is = rustics.INFO_TYPES_I.iloc[type_is]
    type_is = rustics.INFO_TYPES_I

    # Some parameters
    ms = 0.25

    # Select x-/y-axis labels; stars only (k<10)
    x = "time"
    y = "vesc"
    kgroup = rustics.INFO_BSE_K_GROUPS.iloc[0]

    for ti, ti_row in type_is.iterrows():
        filtered = ejdf.df[
            (ejdf.df.kf >= kgroup.lo)
            & (ejdf.df.kf < kgroup.hi)
            & (ejdf.df.type_i == ti)
        ]
        print("\t%s filtered.shape[0] (%% of total): %d (%f)" % (ti_row.label, filtered.shape[0], filtered.shape[0]/ejdf.df[(ejdf.df.kf >= kgroup.lo) & (ejdf.df.kf < kgroup.hi)].shape[0]))
    print("\t #BSCO/10/#esc: ", ejdf.df.shape[0] / 10 / df_esc.shape[0])
