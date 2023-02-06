import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import hvss_utils

################################################################################


N = 10
vbins = hvss_utils.VBINS
binsingle_types = hvss_utils.BINSINGLE_TYPES_CMC_I
bse_k_groups = hvss_utils.BSE_K_GROUPS
age_groups = hvss_utils.AGE_GROUPS

# Get CATALOG_MODELS et al. (note: different from other files!)
catalog_models = hvss_utils.CATALOG_MODELS
N_list = hvss_utils.N_LIST
rv_list = hvss_utils.RV_LIST

# Get cmc_core_collapsed.dat (file w/ cc times)
core_collapsed = pd.read_csv("cmc_core_collapsed.dat")

# initialize full catalog array (n_N, n_rv, n_age_groups, n_binsingle_types, n_bse_k_groups, n_vbins)
catalog = np.zeros(
    hvss_utils.CATALOG_SHAPE,
    dtype=np.int32,
)

for mi, model in catalog_models.iterrows():

    cmc_cluster = (
        "N{}_".format(hvss_utils.ns_to_str(model.loc["N"] * 100000))
        + "rv{:n}_".format(model.loc["rv"])
        + "rg{:n}_".format(model.loc["rg"])
        + "Z{}".format(model.loc["Z"])
    )
    print("\n{}, N={} ({}/{})".format(cmc_cluster, N, mi + 1, len(catalog_models)))

    # Particular shorthands
    data_file_path = (
        hvss_utils.DATA_PATH
        + cmc_cluster
        + "/{}_N-{}".format(hvss_utils.FB_OUTPUT_FILENAME, N)
    )

    # Read saved data file
    print("\tLoading data...")
    try:
        singles = pd.read_csv(
            data_file_path + "_singles.txt",
            names=hvss_utils.SINGLES_HEADERS,
            dtype=np.float64,
        )
    except Exception as e:
        print("\tError in loading singles data:".format(cmc_cluster, N))
        print("\t\t", e)
        print("\tContinuing...")
        continue

    # Get specific core_collapsed row
    cc_row = core_collapsed[
        (core_collapsed.N == model.loc["N"])
        & (core_collapsed.rv == model.loc["rv"])
        & (core_collapsed.rg == model.loc["rg"])
        & (core_collapsed.Z == model.loc["Z"])
    ]

    # remove outliers, filter out mergers
    # TODO: change to energy conservation criterion?
    singles = singles[(singles.type_f != 5)]

    # initialize binned array (n_age_groups, n_binsingle_types, n_bse_k_groups, n_vbins)
    data_binned = np.zeros(
        hvss_utils.BINNED_SHAPE,
        dtype=np.int32,
    )

    # print filtered dataframes
    # print('    dat.shape: ', dat.shape, '\n')
    print("\tsingles.shape: ", singles.shape)

    # vout histogram data
    print("\tBinning data...")
    for bti, binsingle_type in enumerate(binsingle_types):
        for kgi, bse_k_group in enumerate(bse_k_groups):
            for agi, age_group in enumerate(age_groups):
                if agi == 0:
                    age_group.lo = 0.0
                    age_group.hi = cc_row["TotalTime"].iloc[0]
                elif agi == 1:
                    age_group.lo = cc_row["TotalTime"].iloc[0]
                    age_group.hi = np.inf
                else:
                    raise Exception("agi not 0 or 1; unsupported.")
                filtered = singles[
                    (singles.time >= age_group.lo)
                    & (singles.time < age_group.hi)
                    & (singles.kf >= bse_k_group.lo)
                    & (singles.kf < bse_k_group.hi)
                    & (singles.type_i == binsingle_type.id)
                ]
                filtered = filtered.dropna(subset=["vout"])
                hist = [0] * (len(vbins) - 1)
                if not filtered.empty:
                    hist, _ = np.histogram(filtered["vout"], bins=vbins)
                data_binned[bti, kgi, agi, :] += hist
    np.savetxt(
        "{}_binned.txt".format(data_file_path),
        data_binned.flatten(),
        fmt="%d",
    )

    # add to appropriate part of catalog array
    print(
        "\tLooking for N={} in {}; found at index {}".format(
            model.loc["N"],
            N_list,
            np.where(N_list == model.loc["N"]),
        )
    )
    print(
        "\tLooking for rv={} in {}; found at index {}".format(
            model.loc["rv"],
            rv_list,
            np.where(rv_list == model.loc["rv"]),
        )
    )
    catalog[
        np.where(N_list == model.loc["N"]),
        np.where(rv_list == model.loc["rv"]),
        :,
        :,
        :,
        :,
    ] += data_binned

# save full catalog data
data_file_path = hvss_utils.DATA_PATH + "catalog/{}_N-{}".format(
    hvss_utils.FB_OUTPUT_FILENAME, N
)
os.makedirs(hvss_utils.DATA_PATH + "catalog", exist_ok=True)
np.savetxt(
    "{}_catalog.txt".format(data_file_path),
    catalog.flatten(),
    fmt="%d",
)
