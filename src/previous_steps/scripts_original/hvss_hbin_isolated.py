import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import hvss_utils

################################################################################


N = 1000000
vbins = hvss_utils.VBINS
binsingle_types = hvss_utils.BINSINGLE_TYPES_ISOLATED
bse_k_groups = hvss_utils.BSE_K_GROUPS
age_groups = hvss_utils.AGE_GROUPS

# Particular shorthands
data_file_path = hvss_utils.DATA_PATH + "isolated/{}_N-{}".format(
    hvss_utils.FB_OUTPUT_FILENAME, N
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
