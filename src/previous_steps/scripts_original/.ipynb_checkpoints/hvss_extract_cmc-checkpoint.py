import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import hvss_utils


################################################################################


def get_mbhx(data):
    """
    Gets the maximum black hole mass of each encounter.
    """

    masses = data.loc[
        :,
        (
            "m0",
            "m10",
            "m11",
        ),
    ]
    masses *= (
        data.loc[
            :,
            (
                "k0",
                "k10",
                "k11",
            ),
        ]
        == 14
    )

    return masses.max(axis=1)


################################################################################


N = 10

print("Starting iteration...")

cmc_cluster_list = hvss_utils.CMC_CLUSTER_LIST_V2

for cmc_cluster in tqdm(cmc_cluster_list):
    print("{}, N={}".format(cmc_cluster, N))

    # Defines path to data for this cluster
    data_file_path = (
        hvss_utils.DATA_PATH
        + cmc_cluster
        + "/{}_N-{}".format(hvss_utils.FB_OUTPUT_FILENAME, N)
    )

    # Clear old files (DataFrame.to_csv set to append, in order to process in chunks)
    try:
        open("{}_pro.txt".format(data_file_path), "w").close()
    except Exception as e:
        print("\tError in clearing pro file:")
        print("\t\t", e)
    try:
        open("{}_singles.txt".format(data_file_path), "w").close()
    except Exception as e:
        print("\tError in clearing singles file:")
        print("\t\t", e)

    # Load data
    print("\tLoading data...")
    try:
        data = pd.read_csv(
            "{}.txt".format(data_file_path),
            names=hvss_utils.PRO_HEADERS[:-7],
            chunksize=1e4,
        )
    except Exception as e:
        print("\tError in loading data for {}, N={}:".format(cmc_cluster, N))
        print("\t\t", e)
        print("\tContinuing...")
        continue

    # Convert units, calculate additional quantities, save to pro file
    print("\tGenerating pro data...")
    for chunk in data:
        chunk = chunk.astype(np.float32)
        for r in (
            "a_fin",
            "Rmin0",
            "Rmin1",
            "Rmin2",
        ):
            chunk[r] *= chunk["a1"]
        for v in (
            "vinf",
            "vfin0",
            "vfin1",
            "vfin2",
        ):
            chunk[v] *= chunk["v_crit"]
        for L in (
            "Lx",
            "Ly",
            "Lz",
            "Lbinx",
            "Lbiny",
            "Lbinz",
        ):
            chunk[L] *= chunk["v_crit"] ** 2 * chunk["a1"] ** 3
        for ri, row in chunk.iterrows():
            chunk.loc[ri, "type_f"] = hvss_utils.convert_type_f(
                row["type_f"], row["k10"], row["k11"], row["k0"]
            )
        chunk["L"] = (chunk["Lx"] ** 2 + chunk["Ly"] ** 2 + chunk["Lz"] ** 2) ** 0.5
        chunk["Lbin"] = (
            chunk["Lbinx"] ** 2 + chunk["Lbiny"] ** 2 + chunk["Lbinz"] ** 2
        ) ** 0.5
        chunk["cos_theta"] = chunk["Lbinz"] / chunk["Lbin"]
        chunk["mbhx"] = get_mbhx(chunk)
        for i in range(3):
            chunk["vout{}".format(i)] = (
                chunk["vfin{}".format(i)] ** 2 - chunk["vesc"] ** 2
            ) ** 0.5
        chunk.to_csv(
            "{}_pro.txt".format(data_file_path), header=False, index=False, mode="a"
        )

    # Extract single objects
    print("\tGenerating singles data...")
    data = pd.read_csv(
        "{}_pro.txt".format(data_file_path), names=hvss_utils.PRO_HEADERS, chunksize=1e4
    )
    head0 = hvss_utils.SINGLES_HEADERS[:-5] + (
        "vfin0",
        "kf0",
        "Rmin0",
        "Rmin_j0",
        "vout0",
    )
    head1 = hvss_utils.SINGLES_HEADERS[:-5] + (
        "vfin1",
        "kf1",
        "Rmin1",
        "Rmin_j1",
        "vout1",
    )
    head2 = hvss_utils.SINGLES_HEADERS[:-5] + (
        "vfin2",
        "kf2",
        "Rmin2",
        "Rmin_j2",
        "vout2",
    )
    for chunk in data:
        data0 = chunk.loc[:, head0]
        data1 = chunk.loc[:, head1]
        data2 = chunk.loc[:, head2]
        data0.rename(
            columns={
                "vfin0": "vfin",
                "kf0": "kf",
                "Rmin0": "Rmin",
                "Rmin_j0": "Rmin_j",
                "vout0": "vout",
            },
            inplace=True,
        )
        data1.rename(
            columns={
                "vfin1": "vfin",
                "kf1": "kf",
                "Rmin1": "Rmin",
                "Rmin_j1": "Rmin_j",
                "vout1": "vout",
            },
            inplace=True,
        )
        data2.rename(
            columns={
                "vfin2": "vfin",
                "kf2": "kf",
                "Rmin2": "Rmin",
                "Rmin_j2": "Rmin_j",
                "vout2": "vout",
            },
            inplace=True,
        )
        temp = pd.concat(
            (
                data0,
                data1,
                data2,
            )
        )
        temp[temp.kf >= 0].to_csv(
            "{}_singles.txt".format(data_file_path), header=False, index=False, mode="a"
        )
