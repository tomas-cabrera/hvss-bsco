import numpy
import pandas as pd
import matplotlib.pyplot as plt

from cmctools import cmctoolkit
###############################################################################

###############################################################################

def get_unitdict(path):
    """
    Wrapper for cmctoolkit.make_unitdict, to avoid many copies of the readfile lines.
    """
    f = open(path, 'r')
    convfile = f.read().split('\n')
    f.close()
    unitdict = cmctoolkit.make_unitdict(convfile)
    return unitdict   

def load_dyn_dat(path, convert_time=False):
    """
    Load the specified cmc.dyn.dat file, returning as a polars dataframe.
    As per the cmc.dyn.dat convention, the first line is ignored,
        and the second line is used to prescribe the column names.
    Inputs:
        path = path to the cmc.dyn.dat file
    Outputs:
        df = polars dataframe of data
    """

    # Read the second line to get headers
    with open(path) as file:
        for li, l in enumerate(file):
            if li == 1:
                cols = l
                break

    # Process columns, stripping numbers
    # "." split is necessary because of inconsistency in headers
    # "\n" strip applied to last label
    cols = [s.split(":")[-1].split(".")[-1] for s in cols.split(" ")]
    cols[-1] = cols[-1].strip("\n")

    # Load data
    df = pd.read_csv(
        path,
        names=cols,
        delimiter=" ",
        skiprows=2,
    )
    
    return df   

def load_bh_dat(path, convert_time=False):
    """
    Load the specified cmc.dyn.dat file, returning as a polars dataframe.
    As per the cmc.dyn.dat convention, the first line is ignored,
        and the second line is used to prescribe the column names.
    Inputs:
        path = path to the cmc.dyn.dat file
    Outputs:
        df = polars dataframe of data
    """

    # Read the second line to get headers
    with open(path) as file:
        for li, l in enumerate(file):
            if li == 0:
                cols = l
                break

    # Process columns, stripping numbers
    # "." split is necessary because of inconsistency in headers
    # "\n" strip applied to last label
    cols = [s.split(":")[-1].split(".")[-1] for s in cols.split("[")[0].replace("  ", " ").split(" ")]
    cols[-1] = cols[-1].strip("\n")
    print(cols)
    # Load data
    df = pd.read_csv(
        path,
        names=cols,
        delimiter=" ",
        skiprows=2,
    )
    
    return df

def load_lagrad_dat(path):
    """
    Load the specified cmc.lagrad.dat file, returning as a polars dataframe.
    As per the cmc.dyn.dat convention, the first line is ignored,
        and the second line is used to prescribe the column names.
    Inputs:
        path = path to the cmc.dyn.dat file
    Outputs:
        df = polars dataframe of data
    """

    # Read the second line to get headers
    with open(path) as file:
        for li, l in enumerate(file):
            if li == 1:
                cols = l
                break

    # Process columns, stripping numbers
    # "." split is necessary because of inconsistency in headers
    # "\n" strip applied to last label
    cols = [s.split(":")[-1] for s in cols.split(" ")]
    cols[-1] = cols[-1].strip("\n")

    # Load data
    df = pd.read_csv(
        path,
        names=cols,
        delimiter=" ",
        skiprows=1,
    )
    
    return df

def load_esc_dat(path, columns=None):
    """
    Load the specified cmc.esc.dat file, returning as a polars dataframe.
    As per the cmc.esc.dat convention, the first line is used to prescribe the column names.
    Inputs:
        path = path to the cmc.esc.dat file
        columns = esc.dat headers to load (stripped of #/colon prefix)
    Outputs:
        df = polars dataframe of data
    """

    # Read the second line to get headers
    with open(path) as file:
        for li, l in enumerate(file):
            if li == 0:
                cols = l
                break

    # Process columns, stripping numbers
    # "." split is necessary because of inconsistency in headers
    # "\n" strip applied to last label
    cols = [s.split(":")[-1].split(".")[-1] for s in cols.split(" ")]
    cols[-1] = cols[-1].strip("\n")

    # Load data
    df = pd.read_csv(
        path,
        names=cols,
        delimiter=" ",
        skiprows=2,
    )
    
    return df

#     # Read the first line to get headers
#     with open(path) as file:
#         for li, l in enumerate(file):
#             if li == 0:
#                 cols = l
#                 break

#     # Process columns, stripping numbers
#     # "\n" strip applied to last label
#     cols = [s.split(":")[-1] for s in cols.split(" ")]
#     cols[-1] = cols[-1].strip("\n")
    
#     # Generate column indices to load, if applicable
#     cis = [cols.index(c) for c in columns]
#     cols = [cols[ci] for ci in cis]
#     dtypes = [
#             pl.datatypes.UInt64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.UInt64,
#             pl.datatypes.UInt64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.UInt64,
#             pl.datatypes.UInt64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.UInt64,
#             pl.datatypes.UInt64,
#             pl.datatypes.UInt64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.UInt64,
#             pl.datatypes.UInt64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.UInt64,
#             pl.datatypes.UInt64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.Float64,
#             pl.datatypes.UInt64,
#     ]
#     dtypes = [dtypes[ci] for ci in cis]

#     # Load data
#     df = pl.read_csv(
#         path,
#         has_header=False,
#         columns=cis,
#         new_columns=cols,
#         sep=" ",
#         skip_rows=1,
#         dtypes=dtypes,
#     )
    
#     return df

def load_core_dat(path, columns=None):
    """
    Load the specified cmc.core.dat file, returning as a polars dataframe.
    As per the cmc.core.dat convention, the first line is used to prescribe the column names.
    Inputs:
        path = path to the cmc.core.dat file
        columns = core.dat headers to load (stripped of #/colon prefix)
    Outputs:
        df = polars dataframe of data
    """

    # Read the first line to get headers
    with open(path) as file:
        for li, l in enumerate(file):
            if li == 0:
                cols = l
                break

    # Process columns, stripping numbers
    # "\n" strip applied to last label
    cols = [s.split(":")[-1] for s in cols.split(" ")]
    cols[-1] = cols[-1].strip("\n")
    
    # Generate column indices to load, if applicable
    cis = [cols.index(c) for c in columns]
    cols = [cols[ci] for ci in cis]
    dtypes = [
            pl.datatypes.Float64,
            pl.datatypes.Float64,
            pl.datatypes.Float64,
            pl.datatypes.Float64,
            pl.datatypes.Float64,
            pl.datatypes.Float64,
            pl.datatypes.Float64,
            pl.datatypes.UInt64,
            pl.datatypes.Float64,
    ]
    dtypes = [dtypes[ci] for ci in cis]

    # Load data
    df = pl.read_csv(
        path,
        has_header=False,
        columns=cis,
        new_columns=cols,
        sep=" ",
        skip_rows=1,
        dtypes=dtypes,
    )
    
    return df

###############################################################################

# Path/to/matplotlib/style/file
MPL_STYLE_FILE_PATH = "/hildafs/projects/phy200025p/tcabrera/matplotlib_style_file"