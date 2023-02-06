"""! @brief This module loads the standard CMC output files into a pandas DataFrame."""

###############################################################################

import numpy as np
import pandas as pd

import cmctoolkit

###############################################################################


class _dat_file:
    """! The parent class for the individual files. """

    def __init__(
        self, path, header_line_no=0, colon_aliases=[], pd_kwargs={}, convert_units={},
    ):
        """! Read the data.

        @param  path            The path to the output file.
        @param  header_line_no  The line to read the columns from.
        @param  colon_aliases   List-like of character to replace with colons (e.g. ["."] for dyn.dat files)
        @param  pd_kwargs       Dictionary of keyword arguments to pass to pd.read_csv when loading data.
                                Merges the specified dict with the default to preserve defaults.
        @param  convert_units   Dictionary of units to convert (see _dat_file.convert_units)

        """

        # Save path
        self.path = path

        # Get the column names
        names = self._get_column_names(path, header_line_no, colon_aliases)

        # Merge specified pd_kwargs dist with default
        pd_kwarg_defaults = {
            "names": names,
            "delimiter": " ",
            "skiprows": header_line_no + 1,
            "low_memory": False,
        }
        pd_kwargs = {**pd_kwarg_defaults, **pd_kwargs}

        # Load data, passing usecols to select particular columns
        self.df = pd.read_csv(path, **pd_kwargs,)

        # Note that units are not converted from original values
        self.units_converted = {n: False for n in self.df.columns}

        # Convert any specified units on load
        self.convert_units(convert_units)

    def _get_column_names(self, path, header_line_no, colon_aliases=[]):
        """! Parses the specified line in the file to get the header names.

        @return columns     The column names.

        """

        # Read the header line
        with open(path) as file:
            for li, l in enumerate(file):
                if li == header_line_no:
                    names = l
                    break

        # Split at whitespace
        names = names.split()

        # Keep everything after the colon (using aliases if needed
        for ca in colon_aliases:
            names = [c.replace(ca, ":") for c in names]
        names = [c.split(":")[1] for c in names]

        return names

    def convert_units(self, names_units, conv_fname=None, missing_ok=False):
        """! Converts specified column(s) with specified conv.sh file.

        @param  names_units     Dictionary, where the keys are the column names to convert and the values are strings of the conv.sh values to use.
                                " * " and " / " may be used to multiply and divide units, e.g. a value of "cm / nb_s" will make the conversion from nbody velocity to cm/s.
        @param  conv_fname      Path to the conv.sh file.
                                By default, this swaps out the file name for "initial.conv.sh"
        @param  missing_ok      If True, continues if a name is not found in the df column names.

        """

        # Make filename, if not specified
        if conv_fname == None:
            conv_fname = "/".join((*self.path.split("/")[:-1], "initial.conv.sh"))

        # Load unitdict
        unitdict = self._read_unitdict(conv_fname)

        for n, u in names_units.items():
            # Check if the specified unit is in the df
            if n in self.df.columns:
                # Check if the units have already been converted
                if not self.units_converted[n]:
                    # Convert units, and mark as converted
                    self.df[n] *= self._parse_units_string(u, unitdict)
                    self.units_converted[n] = True
            elif not missing_ok:
                raise Exception("Name '%s' not found in columns" % n)

        return 0

    def _read_unitdict(self, conv_fname):
        """! Reads unitdict from conv.sh.  This function is here to wrap the opening of the file with the cmctoolkit fucntion. """

        f = open(conv_fname, "r")
        conv_file = f.read().split("\n")
        f.close()
        unitdict = cmctoolkit.make_unitdict(conv_file)

        return unitdict

    def _parse_units_string(self, string, unitdict):
        factor = 1.0
        op = "*"
        for u in string.split():
            if u == "*":
                op = "*"
            elif u == "/":
                op = "/"
            elif op == "*":
                factor *= unitdict[u]
                op = None
            elif op == "/":
                factor /= unitdict[u]
                op = None
            else:
                raise Exception("Unrecognized sequence in string")

        return factor


class dyn_dat(_dat_file):
    """! Reads the dyn.dat data. """

    def __init__(self, path, **kwargs):
        """! Reads the data, using some defaults for the dyn_dat files. """
        super().__init__(path, header_line_no=1, colon_aliases=["."], **kwargs)

    def convert_tunits(self, **kwargs):
        """! By default converts the t and Dt columns into Myr. """
        self.convert_units({"t": "myr", "Dt": "myr"}, **kwargs)
