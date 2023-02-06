"""! @brief A module to combine data from several catalogs """
###############################################################################

import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import parmap

import cmctoolkit as cmctoolkit

###############################################################################


class CMCCatalog:
    """! The class for a catalog of CMC models. """

    def __init__(self, path, extension="02", mp_nprocs=1):
        """! Initialize the catalog. 

        @param  path            Path to the catalog.
                                Should be a path to a directory that contains a folder for each of the models.
        @param  extension       File extension on names; will be stripped, but is expected at end of all names.
                                Note that default of "02" here a dirty way of keeping Kremer+20-like models.
        @param  mp_nprocs       Number of processes to use for parallel tasks.

        """

        # Store things
        self.path = path
        self.mp_nprocs = mp_nprocs

        # Get list of folder names
        fnames = os.listdir(path)

        # Make catalog dataframe, starting with fnames
        self.df = pd.DataFrame(
            [n for n in fnames if n.endswith(extension)], columns=["fname"]
        ).set_index("fname")

    def parse_names(self, replace={"_v2": ""}, cat_type="Kremer+20"):
        """! Function to extract model parameters from names.
        NB: The dtypes of parameter columns are reset whenever a .dat dataset is loaded.

        @param  replace     Dictionary where the keys will be replaced with the values whenever they appear in the model name.
                            Default removes any "_v2"s in the model names.
        @param  cat_type    Name format.
                            Default is "Kremer+20", i.e. the "N[_v2]_rv_rg_Z" convention from the respective catalog.

        """

        # TODO: move this to _parse_model_name
        # Replace strings
        fnames = self.df.index.get_level_values("fname")
        for item in replace.items():
            fnames = [n.replace(*item) for n in fnames]

        # Extract params
        params = [self._parse_model_name(n, cat_type=cat_type) for n in fnames]
        params = pd.DataFrame(params, index=self.df.index)

        # Join params to dataframe
        self.df = self.df.join(params)

    def _parse_model_name(self, model_name, cat_type="Kremer+20"):
        """! Parse model names.  Currently only the base one is implemented.

        @param  model_name      Name of model
        
        @return model_params    Dictionary of model parameters.

        """

        # For names like the Kremer+20 catalog:
        if cat_type == "Kremer+20":
            # Split and parse
            model_params = dict(zip(["N", "rv", "rg", "Z"], model_name.split("_")))
            for k in model_params.keys():
                if k == "N":
                    model_params[k] = int(float(model_params[k].replace(k, "")))
                else:
                    model_params[k] = float(model_params[k].replace(k, ""))
        else:
            raise Exception("cat_type '%s' not implemented." % cat_type)

        return model_params

    def add_dat_timesteps(
        self, dat_file, tmin=0.0, tmax=14000.0, tnum=50, dat_kwargs={},
    ):
        """! Method for adding data from the .dat files to the current df.

        @param  dat_file        Name of the dat file to read, e.g. "initial.dyn.dat".
                                Should be the same for all models in the catalog.
        @param  tmin            See tnum.
        @param  tmax            See tnum.
        @param  tnum            Number of timesteps to include for each model.
                                Timesteps selected are those closest to the times in np.linspace(tmin, tmax, tnum).
                                If tcount already exists in the dataframe, then these three parameters are ignored.
                                dat_times[np.abs(np.subtract.outer(dat_times, times_to_match)).argmin(axis=0)]
        @param  dat_kwargs      Dictionary of kwargs for _dat_file class initialization.

        """

        # Always load "tcount"
        if "tcount" not in dat_kwargs["pd_kwargs"]["usecols"]:
            dat_kwargs["pd_kwargs"]["usecols"].append("tcount")

        # Split workflow depending on whether there are already dat data
        if "tcount" in self.df.columns:
            # TODO
            pass
        else:
            # Make default timesteps
            timesteps = np.linspace(tmin, tmax, num=tnum)

            # Get data
            if self.mp_nprocs == 1:
                dat_data = parmap.map(
                    self._select_dat_data,
                    self.df.index,
                    dat_file,
                    timesteps,
                    dat_kwargs=dat_kwargs,
                    pm_parallel=False,
                    pm_pbar=True,
                )
            else:
                pool = mp.Pool(self.mp_nprocs)
                dat_data = parmap.map(
                    self._select_dat_data,
                    self.df.index,
                    dat_file,
                    timesteps,
                    dat_kwargs=dat_kwargs,
                    pm_pool=pool,
                    pm_pbar=True,
                )

        # Make df out of output
        dat_data = pd.concat(dat_data)
        # Get a row for every row in dat_data, applying the new index
        self.df = pd.DataFrame(
            [self.df.loc[k] for k, _ in dat_data.index], index=dat_data.index
        )
        # Join the two dfs
        self.df = self.df.join(dat_data)

    def _select_dat_data(self, fname, dat_file, timesteps, dat_kwargs={}):
        """! Method for getting the dat data for one model, for parallelization across models. """

        # Get model row
        model_row = self.df.loc[fname]
        print(fname)

        # Get dat type
        out_type = dat_file.split(".")[
            1
        ]  # TODO: this is not robust, since at least one .dat file has something like "0.1" in the name

        # Load time data too
        tkeys = {"dyn": "t"}
        tkey = tkeys[out_type]
        if tkey not in dat_kwargs["pd_kwargs"]["usecols"]:
            dat_kwargs["pd_kwargs"]["usecols"].append(tkey)

        # Read files
        if out_type == "dyn":
            dat = dyn_dat(
                "/".join((self.path, fname, dat_file)), **dat_kwargs
            )
        else:
            dat = _dat_file(
                "/".join((self.path, fname, dat_file)), **dat_kwargs
            )

        # Convert time
        dat.convert_units({tkey: "myr"})

        # Promote tcount to index
        dat.df["fname"] = fname
        dat.df.set_index(["fname", "tcount"], inplace=True)

        # Throw out all times out of range, returning the empty df if no timesteps remain
        dat.df = dat.df[
            (dat.df[tkey] >= timesteps.min()) & (dat.df[tkey] <= timesteps.max())
        ]
        if dat.df.shape[0] == 0:
            return dat.df

        # Select times and respective data
        time_indices = np.abs(
            np.subtract.outer(dat.df[tkey].to_numpy(), timesteps)
        ).argmin(axis=0)
        dat.df = dat.df.iloc[time_indices]

        return dat.df
###############################################################################

import numpy as np
import pandas as pd

import cmctoolkit as cmctoolkit

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


class GCCatalog:
    def __init__(self, paths, pd_kwargs={}):
        """! Basic things for now."""

        # If a single path is specified, assume it is a loadable csv
        if type(paths) == str:
            self.df = pd.read_csv(paths, **pd_kwargs)

        # If a list of paths is specified, assume it they are baumgardt and harris catalogs 
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

    def match_to_cmc_models(
        self, cmc_path, cmc_kwargs={}, dyn_kwargs={},
    ):
        """! Given a path to a CMC directory, finds the matching CMC models for the GCs."""

        # Save cmc_path
        self.cmc_path = cmc_path

        cmc_models = CMCCatalog(cmc_path, extension=".tar.gz", mp_nprocs=56, **cmc_kwargs)
        cmc_models.add_dat_timesteps(
            "initial.dyn.dat",
            tmin=10000.0,
            tmax=13500.0,
            tnum=10,
            dat_kwargs={
                "pd_kwargs": {"usecols": ["t", "M", "rc_spitzer", "r_h"]},
                "convert_units": {"t": "myr", "M": "msun", "rc_spitzer": "pc", "r_h": "pc"},
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

        # Normalize 
        params_compare = ["logM", "rc/rh"]
        for pi, p in enumerate(params_compare):
            pm = cmc_models.df[p].mean()
            ps = cmc_models.df[p].std()
            pn = "%s_norm" % p
            cmc_models.df[pn] = (cmc_models.df[p] - pm) / ps
            self.df[pn] = (self.df[p] - pm) / ps
            params_compare[pi] = pn

        self.df.to_csv("mwgcs.dat")
        cmc_models.df.to_csv("cmcs.dat")

        # Iterate through rg, met bins (bin edges are average of adjacent CMC rg values)
        rgs_cmc = np.sort(cmc_models.df.rg.unique())
        mets_cmc = np.sort(cmc_models.df["[Fe/H]"].unique())
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
        self.df.to_csv("mwgcs-cmc.dat")
        print(self.df)

mwcat = GCCatalog(
    ["/hildafs/projects/phy200025p/tcabrera/hvss_old_and_complete/holger_baumgardt_clean.txt",
    "/hildafs/projects/phy200025p/tcabrera/hvss_old_and_complete/harris2010_II_clean.txt"],
)
print(mwcat.df)
mwcat.match_to_cmc_models(
    "/hildafs/projects/phy200025p/share/catalog_files",
)
