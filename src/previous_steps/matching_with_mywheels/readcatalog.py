"""! @brief This module concerns reading data from a catalog of CMC models."""

###############################################################################

import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import parmap

import readoutput as readoutput

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
            dat = readoutput.dyn_dat(
                "/".join((self.path, fname, dat_file)), **dat_kwargs
            )
        else:
            dat = readoutput._dat_file(
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
