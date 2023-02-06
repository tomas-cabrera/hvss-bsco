"""! @brief A module to combine data from several catalogs """

###############################################################################

import numpy as np
import pandas as pd

###############################################################################


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

        # Load cmc catalog, and dyn.dat data
        import mywheels.cmcutils.readcatalog as cmccat

        cmc_models = cmccat.CMCCatalog(cmc_path, extension=".tar.gz", mp_nprocs=128, **cmc_kwargs)
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
