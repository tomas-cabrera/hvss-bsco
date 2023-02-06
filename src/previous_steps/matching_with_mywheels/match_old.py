import aggregate as ag

mwcat = ag.GCCatalog(
    ["/hildafs/projects/phy200025p/tcabrera/hvss_old_and_complete/holger_baumgardt_clean.txt",
    "/hildafs/projects/phy200025p/tcabrera/hvss_old_and_complete/harris2010_II_clean.txt"],
)
print(mwcat.df)
mwcat.match_to_cmc_models(
    "/hildafs/projects/phy200025p/share/catalog_files",
)
