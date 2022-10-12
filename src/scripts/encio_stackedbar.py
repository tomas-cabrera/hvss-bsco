import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import rustics

################################################################################

plt.style.use(rustics.PATH_TO_MPLRC)

# Get model info
df_models = rustics.DF_MODELS
#df_models = df_models.iloc[:10,:]

# Toggle MW GC weighting
weight_mw = True

# Initialize stackedbar generator
sbg = rustics.StackedBarGenerator(chunksize=None, nprocs=4, discrim_ejections=True)

# If histograms have been saved:
fname = "df_models_stackedbar.npy"
if os.path.exists(fname):
    temp = np.load(fname)
    df_models["stackedbar"] = [temp[i,:,:,:] for i in range(temp.shape[0])]
else:
# Generate histograms
    temp = sbg.generate_catalog_stackedbars(df_models, pm_pbar=True)
    np.save(fname, np.stack(temp))
    df_models["stackedbar"] = temp 

# Apply MW GC weights mask
if weight_mw:
    df_models["stackedbar"] *= df_models["weights_mw"]

# Build up stacked bars, one type_f at a time
bottoms = {
    "ejected": [0] * (rustics.INFO_TYPES_I.shape[0] - 1),
    "totals": [0] * (rustics.INFO_TYPES_I.shape[0] - 1),
}
cunits = 5
sbs = df_models["stackedbar"].sum() / 10 ** cunits
if sbg.discrim_ejections:
    ti_totals = sbs.sum(axis=(1, 2))
else:
    ti_totals = sbs.sum(axis=1)
print("ti_totals:", ti_totals)
legend_artists = []
legend_labels = []

mosaic = np.array([["totals"],["ejected"]])
fig, axd = plt.subplot_mosaic(
    mosaic,
    figsize=[3,4],
    sharex=True,
    gridspec_kw={
        "hspace": 0.,
    },
)
for tf, tf_row in rustics.INFO_TYPES_F.iterrows():
    y = sbs[1:, tf, :].sum(axis=1)
    print("tf=%d: " % tf, y)
    bars = axd["totals"].bar(
        rustics.INFO_TYPES_I.loc[1:, "label"],
        y,
        bottom=bottoms["totals"],
        label=tf_row.label,
        color=tf_row.color,
    )
    bottoms["totals"] += y
    y = sbs[1:, tf, 1]
    axd["ejected"].bar(
        rustics.INFO_TYPES_I.loc[1:, "label"],
        y,
        bottom=bottoms["ejected"],
        label=tf_row.label,
        color=tf_row.color,
    )
    bottoms["ejected"] += y
    legend_artists.append(bars)
    legend_labels.append(tf_row.label)
axd["totals"].set_ylabel(r"$N_{\rm enc} [10^{%d}$]" % cunits)
axd["ejected"].set_xlabel("Encounter type")
axd["ejected"].set_ylabel(r"$N_{\rm enc,ej} [10^{%d}$]" % cunits)
for ax in axd:
    axd[ax].tick_params(axis="x", which="both", top=False, bottom=False)
axd["totals"].legend(
    title="Result",
)
plt.tight_layout()

# Save and close
fname = "encio_stackedbar_catalog"
if weight_mw:
    fname = "_".join((fname, "wmw"))
if sbg.discrim_ejections:
    fname = "_".join((fname, "dej"))
fname = ".".join((fname, "pdf"))
plt.savefig("/".join((rustics.PATH_TO_FIGURES, fname)))
plt.close()
