from astropy.coordinates import SkyCoord, ICRS, Galactic, Galactocentric
from astropy import units as u

import rustics
import paths

# Load ejections data, while only loading the column that matters
# Column loading is problematic with unit conversions
# Special cases for columns that are functions of the ejections columns
path = paths.data_mwgcs / "NGC_6388" / "output_N-10_ejections.txt" 
# Specify columns to load, and add "kf" if kgroup filtering is enabled
usecols = ["X", "Y", "Z", "U", "V", "W"]
# Load, and convert units
ejdf = rustics.EjectionDf(path, usecols=usecols)
ejdf.convert_from_fewbody()
print(ejdf.df)

# Create coordinate object; convert to ICRS
sc_ejs = {
    "x": ejdf.df.X * u.kpc,
    "y": ejdf.df.Y * u.kpc,
    "z": ejdf.df.Z * u.kpc,
    "v_x": ejdf.df.U * u.km / u.s,
    "v_y": ejdf.df.V * u.km / u.s,
    "v_z": ejdf.df.W * u.km / u.s,
}
print(sc_ejs)
sc_ejs = SkyCoord(frame=Galactocentric, **sc_ejs)
sc_ejs = sc_ejs.transform_to(Galactic)
print(sc_ejs)
ejdf.df["vlos"] = sc_ejs.radial_velocity.value
print(ejdf.df)
