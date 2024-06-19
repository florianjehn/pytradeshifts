# This code is just meant to multiply the data from the
# two scenarios together to get the combined scenario
# which represents the combined impact of nuclear winter
# and global catastrophic infrastructure loss.

import pandas as pd
import country_converter as coco
import os

# Load the data from the two scenarios
crops = ["Wheat", "Maize", "Rice", "Soybean"]
for crop in crops:
    print(f"Calculating combined scenario for {crop}...")
    # First read the data from the nuclear winter scenario
    if crop == "Wheat":
        xia_naming = "swheat"
    elif crop == "Rice":
        xia_naming = "rice"
    elif crop == "Soya Beans":
        xia_naming = "soy"
    elif crop == "Maize":
        xia_naming = "corn"
    print("Reading data for nuclear winter")
    nw = pd.read_csv(f"data{os.sep}scenario_files{os.sep}nuclear_winter{os.sep}xia_37tg_y3_{xia_naming}.csv", index_col=0)

    # Then read the data from the global catastrophic infrastructure loss scenario
    if crop == "Maize":
        crop_name = "Corn"
    else:
        crop_name = crop
    print("Reading data for global catastrophic infrastructure loss")
    gcil = pd.read_csv(f"data{os.sep}scenario_files{os.sep}losing_industry{os.sep}{crop_name}2mean_values.csv", index_col=0)

    # Change to fractions
    nw = nw / 100 + 1
    gcil = gcil / 100 + 1

    # Multiply the two dataframes together, but first make sure that the country names are unified
    cc = coco.CountryConverter()
    print("Converting country names for nuclear winter")
    nw.index = cc.pandas_convert(
        pd.Series(nw.index), to="name_short", not_found=None
    )
    print("Converting country names for global catastrophic infrastructure loss")
    gcil.index = cc.pandas_convert(
        pd.Series(gcil.index), to="name_short", not_found=None
    )

    # Multiply the two dataframes together if the country is in both
    # If the country is not in both, then the value is the value from the scenario
    # that is available
    print("Multiplying the two dataframes")
    combined = pd.concat([nw, gcil], axis=1, join="inner")
    combined = combined.dropna()
    nw = nw.loc[combined.index]
    gcil = gcil.loc[combined.index]
    
    combined = nw["37tg"] * gcil["mean_value"]

    # Change back to percentages
    combined = (combined - 1) * 100

    # Save the combined data
    print("Saving the combined data")
    combined.to_csv(f"data{os.sep}scenario_files{os.sep}GCIL_plus_NW{os.sep}combined_{crop}.csv")
