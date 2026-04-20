from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# flake8: noqa: E501


def hand_grip_strength(
    dominant_hand,
    left_meas1,
    left_meas2,
    left_meas3,
    right_meas1,
    right_meas2,
    right_meas3,
    mean_max="mean",
):
    if mean_max == "mean":
        left = np.nanmean([left_meas1, left_meas2, left_meas3], axis=0)
        right = np.nanmean([right_meas1, right_meas2, right_meas3], axis=0)
    elif mean_max == "max":
        left = np.nanmax([left_meas1, left_meas2, left_meas3], axis=0)
        right = np.nanmax([right_meas1, right_meas2, right_meas3], axis=0)
    else:
        raise ValueError("mean_max must be 'mean' or 'max'")

    # Handle both English and German terms for dominant hand
    is_left_dominant = (dominant_hand == "Left") | (dominant_hand == "Links")

    dominant_hand_strength = np.where(is_left_dominant, left, right)
    non_dominant_hand_strength = np.where(is_left_dominant, right, left)

    return dominant_hand_strength, non_dominant_hand_strength


def study_converter(
    db_csv, additional_file, study_name, dir_path
):  # additional_file if more than one file is needed, else additional_file=1

    if study_name == "Nodaratis":
        hand_grip_strength_dominant, hand_grip_strength_non_dominant = (
            hand_grip_strength(
                db_csv["Dominante Hand"],
                db_csv["Hand Grip links Messung 1"],
                db_csv["Hand Grip links Messung 2"],
                db_csv["Hand Grip links Messung 3"],
                db_csv["Hand Grip rechts Messung 1"],
                db_csv["Hand Grip rechts Messung 2"],
                db_csv["Hand Grip rechts Messung 3"],
                mean_max="max",
            )
        )

        df = pd.DataFrame(
            {
                "UID": db_csv["Record ID"],
                "Gender": db_csv["Geschlecht"],
                "Ethnicity": db_csv["Ethnizität"],
                "Age": db_csv["Alter bei Untersuchung in Jahren"],
                "Weight": db_csv["Gewicht [kg]"],
                "Height": db_csv["Grösse [cm]"],
                "Study name": "Nodaratis",
                "Previous fracture": db_csv["Frühere Fraktur(en)"],
                "Parent fractured hip": db_csv[
                    "Hüftfraktur des Vaters/der Mutter im Erwachsenenalter"
                ],
                "Current smoking": db_csv["Raucher aktuell"],
                "Glucocorticoids": db_csv[
                    "Cortison/Prednison >=3 mg aktuell oder >=5 mg jemals"
                ],
                "Rheumatoid arthritis": db_csv["Rheumatoide Arthritis"],
                "Secondary osteoporosis": db_csv["Sekundäre Osteoporose"],
                "Alcohol 3 or more units/day": db_csv[
                    "Alkoholkonsum >=3 Einheiten/Tag aktuell"
                ],
                "Femoral neck BMD (g/cm2)": db_csv[
                    "Oberschenkelhals Knochenmineraldichte [g/cm2]"
                ],
                "Time difference Frax & HRpQCT [days]": 0,
                # Radius
                "Radius: Measurement number": db_csv[
                    "Radius HRpQCT scan 1 - Messungsnummer"
                ],
                "Radius: Measurement number2": db_csv[
                    "Radius HRpQCT scan 2 - Messungsnummer"
                ],
                "Radius: Measurement number3": db_csv[
                    "Radius HRpQCT scan 3 - Messungsnummer"
                ],
                "Radius: side": db_csv["Radius HRpQCT scan 1 - Seite der Messung"],
                # "Radius: Tot.Ar [mm2]":db_csv["Radius Total Area"],
                "Radius: Ct.Ar [mm2]": db_csv["Radius cortical area [mm2]"],
                "Radius: Tb.Ar [mm2]": db_csv["Radius trabecular area [mm2]"],
                "Radius: Tb.Meta.Ar [mm2]": db_csv["Radius trabecular meta area [mm2]"],
                "Radius: Tb.Inn.Ar [mm2]": db_csv["Radius trabecular inner area [mm2]"],
                "Radius: Ct.Pm [mm]": db_csv["Radius cortical perimeter [mm]"],
                "Radius: Tt.vBMD [mg HA/cmm]": db_csv[
                    "Radius total volumetric bone mineral density [mg HA/ccm]"
                ],
                "Radius: Ct.vBMD [mg HA/cmm]": db_csv[
                    "Radius cortical vBMD  [mg HA/ccm]"
                ],
                "Radius: Tb.vBMD [mg HA/cmm]": db_csv[
                    "Radius trabecular vBMD [mg HA/ccm]"
                ],
                "Radius: Tb.Meta.vBMD [mg HA/cmm]": db_csv[
                    "Radius meta trabecular vBMD (40% of Tb.Ar) [mg HA/ccm]"
                ],
                "Radius: Tb.Inn.vBMD [mg HA/cmm]": db_csv[
                    "Radius inner trabecular vBMD (60% of Tb.Ar) [mg HA/ccm]"
                ],
                "Radius: Tb.BV/TV [1]": db_csv[
                    "Radius trabecular bone volume fraction"
                ],
                "Radius: Ct.Th [mm]": db_csv["Radius cortical thickness [mm]"],
                "Radius: Ct.Po [1]": db_csv["Radius cortical porosity"],
                "Radius: Ct.Po.Dm [mm]": db_csv["Radius cortical pore diameter [mm]"],
                "Radius: Tb.N [1/mm]": db_csv["Radius trabecular number [1/mm]"],
                "Radius: Tb.Th [mm]": db_csv["Radius trabecular thickness [mm]"],
                "Radius: Tb.Sp [mm]": db_csv["Radius trabecular separation [mm]"],
                "Radius: SD of 1/Tb.N [mm]": db_csv[
                    "Radius SD of 1/Tb.N: Inhomogeneity of Network [-]"
                ],
                "Radius: Fmax at failure [N]": db_csv[
                    "Radius maximum force at failure adjusted [N]"
                ],
                "Radius: Stiffness [N/mm]": db_csv["Radius stiffness adjusted [N/mm]"],
                # "Radius: Fmax at failure adjusted [N]": db_csv[
                #     "Radius maximum force at failure adjusted [N]"
                # ],
                # "Radius: Stiffness adjusted [N/mm]": db_csv[
                #     "Radius stiffness adjusted [N/mm]"
                # ],
                # Tibia
                "Tibia: Measurement number": db_csv[
                    "Tibia HRpQCT scan 1 - Messungsnummer"
                ],
                "Tibia: Measurement number2": db_csv[
                    "Tibia HRpQCT scan 2 - Messungsnummer"
                ],
                "Tibia: Measurement number3": db_csv[
                    "Tibia HRpQCT scan 3 - Messungsnummer"
                ],
                "Tibia: side": db_csv["Tibia HRpQCT scan 1 - Seite der Messung"],
                # "Tibia: Tot.Ar [mm2]": db_csv["Tibia Total Area"],
                "Tibia: Ct.Ar [mm2]": db_csv["Tibia cortical area [mm2]"],
                "Tibia: Tb.Ar [mm2]": db_csv["Tibia trabecular area [mm2]"],
                "Tibia: Tb.Meta.Ar [mm2]": db_csv["Tibia trabecular meta area [mm2]"],
                "Tibia: Tb.Inn.Ar [mm2]": db_csv["Tibia trabecular inner area [mm2]"],
                "Tibia: Ct.Pm [mm]": db_csv["Tibia cortical perimeter [mm]"],
                "Tibia: Tt.vBMD [mg HA/cmm]": db_csv[
                    "Tibia total volumetric bone mineral density [mg HA/ccm]"
                ],
                "Tibia: Ct.vBMD [mg HA/cmm]": db_csv[
                    "Tibia cortical vBMD  [mg HA/ccm]"
                ],
                "Tibia: Tb.vBMD [mg HA/cmm]": db_csv[
                    "Tibia trabecular vBMD [mg HA/ccm]"
                ],
                "Tibia: Tb.Meta.vBMD [mg HA/cmm]": db_csv[
                    "Tibia meta trabecular vBMD (40% of Tb.Ar) [mg HA/ccm]"
                ],
                "Tibia: Tb.Inn.vBMD [mg HA/cmm]": db_csv[
                    "Tibia inner trabecular vBMD (60% of Tb.Ar) [mg HA/ccm]"
                ],
                "Tibia: Tb.BV/TV [1]": db_csv["Tibia trabecular bone volume fraction"],
                "Tibia: Ct.Th [mm]": db_csv["Tibia cortical thickness [mm]"],
                "Tibia: Ct.Po [1]": db_csv["Tibia cortical porosity"],
                "Tibia: Ct.Po.Dm [mm]": db_csv["Tibia cortical pore diameter [mm]"],
                "Tibia: Tb.N [1/mm]": db_csv["Tibia trabecular number [1/mm]"],
                "Tibia: Tb.Th [mm]": db_csv["Tibia trabecular thickness [mm]"],
                "Tibia: Tb.Sp [mm]": db_csv["Tibia trabecular separation [mm]"],
                "Tibia: SD of 1/Tb.N [mm]": db_csv[
                    "Tibia SD of 1/Tb.N: Inhomogeneity of Network [-]"
                ],
                "Tibia: Fmax at failure [N]": db_csv[
                    "Tibia maximum force at failure adjusted [N]"
                ],
                "Tibia: Stiffness [N/mm]": db_csv["Tibia stiffness adjusted [N/mm]"],
                # "Tibia: Fmax at failure adjusted [N]": db_csv[
                #     "Tibia maximum force at failure adjusted [N]"
                # ],
                # "Tibia: Stiffness adjusted [N/mm]": db_csv[
                #     "Tibia stiffness adjusted [N/mm]"
                # ],
                "Hand grip strength dominant [kg]": hand_grip_strength_dominant,
                "Hand grip strength non-dominant [kg]": hand_grip_strength_non_dominant,
            }
        )
        # masks radius
        mask2 = df["Radius: Measurement number2"].notnull()
        df.loc[mask2, "Radius: Measurement number"] = df.loc[
            mask2, "Radius: Measurement number2"
        ]
        mask3 = df["Radius: Measurement number3"].notnull()
        df.loc[mask3, "Radius: Measurement number"] = df.loc[
            mask3, "Radius: Measurement number3"
        ]

        # masks tibia
        mask2 = df["Tibia: Measurement number2"].notnull()
        df.loc[mask2, "Tibia: Measurement number"] = df.loc[
            mask2, "Tibia: Measurement number2"
        ]
        mask3 = df["Tibia: Measurement number3"].notnull()
        df.loc[mask3, "Tibia: Measurement number"] = df.loc[
            mask3, "Tibia: Measurement number3"
        ]

        # drop columns Radius: Measurement number2, Radius: Measurement number3, Tibia: Measurement number2, Tibia: Measurement number3
        df = df.drop(
            columns=[
                "Radius: Measurement number2",
                "Radius: Measurement number3",
                "Tibia: Measurement number2",
                "Tibia: Measurement number3",
            ]
        )

        df["Gender"] = df["Gender"].str.replace("Männlich", "Male")
        df["Gender"] = df["Gender"].str.replace("Weiblich", "Female")
        df["Ethnicity"] = df["Ethnicity"].str.replace("Weiss", "White")
        df["Ethnicity"] = df["Ethnicity"].str.replace("Schwarz", "Black")
        df["Ethnicity"] = df["Ethnicity"].str.replace("Asiatisch", "Asian")
        df["Ethnicity"] = df["Ethnicity"].str.replace("Hispanisch", "Hispanic")
        df["Ethnicity"] = df["Ethnicity"].str.replace("Andere", "Other")
        df["Current smoking"] = df["Current smoking"].str.replace("Nein", "No")
        df["Previous fracture"] = df["Previous fracture"].str.replace("Nein", "No")
        df["Previous fracture"] = df["Previous fracture"].str.replace("Ja", "Yes")
        df["Parent fractured hip"] = df["Parent fractured hip"].str.replace(
            "Nein", "No"
        )
        df["Parent fractured hip"] = df["Parent fractured hip"].str.replace("Ja", "Yes")
        df["Glucocorticoids"] = df["Glucocorticoids"].str.replace("Nein", "No")
        df["Glucocorticoids"] = df["Glucocorticoids"].str.replace("Ja", "Yes")
        df["Rheumatoid arthritis"] = df["Rheumatoid arthritis"].str.replace(
            "Nein", "No"
        )
        df["Rheumatoid arthritis"] = df["Rheumatoid arthritis"].str.replace("Ja", "Yes")
        df["Secondary osteoporosis"] = df["Secondary osteoporosis"].str.replace(
            "Nein", "No"
        )
        df["Secondary osteoporosis"] = df["Secondary osteoporosis"].str.replace(
            "Ja", "Yes"
        )
        df["Alcohol 3 or more units/day"] = df[
            "Alcohol 3 or more units/day"
        ].str.replace("Nein", "No")
        df["Alcohol 3 or more units/day"] = df[
            "Alcohol 3 or more units/day"
        ].str.replace("Ja", "Yes")
        df["Current smoking"] = df["Current smoking"].str.replace("Ja", "Yes")
        df["Radius: side"] = df["Radius: side"].str.replace("Links", "Left")
        df["Radius: side"] = df["Radius: side"].str.replace("Rechts", "Right")
        df["Tibia: side"] = df["Tibia: side"].str.replace("Links", "Left")
        df["Tibia: side"] = df["Tibia: side"].str.replace("Rechts", "Right")

        df["Radius: Fmax at failure [N]"] = (df["Radius: Fmax at failure [N]"]).abs()
        df["Tibia: Fmax at failure [N]"] = (df["Tibia: Fmax at failure [N]"]).abs()

    if study_name == "AFFIRM-CT":
        gender_mapping = {1: "Female", 2: "Male"}

        ethnicity_mapping = {
            "1": "White",
            "2": "Black",
            "3": "Asian",
            "4": "Hispanic",
            "88": "Other",
        }

        yes_no_map_str = {"1": "Yes", "0": "No"}
        yes_no_map_int = {1: "Yes", 0: "No"}

        side_xct_mapping_str = {1.0: "Right", 2.0: "Left"}
        side_xct_mapping_int = {1: "Right", 2: "Left"}

        is_left_dominant = db_csv["monito_dominant"] == "left"

        dominant_hand_strength = np.where(
            is_left_dominant, db_csv["grip_left"], db_csv["grip_right"]
        )
        non_dominant_hand_strength = np.where(
            is_left_dominant, db_csv["grip_right"], db_csv["grip_left"]
        )

        df = pd.DataFrame(
            {
                "UID": db_csv["record_id"],
                "Gender": db_csv["gender"].map(gender_mapping),
                "Ethnicity": db_csv["ethnicity"].map(ethnicity_mapping),
                "Age": db_csv["age_calc"],
                "Weight": db_csv["weight"],
                "Height": db_csv["height"],
                "Study name": "AFFIRM-CT",
                "Previous fracture": db_csv["frax_fracture_frax"].map(yes_no_map_str),
                "Parent fractured hip": db_csv["frax_parental_fracture_frax"].map(
                    yes_no_map_str
                ),
                "Current smoking": db_csv["frax_smoking_frax"].map(yes_no_map_str),
                "Glucocorticoids": db_csv["frax_glucocorticoid_frax"].map(
                    yes_no_map_str
                ),
                "Rheumatoid arthritis": db_csv["frax_arthritis_frax"].map(
                    yes_no_map_int
                ),
                "Secondary osteoporosis": db_csv["frax_sec_osteoporosis_frax"].map(
                    yes_no_map_int
                ),
                "Alcohol 3 or more units/day": db_csv["frax_alcohol_frax"].map(
                    yes_no_map_int
                ),
                "Femoral neck BMD (g/cm2)": db_csv["frax_bmd_frax_calc"],
                "Time difference Frax & HRpQCT [days]": 0,
                #
                "Hand grip strength dominant [kg]": dominant_hand_strength,
                "Hand grip strength non-dominant [kg]": non_dominant_hand_strength,
                #
                # Radius
                "Radius: Measurement number": db_csv["xct_radius_noanalysis"],
                "Radius: side": db_csv["radius_side_xct"].map(side_xct_mapping_str),
                # "Radius Tot.Ar [mm2]": db_csv["Radius Total Area"],
                "Radius: Ct.Ar [mm2]": db_csv["xct_radius_pateval_18"],
                "Radius: Tb.Ar [mm2]": db_csv["xct_radius_pateval_15"],
                "Radius: Tb.Meta.Ar [mm2]": db_csv["xct_radius_pateval_16"],
                "Radius: Tb.Inn.Ar [mm2]": db_csv["xct_radius_pateval_17"],
                "Radius: Ct.Pm [mm]": db_csv["xct_radius_pateval_6"],
                "Radius: Tt.vBMD [mg HA/cmm]": db_csv["xct_radius_pateval_1"],
                "Radius: Ct.vBMD [mg HA/cmm]": db_csv["xct_radius_pateval_5"],
                "Radius: Tb.vBMD [mg HA/cmm]": db_csv["xct_radius_pateval_2"],
                "Radius: Tb.Meta.vBMD [mg HA/cmm]": db_csv["xct_radius_pateval_3"],
                "Radius: Tb.Inn.vBMD [mg HA/cmm]": db_csv["xct_radius_pateval_4"],
                "Radius: Tb.BV/TV [1]": db_csv["xct_radius_pateval_8"],
                "Radius: Ct.Th [mm]": db_csv["xct_radius_pateval_13"],
                "Radius: Ct.Po [1]": db_csv["xct_radius_pateval_7"],
                "Radius: Ct.Po.Dm [mm]": db_csv["xct_radius_pateval_14"],
                "Radius: Tb.N [1/mm]": db_csv["xct_radius_pateval_9"],
                "Radius: Tb.Th [mm]": db_csv["xct_radius_pateval_10"],
                "Radius: Tb.Sp [mm]": db_csv["xct_radius_pateval_11"],
                "Radius: SD of 1/Tb.N [mm]": db_csv["xct_radius_pateval_12"],
                "Radius: Fmax at failure [N]": db_csv["xct_radius_hfea_1"],
                "Radius: Stiffness [N/mm]": db_csv["xct_radius_hfea_2"],
                #
                # Tibia
                # 'Radius: Tt Ar [mm2]': db_csv[''],
                "Tibia: Measurement number": db_csv["xct_tibia_noanalysis"],
                "Tibia: side": db_csv["tibia_side_xct"].map(side_xct_mapping_int),
                # "Tibia Tot.Ar [mm2]": db_csv["Tibia Total Area"],
                "Tibia: Ct.Ar [mm2]": db_csv["xct_tibia_pateval_18"],
                "Tibia: Tb.Ar [mm2]": db_csv["xct_tibia_pateval_15"],
                "Tibia: Tb.Meta.Ar [mm2]": db_csv["xct_tibia_pateval_16"],
                "Tibia: Tb.Inn.Ar [mm2]": db_csv["xct_tibia_pateval_17"],
                "Tibia: Ct.Pm [mm]": db_csv["xct_tibia_pateval_6"],
                "Tibia: Tt.vBMD [mg HA/cmm]": db_csv["xct_tibia_pateval_1"],
                "Tibia: Ct.vBMD [mg HA/cmm]": db_csv["xct_tibia_pateval_5"],
                "Tibia: Tb.vBMD [mg HA/cmm]": db_csv["xct_tibia_pateval_2"],
                "Tibia: Tb.Meta.vBMD [mg HA/cmm]": db_csv["xct_tibia_pateval_3"],
                "Tibia: Tb.Inn.vBMD [mg HA/cmm]": db_csv["xct_tibia_pateval_4"],
                "Tibia: Tb.BV/TV [1]": db_csv["xct_tibia_pateval_8"],
                "Tibia: Ct.Th [mm]": db_csv["xct_tibia_pateval_13"],
                "Tibia: Ct.Po [1]": db_csv["xct_tibia_pateval_7"],
                "Tibia: Ct.Po.Dm [mm]": db_csv["xct_tibia_pateval_14"],
                "Tibia: Tb.N [1/mm]": db_csv["xct_tibia_pateval_9"],
                "Tibia: Tb.Th [mm]": db_csv["xct_tibia_pateval_10"],
                "Tibia: Tb.Sp [mm]": db_csv["xct_tibia_pateval_11"],
                "Tibia: SD of 1/Tb.N [mm]": db_csv["xct_tibia_pateval_12"],
                "Tibia: Fmax at failure [N]": db_csv["xct_tibia_hfea_1"],
                "Tibia: Stiffness [N/mm]": db_csv["xct_tibia_hfea_2"],
            }
        )

        # convert AffirmCT data
        df = df.groupby("UID").first()  # Combine alle rows of same subject
        df = df[
            ~df["Radius: Ct.Ar [mm2]"].isnull()
        ]  # use only subject with HR-pQCT results
        df = df.reset_index()
        df["Radius: Fmax at failure [N]"] = (df["Radius: Fmax at failure [N]"]).abs()
        df["Tibia: Fmax at failure [N]"] = (df["Tibia: Fmax at failure [N]"]).abs()

        df.to_csv(dir_path + "affirm_ct_data.csv", sep="\t", index=False)

    if study_name == "Reproducibility":
        data_reproducibility = db_csv
        # convert date of birth to age
        data_reproducibility["Meas_year"] = pd.to_datetime(
            data_reproducibility["Meas-Date1"]
        ).dt.year
        data_reproducibility["Age"] = (
            data_reproducibility["Meas_year"] - data_reproducibility["DOB"]
        )
        # Extract limb side from column
        data_reproducibility["Site_limb"] = data_reproducibility["Site"].str.extract(
            "(L)"
        )
        data_reproducibility["Site_limb"] = data_reproducibility["Site_limb"].fillna(
            "R"
        )
        # Copy column names
        columns_bone_data_r = data_reproducibility.add_suffix(" Radius").columns[
            27:45
        ]  # .add_suffix(' Radius') # Bone data
        columns_bone_hFE_r = data_reproducibility.add_suffix(" Radius").columns[
            65:67
        ]  # .add_suffix(' Radius') # Bone data
        columns_bone_data_t = data_reproducibility.add_suffix(" Tibia").columns[
            27:45
        ]  # .add_suffix(' Tibia') # Bone data
        columns_bone_hFE_t = data_reproducibility.add_suffix(" Tibia").columns[
            65:67
        ]  # .add_suffix(' Tibia') # Bone data
        columns_PatNo = data_reproducibility.columns[2:4]  # Patient number and gender
        column_age = data_reproducibility.columns[170:171]  # Age
        column_meas_num_r = data_reproducibility.add_suffix(" Radius").columns[22:23]
        column_meas_num_t = data_reproducibility.add_suffix(" Tibia").columns[22:23]
        column_limb_side_r = data_reproducibility.add_suffix(" Radius").columns[
            171:172
        ]  # limb-side Radius
        column_limb_side_t = data_reproducibility.add_suffix(" Tibia").columns[
            171:172
        ]  # limb-side Tibia
        column_frax = data_reproducibility.columns[4:15]  # FRAX
        # column_totAr_r = data_reproducibility.add_suffix(" Radius").columns[19:20]  # Total Area Radius
        # column_totAr_t = data_reproducibility.add_suffix(" Tibia").columns[19:20]  # Total Area Radius
        # create empty database with same column names for radius and tibia
        data_reproducibility_adapted = pd.DataFrame(
            columns=[
                columns_PatNo.append(column_age)
                .append(column_meas_num_r)
                .append(columns_bone_data_r)
                .append(columns_bone_hFE_r)
                .append(column_limb_side_r)
                .append(column_meas_num_t)
                .append(columns_bone_data_t)
                .append(columns_bone_hFE_t)
                .append(column_limb_side_t)
                .append(column_frax)
                # .append(column_totAr_r)
                # .append(column_totAr_t)
            ],
            index=range(len(data_reproducibility)),
        )

        # save values for radius and tibia in new data base
        for x in range(len(data_reproducibility)):
            if (
                data_reproducibility.iloc[x]["PatNo"]
                == data_reproducibility.iloc[x - 1]["PatNo"]
            ):  # if same patient as before
                if data_reproducibility.iloc[x]["Site.short"] == "R":
                    data_reproducibility_adapted.iloc[x - 1][3:4] = (
                        data_reproducibility.iloc[x, 22:23]
                    )  # assign meas number
                    data_reproducibility_adapted.iloc[x - 1][4:22] = (
                        data_reproducibility.iloc[x, 27:45]
                    )  # if radius: assign bone data
                    data_reproducibility_adapted.iloc[x - 1][22:24] = (
                        data_reproducibility.iloc[x, 65:67]
                    )  # if radius: assign bone data
                    data_reproducibility_adapted.iloc[x - 1][24:25] = (
                        data_reproducibility.iloc[x, 171:172]
                    )  # assign limb side
                    # data_reproducibility_adapted.iloc[x - 1][58:59] = data_reproducibility.iloc[x,19:20]  # assign total area
                else:
                    data_reproducibility_adapted.iloc[x - 1][25:26] = (
                        data_reproducibility.iloc[x, 22:23]
                    )  # assign meas number
                    data_reproducibility_adapted.iloc[x - 1][26:44] = (
                        data_reproducibility.iloc[x, 27:45]
                    )  # if tibia: assign bone data
                    data_reproducibility_adapted.iloc[x - 1][44:46] = (
                        data_reproducibility.iloc[x, 65:67]
                    )  # if tibia: assign bone data
                    data_reproducibility_adapted.iloc[x - 1][46:47] = (
                        data_reproducibility.iloc[x, 171:172]
                    )  # assign limb side
                    # data_reproducibility_adapted.iloc[x - 1][59:60] = data_reproducibility.iloc[x,19:20]  # assign total area

            else:
                if data_reproducibility.iloc[x]["Site.short"] == "R":  # if radius
                    data_reproducibility_adapted.iloc[x][0:2] = (
                        data_reproducibility.iloc[x, 2:4]
                    )  # assign PatNo and remark
                    data_reproducibility_adapted.iloc[x][2:3] = (
                        data_reproducibility.iloc[x, 170:171]
                    )  # assign age
                    data_reproducibility_adapted.iloc[x][3:4] = (
                        data_reproducibility.iloc[x, 22:23]
                    )  # assign meas number
                    data_reproducibility_adapted.iloc[x][4:22] = (
                        data_reproducibility.iloc[x, 27:45]
                    )  # assign bone data
                    data_reproducibility_adapted.iloc[x][22:24] = (
                        data_reproducibility.iloc[x, 65:67]
                    )  # assign bone data
                    data_reproducibility_adapted.iloc[x][24:25] = (
                        data_reproducibility.iloc[x, 171:172]
                    )  # assign limb side
                    data_reproducibility_adapted.iloc[x][47:58] = (
                        data_reproducibility.iloc[x, 4:15]
                    )  # assign frax data
                    # data_reproducibility_adapted.iloc[x][58:59] = data_reproducibility.iloc[x,19:20]  # assign total area

                else:  # if tibia
                    data_reproducibility_adapted.iloc[x][0:2] = (
                        data_reproducibility.iloc[x, 2:4]
                    )  # assign PatNo and remark
                    data_reproducibility_adapted.iloc[x][2:3] = (
                        data_reproducibility.iloc[x, 170:171]
                    )  # assign age
                    data_reproducibility_adapted.iloc[x][25:26] = (
                        data_reproducibility.iloc[x, 22:23]
                    )  # assign meas number
                    data_reproducibility_adapted.iloc[x][26:44] = (
                        data_reproducibility.iloc[x, 27:45]
                    )  # assign bone data
                    data_reproducibility_adapted.iloc[x][44:46] = (
                        data_reproducibility.iloc[x, 65:67]
                    )  # assign bone data
                    data_reproducibility_adapted.iloc[x][46:47] = (
                        data_reproducibility.iloc[x, 171:172]
                    )  # assign limb side
                    data_reproducibility_adapted.iloc[x][47:58] = (
                        data_reproducibility.iloc[x, 4:15]
                    )  # assign frax data
                    # data_reproducibility_adapted.iloc[x][59:60] = data_reproducibility.iloc[x,19:20]  # assign tottal area

            # remove empty lines
        data_reproducibility_adapted = data_reproducibility_adapted.dropna(
            how="all"
        ).reset_index(drop=True)

        data_reproducibility_adapted["Study name"] = "Reproducibility"
        # data_reproducibility_adapted["Radius: Fmax at failure adjusted [N]"] = np.nan
        # data_reproducibility_adapted["Radius: Stiffness adjusted [N/mm]"] = np.nan
        # data_reproducibility_adapted["Tibia: Fmax at failure adjusted [N]"] = np.nan
        # data_reproducibility_adapted["Tibia: Stiffness adjusted [N/mm]"] = np.nan
        data_reproducibility_adapted["Time difference Frax & HRpQCT [days]"] = 0

        df = data_reproducibility_adapted[
            [
                "PatNo",
                "PatRemark",
                "Ethnicity",
                "Age",
                "Weight",
                "Height",
                "Study name",
                "Previous fracture",
                "Parent fractured hip",
                "Current smoking",
                "Glucocorticoids",
                "Rheumatoid arthritis",
                "Secondary osteoporosis",
                "Alcohol 3 or more units/day",
                "Femoral neck BMD (g/cm2)",
                "Time difference Frax & HRpQCT [days]",
                "MeasNo1 Radius",
                "Site_limb Radius",
                # "Total-Area Radius",
                "Ct.Ar1 Radius",
                "Tb.Ar1 Radius",
                "Tb.Meta.Ar1 Radius",
                "Tb.Inn.Ar1 Radius",
                "Ct.Pm1 Radius",
                "Tot.vBMD1 Radius",
                "Ct.vBMD1 Radius",
                "Tb.vBMD1 Radius",
                "Tb.Meta.vBMD1 Radius",
                "Tb.Inn.vBMD1 Radius",
                "Tb.BV.TV1 Radius",
                "Ct.Th1 Radius",
                "Ct.Po1 Radius",
                "Ct.Po.Dm1 Radius",
                "Tb.N1 Radius",
                "Tb.Th1 Radius",
                "Tb.Sp1 Radius",
                "Tb.1/N.SD1 Radius",
                "F.load1 Radius",
                "Stiffness1 Radius",
                # "Radius: Fmax at failure adjusted [N]",
                # "Radius: Stiffness adjusted [N/mm]",
                "MeasNo1 Tibia",
                "Site_limb Tibia",
                # "Total-Area Tibia",
                "Ct.Ar1 Tibia",
                "Tb.Ar1 Tibia",
                "Tb.Meta.Ar1 Tibia",
                "Tb.Inn.Ar1 Tibia",
                "Ct.Pm1 Tibia",
                "Tot.vBMD1 Tibia",
                "Ct.vBMD1 Tibia",
                "Tb.vBMD1 Tibia",
                "Tb.Meta.vBMD1 Tibia",
                "Tb.Inn.vBMD1 Tibia",
                "Tb.BV.TV1 Tibia",
                "Ct.Th1 Tibia",
                "Ct.Po1 Tibia",
                "Ct.Po.Dm1 Tibia",
                "Tb.N1 Tibia",
                "Tb.Th1 Tibia",
                "Tb.Sp1 Tibia",
                "Tb.1/N.SD1 Tibia",
                "F.load1 Tibia",
                "Stiffness1 Tibia",
                # "Tibia: Fmax at failure adjusted [N]",
                # "Tibia: Stiffness adjusted [N/mm]",
            ]
        ].copy()

        df = df.rename(
            columns={
                "PatNo": "UID",
                "PatRemark": "Gender",
                "MeasNo1 Radius": "Radius: Measurement number",
                "Site_limb Radius": "Radius: side",
                # "Total-Area Radius": "Radius: Tot.Ar [mm2]",
                "Ct.Ar1 Radius": "Radius: Ct.Ar [mm2]",
                "Tb.Ar1 Radius": "Radius: Tb.Ar [mm2]",
                "Tb.Meta.Ar1 Radius": "Radius: Tb.Meta.Ar [mm2]",
                "Tb.Inn.Ar1 Radius": "Radius: Tb.Inn.Ar [mm2]",
                "Ct.Pm1 Radius": "Radius: Ct.Pm [mm]",
                "Tot.vBMD1 Radius": "Radius: Tt.vBMD [mg HA/cmm]",
                "Ct.vBMD1 Radius": "Radius: Ct.vBMD [mg HA/cmm]",
                "Tb.vBMD1 Radius": "Radius: Tb.vBMD [mg HA/cmm]",
                "Tb.Meta.vBMD1 Radius": "Radius: Tb.Meta.vBMD [mg HA/cmm]",
                "Tb.Inn.vBMD1 Radius": "Radius: Tb.Inn.vBMD [mg HA/cmm]",
                "Tb.BV.TV1 Radius": "Radius: Tb.BV/TV [1]",
                "Ct.Th1 Radius": "Radius: Ct.Th [mm]",
                "Ct.Po1 Radius": "Radius: Ct.Po [1]",
                "Ct.Po.Dm1 Radius": "Radius: Ct.Po.Dm [mm]",
                "Tb.N1 Radius": "Radius: Tb.N [1/mm]",
                "Tb.Th1 Radius": "Radius: Tb.Th [mm]",
                "Tb.Sp1 Radius": "Radius: Tb.Sp [mm]",
                "Tb.1/N.SD1 Radius": "Radius: SD of 1/Tb.N [mm]",
                "F.load1 Radius": "Radius: Fmax at failure [N]",
                "Stiffness1 Radius": "Radius: Stiffness [N/mm]",
                "MeasNo1 Tibia": "Tibia: Measurement number",
                "Site_limb Tibia": "Tibia: side",
                # "Total-Area Tibia": "Tibia: Tot.Ar [mm2]",
                "Ct.Ar1 Tibia": "Tibia: Ct.Ar [mm2]",
                "Tb.Ar1 Tibia": "Tibia: Tb.Ar [mm2]",
                "Tb.Meta.Ar1 Tibia": "Tibia: Tb.Meta.Ar [mm2]",
                "Tb.Inn.Ar1 Tibia": "Tibia: Tb.Inn.Ar [mm2]",
                "Ct.Pm1 Tibia": "Tibia: Ct.Pm [mm]",
                "Tot.vBMD1 Tibia": "Tibia: Tt.vBMD [mg HA/cmm]",
                "Ct.vBMD1 Tibia": "Tibia: Ct.vBMD [mg HA/cmm]",
                "Tb.vBMD1 Tibia": "Tibia: Tb.vBMD [mg HA/cmm]",
                "Tb.Meta.vBMD1 Tibia": "Tibia: Tb.Meta.vBMD [mg HA/cmm]",
                "Tb.Inn.vBMD1 Tibia": "Tibia: Tb.Inn.vBMD [mg HA/cmm]",
                "Tb.BV.TV1 Tibia": "Tibia: Tb.BV/TV [1]",
                "Ct.Th1 Tibia": "Tibia: Ct.Th [mm]",
                "Ct.Po1 Tibia": "Tibia: Ct.Po [1]",
                "Ct.Po.Dm1 Tibia": "Tibia: Ct.Po.Dm [mm]",
                "Tb.N1 Tibia": "Tibia: Tb.N [1/mm]",
                "Tb.Th1 Tibia": "Tibia: Tb.Th [mm]",
                "Tb.Sp1 Tibia": "Tibia: Tb.Sp [mm]",
                "Tb.1/N.SD1 Tibia": "Tibia: SD of 1/Tb.N [mm]",
                "F.load1 Tibia": "Tibia: Fmax at failure [N]",
                "Stiffness1 Tibia": "Tibia: Stiffness [N/mm]",
            }
        )

        # convert Reproducibility data (German to English)
        df["Gender"] = df["Gender"].replace("M", "Male")
        df["Gender"] = df["Gender"].replace("F", "Female")
        df["Radius: Fmax at failure [N]"] = (df["Radius: Fmax at failure [N]"]).abs()
        df["Tibia: Fmax at failure [N]"] = (df["Tibia: Fmax at failure [N]"]).abs()
        df["Radius: side"] = df["Radius: side"].replace("L", "Left")
        df["Radius: side"] = df["Radius: side"].replace("R", "Right")
        df["Tibia: side"] = df["Tibia: side"].replace("L", "Left")
        df["Tibia: side"] = df["Tibia: side"].replace("R", "Right")

        df.columns = df.columns.map("".join)

    if study_name == "BOLD":
        data_bold_patient = additional_file
        data_bold_bone = db_csv

        try:
            hgs_path = Path(dir_path) / "bold_hand_grip_strength.csv"
            hgs_df = pd.read_csv(hgs_path)

            hgs_df["Dominant_Side"] = hgs_df[
                "Dominant_Side"
            ].str.title()  # Convert to Title case (Right/Left)

            # Calculate dominant and non-dominant hand strength
            dominant_hand_strength, non_dominant_hand_strength = hand_grip_strength(
                hgs_df["Dominant_Side"],
                hgs_df["Left_1"],
                hgs_df["Left_2"],
                hgs_df["Left_3"],
                hgs_df["Right_1"],
                hgs_df["Right_2"],
                hgs_df["Right_3"],
                mean_max="max",
            )

            # Create filtered dataframe with calculated values
            hgs_filtered = pd.DataFrame(
                {
                    "UID": hgs_df["UID"],
                    "Hand grip strength dominant [kg]": dominant_hand_strength,
                    "Hand grip strength non-dominant [kg]": non_dominant_hand_strength,
                }
            )

            # Check for duplicates
            duplicates = hgs_filtered[
                hgs_filtered.duplicated(subset=["UID"], keep=False)
            ]
            if not duplicates.empty:
                print("Duplicates found:")
                print(duplicates)

        except FileNotFoundError:
            print("Hand grip strength data for BOLD study not included.")
            hgs_filtered = None

        # Then merge if hand grip data exists
        if hgs_filtered is not None:
            data_bold_bone = pd.merge(
                data_bold_bone,
                hgs_filtered,
                left_on="PatNo",
                right_on="UID",
                how="left",
            )

        # convert date of birth to age
        data_bold_patient["Birth_year"] = pd.to_datetime(
            data_bold_patient["Geburtsdatum"], dayfirst=False
        )
        data_bold_patient["Meas_year_HRpQCT"] = pd.to_datetime(
            data_bold_patient["HRpQCT Bern Termin"], dayfirst=True
        )

        # Calculate age in years using dt.days and dividing by 365.25
        data_bold_patient["Age"] = (
            (
                (
                    data_bold_patient["Meas_year_HRpQCT"]
                    - data_bold_patient["Birth_year"]
                ).dt.days
                / 365.25
            )
            .round()
            .astype("Int64")
        )  # Use Int64 to handle NaN values

        # Extract limb side from column
        data_bold_bone["Site_limb"] = data_bold_bone["Site"].str.extract("(L)")
        data_bold_bone["Site_limb"] = data_bold_bone["Site_limb"].fillna("R")

        # Calculate time difference between frax and hrpqct
        for x in range(112):
            try:
                frax_date = pd.to_datetime(
                    data_bold_bone.loc[x * 2, "Frax termin"], dayfirst=True
                )
                hrpqct_date = data_bold_patient.loc[x, "Meas_year_HRpQCT"]

                if pd.notna(frax_date) and pd.notna(hrpqct_date):
                    data_bold_bone.loc[x * 2, "Diff time meas"] = (
                        hrpqct_date - frax_date
                    ).days
                else:
                    data_bold_bone.loc[x * 2, "Diff time meas"] = np.nan
            except (KeyError, IndexError, ValueError):
                data_bold_bone.loc[x * 2, "Diff time meas"] = np.nan

        # Copy column names
        columns_bone_data_r = data_bold_bone.add_suffix(" Radius").columns[
            7:27
        ]  # .add_suffix(' Radius') # Bone data
        columns_bone_data_t = data_bold_bone.add_suffix(" Tibia").columns[
            7:27
        ]  # .add_suffix(' Tibia') # Bone data

        columns_PatNo = data_bold_bone.columns[0:1]  # Patient number
        columns_PatRemark = data_bold_bone.columns[4:5]  # Patient gender
        column_age = data_bold_patient.columns[7:8]  # Age
        column_limb_side_r = data_bold_bone.add_suffix(" Radius").columns[
            5:6
        ]  # limb-side Radius
        column_limb_side_t = data_bold_bone.add_suffix(" Tibia").columns[
            5:6
        ]  # limb-side Tibia
        column_meas_num_r = data_bold_bone.add_suffix(" Radius").columns[1:2]
        column_meas_num_t = data_bold_bone.add_suffix(" Tibia").columns[1:2]
        column_frax = data_bold_bone.columns[30:41]

        column_time_diff_meas = data_bold_bone.columns[
            47:48
        ]  # Fixed: was 48:49, now 47:48

        # Add hand grip strength columns if they exist
        column_hgs_dom = column_hgs_non_dom = pd.Index([])
        if "Hand grip strength dominant [kg]" in data_bold_bone.columns:
            column_hgs_dom = data_bold_bone.columns[
                44:45
            ]  # Hand grip dominant (column 44)
            column_hgs_non_dom = data_bold_bone.columns[
                45:46
            ]  # Hand grip non-dominant (column 45)

        # create empty database with same column names for radius and tibia
        all_columns = (
            list(columns_PatNo)
            + list(columns_PatRemark)
            + list(column_age)
            + list(column_meas_num_r)
            + list(columns_bone_data_r)
            + list(column_limb_side_r)
            + list(column_meas_num_t)
            + list(columns_bone_data_t)
            + list(column_limb_side_t)
            + list(column_frax)
            + list(column_time_diff_meas)
        )

        # Only add hand grip columns if they exist
        if "Hand grip strength dominant [kg]" in data_bold_bone.columns:
            all_columns = all_columns + list(column_hgs_dom) + list(column_hgs_non_dom)

        data_bold_adapted = pd.DataFrame(
            columns=all_columns,
            index=range(len(data_bold_bone)),
        )

        # save values for radius and tibia in new data base
        for x in range(len(data_bold_bone)):
            if data_bold_bone.iloc[x]["Diabetes"] == "No":
                if (
                    data_bold_bone.iloc[x]["PatNo"]
                    == data_bold_bone.iloc[x - 1]["PatNo"]
                ):  # if same patient as before
                    if (data_bold_bone.loc[x]["Site"] == "Radius L") or (
                        data_bold_bone.loc[x]["Site"] == "Radius R"
                    ):
                        data_bold_adapted.iloc[x - 1][3:4] = data_bold_bone.iloc[
                            x, 1:2
                        ]  # assign meas number
                        data_bold_adapted.iloc[x - 1][4:24] = data_bold_bone.iloc[
                            x, 7:27
                        ]  # if radius
                        data_bold_adapted.iloc[x - 1][24:25] = data_bold_bone.iloc[
                            x, 47:48
                        ]  # assign limb side (Site_limb column)
                    else:
                        data_bold_adapted.iloc[x - 1][25:26] = data_bold_bone.iloc[
                            x, 1:2
                        ]  # assign meas number
                        data_bold_adapted.iloc[x - 1][26:46] = data_bold_bone.iloc[
                            x, 7:27
                        ]  # if tibia
                        data_bold_adapted.iloc[x - 1][46:47] = data_bold_bone.iloc[
                            x, 47:48
                        ]  # assign limb side (Site_limb column)
                else:
                    if (data_bold_bone.loc[x]["Site"] == "Radius L") or (
                        data_bold_bone.loc[x]["Site"] == "Radius R"
                    ):  # if radius
                        data_bold_adapted.iloc[x][0:1] = data_bold_bone.iloc[
                            x, 0:1
                        ]  # assign PatNo
                        data_bold_adapted.iloc[x][1:2] = data_bold_bone.iloc[
                            x, 4:5
                        ]  # assign remark
                        data_bold_adapted.iloc[x][3:4] = data_bold_bone.iloc[
                            x, 1:2
                        ]  # assign meas number
                        data_bold_adapted.iloc[x][4:24] = data_bold_bone.iloc[
                            x, 7:27
                        ]  # assign bone data
                        data_bold_adapted.iloc[x][24:25] = data_bold_bone.iloc[
                            x, 47:48
                        ]  # assign limb side (Site_limb column)
                        data_bold_adapted.iloc[x][47:58] = data_bold_bone.iloc[
                            x, 30:41
                        ]  # assign FRAX
                        data_bold_adapted.iloc[x, 58] = data_bold_bone.iloc[
                            x, 47  # Fixed: use column 47 for "Diff time meas"
                        ]  # time difference between measurements (Diff time meas column)
                        # Add hand grip strength if available
                        if "Hand grip strength dominant [kg]" in data_bold_bone.columns:
                            # Check if columns exist in data_bold_adapted before assigning
                            if len(data_bold_adapted.columns) > 59:
                                data_bold_adapted.iloc[x, 59] = data_bold_bone.iloc[
                                    x, 44
                                ]  # hand grip dominant
                            if len(data_bold_adapted.columns) > 60:
                                data_bold_adapted.iloc[x, 60] = data_bold_bone.iloc[
                                    x, 45
                                ]  # hand grip non-dominant

                    else:  # if tibia
                        data_bold_adapted.iloc[x][0:1] = data_bold_bone.iloc[
                            x, 0:1
                        ]  # assign PatNo
                        data_bold_adapted.iloc[x][1:2] = data_bold_bone.iloc[
                            x, 4:5
                        ]  # assign remark
                        data_bold_adapted.iloc[x][25:26] = data_bold_bone.iloc[
                            x, 1:2
                        ]  # assign meas number
                        data_bold_adapted.iloc[x][26:46] = data_bold_bone.iloc[
                            x, 7:27
                        ]  # assign bone data
                        data_bold_adapted.iloc[x][46:47] = data_bold_bone.iloc[
                            x, 47:48
                        ]  # assign limb side (Site_limb column)
                        data_bold_adapted.iloc[x][47:58] = data_bold_bone.iloc[
                            x, 30:41
                        ]  # assign FRAX
                        data_bold_adapted.iloc[x, 58] = data_bold_bone.iloc[
                            x, 47  # Fixed: was 48, now 47 (Diff time meas)
                        ]  # time difference between measurements (Diff time meas column)
                        # Add hand grip strength if available
                        if "Hand grip strength dominant [kg]" in data_bold_bone.columns:
                            data_bold_adapted.iloc[x, 59] = data_bold_bone.iloc[
                                x, 44
                            ]  # hand grip dominant
                            data_bold_adapted.iloc[x, 60] = data_bold_bone.iloc[
                                x, 45
                            ]  # hand grip non-dominant

        data_bold_adapted = data_bold_adapted.dropna(how="all").reset_index(drop=True)

        for x in range(len(data_bold_adapted)):
            for n in range(len(data_bold_patient)):
                if (
                    data_bold_adapted.loc[x]["PatNo"]
                    == data_bold_patient.iloc[n]["Pat  ID"]
                ):  # if same patient Number in both files
                    data_bold_adapted.loc[x]["Age"] = data_bold_patient.iloc[n][
                        "Age"
                    ]  # assign age

        data_bold_adapted["Study name"] = "BOLD"
        # data_bold_adapted["Radius: Fmax at failure adjusted [N]"] = np.nan
        # data_bold_adapted["Radius: Stiffness adjusted [N/mm]"] = np.nan
        # data_bold_adapted["Tibia: Fmax at failure adjusted [N]"] = np.nan
        # data_bold_adapted["Tibia: Stiffness adjusted [N/mm]"] = np.nan

        df = data_bold_adapted[
            [
                "PatNo",
                "PatRemark",
                "Ethnicity",
                "Age",
                "Weight",
                "Height",
                "Study name",
                "Previous fracture",
                "Parent fractured hip",
                "Current smoking",
                "Glucocorticoids",
                "Rheumatoid arthritis",
                "Secondary osteoporosis",
                "Alcohol 3 or more units/day",
                "Femoral neck BMD (g/cm2)",
                "Diff time meas",
                "MeasNo Radius",
                "Site Radius",
                # "Total-Area Radius",
                "Ct.Ar Radius",
                "Tb.Ar Radius",
                "Tb.Meta.Ar Radius",
                "Tb.Inn.Ar Radius",
                "Ct.Pm Radius",
                "Tot.vBMD Radius",
                "Ct.vBMD Radius",
                "Tb.vBMD Radius",
                "Tb.Meta.vBMD Radius",
                "Tb.Inn.vBMD Radius",
                "Tb.BV/TV Radius",
                "Ct.Th Radius",
                "Ct.Po Radius",
                "Ct.Po.Dm Radius",
                "Tb.N Radius",
                "Tb.Th Radius",
                "Tb.Sp Radius",
                "Tb./N.SD Radius",
                "Strength [N] Radius",
                "Stiffness [N/mm] Radius",
                # "Radius: Fmax at failure adjusted [N]",
                # "Radius: Stiffness adjusted [N/mm]",
                "MeasNo Tibia",
                "Site Tibia",
                # "Total-Area Tibia",
                "Ct.Ar Tibia",
                "Tb.Ar Tibia",
                "Tb.Meta.Ar Tibia",
                "Tb.Inn.Ar Tibia",
                "Ct.Pm Tibia",
                "Tot.vBMD Tibia",
                "Ct.vBMD Tibia",
                "Tb.vBMD Tibia",
                "Tb.Meta.vBMD Tibia",
                "Tb.Inn.vBMD Tibia",
                "Tb.BV/TV Tibia",
                "Ct.Th Tibia",
                "Ct.Po Tibia",
                "Ct.Po.Dm Tibia",
                "Tb.N Tibia",
                "Tb.Th Tibia",
                "Tb.Sp Tibia",
                "Tb./N.SD Tibia",
                "Strength [N] Tibia",
                "Stiffness [N/mm] Tibia",
                # "Tibia: Fmax at failure adjusted [N]",
                # "Tibia: Stiffness adjusted [N/mm]",
            ]
            + (
                [
                    "Hand grip strength dominant [kg]",
                    "Hand grip strength non-dominant [kg]",
                ]
                if "Hand grip strength dominant [kg]" in data_bold_adapted.columns
                else []
            )
        ].copy()

        df = df.rename(
            columns={
                "PatNo": "UID",
                "PatRemark": "Gender",
                "Diff time meas": "Time difference Frax & HRpQCT [days]",
                "MeasNo Radius": "Radius: Measurement number",
                "Site Radius": "Radius: side",
                # "Total-Area Radius": "Radius: Tot.Ar [mm2]",
                "Ct.Ar Radius": "Radius: Ct.Ar [mm2]",
                "Tb.Ar Radius": "Radius: Tb.Ar [mm2]",
                "Tb.Meta.Ar Radius": "Radius: Tb.Meta.Ar [mm2]",
                "Tb.Inn.Ar Radius": "Radius: Tb.Inn.Ar [mm2]",
                "Ct.Pm Radius": "Radius: Ct.Pm [mm]",
                "Tot.vBMD Radius": "Radius: Tt.vBMD [mg HA/cmm]",
                "Ct.vBMD Radius": "Radius: Ct.vBMD [mg HA/cmm]",
                "Tb.vBMD Radius": "Radius: Tb.vBMD [mg HA/cmm]",
                "Tb.Meta.vBMD Radius": "Radius: Tb.Meta.vBMD [mg HA/cmm]",
                "Tb.Inn.vBMD Radius": "Radius: Tb.Inn.vBMD [mg HA/cmm]",
                "Tb.BV/TV Radius": "Radius: Tb.BV/TV [1]",
                "Ct.Th Radius": "Radius: Ct.Th [mm]",
                "Ct.Po Radius": "Radius: Ct.Po [1]",
                "Ct.Po.Dm Radius": "Radius: Ct.Po.Dm [mm]",
                "Tb.N Radius": "Radius: Tb.N [1/mm]",
                "Tb.Th Radius": "Radius: Tb.Th [mm]",
                "Tb.Sp Radius": "Radius: Tb.Sp [mm]",
                "Tb./N.SD Radius": "Radius: SD of 1/Tb.N [mm]",
                "Strength [N] Radius": "Radius: Fmax at failure [N]",
                "Stiffness [N/mm] Radius": "Radius: Stiffness [N/mm]",
                "MeasNo Tibia": "Tibia: Measurement number",
                "Site Tibia": "Tibia: side",
                # "Total-Area Tibia": "Tibia: Tot.Ar [mm2]",
                "Ct.Ar Tibia": "Tibia: Ct.Ar [mm2]",
                "Tb.Ar Tibia": "Tibia: Tb.Ar [mm2]",
                "Tb.Meta.Ar Tibia": "Tibia: Tb.Meta.Ar [mm2]",
                "Tb.Inn.Ar Tibia": "Tibia: Tb.Inn.Ar [mm2]",
                "Ct.Pm Tibia": "Tibia: Ct.Pm [mm]",
                "Tot.vBMD Tibia": "Tibia: Tt.vBMD [mg HA/cmm]",
                "Ct.vBMD Tibia": "Tibia: Ct.vBMD [mg HA/cmm]",
                "Tb.vBMD Tibia": "Tibia: Tb.vBMD [mg HA/cmm]",
                "Tb.Meta.vBMD Tibia": "Tibia: Tb.Meta.vBMD [mg HA/cmm]",
                "Tb.Inn.vBMD Tibia": "Tibia: Tb.Inn.vBMD [mg HA/cmm]",
                "Tb.BV/TV Tibia": "Tibia: Tb.BV/TV [1]",
                "Ct.Th Tibia": "Tibia: Ct.Th [mm]",
                "Ct.Po Tibia": "Tibia: Ct.Po [1]",
                "Ct.Po.Dm Tibia": "Tibia: Ct.Po.Dm [mm]",
                "Tb.N Tibia": "Tibia: Tb.N [1/mm]",
                "Tb.Th Tibia": "Tibia: Tb.Th [mm]",
                "Tb.Sp Tibia": "Tibia: Tb.Sp [mm]",
                "Tb./N.SD Tibia": "Tibia: SD of 1/Tb.N [mm]",
                "Strength [N] Tibia": "Tibia: Fmax at failure [N]",
                "Stiffness [N/mm] Tibia": "Tibia: Stiffness [N/mm]",
            }
        )

        # convert Bold data
        df["Gender"] = df["Gender"].replace("m", "Male")
        df["Gender"] = df["Gender"].replace("f", "Female")
        df["Gender"] = df["Gender"].replace("w", "Female")
        df["Radius: Fmax at failure [N]"] = (df["Radius: Fmax at failure [N]"]).abs()
        df["Tibia: Fmax at failure [N]"] = (df["Tibia: Fmax at failure [N]"]).abs()
        df["Radius: side"] = df["Radius: side"].replace("L", "Left")
        df["Radius: side"] = df["Radius: side"].replace("R", "Right")
        df["Tibia: side"] = df["Tibia: side"].replace("L", "Left")
        df["Tibia: side"] = df["Tibia: side"].replace("R", "Right")

        df.columns = df.columns.map("".join)

    if study_name == "OIFRAC":
        df = pd.DataFrame(
            {
                "UID": db_csv["Record ID"],
                "Gender": db_csv["Gender"],
                "Age": db_csv["Age "],
                "Ethnicity": None,
                "Weight": db_csv["Weight"],
                "Height": db_csv["Height"],
                "Study name": "OIFRAC",
                "Previous fracture": db_csv["Previous fracture"],
                "Parent fractured hip": db_csv["Parental hip fracture"],
                "Current smoking": db_csv["Current smoking"],
                "Glucocorticoids": db_csv["Glucocorticoid intake "],
                "Rheumatoid arthritis": db_csv["Rheumatoid arthritis"],
                "Secondary osteoporosis": db_csv["Secondary osteoporosis"],
                "Alcohol 3 or more units/day": db_csv["> 3 units/day of alcohol"],
                "Time difference Frax & HRpQCT [days]": 0,
                "Femoral neck BMD (g/cm2)": None,
                # Radius
                # 'Radius: Tt Ar [mm2]': db_csv[''],
                "Radius: Measurement number": db_csv[
                    "Measurement number of analysed scan"
                ],
                "Radius: side": db_csv["Side"],
                "Radius Tot.Ar [mm2]": db_csv["Radius Total area [mm2]"],
                "Radius: Ct.Ar [mm2]": db_csv["Radius cortical area [mm2]"],
                "Radius: Tb.Ar [mm2]": db_csv["Radius trabecular area [mm2]"],
                "Radius: Tb.Meta.Ar [mm2]": db_csv["Radius trabecular meta area [mm2]"],
                "Radius: Tb.Inn.Ar [mm2]": db_csv["Radius trabecular inner area [mm2]"],
                "Radius: Ct.Pm [mm]": db_csv["Radius cortical perimeter [mm2]"],
                "Radius: Tt.vBMD [mg HA/cmm]": db_csv[
                    "Radius total volumetric bone mineral density [mg HA/ccm]"
                ],
                "Radius: Ct.vBMD [mg HA/cmm]": db_csv[
                    "Radius cortical vBMD [mg HA/ccm]"
                ],
                "Radius: Tb.vBMD [mg HA/cmm]": db_csv[
                    "Radius trabecular vBMD [mg HA/ccm]"
                ],
                "Radius: Tb.Meta.vBMD [mg HA/cmm]": db_csv[
                    "Radius meta trabecular vBMD (40% of Tb.Ar.) [mg HA/ccm]"
                ],
                "Radius: Tb.Inn.vBMD [mg HA/cmm]": db_csv[
                    "Radius inner trabecular vBMD (60% of Tb.Ar) [mg HA/ccm]"
                ],
                "Radius: Tb.BV/TV [1]": db_csv[
                    "Radius trabecular bone volume fraction"
                ],
                "Radius: Ct.Th [mm]": db_csv["Radius cortical thickness [mm]"],
                "Radius: Ct.Po [1]": db_csv["Radius cortical porosity"],
                "Radius: Ct.Po.Dm [mm]": db_csv["Radius cortical pore diameter [mm]"],
                "Radius: Tb.N [1/mm]": db_csv["Radius trabecular number [1/mm]"],
                "Radius: Tb.Th [mm]": db_csv["Radius trabecular thickness [mm]"],
                "Radius: Tb.Sp [mm]": db_csv["Radius trabecular separation [mm]"],
                "Radius: SD of 1/Tb.N [mm]": db_csv[
                    "Radius SD of 1/Tb.N: Inhomogeneity of Network [-]"
                ],
                "Radius: Fmax at failure [N]": np.abs(
                    db_csv["Radius maximum force at failure [N]"]
                ),
                "Radius: Stiffness [N/mm]": db_csv["Radius stiffness [N/mm]"],
                # "Radius: Fmax at failure adjusted [N]": db_csv[
                #     "Radius maximum force at failure adjusted [N]"
                # ],
                # "Radius: Stiffness adjusted [N/mm]": db_csv[
                #     "Radius stiffness adjusted [N/mm]"
                # ],
                # Tibia
                "Tibia: Measurement number": db_csv[
                    "Measurement number of analysed scan.1"
                ],
                "Tibia: side": db_csv["Side.1"],
                "Tibia Tot.Ar [mm2]": db_csv["Tibia Total area[mm2]"],
                "Tibia: Ct.Ar [mm2]": db_csv["Tibia cortical area [mm2]"],
                "Tibia: Tb.Ar [mm2]": db_csv["Tibia trabecular area [mm2]"],
                "Tibia: Tb.Meta.Ar [mm2]": db_csv["Tibia trabecular meta area [mm2]"],
                "Tibia: Tb.Inn.Ar [mm2]": db_csv["Tibia trabecular inner area [mm2]"],
                "Tibia: Ct.Pm [mm]": db_csv["Tibia cortical perimeter [mm]"],
                "Tibia: Tt.vBMD [mg HA/cmm]": db_csv[
                    "Tibia total volumetric bone mineral density [mg HA/ccm]"
                ],
                "Tibia: Ct.vBMD [mg HA/cmm]": db_csv["Tibia cortical vBMD [mg HA/ccm]"],
                "Tibia: Tb.vBMD [mg HA/cmm]": db_csv[
                    "Tibia trabecular vBMD [mg HA/ccm]"
                ],
                "Tibia: Tb.Meta.vBMD [mg HA/cmm]": db_csv[
                    "Tibia meta trabecular vBMD (40% of Tb.Ar.) [mg HA/ccm]"
                ],
                "Tibia: Tb.Inn.vBMD [mg HA/cmm]": db_csv[
                    "Tibia inner trabecular vBMD (60% of Tb.Ar) [mg HA/ccm]"
                ],
                "Tibia: Tb.BV/TV [1]": db_csv["Tibia trabecular bone volume fraction"],
                "Tibia: Ct.Th [mm]": db_csv["Tibia cortical thickness [mm]"],
                "Tibia: Ct.Po [1]": db_csv["Tibia cortical porosity"],
                "Tibia: Ct.Po.Dm [mm]": db_csv["Tibia cortical pore diameter [mm]"],
                "Tibia: Tb.N [1/mm]": db_csv["Tibia trabecular number [1/mm]"],
                "Tibia: Tb.Th [mm]": db_csv["Tibia trabecular thickness [mm]"],
                "Tibia: Tb.Sp [mm]": db_csv["Tibia trabecular separation [mm]"],
                "Tibia: SD of 1/Tb.N [mm]": db_csv[
                    "Tibia SD of 1/Tb.N: Inhomogeneity of Network [-]"
                ],
                "Tibia: Fmax at failure [N]": np.abs(
                    db_csv["Tibia maximum force at failure [N]"]
                ),
                "Tibia: Stiffness [N/mm]": db_csv["Tibia stiffness [N/mm]"],
                # "Tibia: Fmax at failure adjusted [N]": db_csv[
                #     "Tibia maximum force at failure adjusted [N]"
                # ],
                # "Tibia: Stiffness adjusted [N/mm]": db_csv[
                #     "Tibia stiffness adjusted [N/mm]"
                # ],
            }
        )

        # Filter to keep only rows with HR-pQCT scan results for either the Radius or Tibia
        hrpqct_columns = ["Radius Tot.Ar [mm2]", "Tibia Tot.Ar [mm2]"]
        df = df.dropna(
            how="all", subset=hrpqct_columns
        )  # Keep rows where at least one column has a non-null value

        return df

    if study_name == "PARATY":
        gender_mapping = {"Yes": "Male", "No": "Female"}
        # ---------------------------
        # Clean the Data
        # ---------------------------

        # Remove double quotes from column names
        db_csv.columns = db_csv.columns.str.strip()
        db_csv.columns = db_csv.columns.str.replace('"', "", regex=False)

        # Clean numeric columns
        numeric_columns = [
            "Age",
            "Weight",
            "Height",
            "Radius total volumetric bone mineral density [mg HA/ccm]",
            "Radius cortical vBMD [mg HA/ccm]",
            "Radius trabecular vBMD [mg HA/ccm]",
            "Radius meta trabecular vBMD (40% of Tb.Ar.) [mg HA/ccm]",
            "Radius inner trabecular vBMD (60% of Tb.Ar) [mg HA/ccm]",
            "Radius trabecular bone volume fraction",
            "Radius cortical thickness [mm]",
            "Radius cortical porosity",
            "Radius cortical pore diameter [mm]",
            "Radius trabecular number [1/mm]",
            "Radius trabecular thickness [mm]",
            "Radius trabecular separation [mm]",
            "Radius SD of 1/Tb.N: Inhomogeneity of Network [-]",
            "Radius maximun force at failure [N]",
            "Radius stiffness [N/mm]",
            "Tibia total volumetric bone mineral density [mg HA/ccm]",
            "Tibia cortical vBMD [mg HA/ccm]",
            "Tibia trabecular vBMD [mg HA/ccm]",
            "Tibia meta trabecular vBMD (40% of Tb.Ar.) [mg HA/ccm]",
            "Tibia inner trabecular vBMD (60% of Tb.Ar) [mg HA/ccm]",
            "Tibia trabecular bone volume fraction",
            "Tibia cortical thickness [mm]",
            "Tibia cortical porosity",
            "Tibia cortical pore diameter [mm]",
            "Tibia trabecular number [1/mm]",
            "Tibia trabecular thickness [mm]",
            "Tibia trabecular separation [mm]",
            "Tibia SD of 1/Tb.N: Inhomogeneity of Network [-]",
            "Tibia maximun force at failure [N]",
            "Tibia stiffness [N/mm]",
        ]
        for col in numeric_columns:
            if col in db_csv.columns:
                db_csv[col] = pd.to_numeric(db_csv[col], errors="coerce")
        # ---------------------------
        # Create the DataFrame
        # ---------------------------
        df = pd.DataFrame(
            {
                "UID": db_csv["Record ID"],
                "Gender": db_csv["Male gender"].map(gender_mapping),
                "Age": db_csv["Age"],
                "Ethnicity": db_csv["Ethnicity"],
                "Weight": db_csv["Weight"],
                "Height": db_csv["Height"],
                "Study name": "PARATY",
                "Previous fracture": db_csv["Previous fractures"],
                "Parent fractured hip": db_csv["Parental hip fracture"],
                "Current smoking": db_csv["Current smoking"],
                "Glucocorticoids": db_csv["Glucocorticoid intake"],
                "Rheumatoid arthritis": db_csv["Rheumatoid arthritis"],
                "Secondary osteoporosis": db_csv["Secondary osteoporosis"],
                "Alcohol 3 or more units/day": db_csv["> 3 units/day of alcohol"],
                "Time difference Frax & HRpQCT [days]": 0,
                "Femoral neck BMD (g/cm2)": None,
                "Measurement date": db_csv["Date of exam"],
                "Accident date": db_csv["Date of SCI"],
                # Radius
                "Radius: Measurement number": db_csv[
                    "Measurement number of analysed scan"
                ],
                "Radius: side": db_csv["Side"],
                "Radius: Ct.Pm [mm]": db_csv["Radius cortical perimeter [mm]"],
                "Radius: Ct.Ar [mm2]": db_csv["Radius cortical area [mm2]"],
                "Radius: Tb.Ar [mm2]": db_csv["Radius trabecular area [mm2]"],
                "Radius: Tb.Meta.Ar [mm2]": db_csv["Radius trabecular meta area [mm2]"],
                "Radius: Tb.Inn.Ar [mm2]": db_csv["Radius trabecular inner area [mm2]"],
                "Radius: Tt.vBMD [mg HA/cmm]": db_csv[
                    "Radius total volumetric bone mineral density [mg HA/ccm]"
                ],
                "Radius: Ct.vBMD [mg HA/cmm]": db_csv[
                    "Radius cortical vBMD [mg HA/ccm]"
                ],
                "Radius: Tb.vBMD [mg HA/cmm]": db_csv[
                    "Radius trabecular vBMD [mg HA/ccm]"
                ],
                "Radius: Tb.Meta.vBMD [mg HA/cmm]": db_csv[
                    "Radius meta trabecular vBMD (40% of Tb.Ar.) [mg HA/ccm]"
                ],
                "Radius: Tb.Inn.vBMD [mg HA/cmm]": db_csv[
                    "Radius inner trabecular vBMD (60% of Tb.Ar) [mg HA/ccm]"
                ],
                "Radius: Tb.BV/TV [1]": db_csv[
                    "Radius trabecular bone volume fraction"
                ],
                "Radius: Ct.Th [mm]": db_csv["Radius cortical thickness [mm]"],
                "Radius: Ct.Po [1]": db_csv["Radius cortical porosity"],
                "Radius: Ct.Po.Dm [mm]": db_csv["Radius cortical pore diameter [mm]"],
                "Radius: Tb.N [1/mm]": db_csv["Radius trabecular number [1/mm]"],
                "Radius: Tb.Th [mm]": db_csv["Radius trabecular thickness [mm]"],
                "Radius: Tb.Sp [mm]": db_csv["Radius trabecular separation [mm]"],
                "Radius: SD of 1/Tb.N [mm]": db_csv[
                    "Radius SD of 1/Tb.N: Inhomogeneity of Network [-]"
                ],
                "Radius: Fmax at failure [N]": np.abs(
                    db_csv["Radius maximun force at failure [N]"]
                ),
                "Radius: Stiffness [N/mm]": db_csv["Radius stiffness [N/mm]"],
                # Tibia
                "Tibia: Measurement number": db_csv[
                    "Measurement number of analysed scan.1"
                ],
                "Tibia: side": db_csv["Side.1"],
                "Tibia: Ct.Pm [mm]": db_csv["Tibia cortical perimeter [mm]"],
                "Tibia: Ct.Ar [mm2]": db_csv["Tibia cortical area [mm2]"],
                "Tibia: Tb.Ar [mm2]": db_csv["Tibia trabecular area [mm2]"],
                "Tibia: Tb.Meta.Ar [mm2]": db_csv["Tibia trabecular meta area [mm2]"],
                "Tibia: Tb.Inn.Ar [mm2]": db_csv["Tibia trabecular inner area [mm2]"],
                "Tibia: Ct.Pm [mm]": db_csv["Tibia cortical perimeter [mm]"],
                "Tibia: Tt.vBMD [mg HA/cmm]": db_csv[
                    "Tibia total volumetric bone mineral density [mg HA/ccm]"
                ],
                "Tibia: Ct.vBMD [mg HA/cmm]": db_csv["Tibia cortical vBMD [mg HA/ccm]"],
                "Tibia: Tb.vBMD [mg HA/cmm]": db_csv[
                    "Tibia trabecular vBMD [mg HA/ccm]"
                ],
                "Tibia: Tb.Meta.vBMD [mg HA/cmm]": db_csv[
                    "Tibia meta trabecular vBMD (40% of Tb.Ar.) [mg HA/ccm]"
                ],
                "Tibia: Tb.Inn.vBMD [mg HA/cmm]": db_csv[
                    "Tibia inner trabecular vBMD (60% of Tb.Ar) [mg HA/ccm]"
                ],
                "Tibia: Tb.BV/TV [1]": db_csv["Tibia trabecular bone volume fraction"],
                "Tibia: Ct.Th [mm]": db_csv["Tibia cortical thickness [mm]"],
                "Tibia: Ct.Po [1]": db_csv["Tibia cortical porosity"],
                "Tibia: Ct.Po.Dm [mm]": db_csv["Tibia cortical pore diameter [mm]"],
                "Tibia: Tb.N [1/mm]": db_csv["Tibia trabecular number [1/mm]"],
                "Tibia: Tb.Th [mm]": db_csv["Tibia trabecular thickness [mm]"],
                "Tibia: Tb.Sp [mm]": db_csv["Tibia trabecular separation [mm]"],
                "Tibia: SD of 1/Tb.N [mm]": db_csv[
                    "Tibia SD of 1/Tb.N: Inhomogeneity of Network [-]"
                ],
                "Tibia: Fmax at failure [N]": np.abs(
                    db_csv["Tibia maximun force at failure [N]"]
                ),
                "Tibia: Stiffness [N/mm]": db_csv["Tibia stiffness [N/mm]"],
            }
        )
        # ---------------------------
        # Fill Missing Values
        # ---------------------------
        # Fill missing values for 'Age' and 'Gender' within each 'UID'
        df[["Age", "Gender"]] = (
            df.groupby("UID")[["Age", "Gender"]]
            .apply(lambda group: group.fillna(method="ffill").fillna(method="bfill"))
            .reset_index(drop=True)
        )
        # Filter to keep only rows with HR-pQCT scan results for either the Radius or Tibia
        hrpqct_columns = ["Radius: Measurement number", "Tibia: Measurement number"]
        df = df.dropna(
            how="all", subset=hrpqct_columns
        )  # Keep rows where at least one column has a non-null value
        return df


def main():
    # Import raw datafile from csv raw filepath
    dir_path = r"00_DB/"
    print(Path.cwd())
    # Nodaratis
    db_nodaratis = (
        r"843NodaratisStrength-NODARATIScommonDB_DATA_LABELS_2022-11-03_1342.csv"
    )
    db_nodaratis = (
        r"843NodaratisStrength-NODARATIScommonDB_DATA_LABELS_2023-05-31_1710.csv"
    )

    # TODO: update here the csv with the student's data
    db_nodaratis = (
        r"LB0843NodaratisStren-NODARATIScommonDB_DATA_LABELS_2024-09-04_1540.csv"
    )
    study_nodaratis = "Nodaratis"
    raw_filepath_nodaratis = Path(Path.cwd(), dir_path, db_nodaratis)
    data_nodaratis = pd.read_csv(raw_filepath_nodaratis, sep="\t")

    # AFFIRM-CT
    # db_affirmCT = r"LB1118AFFIRMCT_DATA_LABELS_2022-12-07_1514_aktuell.csv"
    # db_affirmCT = r"LB1118AFFIRMCT_DATA_2023-12-18_1157.csv"
    db_affirmCT = r"LB1118AFFIRMCT_DATA_2025-01-08_1529.csv"
    study_affirmCT = "AFFIRM-CT"
    raw_filepath_affirmCT = Path(Path.cwd(), dir_path, db_affirmCT)
    data_affirmCT = pd.read_csv(raw_filepath_affirmCT, sep=",", low_memory=False)

    # Reproducibility
    db_reproducibility = (
        r"Repro_UPAT_TOTAL_Denis_Cleaned_004_incl_nm_003_includedFRAX.csv"
    )
    study_reproducibility = "Reproducibility"
    raw_filepath_reproducibility = Path(Path.cwd(), dir_path, db_reproducibility)
    data_reproducibility = pd.read_csv(raw_filepath_reproducibility, sep=",")

    # BOLD
    db_bold_bone = r"HRpQCT_data_BOLD_final_hFE.csv"
    db_bold_patient = r"Aktuelle Liste Teilnehmer 20.02.2019 BOLD-1.csv"
    study_bold = "BOLD"
    raw_filepath_bold_bone = Path(Path.cwd(), dir_path, db_bold_bone)
    raw_filepath_bold_patient = Path(Path.cwd(), dir_path, db_bold_patient)
    data_bold_bone = pd.read_csv(raw_filepath_bold_bone, sep=",")
    data_bold_patient = pd.read_csv(raw_filepath_bold_patient, sep=",").sort_values(
        "Pat  ID"
    )

    # OIFRAC
    db_oifrac = r"OIFRAC_DATA_LABELS_2025-11-25_1128.csv"
    study_oifrac = "OIFRAC"
    raw_filepath_oifrac = Path(Path.cwd(), dir_path, db_oifrac)
    data_oifrac = pd.read_csv(raw_filepath_oifrac, sep="\t")

    # PARATY
    # db_paraty = r"LB1130PARATY_DATA_LABELS_2025-04-09_1622.csv"
    # study_paraty = "PARATY"
    # raw_filepath_paraty = Path(Path.cwd(), dir_path, db_paraty)
    # data_paraty = pd.read_csv(raw_filepath_paraty, sep="\t")

    df_nodaratis = study_converter(
        data_nodaratis, 1, study_nodaratis, dir_path=dir_path
    )
    df_affirmCT = study_converter(data_affirmCT, 1, study_affirmCT, dir_path=dir_path)
    df_reproducibility = study_converter(
        data_reproducibility, 1, study_reproducibility, dir_path=dir_path
    )
    df_bold = study_converter(
        data_bold_bone, data_bold_patient, study_bold, dir_path=dir_path
    )

    df_oifrac = study_converter(data_oifrac, 1, study_oifrac, dir_path=dir_path)

    # df_paraty = study_converter(data_paraty, 1, study_paraty, dir_path=dir_path)

    # Combine all studies to a single one
    frames = [
        df_nodaratis,
        df_affirmCT,
        df_reproducibility,
        df_bold,
        df_oifrac,
    ]  #! removed OIFRAC (not healthy dataset), df_oifrac]
    df_common = pd.concat(frames)

    # assign right data type to common dataframe
    df_common = df_common.astype(
        {
            "UID": "int64",
            "Gender": "object",
            "Ethnicity": "object",
            "Age": "int64",
            "Weight": "float64",
            "Height": "float64",
            "Study name": "object",
            "Previous fracture": "string",
            "Parent fractured hip": "string",
            "Current smoking": "string",
            "Glucocorticoids": "string",
            "Rheumatoid arthritis": "string",
            "Secondary osteoporosis": "string",
            "Alcohol 3 or more units/day": "string",
            "Femoral neck BMD (g/cm2)": "float64",
            "Time difference Frax & HRpQCT [days]": "object",
            "Radius: Measurement number": "float64",
            "Radius: side": "object",
            # "Radius: Tot.Ar [mm2]": "float64",
            "Radius: Ct.Ar [mm2]": "float64",
            "Radius: Tb.Ar [mm2]": "float64",
            "Radius: Tb.Meta.Ar [mm2]": "float64",
            "Radius: Tb.Inn.Ar [mm2]": "float64",
            "Radius: Ct.Pm [mm]": "float64",
            "Radius: Tt.vBMD [mg HA/cmm]": "float64",
            "Radius: Ct.vBMD [mg HA/cmm]": "float64",
            "Radius: Tb.vBMD [mg HA/cmm]": "float64",
            "Radius: Tb.Meta.vBMD [mg HA/cmm]": "float64",
            "Radius: Tb.Inn.vBMD [mg HA/cmm]": "float64",
            "Radius: Tb.BV/TV [1]": "float64",
            "Radius: Ct.Th [mm]": "float64",
            "Radius: Ct.Po [1]": "float64",
            "Radius: Ct.Po.Dm [mm]": "float64",
            "Radius: Tb.N [1/mm]": "float64",
            "Radius: Tb.Th [mm]": "float64",
            "Radius: Tb.Sp [mm]": "float64",
            "Radius: SD of 1/Tb.N [mm]": "float64",
            "Radius: Fmax at failure [N]": "float64",
            "Radius: Stiffness [N/mm]": "float64",
            # "Radius: Fmax at failure adjusted [N]": "float64",
            # "Radius: Stiffness adjusted [N/mm]": "float64",
            "Tibia: Measurement number": "float64",
            "Tibia: side": "object",
            # "Tibia: Tot.Ar [mm2]": "float64",
            "Tibia: Ct.Ar [mm2]": "float64",
            "Tibia: Tb.Ar [mm2]": "float64",
            "Tibia: Tb.Meta.Ar [mm2]": "float64",
            "Tibia: Tb.Inn.Ar [mm2]": "float64",
            "Tibia: Ct.Pm [mm]": "float64",
            "Tibia: Tt.vBMD [mg HA/cmm]": "float64",
            "Tibia: Ct.vBMD [mg HA/cmm]": "float64",
            "Tibia: Tb.vBMD [mg HA/cmm]": "float64",
            "Tibia: Tb.Meta.vBMD [mg HA/cmm]": "float64",
            "Tibia: Tb.Inn.vBMD [mg HA/cmm]": "float64",
            "Tibia: Tb.BV/TV [1]": "float64",
            "Tibia: Ct.Th [mm]": "float64",
            "Tibia: Ct.Po [1]": "float64",
            "Tibia: Ct.Po.Dm [mm]": "float64",
            "Tibia: Tb.N [1/mm]": "float64",
            "Tibia: Tb.Th [mm]": "float64",
            "Tibia: Tb.Sp [mm]": "float64",
            "Tibia: SD of 1/Tb.N [mm]": "float64",
            "Tibia: Fmax at failure [N]": "float64",
            "Tibia: Stiffness [N/mm]": "float64",
            # "Tibia: Fmax at failure adjusted [N]": "float64",
            # "Tibia: Stiffness adjusted [N/mm]": "float64",
        }
    )

    date_today = datetime.now().strftime("%Y-%m-%d")
    csv_path = Path(dir_path) / f"HR-pQCT_database_{date_today}.csv"
    df_common.to_csv(csv_path, sep=",", index=False)

    # csv_path = Path(dir_path) / f"PARATY_db_{date_today}.csv"
    # df_paraty.to_csv(csv_path, sep=",", index=False)

    return df_common


if __name__ == "__main__":
    main()
