import configparser
from datetime import datetime
from pathlib import Path

import pandas as pd
import hrpqct_database.dataclasses_hrpqct as dataclass_hrpqct
import hrpqct_database.statistics_hrpqct as statistics_hrpqct
import hrpqct_database.utils.expand_database as db_utils
from hrpqct_database.gen_pdf import PDF

# flake8: noqa: E501


def main(
    common_df: pd.DataFrame,
    patient_UID: int,
    originator_s: str = "POS",
    include_hfe: bool = False,
    export_pdf: bool = False,
):
    dataset = dataclass_hrpqct.HRpQCT_Dataset(df=common_df, hfe_expansion=include_hfe)
    results_dir = Path(f"02_OUTPUT/{patient_UID}")
    # Path(results_dir).mkdir(parents=True, exist_ok=True)

    # extract only dataclass for patient_UID
    df_patient = common_df[common_df["UID"] == patient_UID]
    dataset_patient = dataclass_hrpqct.HRpQCT_Dataset(
        df=df_patient, hfe_expansion=include_hfe
    )

    db_stats = statistics_hrpqct.Statistics(
        df=dataset.df,
        df_patient=dataset_patient.df,
        name="HRpQCT common database",
        originator=originator_s,
        showfig=False,
        savefig=True,
    )

    patient_gender = dataset_patient.Gender.values[0]

    patient_properties_tibia = [
        dataset_patient.Tibia_total_volumetric_bone_mineral_density_mg_HA_ccm,
        dataset_patient.Tibia_cortical_vBMD_mg_HA_ccm,
        dataset_patient.Tibia_trabecular_bone_volume_fraction,
        dataset_patient.Tibia_rel_cortical_thickness,
        dataset_patient.Tibia_da,
        dataset_patient.Tibia_yield_stress,
    ]

    dataset_roperties_tibia = [
        dataset.Tibia_total_volumetric_bone_mineral_density_mg_HA_ccm,
        dataset.Tibia_cortical_vBMD_mg_HA_ccm,
        dataset.Tibia_trabecular_bone_volume_fraction,
        dataset.Tibia_rel_cortical_thickness,
        dataset.Tibia_da,
        dataset.Tibia_yield_stress,
    ]

    patient_properties_radius = [
        dataset_patient.Radius_total_volumetric_bone_mineral_density_mg_HA_ccm,
        dataset_patient.Radius_cortical_vBMD_mg_HA_ccm,
        dataset_patient.Radius_trabecular_bone_volume_fraction,
        dataset_patient.Radius_rel_cortical_thickness,
        dataset_patient.Radius_da,
        dataset_patient.Radius_yield_stress,
    ]

    dataset_properties_radius = [
        dataset.Radius_total_volumetric_bone_mineral_density_mg_HA_ccm,
        dataset.Radius_cortical_vBMD_mg_HA_ccm,
        dataset.Radius_trabecular_bone_volume_fraction,
        dataset.Radius_rel_cortical_thickness,
        dataset.Radius_da,
        dataset.Radius_yield_stress,
    ]

    for site in ["Radius", "Tibia"]:
        _site = site.lower()
        if _site == "radius":
            patient_properties = patient_properties_radius
            dataset_properties = dataset_properties_radius
        elif _site == "tibia":
            patient_properties = patient_properties_tibia
            dataset_properties = dataset_roperties_tibia

        t_scores_patient = []
        for i in range(len(dataset_properties)):
            ref_avg_i, ref_std_i = dataset.__avg_std__(
                dataset_properties[i], patient_gender
            )
            t_score_i = db_stats.t_score(patient_properties[i], ref_avg_i, ref_std_i)
            t_scores_patient.append(t_score_i)

        if _site == "tibia":
            db_stats.t_scores_tibia = t_scores_patient
        elif _site == "radius":
            db_stats.t_scores_radius = t_scores_patient
        db_stats.get_radar_chart(t_scores_patient, patient_UID)

        # * THIS IS AN EXAMPLE OF HOW TO CREATE SCATTER PLOTS
        for i in range(len(dataset_properties)):
            db_stats.patient_assessment(
                dataset_patient.Age,
                patient_properties[i],
                dataset.Age,
                dataset_properties[i],
                dataset.__avg_std__(dataset_properties[i], patient_gender),
                PAPER=True,
                QUADRATIC=True,
            )

    if export_pdf:
        patient_pdf = PDF(
            dataset_patient=dataset_patient,
            gender=patient_gender,
            study_name=dataset_patient.Study_name,
            site=_site,
            pat_no=patient_UID,
            Radius_measurement_number=dataset_patient.Radius_measurement_number.values[
                0
            ],
            Tibia_measurement_number=dataset_patient.Tibia_measurement_number.values[0],
            born=dataset_patient.Age.values[0],
            Tibia_height_mm=dataset_patient.Tibia_height_mm,
            Radius_height_mm=dataset_patient.Radius_height_mm,
            # Measurement_date=dataset_patient.Measurement_date.values[0],
            Measurement_date=datetime.now().strftime("%Y-%m-%d"),
            save_dir=db_stats.outputdir,
            images_3d_path=r"01_CODE\PDF-EXPORT\images_3d",
            images_section_path=r"01_CODE\PDF-EXPORT\images_section",
            t_scores={
                "Tibia": db_stats.t_scores_tibia,
                "Radius": db_stats.t_scores_radius,
            },
        )
        patient_pdf.gen_pdf()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    settings_path = Path(__file__).parent / "settings_main.cfg"
    config.read_file(open(settings_path, "r"))

    common_data_csv_relative_path = config.get(
        "DEFAULT", "common_data_csv_relative_path"
    )
    originator_acronym = config.get("DEFAULT", "originator_acronym")
    patient_UID = config.getint("DEFAULT", "patient_UID")
    INCLUDE_HFE = config.getboolean("DEFAULT", "include_hfe")
    export_pdf_bool = config.getboolean("DEFAULT", "export_pdf_bool")

    basepath = Path("00_DB")
    common_data_csv_relative_path = basepath / common_data_csv_relative_path

    if INCLUDE_HFE is True:
        path_hfe = basepath / config.get("DEFAULT", "path_hfe_results_csv")
        correspondences_path = basepath / config.get("DEFAULT", "correspondences_path")

        common_df = db_utils.expand_with_hfe(
            df=common_data_csv_relative_path,
            hfe_path=path_hfe,
            correspondences_path=correspondences_path,
        )
    else:
        path_common_csv = (common_data_csv_relative_path,)
        common_df = pd.read_csv(path_common_csv, sep=",", header=0)
    main(
        common_df,
        patient_UID=patient_UID,
        originator_s=originator_acronym,
        include_hfe=INCLUDE_HFE,
        export_pdf=export_pdf_bool,
    )
