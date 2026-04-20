from dataclasses import dataclass

import numpy as np
import pandas as pd

# flake8: noqa: E501


@dataclass
class HRpQCT_Dataset:
    """
    Class to hold the dataset and its metadata
    Args: (pandas.DataFrame) dataset: the dataset from REDCap
    Output: (dataclasses_hrpqct.Dataset) dataset: the dataset that can be used for analysis
    """

    df: pd.DataFrame
    hfe_expansion: bool = False

    def _get_stpe(self):
        """Get short-term precision error with default of 1e6 (%) if not set"""
        return getattr(self, "_stpe", 1e6)

    def _set_stpe(self, value: float):
        """Set short-term precision error (in %)"""
        self._stpe = float(value)

    # Add stpe property to all pandas Series objects
    pd.Series.stpe = property(_get_stpe, _set_stpe)

    def __avg_std__(self, attr, gender):
        """
        This also contains the few 'Repro' participants < self.age_limit
        """
        ref_avg = (attr[(self.Age <= self.age_limit) & (self.Gender == gender)]).mean()
        ref_std = (attr[(self.Age <= self.age_limit) & (self.Gender == gender)]).std()
        return ref_avg, ref_std

    def __bmi_calculator__(self):
        """
        Calculates the BMI from the height and weight of each participant
        Returns:
            float64: BMI value stored in the dataset
        """
        return self.Weight_kg / (self.Height_cm / 100) ** 2

    def radius_total_area(self):
        """
        Calculates the total area of the radius as the sum of the cortical and trabecular area of each participant
        Returns:
            float64: total area stored in the dataset
        """
        return self.Radius_cortical_area_mm2 + self.Radius_trabecular_area_mm2

    def tibia_total_area(self):
        """
        Calculates the total area of the tibia as the sum of the cortical and trabecular area of each participant
        Returns:
            float64: total area stored in the dataset
        """
        return self.Tibia_cortical_area_mm2 + self.Tibia_trabecular_area_mm2

    def radius_ultimate_stress(self):
        """
        Calculates the ultimate stress of the radius (ultimate force / total area) of each participant
        Returns:
            float64: ultimate stress stored in the dataset
        """
        return self.Radius_maximum_force_at_failure_N / self.Radius_total_area

    def tibia_ultimate_stress(self):
        """
        Calculates the ultimate stress of the tibia (ultimate force / total area) of each participant
        Returns:
            float64: ultimate stress stored in the dataset
        """
        return self.Tibia_maximum_force_at_failure_N / self.Tibia_total_area

    def __yield_stress__(self, force, area):
        """
        Calculates the yield stress of the radius (yield force / total area) of each participant
        Returns:
            float64: yield stress stored in the dataset
        """
        return force / area

    def radius_bmc(self):
        """
        Calculates the bone mineral content of the radius (vBMD/volume) of each participant
        Returns:
            float64: bone mineral content stored in the dataset
        """
        return (
            self.Radius_total_volumetric_bone_mineral_density_mg_HA_ccm
            * 10**-3
            * self.Radius_height_mm
            * self.Radius_total_area
        )

    def radius_cortical_bmc(self):
        """
        Calculates the bone mineral content of the radius (vBMD/volume) of each participant
        Returns:
            float64: bone mineral content stored in the dataset
        """
        return (
            self.Radius_cortical_vBMD_mg_HA_ccm
            * 10**-3
            * self.Radius_height_mm
            * self.Radius_cortical_area_mm2
        )

    def radius_trabecular_bmc(self):
        """
        Calculates the bone mineral content of the radius (vBMD/volume) of each participant
        Returns:
            float64: bone mineral content stored in the dataset
        """
        return (
            self.Radius_trabecular_vBMD_mg_HA_ccm
            * 10**-3
            * self.Radius_height_mm
            * self.Radius_trabecular_area_mm2
        )

    def radius_trabecular_to_cortical_bmc(self):
        """
        Calculates the bone mineral content of the radius (vBMD/volume) of each participant
        Returns:
            float64: bone mineral content stored in the dataset
        """
        return self.Radius_trabecular_bmc / self.Radius_cortical_bmc

    def radius_cortical_to_trabecular_bmc(self):
        """
        Calculates the bone mineral content of the radius (vBMD/volume) of each participant
        Returns:
            float64: bone mineral content stored in the dataset
        """
        return self.Radius_cortical_bmc / self.Radius_trabecular_bmc

    def tibia_bmc(self):
        """
        Calculates the bone mineral content of the tibia (vBMD/volume) of each participant
        Returns:
            float64: bone mineral content stored in the dataset
        """
        return (
            self.Tibia_total_volumetric_bone_mineral_density_mg_HA_ccm
            * 10**-3
            * self.Tibia_height_mm
            * self.Tibia_total_area
        )

    def tibia_cortical_bmc(self):
        """
        Calculates the bone mineral content of the radius (vBMD/volume) of each participant
        Returns:
            float64: bone mineral content stored in the dataset
        """
        return (
            self.Tibia_cortical_vBMD_mg_HA_ccm
            * 10**-3
            * self.Tibia_height_mm
            * self.Tibia_cortical_area_mm2
        )

    def tibia_trabecular_bmc(self):
        """
        Calculates the bone mineral content of the radius (vBMD/volume) of each participant
        Returns:
            float64: bone mineral content stored in the dataset
        """
        return (
            self.Tibia_trabecular_vBMD_mg_HA_ccm
            * 10**-3
            * self.Tibia_height_mm
            * self.Tibia_trabecular_area_mm2
        )

    def tibia_trabecular_to_cortical_bmc(self):
        """
        Calculates the bone mineral content of the radius (vBMD/volume) of each participant
        Returns:
            float64: bone mineral content stored in the dataset
        """
        return self.Tibia_trabecular_bmc / self.Tibia_cortical_bmc

    def tibia_cortical_to_trabecular_bmc(self):
        """
        Calculates the bone mineral content of the radius (vBMD/volume) of each participant
        Returns:
            float64: bone mineral content stored in the dataset
        """
        return self.Tibia_cortical_bmc / self.Tibia_trabecular_bmc

    def radius_rel_cortical_thickness(self):
        """
        Calculates the bone mineral content of the radius (vBMD/volume) of each participant
        Returns:
            float64: bone mineral content stored in the dataset
        """
        return self.Radius_cortical_thickness_mm / np.sqrt(
            self.Radius_total_area / np.pi
        )

    def tibia_rel_cortical_thickness(self):
        """
        Calculates the bone mineral content of the radius (vBMD/volume) of each participant
        Returns:
            float64: bone mineral content stored in the dataset
        """
        return self.Tibia_cortical_thickness_mm / np.sqrt(self.Tibia_total_area / np.pi)

    def __radius_apparent_modulus__(self):
        """
        Calculates the apparent modulus of the radius (stiffness*L_0/A_tot) of each participant
        Returns:
            float64: apparent modulus stored in the dataset
        """
        return (
            self.Radius_stiffness_N_mm * self.Radius_height_mm / self.Radius_total_area
        )

    def __tibia_apparent_modulus__(self):
        """
        Calculates the apparent modulus of the tibia (stiffness*L_0/A_tot) of each participant
        Returns:
            float64: apparent modulus stored in the dataset
        """
        return self.Tibia_stiffness_N_mm * self.Tibia_height_mm / self.Tibia_total_area

    def __radius_percentStiffness_f__(self):
        """
        Calculates the stiffness in percent for females of the radius of each participant
        Returns:
            float64: stiffness in percent stored in the dataset
        """
        ref_f = (
            self.Radius_stiffness_N_mm[
                (self.Study_name == "Nodaratis") & (self.Gender == "Female")
            ]
        ).mean()
        return (self.Radius_stiffness_N_mm[self.Gender == "Female"] / ref_f) * 100

    def __radius_percentStiffness_m__(self):
        """
        Calculates the stiffness in percent for the males of the radius of each participant
        Returns:
            float64: stiffness in percent stored in the dataset
        """
        ref_m = (
            self.Radius_stiffness_N_mm[
                (self.Study_name == "Nodaratis") & (self.Gender == "Male")
            ]
        ).mean()
        return (self.Radius_stiffness_N_mm[self.Gender == "Male"] / ref_m) * 100

    def __tibia_percentStiffness_f__(self):
        """
        Calculates the stiffness in percent for the females of the tibia of each participant
        Returns:
            float64: stiffness in percent stored in the dataset
        """
        ref_f = (
            self.Tibia_stiffness_N_mm[
                (self.Study_name == "Nodaratis") & (self.Gender == "Female")
            ]
        ).mean()
        return (self.Tibia_stiffness_N_mm[self.Gender == "Female"] / ref_f) * 100

    def __tibia_percentStiffness_m__(self):
        """
        Calculates the stiffness in percent for the males of the tibia of each participant
        Returns:
            float64: stiffness in percent stored in the dataset
        """
        ref_m = (
            self.Tibia_stiffness_N_mm[
                (self.Study_name == "Nodaratis") & (self.Gender == "Male")
            ]
        ).mean()
        return (self.Tibia_stiffness_N_mm[self.Gender == "Female"] / ref_m) * 100

    def radius_area_ratio(self):
        """
        Calculates the area ratio (cortical area / trabecular area) of the radius of each participant
        Returns:
            float64: stiffness in percent stored in the dataset
        """
        return self.Radius_cortical_area_mm2 / self.Radius_trabecular_area_mm2

    def tibia_area_ratio(self):
        """
        Calculates the area ratio (cortical area / trabecular area) of the tibia of each participant
        Returns:
            float64: stiffness in percent stored in the dataset
        """
        return self.Tibia_cortical_area_mm2 / self.Tibia_trabecular_area_mm2

    def _export_to_csv(self, filename):
        def _post_init_to_col_(col):
            return col.to_frame().rename(columns={col.name: col.name})

        # Create a pandas dataframe of all methods contained in __post__init__()
        df = pd.DataFrame()
        columns = [
            self.UID,
            self.Gender,
            self.Ethnicity,
            self.Age,
            self.Weight_kg,
            self.Height_cm,
            self.Study_name,
            self.Previous_fractures,
            self.Parents_hip_fracture_in_adulthood,
            self.Smoker,
            self.Glucocorticoids,
            self.Rheumatoid_arthritis,
            self.Secondary_osteoporosis,
            self.Alcohol_consumption_3_u_day_or_more,
            self.dexa_femoral_neck_BMD_g_cm2,
            self.Radius_measurement_number,
            self.Radius_side,
            self.Radius_total_volumetric_bone_mineral_density_mg_HA_ccm,
            self.Radius_trabecular_vBMD_mg_HA_ccm,
            self.Radius_meta_trabecular_vBMD_40_Tb_Ar_mg_HA_ccm,
            self.Radius_inner_trabecular_vBMD_60_Tb_Ar_mg_HA_ccm,
            self.Radius_cortical_vBMD_mg_HA_ccm,
            self.Radius_cortical_perimeter_mm,
            self.Radius_cortical_porosity,
            self.Radius_trabecular_bone_volume_fraction,
            self.Radius_trabecular_number_1_mm,
            self.Radius_trabecular_thickness_mm,
            self.Radius_trabecular_separation_mm,
            self.Radius_SD_of_1_Tb_N_Inhomogeneity_of_Network,
            self.Radius_cortical_thickness_mm,
            self.Radius_cortical_pore_diameter_mm,
            self.Radius_trabecular_area_mm2,
            self.Radius_trabecular_meta_area_mm2,
            self.Radius_trabecular_inner_area_mm2,
            self.Radius_cortical_area_mm2,
            self.Radius_maximum_force_at_failure_N,
            self.Radius_stiffness_N_mm,
            self.Radius_area_ratio,
            self.Tibia_measurement_number,
            self.Tibia_side,
            self.Tibia_total_volumetric_bone_mineral_density_mg_HA_ccm,
            self.Tibia_trabecular_vBMD_mg_HA_ccm,
            self.Tibia_meta_trabecular_vBMD_40_Tb_Ar_mg_HA_ccm,
            self.Tibia_inner_trabecular_vBMD_60_Tb_Ar_mg_HA_ccm,
            self.Tibia_cortical_vBMD_mg_HA_ccm,
            self.Tibia_cortical_perimeter_mm,
            self.Tibia_cortical_porosity,
            self.Tibia_trabecular_bone_volume_fraction,
            self.Tibia_trabecular_number_1_mm,
            self.Tibia_trabecular_thickness_mm,
            self.Tibia_trabecular_separation_mm,
            self.Tibia_SD_of_1_Tb_N_Inhomogeneity_of_Network,
            self.Tibia_cortical_thickness_mm,
            self.Tibia_cortical_pore_diameter_mm,
            self.Tibia_trabecular_area_mm2,
            self.Tibia_trabecular_meta_area_mm2,
            self.Tibia_trabecular_inner_area_mm2,
            self.Tibia_cortical_area_mm2,
            self.Tibia_maximum_force_at_failure_N,
            self.Tibia_stiffness_N_mm,
            self.Tibia_area_ratio,
            self.bmi,
            self.Radius_total_area,
            self.Tibia_total_area,
            self.Radius_ultimate_stress,
            self.Tibia_ultimate_stress,
            self.Radius_bmc,
            self.Radius_cortical_bmc,
            self.Radius_trabecular_bmc,
            self.Radius_trabecular_to_cortical_bmc,
            self.Radius_cortical_to_trabecular_bmc,
            self.Tibia_bmc,
            self.Tibia_cortical_bmc,
            self.Tibia_trabecular_bmc,
            self.Tibia_trabecular_to_cortical_bmc,
            self.Tibia_cortical_to_trabecular_bmc,
            self.Radius_rel_cortical_thickness,
            self.Tibia_rel_cortical_thickness,
            self.Radius_apparent_modulus,
            self.Tibia_apparent_modulus,
            self.Radius_percentStiffness_f,
            self.Radius_percentStiffness_m,
            self.Tibia_stiffness_hFE,
            self.Radius_stiffness_hFE,
            self.Tibia_yield_force,
            self.Radius_yield_force,
            self.Tibia_yield_stress,
            self.Radius_yield_stress,
            # self.Tibia_yield_stress_hfe,
            # self.Radius_yield_stress_hfe,
            self.Tibia_da,
            self.Radius_da,
            self.Tibia_simulation_time,
            self.Radius_simulation_time,
        ]

        for col in columns:
            df = pd.concat([df, _post_init_to_col_(col)], axis=1)
        df.to_csv(filename, index=False)
        return None

    def __post_init__(self):
        # fmt: off
        self.Radius_height_mm = float(20.4)
        self.Tibia_height_mm = float(30.6)
        self.age_limit = int(37)

        self.UID = self.df['UID']
        self.Gender = self.df['Gender']
        self.Ethnicity = self.df['Ethnicity']
        self.Age = self.df['Age']
        self.Age.name = 'Age (years)'
        self.Weight_kg = self.df['Weight']
        self.Height_cm = self.df['Height']
        self.Study_name = self.df['Study name']
        self.Previous_fractures = self.df['Previous fracture']
        self.Parents_hip_fracture_in_adulthood = self.df['Parent fractured hip']
        self.Smoker = self.df['Current smoking']
        self.Glucocorticoids = self.df['Glucocorticoids']
        self.Rheumatoid_arthritis = self.df['Rheumatoid arthritis']
        self.Secondary_osteoporosis = self.df['Secondary osteoporosis']
        self.Alcohol_consumption_3_u_day_or_more = self.df['Alcohol 3 or more units/day']
        self.dexa_femoral_neck_BMD_g_cm2 = self.df['Femoral neck BMD (g/cm2)']
        # self.Measurement_date =self.df['Measurement date']

        self.Radius_measurement_number = self.df['Radius: Measurement number'].astype(pd.Int64Dtype()) # pd.Int64Dtype() is used to allow NaN values
        self.Radius_side = self.df['Radius: side']
        try:
            self.Radius_total_volumetric_bone_mineral_density_mg_HA_ccm = self.df['Radius: Tot.vBMD [mg HA/cmm]']
        except KeyError:
            self.Radius_total_volumetric_bone_mineral_density_mg_HA_ccm = self.df['Radius: Tt.vBMD [mg HA/cmm]']
        self.Radius_total_volumetric_bone_mineral_density_mg_HA_ccm.name = 'Radius: Tot.vBMD [mg HA/cmm]'
        self.Radius_trabecular_vBMD_mg_HA_ccm = self.df['Radius: Tb.vBMD [mg HA/cmm]']
        self.Radius_meta_trabecular_vBMD_40_Tb_Ar_mg_HA_ccm = self.df['Radius: Tb.Meta.vBMD [mg HA/cmm]']
        self.Radius_inner_trabecular_vBMD_60_Tb_Ar_mg_HA_ccm = self.df['Radius: Tb.Inn.vBMD [mg HA/cmm]']
        self.Radius_cortical_vBMD_mg_HA_ccm = self.df['Radius: Ct.vBMD [mg HA/cmm]']
        self.Radius_cortical_perimeter_mm = self.df['Radius: Ct.Pm [mm]']
        self.Radius_cortical_porosity = self.df['Radius: Ct.Po [1]']
        self.Radius_trabecular_bone_volume_fraction = self.df['Radius: Tb.BV/TV [1]']
        self.Radius_trabecular_number_1_mm = self.df['Radius: Tb.N [1/mm]']
        self.Radius_trabecular_thickness_mm = self.df['Radius: Tb.Th [mm]']
        self.Radius_trabecular_separation_mm = self.df['Radius: Tb.Sp [mm]']
        self.Radius_SD_of_1_Tb_N_Inhomogeneity_of_Network = self.df['Radius: SD of 1/Tb.N [mm]']
        self.Radius_cortical_thickness_mm = self.df['Radius: Ct.Th [mm]']
        self.Radius_cortical_pore_diameter_mm = self.df['Radius: Ct.Po.Dm [mm]']
        self.Radius_trabecular_area_mm2 = self.df['Radius: Tb.Ar [mm2]']
        self.Radius_trabecular_meta_area_mm2 = self.df['Radius: Tb.Meta.Ar [mm2]']
        self.Radius_trabecular_inner_area_mm2 = self.df['Radius: Tb.Inn.Ar [mm2]']
        self.Radius_cortical_area_mm2 = self.df['Radius: Ct.Ar [mm2]']
        self.Radius_maximum_force_at_failure_N = self.df['Radius: Fmax at failure [N]']
        self.Radius_stiffness_N_mm = self.df['Radius: Stiffness [N/mm]']
        # self.Radius_maximum_force_at_failure_adjusted_N = self.df['Radius: Fmax at failure adjusted [N]']
        # self.Radius_stiffness_adjusted_N_mm = self.df['Radius: Stiffness adjusted [N/mm]']

        self.Radius_area_ratio = self.radius_area_ratio()
        self.Radius_area_ratio.name = 'Radius: Ct./Tb. [-]'

        self.Tibia_measurement_number = self.df['Tibia: Measurement number'].astype(pd.Int64Dtype()) # pd.Int64Dtype() is used to allow NaN values
        self.Tibia_side = self.df['Tibia: side']
        try:
            self.Tibia_total_volumetric_bone_mineral_density_mg_HA_ccm = self.df['Tibia: Tot.vBMD [mg HA/cmm]']
        except KeyError:
            self.Tibia_total_volumetric_bone_mineral_density_mg_HA_ccm = self.df['Tibia: Tt.vBMD [mg HA/cmm]']
        self.Tibia_total_volumetric_bone_mineral_density_mg_HA_ccm.name = 'Tibia: Tot.vBMD [mg HA/cmm]'
        self.Tibia_trabecular_vBMD_mg_HA_ccm = self.df['Tibia: Tb.vBMD [mg HA/cmm]']
        self.Tibia_meta_trabecular_vBMD_40_Tb_Ar_mg_HA_ccm = self.df['Tibia: Tb.Meta.vBMD [mg HA/cmm]']
        self.Tibia_inner_trabecular_vBMD_60_Tb_Ar_mg_HA_ccm = self.df['Tibia: Tb.Inn.vBMD [mg HA/cmm]']
        self.Tibia_cortical_vBMD_mg_HA_ccm = self.df['Tibia: Ct.vBMD [mg HA/cmm]']
        self.Tibia_cortical_perimeter_mm = self.df['Tibia: Ct.Pm [mm]']
        self.Tibia_cortical_porosity = self.df['Tibia: Ct.Po [1]']
        self.Tibia_trabecular_bone_volume_fraction = self.df['Tibia: Tb.BV/TV [1]']
        self.Tibia_trabecular_number_1_mm = self.df['Tibia: Tb.N [1/mm]']
        self.Tibia_trabecular_thickness_mm = self.df['Tibia: Tb.Th [mm]']
        self.Tibia_trabecular_separation_mm = self.df['Tibia: Tb.Sp [mm]']
        self.Tibia_SD_of_1_Tb_N_Inhomogeneity_of_Network = self.df['Tibia: SD of 1/Tb.N [mm]']
        self.Tibia_cortical_thickness_mm = self.df['Tibia: Ct.Th [mm]']
        self.Tibia_cortical_pore_diameter_mm = self.df['Tibia: Ct.Po.Dm [mm]']
        self.Tibia_trabecular_area_mm2 = self.df['Tibia: Tb.Ar [mm2]']
        self.Tibia_trabecular_meta_area_mm2 = self.df['Tibia: Tb.Meta.Ar [mm2]']
        self.Tibia_trabecular_inner_area_mm2 = self.df['Tibia: Tb.Inn.Ar [mm2]']
        self.Tibia_cortical_area_mm2 = self.df['Tibia: Ct.Ar [mm2]']
        self.Tibia_maximum_force_at_failure_N = self.df['Tibia: Fmax at failure [N]']
        self.Tibia_stiffness_N_mm = self.df['Tibia: Stiffness [N/mm]']

        # self.Tibia_maximum_force_at_failure_adjusted_N = self.df['Tibia: Fmax at failure adjusted [N]']
        # self.Tibia_stiffness_adjusted_N_mm = self.df['Tibia: Stiffness adjusted [N/mm]']
        # self.Tibia_maximum_force_at_failure_adjusted_N = self.df['Tibia: Fmax at failure adjusted [N]']
        # self.Tibia_stiffness_adjusted_N_mm = self.df['Tibia: Stiffness adjusted [N/mm]']
        # self.Tibia_tot_area_scanco = self.df['Tibia: Tot.Ar [mm2]']

        self.Tibia_area_ratio = self.tibia_area_ratio()
        self.Tibia_area_ratio.name = 'Tibia: Ct./Tb. [-]'

        self.bmi = self.__bmi_calculator__()
        self.bmi.name = 'BMI (kg/m²)'
        self.Radius_total_area = self.radius_total_area()
        self.Radius_total_area.name = 'Radius: Tot.Ar [mm2]'
        self.Tibia_total_area = self.tibia_total_area()
        self.Tibia_total_area.name = 'Tibia: Tot.Ar [mm2]'
        self.Radius_ultimate_stress = self.radius_ultimate_stress()
        self.Radius_ultimate_stress.name = 'Radius: Ult. stress [MPa]'
        self.Tibia_ultimate_stress = self.tibia_ultimate_stress()
        self.Tibia_ultimate_stress.name = 'Tibia: Ult. stress [MPa]'
        self.Radius_bmc = self.radius_bmc()
        self.Radius_bmc.name = 'Radius: BMC [g HA]'
        self.Radius_cortical_bmc = self.radius_cortical_bmc()
        self.Radius_cortical_bmc.name = 'Radius: Cortical BMC [g HA]'
        self.Radius_trabecular_bmc = self.radius_trabecular_bmc()
        self.Radius_trabecular_bmc.name = 'Radius: Trabecular BMC [g HA]'
        self.Radius_trabecular_to_cortical_bmc = self.radius_trabecular_to_cortical_bmc()
        self.Radius_trabecular_to_cortical_bmc.name = 'Radius: Trabecular to cortical BMC [g HA]'
        self.Radius_cortical_to_trabecular_bmc = self.radius_cortical_to_trabecular_bmc()
        self.Radius_cortical_to_trabecular_bmc.name = 'Radius: Cortical to trabecular BMC [g HA]'
        self.Tibia_bmc = self.tibia_bmc()
        self.Tibia_bmc.name = 'Tibia: BMC [g HA]'
        self.Tibia_cortical_bmc = self.tibia_cortical_bmc()
        self.Tibia_cortical_bmc.name = 'Tibia: Cortical BMC [g HA]'
        self.Tibia_trabecular_bmc = self.tibia_trabecular_bmc()
        self.Tibia_trabecular_bmc.name = 'Tibia: Trabecular BMC [g HA]'
        self.Tibia_trabecular_to_cortical_bmc = self.tibia_trabecular_to_cortical_bmc()
        self.Tibia_trabecular_to_cortical_bmc.name = 'Tibia: Trabecular to cortical BMC [g HA]'
        self.Tibia_cortical_to_trabecular_bmc = self.tibia_cortical_to_trabecular_bmc()
        self.Tibia_cortical_to_trabecular_bmc.name = 'Tibia: Cortical to trabecular BMC [g HA]'
        self.Radius_rel_cortical_thickness = self.radius_rel_cortical_thickness()
        self.Radius_rel_cortical_thickness.name = 'Radius: Rel.Ct.Th [-]'
        self.Tibia_rel_cortical_thickness = self.tibia_rel_cortical_thickness()
        self.Tibia_rel_cortical_thickness.name = 'Tibia: Rel.Ct.Th [-]'
        self.Radius_apparent_modulus = self.__radius_apparent_modulus__()
        self.Radius_apparent_modulus.name = 'Radius: Apparent modulus [MPa]'
        self.Tibia_apparent_modulus = self.__tibia_apparent_modulus__()
        self.Tibia_apparent_modulus.name = 'Tibia: Apparent modulus [MPa]'
        self.Radius_percentStiffness_f = self.__radius_percentStiffness_f__()
        self.Radius_percentStiffness_f.name = 'Percentage of female radius stiffness'
        self.Radius_percentStiffness_m = self.__radius_percentStiffness_m__()
        self.Radius_percentStiffness_m.name = 'Percentage of male radius stiffness'

        self.Tibia_percentStiffness_f = self.__tibia_percentStiffness_f__()
        self.Tibia_percentStiffness_f.name = 'Percentage of female tibia stiffness'
        self.Tibia_percentStiffness_m = self.__tibia_percentStiffness_m__()
        self.Tibia_percentStiffness_m.name = 'Percentage of male tibia stiffness'
        
        try:
            self.hand_grip_strength_dominant_kg = self.df['Hand grip strength dominant [kg]']
            self.hand_grip_strength_dominant_kg.name = 'Hand grip strength: dominant hand [kg]'
            self.hand_grip_strength_nondominant_kg = self.df['Hand grip strength non-dominant [kg]']
            self.hand_grip_strength_nondominant_kg.name = 'Hand grip strength: non-dominant hand [kg]'
        except KeyError:
            print("Hand grip strength data not available in the dataset.")
            pass
        
        if self.hfe_expansion is True:
            # Try to open results updated from pipeline, else pass
            # Trabecular degree of anisotropy
            try:
                self.Tibia_da = self.df['Tibia: Tb.DA']
            except KeyError:
                self.Tibia_da = self.df['Tibia: HFE Trabecular DA']
            self.Tibia_da.name = 'Tibia: Tb.DA'
            try:
                self.Radius_da = self.df['Radius: Tb.DA']
            except KeyError:
                self.Radius_da = self.df['Radius: HFE Trabecular DA']
            self.Radius_da.name = 'Radius: Tb.DA'
            
            # Yield force from hFE analysis
            try:
                self.Tibia_yield_force = self.df['Tibia: y.Force [N]']
            except KeyError:
                self.Tibia_yield_force = self.df['Tibia: HFE Yield Force [N]']
            self.Tibia_yield_force.name = 'Tibia: y.Force [N]'
            try:
                self.Radius_yield_force = self.df['Radius: y.Force [N]']
            except KeyError:
                self.Radius_yield_force = self.df['Radius: HFE Yield Force [N]']
            self.Radius_yield_force.name = 'Radius: y.Force [N]'

            # Yield stress from hFE analysis
            self.Tibia_yield_stress = self.__yield_stress__(self.Tibia_yield_force, self.Tibia_total_area)
            self.Tibia_yield_stress.name = 'Tibia: y.Stress [MPa]'
            self.Radius_yield_stress = self.__yield_stress__(self.Radius_yield_force, self.Radius_total_area)
            self.Radius_yield_stress.name = 'Radius: y.Stress [MPa]'
            
            # Apparent modulus from hFE analysis
            try:
                self.Tibia_stiffness_hFE = self.df['Tibia: hFE Stiffness [N/mm]']
            except KeyError:
                self.Tibia_stiffness_hFE = self.df['Tibia: HFE Stiffness [N/mm]']
            self.Tibia_stiffness_hFE.name = 'Tibia: hFE Stiffness [N/mm]'
    
            try:
                self.Radius_stiffness_hFE = self.df['Radius: hFE Stiffness [N/mm]']
            except KeyError:
                self.Radius_stiffness_hFE = self.df['Radius: HFE Stiffness [N/mm]']
            self.Radius_stiffness_hFE.name = 'Radius: hFE Stiffness [N/mm]'
            
            # Apparent yield stress from hFE analysis
            # try:
            #     self.Tibia_yield_stress_hfe = self.df['Tibia: HFE Apparent Yield Stress [MPa]']
            # except KeyError:
            #     self.Tibia_yield_stress_hfe = self.df['Tibia: HFE Apparent Yield Stress [MPa]']
            # self.Tibia_yield_stress_hfe.name = 'Tibia: HFE Apparent Yield Stress [MPa]'
            # try:
            #     self.Radius_yield_stress_hfe = self.df['Radius: HFE Apparent Yield Stress [MPa]']
            # except KeyError:
            #     self.Radius_yield_stress_hfe = self.df['Radius: HFE Apparent Yield Stress [MPa]']
            # self.Radius_yield_stress_hfe.name = 'Radius: HFE Apparent Yield Stress [MPa]'
            
            # Simulation time from hFE analysis
            self.Tibia_simulation_time = self.df['Tibia: HFE Simulation Time [s]']
            self.Radius_simulation_time = self.df['Radius: HFE Simulation Time [s]']
        else:
            self.Tibia_da = None
            self.Radius_da = None

        # self.T_score_femoral_neck_aBMD = self.t_score_femoral_neck_aBMD()
        # self.T_score_femoral_neck_aBMD.name = 'T-score femoral neck [-]'
        # fmt: on
