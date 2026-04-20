import pandas as pd
from dataclasses import dataclass


@dataclass
class ReferenceValues:
    gender: str

    def __post_init__(self):
        df_masked = self.__mask_to_reference_values__(
            study="Nodaratis", gender=self.gender
        )
        df_masked = self.__get_average_per_column__(df_masked)
        df_masked = self.__get_stdev_per_column__(df_masked)
        self.ref_male = df_masked[df_masked.index.isin(["Average", "Stdev"])]
        self.ref_female = df_masked[df_masked.index.isin(["Average", "Stdev"])]

    def __mask_to_reference_values__(self, study, gender):
        df_masked = self.df[self.df["Study name"] == study]
        df_masked = df_masked[df_masked["Gender"] == gender]
        return df_masked

    def __get_average_per_column__(self, df_masked):
        return pd.concat(
            [
                df_masked,
                pd.DataFrame(
                    {"Average": df_masked.mean(axis=0, numeric_only=True)}
                ).transpose(),
            ]
        )

    def __get_stdev_per_column__(self, df_masked):
        return pd.concat(
            [
                df_masked,
                pd.DataFrame(
                    {"Stdev": df_masked.std(axis=0, numeric_only=True)}
                ).transpose(),
            ]
        )
