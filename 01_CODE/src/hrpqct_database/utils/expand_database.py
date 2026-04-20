from pathlib import Path
import pandas as pd

# flake8: noqa: E501


def __hfe_correspondences_expansion__(
    hfe_df: pd.DataFrame, correspondences_df: pd.DataFrame
) -> pd.DataFrame:
    hfe_expanded_df = pd.merge(
        hfe_df, correspondences_df, left_on="Sample", right_on="Filename"
    )

    # sort columns
    hfe_expanded_df = hfe_expanded_df[
        [
            "ParentZero",
            "ParentOne",
            "Filename",
            "stiffness_1D_FZ_MAX",
            "yield_force_FZ_MAX",
            "apparent_yield_stress",
            "trabecular_average_DA",
            "simulation_time",
        ]
    ]

    hfe_expanded_df["ParentZero"] = hfe_expanded_df["ParentZero"].apply(
        lambda x: str(x).zfill(8)
    )
    hfe_expanded_df["ParentOne"] = hfe_expanded_df["ParentOne"].apply(
        lambda x: str(x).zfill(8)
    )
    return hfe_expanded_df


def __expand_database_with_hfe__(
    df: pd.DataFrame, hfe_expanded_df: pd.DataFrame
) -> pd.DataFrame:
    # Zero-pad '...: Measurement number' to 8 digits
    df["Tibia: Measurement number"] = df["Tibia: Measurement number"].apply(
        lambda x: str(int(x)).zfill(8) if not pd.isna(x) else x
    )
    df["Radius: Measurement number"] = df["Radius: Measurement number"].apply(
        lambda x: str(int(x)).zfill(8) if not pd.isna(x) else x
    )

    # Create a copy of simulation data for tibia and radius merges
    tibia_hfe_df = hfe_expanded_df.copy()
    radius_hfe_df = hfe_expanded_df.copy()

    # Rename columns in the tibia simulation data to add 'Tibia: ' prefix
    tibia_columns = {
        "stiffness_1D_FZ_MAX": "Tibia: HFE Stiffness [N/mm]",
        "yield_force_FZ_MAX": "Tibia: HFE Yield Force [N]",
        "apparent_yield_stress": "Tibia: HFE Apparent Yield Stress [MPa]",
        "trabecular_average_DA": "Tibia: HFE Trabecular DA",
        "simulation_time": "Tibia: HFE Simulation Time [s]",
    }
    tibia_hfe_df = tibia_hfe_df.rename(columns=tibia_columns)

    # Rename columns in the radius simulation data to add 'Radius: ' prefix
    radius_columns = {
        "stiffness_1D_FZ_MAX": "Radius: HFE Stiffness [N/mm]",
        "yield_force_FZ_MAX": "Radius: HFE Yield Force [N]",
        "apparent_yield_stress": "Radius: HFE Apparent Yield Stress [MPa]",
        "trabecular_average_DA": "Radius: HFE Trabecular DA",
        "simulation_time": "Radius: HFE Simulation Time [s]",
    }
    radius_hfe_df = radius_hfe_df.rename(columns=radius_columns)

    # Create a combined dataframe with both tibia and radius data
    combined_df = df.copy()

    # Merge tibia simulation data
    combined_df = pd.merge(
        combined_df,
        tibia_hfe_df[["ParentOne"] + list(tibia_columns.values())],
        left_on="Tibia: Measurement number",
        right_on="ParentOne",
        how="left",
    ).drop("ParentOne", axis=1)

    # Fix for duplicate columns
    for col in combined_df.columns:
        if col in list(radius_columns.values()):
            combined_df = combined_df.rename(columns={col: f"{col}_tibia_temp"})

    # Merge radius simulation data
    combined_df = pd.merge(
        combined_df,
        radius_hfe_df[["ParentOne"] + list(radius_columns.values())],
        left_on="Radius: Measurement number",
        right_on="ParentOne",
        how="left",
    ).drop("ParentOne", axis=1)

    # Restore original tibia columns
    for col in tibia_columns.values():
        if f"{col}_tibia_temp" in combined_df.columns:
            combined_df = combined_df.rename(columns={f"{col}_tibia_temp": col})

    # Make sure no "_y" suffix columns remain
    rename_cols = {}
    for col in combined_df.columns:
        if col.endswith("_y"):
            original_col = col.replace("_y", "")
            rename_cols[col] = original_col

    if rename_cols:
        combined_df = combined_df.rename(columns=rename_cols)

    return combined_df


def expand_with_hfe(
    df: Path, hfe_path: Path, correspondences_path: Path
) -> pd.DataFrame:
    """
    Expand the database with HFE data if hfe_expansion is True.
    """
    common_df = pd.read_csv(df, sep=",", header=0)
    hfe_df = pd.read_csv(hfe_path, sep=",", header=0)
    correspondences_df = pd.read_csv(correspondences_path, sep=",", header=0)

    correspondences_df = __hfe_correspondences_expansion__(
        hfe_df=hfe_df, correspondences_df=correspondences_df
    )

    expanded_df = __expand_database_with_hfe__(
        df=common_df,
        hfe_expanded_df=correspondences_df,
    )

    return expanded_df
