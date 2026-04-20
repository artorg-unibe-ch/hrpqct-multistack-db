from pathlib import Path

import pandas as pd

# flake8: noqa: E501


def read_database(database_path: Path):
    database = pd.read_csv(database_path)
    database["Radius: Measurement number"] = database[
        "Radius: Measurement number"
    ].apply(lambda x: f"{x:08.0f}")
    database["Tibia: Measurement number"] = database["Tibia: Measurement number"].apply(
        lambda x: f"{x:08.0f}"
    )
    return database


def read_upat_list(upat_list_path: Path):
    upat_df = pd.read_fwf(upat_list_path)

    upat_df["S-Nr"] = upat_df["S-Nr"].apply(lambda x: f"{x:08.0f}")
    upat_df["M-Nr"] = upat_df["M-Nr"].apply(lambda x: f"{x:08.0f}")

    upat_subset = upat_df[["S-Nr", "M-Nr", "Filename IMA-Label"]].copy()
    upat_subset["Filename IMA-Label"] = (
        upat_subset["Filename IMA-Label"].str.split().str[0]
    )
    return upat_subset


def read_uct_list(uct_list_path: Path):
    uct_list = pd.read_csv(uct_list_path, sep="\t")

    uct_df = pd.DataFrame()
    uct_df["S-Nr"] = uct_list["SampNo"].apply(lambda x: f"{x:08.0f}")
    uct_df["M-Nr"] = uct_list["MeasNo"].apply(lambda x: f"{x:08.0f}")
    # uct_df["Filename IMA-Label"] = uct_list["IMA-Dir"]
    uct_df["Filename IMA-Label"] = uct_list["Filename"]
    uct_df["TRI-DA"] = uct_list["TRI-DA"]
    print(uct_df.head(5))
    return uct_df


def read_simulation_results(simulation_results_path: Path):
    simulation_results = pd.read_csv(simulation_results_path)

    # Create dictionaries to map from Sample ID to each metric
    stiffness_map = dict(
        zip(simulation_results["Sample"], simulation_results["stiffness_1D_FZ_MAX"])
    )
    yield_force_map = dict(
        zip(simulation_results["Sample"], simulation_results["yield_force_FZ_MAX"])
    )
    sim_time_map = dict(
        zip(simulation_results["Sample"], simulation_results["simulation_time"])
    )
    trab_avg_da_map = dict(
        zip(simulation_results["Sample"], simulation_results["trabecular_average_DA"])
    )
    return (
        simulation_results,
        stiffness_map,
        yield_force_map,
        sim_time_map,
        trab_avg_da_map,
    )


def check_missing_results(database, simulation_results_path):
    # only print rows where Radius: poncioni_apparent_modulus or Tibia: poncioni_apparent_modulus is NaN
    missing_simulation_results = database[
        database["Radius: poncioni_apparent_modulus"].isna()
        | database["Tibia: poncioni_apparent_modulus"].isna()
    ]

    # Correct filtering by study name
    missing_nodaratis = missing_simulation_results[
        missing_simulation_results["Study name"] == "Nodaratis"
    ]
    missing_affirm_ct = missing_simulation_results[
        missing_simulation_results["Study name"] == "AFFIRM-CT"
    ]
    missing_bold = missing_simulation_results[
        missing_simulation_results["Study name"] == "BOLD"
    ]
    missing_repro = missing_simulation_results[
        missing_simulation_results["Study name"] == "Reproducibility"
    ]

    print(f"Missing in Nodaratis: {len(missing_nodaratis)}")
    print(f"Missing in AFFIRM-CT: {len(missing_affirm_ct)}")
    print(f"Missing in BOLD: {len(missing_bold)}")
    print(f"Missing in Reproducibility: {len(missing_repro)}")

    missing_basepath = simulation_results_path.parent / "missing"
    missing_basepath.mkdir(exist_ok=True)

    missing_nodaratis_path = missing_basepath / "missing_nodaratis.csv"
    missing_affirm_ct_path = missing_basepath / "missing_affirm_ct.csv"
    missing_bold_path = missing_basepath / "missing_bold.csv"
    missing_repro_path = missing_basepath / "missing_repro.csv"

    missing_paths = [
        missing_nodaratis_path,
        missing_affirm_ct_path,
        missing_bold_path,
        missing_repro_path,
    ]
    missing_dfs = [missing_nodaratis, missing_affirm_ct, missing_bold, missing_repro]

    for path, df in zip(missing_paths, missing_dfs):
        pd.concat(
            [df["Filename radius"].dropna(), df["Filename tibia"].dropna()]
        ).to_frame(name="Filename").to_csv(path, index=False)
    return None


def merge_databases(database, simulation_results):
    # First, create a copy of the database to work with
    result_database = database.copy()

    # Create a mapping dictionary from Sample to itself
    sample_map = dict(zip(simulation_results["Sample"], simulation_results["Sample"]))

    # Map the filenames to their corresponding sample IDs
    result_database["Radius: sample"] = result_database["Filename radius"].map(
        sample_map
    )
    result_database["Tibia: sample"] = result_database["Filename tibia"].map(sample_map)

    # Merge the original database with the simulation results
    merged_database = pd.merge(
        result_database,
        simulation_results,
        how="left",
        left_on=["Filename radius", "Filename tibia"],
        right_on=["Sample", "Sample"],
        suffixes=("", "_poncioni"),
    )

    # Drop the redundant columns from the simulation results
    columns_to_drop = [
        "Sample",
        "simulation_time",
        "stiffness_1D_FZ_MAX",
        "yield_force_FZ_MAX",
    ]
    merged_database = merged_database.drop(columns=columns_to_drop, errors="ignore")
    return merged_database


def map_simulation_results(
    database, stiffness_map, yield_force_map, sim_time_map, trab_avg_da_map
):
    # Map simulation results to radius and tibia measurements
    database["Radius: poncioni_apparent_modulus"] = database["Filename radius"].map(
        stiffness_map
    )
    database["Radius: poncioni_yield_force"] = database["Filename radius"].map(
        yield_force_map
    )
    database["Radius: poncioni_sim_time"] = database["Filename radius"].map(
        sim_time_map
    )
    database["Radius: trabecular_average_DA"] = database["Filename radius"].map(
        trab_avg_da_map
    )

    database["Tibia: poncioni_apparent_modulus"] = database["Filename tibia"].map(
        stiffness_map
    )
    database["Tibia: poncioni_yield_force"] = database["Filename tibia"].map(
        yield_force_map
    )
    database["Tibia: poncioni_sim_time"] = database["Filename tibia"].map(sim_time_map)
    database["Tibia: trabecular_average_DA"] = database["Filename tibia"].map(
        trab_avg_da_map
    )
    return database


def map_m_nr_to_c_filename(database, upat_subset):
    # Map radius and tibia measurement numbers to SCANCO C-filenames
    database["Filename radius"] = database["Radius: Measurement number"].map(
        dict(zip(upat_subset["M-Nr"], upat_subset["Filename IMA-Label"]))
    )

    database["Filename tibia"] = database["Tibia: Measurement number"].map(
        dict(zip(upat_subset["M-Nr"], upat_subset["Filename IMA-Label"]))
    )
    return database


def main():
    # Paths
    basepath = Path(
        "/home/simoneponcioni/Documents/01_PHD/03_Methods/HR-pQCT_database/00_DB/"
    )
    # original database
    database_path = basepath / "HR-pQCT_database_full.csv"
    new_database_path = basepath / "HR-pQCT_database_full_2025-01-28_with_filename.csv"
    uct_list = basepath / "UCT_LIST_20240911.TXT"
    simulation_results_path = (
        basepath
        / "2025-simulation-results"
        / "new-da"
        / "2025-simulation-results-new-da.csv"
    )

    database = read_database(database_path)
    # upat_list = read_upat_list(upat_list_path)
    uct_list = read_uct_list(uct_list)
    # database = map_m_nr_to_c_filename(database, upat_list)
    database = map_m_nr_to_c_filename(database, uct_list)
    (
        simulation_results,
        stiffness_map,
        yield_force_map,
        sim_time_map,
        trab_avg_da_map,
    ) = read_simulation_results(simulation_results_path)
    database = map_simulation_results(
        database, stiffness_map, yield_force_map, sim_time_map, trab_avg_da_map
    )
    check_missing_results(database, simulation_results_path)
    merged_database = merge_databases(database, simulation_results)
    # merged_database = merge_tri_da(merged_database, uct_list)
    print(merged_database)
    print("***")
    merged_database["Tibia: yield_stress"] = (
        merged_database["Tibia: yield_stress"] / merged_database["Tibia: total_area"]
    )

    merged_database.to_csv(new_database_path, index=False)
    return None


if __name__ == "__main__":
    main()
