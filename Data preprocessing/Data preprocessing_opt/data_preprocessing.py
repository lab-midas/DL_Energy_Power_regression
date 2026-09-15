# ⚙️ Data Preprocessing
#
# ------------------------------------------------------------------------------
# 🗂️ Data Structure
#
# The folder structure for the data is organized as follows:
#
# 📂 Data storage Siemens
# ├── 📂 MRI data
# │   ├── 📂 single_MRI_machine_1
# │   │   ├── 📂 2024-12-11
# │   │   │   ├── 📄 Scanner.csv
# │   │   │   ├── 📄 Examinations.csv
# │   │   │   ├── 📄 Measurements.csv
# │   │   │   ├── 📄 Events.csv
# │   │   │   ├── 📄 ProtocolParameters.csv
# │   │   │   ├── 📄 ProtocolParametersDescription.csv
# │   │   │   ├── 📄 powerdata_1.csv
# │   │   │   └── ...
# │   │   ├── 📂 2024-12-12
# │   │   │   ├── 📄 Scanner.csv
# │   │   │   ├── 📄 Examinations.csv
# │   │   │   ├── 📄 Measurements.csv
# │   │   │   ├── 📄 Events.csv
# │   │   │   ├── 📄 ProtocolParameters.csv
# │   │   │   ├── 📄 ProtocolParametersDescription.csv
# │   │   │   ├── 📄 powerdata_1.csv
# │   │   │   └── ...
# │   ├── 📂 single_MRI_machine_2
# │   │   ├── 📂 2024-12-11
# │   │   │   ├── 📄 Scanner.csv
# │   │   │   ├── 📄 Examinations.csv
# │   │   │   ├── 📄 Measurements.csv
# │   │   │   ├── 📄 Events.csv
# │   │   │   ├── 📄 ProtocolParameters.csv
# │   │   │   ├── 📄 ProtocolParametersDescription.csv
# │   │   │   ├── 📄 powerdata_2.csv
# │   │   │   └── ...
# │   │   └── ...
# │   └── ...
# ├── 📂 CT data
# ├── 📂 Playground
# ├── 📂 Raw data
# ...
#
# ------------------------------------------------------------------------------
# 📁 Explanation of Folder Structure
#
# Raw Data 🗄️
# - Contains all the data downloaded by running the Half Automated Data Pulling Pipeline.
# - Includes both MRI and CT data. The corresponding data is manually copied into
#   the 'MRI data' and 'CT data' folders.
#
# MRI Data 🧲
# - Contains subfolders for each MRI machine (e.g., 'single_MRI_machine_1', 'single_MRI_machine_2').
# - Each machine folder contains subfolders for each date (e.g., '2024-12-11', '2024-12-12').
# - Each date folder contains the corresponding data and CSV files.
#
# CT Data 🩻
# - Similar structure to MRI data, but for CT machines.
#
# Test data 🛠️
# - Contains a subsample of data used for debugging and testing the data processing pipeline.
#
# ------------------------------------------------------------------------------
# 📄 Explanation of CSV Files
#
# Scanner.csv: General information about the scanner and MRI machines.
# Examinations.csv: Details about the examinations performed on the MRI machines.
# Measurements.csv: Information about individual measurements during the examinations.
# Events.csv: Logs of all events, mapping examinations and measurements.
# ProtocolParameters.csv: Parameters used in each measurement (abbreviated format).
# ProtocolParametersDescription.csv: Explanation of parameter abbreviations used in ProtocolParameters.csv.
# powerdata_*: Data collected from the powermeters at a 1 Hz sampling rate.
#   - Each scanner is connected to a specific powermeter, and the filename reflects this association.
#   - The filename always starts with 'powerdata_', followed by the name of the
#     corresponding powermeter (e.g., 'powerdata_Powermeter1.csv', 'powerdata_Powermeter2.csv').
#   - These files contain detailed power and energy consumption data for the scanner.
#
# ------------------------------------------------------------------------------
# 🗂️ Structure of the Merged DataFrame
#
# The resulting DataFrame from the MRI data preprocessing pipeline is hierarchical:
#
# Scanner ⟶ Examinations ⟶ Measurements ⟶ Parameters
#
# - Each scanner can have multiple examinations.
# - Each examination can have multiple measurements.
# - Each measurement is associated with multiple scanning parameters.
# - Additional information about daily energy consumption is included at the scanner level.
#
# ------------------------------------------------------------------------------
# 🔖 Column Suffixes Explained
#
# To clarify the origin of each column, suffixes are used:
#
# _scan     : Information about the scanner (e.g., model, serial number)
# _exam     : Information about the examinations (e.g., examination date)
# _meas     : Information about the measurements (e.g., measurement time, results)
# _param    : Information about the scanning parameters (e.g., protocol settings, technical details)
# _energy   : Daily energy consumption and power-related metrics (aggregated per scanner per day)
#
# This structure allows for comprehensive analysis, enabling you to trace each
# measurement back to its examination and scanner, while also incorporating
# relevant technical and energy data.
# ------------------------------------------------------------------------------

import os
import pandas as pd
import duckdb as dd
import sys
import glob
import warnings

# Get the current working directory
CWD = os.getcwd()

GLOBAL_UTILS_DIR = os.path.normpath(
    os.path.join(CWD, "..", "..", "Global utils_opt")
)

# Append this directory to the system path
sys.path.append(GLOBAL_UTILS_DIR)

from global_utils import *


# ------------------------------------------------------------------------------
# 🚀 Preparing the Merging Process
# ------------------------------------------------------------------------------
#
# To successfully prepare the merging process for MRI data, these steps are performed:
#
# 1. Get the Data Directory 📂
#    - Checks if the Data Directory exists. This directory has to contain the data.
#    - Function: get_data_directory
#
# 2. Set Up the DuckDB Connection 🛠️
#    - Initializes a DuckDB database by creating a dedicated folder and database file.
#    - Establishes a connection to the DuckDB database, enabling efficient storage
#      and querying of the data.
#    - Function: setup_duckdb_connection
#
# 3. Retrieve MRI Folders 🗂️
#    - Identify the first-level folders in the data directory. These folders
#      correspond to individual MRI scanners.
#    - Function: retrieve_mri_folders
#
# 4. Validate MRI Folders ✅
#    - Ensure that the data directory contains folders corresponding to MRI scanners.
#    - This is done by checking if the Scanner.csv file in each folder
#      contains a 'MR' pattern in the SiteSecondaryName column, which indicates
#      that the parent folder corresponds to an MRI scanner.
#    - Function: validate_mri_folders
#
# 5. Create Scanner Mapping 🔗
#    - Create a mapping between ScannerID, Serial, and PowermeterID by
#      iterating over the MRI folders.
#    - Extract the ScannerID and Serial from the Scanner.csv files.
#    - Retrieve the filenames of the power data CSV files, which contain the PowermeterID.
#    - Compile this information into a DataFrame for further processing.
#    - Function: create_scanner_mapping
#
# 6. Retrieve Power Boundaries ⚡
#    - Locate the powerboundaries_scanner_mapping.csv file in the data directory.
#    - This file contains the power boundaries for various scanner modes, including:
#        - EcoPowerMode 🌱
#        - Idle Mode 💤
#        - Scanning Mode 🧲
#    - The file specifies the power levels associated with each mode for different scanners.
#    - Note: This file must be added manually to the data directory before running the process.
#    - Function: retrieve_power_boundaries
#


def get_data_dir(data_dir: str) -> str:
    """
    Create the path to the data directory and check if it exists.
    If it does, return the path. If it does not, raise an error.

    """

    # Check if the data storage directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"❌ The data storage directory {data_dir} does not exist."
        )
    else:
        print(f"✅ Data storage directory found: {data_dir}")

    return data_dir


def set_up_duckdb(cwd: str) -> dd.DuckDBPyConnection:
    """
    Create a DuckDB folder in the current working directory if it does not exist.
    Set up a DuckDB database file in that folder and return a connection to it.

    Parameters:
        cwd (str): The current working directory.

    Returns:
        dd.DuckDBPyConnection: A connection object to the DuckDB database.
    """

    # Create a duckdb folder if it doesn't exist
    duckdb_dir = os.path.join(cwd, "duckdb_files")
    os.makedirs(duckdb_dir, exist_ok=True)
    print(f"✅ DuckDB files directory is at: {duckdb_dir}")

    # Connect to a DuckDB database file in the duckdb_files directory
    duckdb_path = os.path.join(duckdb_dir, "MRI_data.duckdb")
    CON = dd.connect(
        duckdb_path, read_only=False
    )  # Ensure read_only=False for write access
    print(f"✅ DuckDB database created at: {duckdb_path}")

    # Set the temporary directory for DuckDB to the duckdb_files directory, this
    # allows DuckDB to store files on disk when performing operations
    # exceed RAM memory limits
    CON.execute(f"SET temp_directory='{duckdb_dir}'")
    print(f"✅ Temporary directory for DuckDB set to: {duckdb_dir}")

    return CON


def get_mri_folders(data_dir: str, dir_exclude: list = []) -> list:
    """
    The data of the MRI scanners is stored in dedictaed folders for each MRI
    scanner. For further processing get a list of all MRI folder. Always
    exclude the DuckDB folder and CSV files which were created to allow the
    merging. Additionally, the user can specify other folders to exclude by
    providing a list of folder names to exclude.

    Parameters:
        data_dir (str): Path to the data directory.
        dir_exclude (list): List of directory names to exclude.

    Returns:
        list: List of MRI folder names.
    """

    # Get the list of all folders which should correspond to MRI scanners.
    # Exclude the folder that contains the DuckDB temp files and any CSV files which
    # were created to allow the merging
    return [
        folder
        for folder in os.listdir(data_dir)
        if (
            not folder.startswith("duckdb")
            and not folder.endswith(".csv")
            and folder not in dir_exclude
        )
    ]


def check_mri_folder(
    data_dir: str, CON: dd.DuckDBPyConnection, mri_folders: list
) -> list:
    """
    Control if the data directory contains folders corresponding to MRI scanners by
    checking if the given Scanner.csv file contains a 'MR' pattern in the
    SiteSecondaryName column which indicates that the parent folder corresponds
    to an MRI scanner.

    Parameters:
        data_dir (str): Path to the data directory.
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.
        mri_folders (list): List of MRI folder names.

    Returns:
        list: List of valid MRI folder names.
    """
    # Initialize an empty list to store valid MRI folder names
    valid_mri_folders = []

    # Iterate over MRI folders
    for mri_folder in tqdm(mri_folders, desc="Checking MRI folders"):

        # Construct the path to the MRI folder
        mri_folder_path = os.path.join(data_dir, mri_folder)

        # Get the distinct SiteSecondaryName values from all Scanner.csv files in
        # the MRI folder
        scanner_df = CON.execute(
            f"""
            SELECT DISTINCT
                SiteSecondaryName
            FROM 
                read_csv_auto('{mri_folder_path}/*/Scanner.csv', union_by_name=true)
            WHERE 
                SiteSecondaryName IS NOT NULL AND
                SiteSecondaryName LIKE 'MR%'
            """
        ).df()

        # Check if the column SiteSecondaryName is empty in the DataFrame, which
        # indicates that no MRI scanner was found
        if scanner_df.empty:
            raise ValueError(
                f"""
                Folder {mri_folder} does not contain any Scanner.csv files with 
                a 'MR' pattern in the SiteSecondaryName column. Indicating
                that this folder does not correspond to an MRI scanner. Please
                check the folder structure and ensure that only MRI scanner folders
                are present in the data directory.
                """
            )
        else:
            valid_mri_folders.append(mri_folder)

    print(
        f"✅ No invalid MRI folders found.\nValid MRI folders:\n"
        + "\n".join(f"\t📂 {folder}" for folder in valid_mri_folders)
    )

    return valid_mri_folders


def create_mapping_scanner_mapping(
    data_dir: str, CON: dd.DuckDBPyConnection, mri_folders: list
) -> dd.DuckDBPyConnection:
    """
    Create a mapping between ScannerID, Serial, and PowermeterID by iterating over
    the MRI folders. Extract the ScannerID and Serial from the Scanner.csv files,
    get the filenames of the powerdata CSV files which contain the PowermeterID
    and compile this information into a DataFrame. The mapping is saved to a
    CSV file in the data directory and also registered as a DuckDB table.

    Parameters:
        data_dir (str): Path to the data directory.
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.
        mri_folders (list): List of MRI folder names.

    Returns:
        dd.DuckDBPyConnection: The DuckDB connection with the powermeter_scanner_mapping table registered.
    """

    # Initialize an empty list to store the mappings
    mappings = []

    # Iterate over MRI folders, use tqdm to show progress
    for mri_folder in tqdm(mri_folders, desc="Processing MRI folders"):

        # Get unique ScannerID and Serial assuming one per folder
        scanner_df = CON.execute(
            f"""
            SELECT DISTINCT 
                ScannerID,
                Serial
            FROM 
                read_csv_auto('{data_dir}/{mri_folder}/*/Scanner.csv', union_by_name=true)
            """
        ).df()

        # Skip if no scanner data
        if scanner_df.empty:
            warnings.warn(f"No scanner data found in {mri_folder}, skipping...")
            continue

        # Get unique ScannerIDs, which should be one per folder
        scanner_ids = scanner_df["ScannerID"].unique()
        # Get unique Serials, which should be one per folder
        serials = scanner_df["Serial"].unique()

        # If there are multiple unique ScannerIDs or Serials,
        # raise an error since we expect only 1
        if len(scanner_ids) > 1 or len(serials) > 1:
            raise ValueError(
                f"Multiple unique ScannerIDs/Serials in {mri_folder}: "
                f"{scanner_ids}, {serials}"
            )

        # Get the single ScannerID and Serial for the current MRI folder
        scanner_id = scanner_ids[0]
        serial = serials[0]

        # Get powerdataID by scanning the folder structure for
        # powerdata CSV files which should be in the format powerdata_{PowermeterID}.csv
        power_files = glob.glob(
            os.path.join(data_dir, mri_folder, "*", "powerdata_*.csv")
        )

        # Skip if no power data files found
        if not power_files:
            warnings.warn(f"No power data files found in {mri_folder}, skipping...")
            continue

        # Extract PowermeterID from filename by splitting the filename
        # assuming format powerdata_{PowermeterID}.csv
        power_ids = {
            os.path.splitext(os.path.basename(f))[0].split("_", 1)[1]
            for f in power_files
        }

        # Create mappings for each PowermeterID
        for power_id in power_ids:
            mappings.append(
                {
                    "ScannerID": scanner_id,
                    "Serial": serial,
                    "PowermeterID": power_id,
                }
            )

    # Create DataFrame and save to CSV, ensuring to drop duplicates and sort for
    # better readability
    df = (
        pd.DataFrame(mappings)
        .drop_duplicates()
        .sort_values(by=["ScannerID", "Serial", "PowermeterID"])
        .reset_index(drop=True)
    )

    # Save the mapping to a CSV file in the data directory
    output_path = os.path.join(data_dir, "powermeter_scanner_mapping.csv")
    df.to_csv(output_path, index=False, sep=";")

    # Register the DataFrame as a DuckDB table for further use in SQL queries
    CON.register("powermeter_scanner_mapping", df)

    return CON


def get_powerboundries_scanner(
    data_dir: str, CON: dd.DuckDBPyConnection
) -> dd.DuckDBPyConnection:
    """
    Retrieve the powerboundaries_scanner_mapping.csv file, which contains the
    power boundaries for various scanner modes, including EcoPowerMode, Idle,
    and Scanning mode. This file specifies the power levels associated with
    each mode for different scanners. It has to be added manually to the
    data directory.

    Parameters:
        data_dir (str): Path to the data directory.
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.

    Returns:
        dd.DuckDBPyConnection: The DuckDB connection with the powerboundries_scanner table registered.
    """

    # Ensure that the powerboundries_scanner_mapping.csv file exists in the data directory
    powerboundries_scanner_path = os.path.join(
        data_dir, "powerboundries_scanner_mapping.csv"
    )
    if not os.path.exists(powerboundries_scanner_path):
        raise FileNotFoundError(
            f"""❌ The file powerboundries_scanner_mapping.csv was not found in the data directory {data_dir}.
        Please ensure that this file is present in the data directory and contains the necessary power boundaries
        for the different scanner modes mapped to the corresponding Serial of the MRI scanner. 🧲⚡"""
        )
    else:
        print(
            f"""✅ Poweboundries file found: {powerboundries_scanner_path}\t
        Ready to load power boundaries for MRI scanners! 🧲"""
        )

    # Read the powerboundries_scanner_mapping.csv file into a DataFrame
    powerboundries_scanner_df = CON.execute(
        f"""
        SELECT
            *
        FROM
            read_csv_auto('{data_dir}/powerboundries_scanner_mapping.csv') as powerboundries_scanner
        """
    ).df()

    # Register the scannermode_powermeter DataFrame as a DuckDB table
    CON.register("powerboundries_scanner", powerboundries_scanner_df)

    return CON


# ------------------------------------------------------------------------------
# 🚀 Explanation of merging functions
# ------------------------------------------------------------------------------
#
# The following functions are used to ensure a successful and efficient merging process:
#
# 1. Retrieve Protocol Parameters Descriptions 📄
#    - Collect all `ProtocolParametersDescription.csv` files from the data directory.
#    - Combine these files into a unified dataset containing all possible protocol descriptions.
#    - Function: get_parameters_description
#
# 2. Validate Data Files ✅
#    - Verify that each date folder contains the required CSV files:
#        - Events.csv
#        - Examinations.csv
#        - Measurements.csv
#        - ProtocolParameters.csv
#        - powerdata_*.csv
#    - Ensures all necessary data is available for merging.
#    - Function: check_data
#
# 3. Retrieve Data from CSV Files 📂
#    - Helper function to extract data from a specified CSV file in a given directory.
#    - Utilizes DuckDB for efficient reading and querying.
#    - Function: get_data
#
# 4. Unify Energy Measurements ⚡
#    - Identify all columns in Measurements and Examinations DataFrames starting
#      with `TotalEnergy`.
#    - Sum these columns to create `SiemensTotalEnergy_KWh`.
#    - Drop the original `TotalEnergy` columns.
#    - Unifies energy measurements from different MRI scanners, handling missing values.
#    - Function: unify_energy_meas_exams
#
# 5. Rename Parameter Columns 🏷️
#    - Standardize column names in the Parameters DataFrame using a renaming dictionary.
#    - Ensures consistency and clarity.
#    - Function: rename_parameter
#
# 6. Merge Measurements and Parameters 🔗
#    - Merge Measurements and renamed Parameters DataFrames.
#    - Use MeasurementID and FK_MeasurementID as join keys.
#    - Only measurements with corresponding protocol parameters are included.
#    - Function: merge_meas_params
#
# 7. Merge Scanner, Examinations, Measurements and Parameters 📊
#    - Integrate the `Scanner`, `Examinations`, and `Measurements + Parameters`
#      DataFrames into a single comprehensive dataset.
#    - Merge the `Scanner` DataFrame to both `Examinations` and `Measurements + Parameters`
#      using the relationships defined in the `Events.csv` file (via the relevant IDs).
#    - Merge `Examinations` and `Measurements + Parameters` based on time intervals:
#      a measurement is included only if its `MeasurementStart` and `MeasurementEnd`
#      fall within the `ExaminationStart` and `ExaminationEnd` of the corresponding examination.
#    - Function: merge_scan_exams_meas_params
#
# 8. Merge Power Data with Scanner Mapping ⚡
#    - Create a dataset including ScannerID, Serial, and all relevant power data.
#    - Connects power measurements to corresponding MRI scanners.
#    - Function: merge_power_mapping
#
# 9. Enrich Power Data ⚡
#    - Enhance power data with new columns for total energy, apparent power,
#      active power, and reactive power.
#    - Sum values across phases (L1, L2, L3) for unified columns.
#    - Create a date key from the Time column for daily aggregation.
#    - Map power boundaries for scanner modes by joining with powerboundaries_scanner.
#    - Function: enrich_power_data
#
# 10. Aggregate Power and Energy per Scanner 📊
#     - Aggregate energy and power data at the scanner level (for scanners
#       with multiple powermeters).
#     - Sum energy columns and average power columns across powermeters.
#     - Function: agg_power_boundaries
#
# 11. Calculate Daily Energy Consumption 📅
#     - Compute daily energy consumption for different scanner modes:
#         - Idle Mode 💤
#         - Eco Power Mode 🌱
#         - Scanning Mode 🧲
#     - Function: calculate_daily_energy_consumption
#
# 12. Merge Measurements with Aggregated Power Data 🔗
#     - Combine scan_exams_meas_params with agg_power_boundaries.
#     - Join based on scanner serial and examination interval.
#     - Calculate total energy and average power for each measurement period.
#     - Function: merge_measurements_with_power
#
# 13. Merge Examinations with Aggregated Power Data 🔗
#     - Combine scan_exams_meas_params with agg_power_boundaries.
#     - Join based on scanner serial and examination interval.
#     - Calculate total energy and average power for each examination period.
#     - Function: merge_examinations_with_power
#
# 14. Merge Measurements Parameters with Examinations and Power Data 🔗
#     - Integrate Measurements Parameters with Examinations and aggregated power data.
#     - Ensures a comprehensive dataset for further analysis.
#     - Function: merge_measurements_params_with_examinations
#
# 15. Merge Daily Aggregated Power Data 📈
#     - Combine daily aggregated power data with Measurements Parameters and Examinations.
#     - Links daily power data with corresponding measurements and examinations.
#     - Function: merge_daily_power_data
#
# 16. Reorder DataFrame Columns 🗂️
#     - Reorganize columns of the final DataFrame based on suffix priority.
#     - After applying suffix priority, sort columns alphabetically for readability
#       and consistency.
#     - Function: reorder_columns


def get_parameters_description(
    data_dir: str, CON: dd.DuckDBPyConnection
) -> Tuple[pd.DataFrame, dict]:
    """
    Get the ProtocolParametersDescription.csv files from all MRI folders, join them
    into a single DataFrame, and drop duplicates. This ensures that we have a
    comprehensive and unique set of protocol parameter descriptions across all
    MRI scanners.

    Parameters:
        data_dir (str): Path to the data directory.
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame containing the joined ProtocolParametersDescription data
                            with duplicates dropped.
            - dict: A dictionary for renaming columns, mapping ProtocolParameterName to Explanation.
    """

    # Get all joined ProtocolParametersDescription.csv files into a single DataFrame
    joined_protcol_parameters_df = CON.execute(
        f"""
        SELECT 
            *
        FROM 
            read_csv_auto('{data_dir}/*/*/ProtocolParametersDescription.csv', union_by_name=true)
        AS 
            protcol_params   
        WHERE 
            protcol_params.Explanation IS NOT NULL AND
            protcol_params.ProtocolParameterName IS NOT NULL
        ORDER BY
            protcol_params.ProtocolParameterName
        """
    ).df()

    # Drop duplicates
    param_renaming_df = joined_protcol_parameters_df.drop_duplicates()

    # Sort by the ProtocolParameterName for better readability
    param_renaming_df = param_renaming_df.sort_values(
        by="ProtocolParameterName"
    ).reset_index(drop=True)

    # Define renaming dictionary by zipping the ProtocolParameterName and Explanation
    # columns from the unique protocol parameters DataFrame
    param_renaming_dict = dict(
        zip(
            param_renaming_df.ProtocolParameterName,
            param_renaming_df.Explanation,
        )
    )

    return param_renaming_df, param_renaming_dict


def check_data(data_dir: str) -> bool:
    """
    Verify that the data directory contains all the required files and folders
    for the MRI data analysis. This includes the following files: Events.csv,
    Examinations.csv, Measurements.csv, ProtocolParameters.csv, and any
    powerdata_*.csv files.

    Parameters:
        data_dir (str): Path to the data directory.

    Returns:
        bool: True if all expected files and folders are found, False otherwise.
    """

    # Use glob to check for the presence of the expected files in the data directory
    events_files = glob.glob(os.path.join(data_dir, "Events.csv"))
    examinations_files = glob.glob(os.path.join(data_dir, "Examinations.csv"))
    measurements_files = glob.glob(os.path.join(data_dir, "Measurements.csv"))
    protocol_parameters_files = glob.glob(
        os.path.join(data_dir, "ProtocolParameters.csv")
    )
    powerdata_files = glob.glob(os.path.join(data_dir, "powerdata_*.csv"))

    # Check if all expected files are found
    all_files_available = all(
        [
            events_files,
            examinations_files,
            measurements_files,
            protocol_parameters_files,
            powerdata_files,
        ]
    )

    return all_files_available


def get_data(data_dir: str, csv_file: str) -> pd.DataFrame:
    """
    Get the data from a defined CSV file located in the specified directory
    by using DuckDB to read the CSV file.

    Parameters:
        data_dir (str): Path to the data directory.
        csv_file (str): Name of the CSV file to read.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the specified CSV file.
    """

    # Get the data from a CSV file located in the specified directory.
    data_df = CON.execute(
        f"""
    SELECT 
        *
    FROM 
        read_csv_auto('{data_dir}/{csv_file}', union_by_name=true)
    """
    ).df()

    return data_df


def unify_energy_meas_exams(
    meas_df: pd.DataFrame, exams_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unify the energy measurements in the measurements and examinations DataFrame
    by identifying all columns that start with "TotalEnergy", summing them up
    to create a new column "SiemensTotalEnergy_KWh", and then dropping the
    original "TotalEnergy" columns.This process allows us to unify the energy
    measurements from different MRI scanners into a single column while
    ensuring that we handle missing values appropriately.

    Parameters:
        meas_df (pd.DataFrame): The DataFrame containing the measurements data with energy columns.
        exams_df (pd.DataFrame): The DataFrame containing the examinations data with energy columns.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A cleaned DataFrame containing the measurements data with a unified energy column.
            - pd.DataFrame: A cleaned DataFrame containing the examinations data with a unified energy column.
    """

    # Get the columns that start with "TotalEnergy" which indicate the energy
    # measurements. For each MRI scanner the energy column is differently named
    # but they all start with "TotalEnergy" which allows us to identify them.
    energy_columns_meas = [
        col for col in meas_df.columns if col.startswith("TotalEnergy")
    ]
    energy_clumns_exams = [
        col for col in exams_df.columns if col.startswith("TotalEnergy")
    ]

    # Get all the rows where the TotalEnergy columns are all NaN to check if there
    # are measurements without energy data
    meas_without_energy = meas_df[energy_columns_meas].isna().all(axis=1)
    exams_without_energy = exams_df[energy_clumns_exams].isna().all(axis=1)

    # Add all columns starting with "TotalEnergy" to create a new column "IntermediateEnergy".
    # By default, pandas sum ignores NaN values and sums the available values.
    # However, if all values in a row are NaN, the result will be 0.
    # To ensure the result is NaN when all values are NaN,
    # we identify such rows beforehandand set the sum to NaN for them afterwards.
    meas_df["SiemensTotalEnergy_KWh"] = meas_df[energy_columns_meas].sum(
        axis=1, skipna=True
    )
    exams_df["SiemensTotalEnergy_KWh"] = exams_df[energy_clumns_exams].sum(
        axis=1, skipna=True
    )

    # Drop all columns starting with "TotalEnergy"
    meas_df = meas_df.drop(columns=energy_columns_meas, inplace=False)
    exams_df = exams_df.drop(columns=energy_clumns_exams, inplace=False)

    # Set the measurements without energy data to NaN in the SiemensTotalEnergy_KWh column
    meas_df.loc[meas_without_energy, "SiemensTotalEnergy_KWh"] = pd.NA
    exams_df.loc[exams_without_energy, "SiemensTotalEnergy_KWh"] = pd.NA

    return meas_df, exams_df


def rename_parameter(
    param_renaming_dict: dict, params_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Rename the columns of the parameters DataFrame using the provided renaming dictionary.

    Parameters:
        param_renaming_dict (dict): A dictionary mapping old column names to new column names.
        params_df (pd.DataFrame): The DataFrame containing the parameters to be renamed.

    Returns:
        pd.DataFrame: A DataFrame with renamed columns based on the renaming dictionary.

    """

    # Rename the columns of the parameters DataFrame using the renaming dictionary
    renamed_params_df = params_df.rename(columns=param_renaming_dict, inplace=False)

    return renamed_params_df


def merge_meas_params(CON: dd.DuckDBPyConnection) -> pd.DataFrame:
    """
    Merge the measurements DataFrame with the renamed parameters DataFrame
    using an inner join on MeasurementID and FK_MeasurementID to ensure that
    only measurements with corresponding protocol parameters are included in the
    merged DataFrame.

    Parameters:
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.
    Returns:
        pd.DataFrame: A merged DataFrame containing both measurements and protocol parameters.
    """

    # Merge the measurements DataFrame with the renamed parameters DataFrame
    meas_params_df = CON.execute(
        f"""
        SELECT
            meas.*, 
            renamed_params.*
        FROM
            meas
        INNER JOIN
            renamed_params
        ON
            meas.MeasurementID_meas = renamed_params.FK_MeasurementID_param
        ORDER BY
            meas.MeasurementID_meas
        """
    ).df()

    return meas_params_df


def merge_scan_exams_meas_params(CON: dd.DuckDBPyConnection) -> pd.DataFrame:
    """
    Merge the scanner, examinations, measurements + parameters DataFrames into
    a single DataFrame using inner joins on the relevant IDs and time intervals
    to retrieve a maximum amount of data. The scan DataFrame is merged
    with the examinations and measurements + parameters DataFrames based on
    the IDs which are included in the Event.csv files. The examinations and
    measurements + parameters DataFrames are merged based on the time intervals
    of the examinations and measurements. Meaning the MeasurementStart_meas and
    MeasurementEnd_meas has to be between the ExaminationStart_exam and ExaminationEnd_exam
    of the examinations DataFrame.

    Parameters:
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.
    Returns:
        pd.DataFrame: A merged DataFrame containing scanner, measurements, and protocol parameters data.
    """

    # Merge the scanner, examinations, measurements + parameters DataFrames into a single DataFrame.
    # The merging of the examinations and measurements + parameters DataFrames is
    # based on the time intervals of the examinations and measurements.
    scan_exams_meas_params_df = CON.execute(
        f"""
    SELECT DISTINCT
        events.EventID,
        scan.*,
        exams.*,
        meas_params.*
    FROM
        events
    INNER JOIN
        scan
    ON
        events.FK_ScannerID = scan.ScannerID_scan
    INNER JOIN
        meas_params
    ON
        events.EventID = meas_params.FK_EventID_meas
    INNER JOIN
        exams
    ON
        meas_params.MeasurementStart_meas BETWEEN exams.ExaminationStart_exam AND exams.ExaminationEnd_exam
        AND meas_params.MeasurementEnd_meas BETWEEN exams.ExaminationStart_exam AND exams.ExaminationEnd_exam
    ORDER BY
        scan.SystemType_scan asc,
        exams.ExaminationID_exam asc,
        meas_params.MeasurementID_meas asc
        
    """
    ).df()

    # Drop the EventID column since it is not needed
    scan_exams_meas_params_df.drop(columns=["EventID"], inplace=True)

    return scan_exams_meas_params_df


def merge_power_mapping(CON: dd.DuckDBPyConnection) -> pd.DataFrame:
    """
    Merge the power data from the powerdata_*.csv files with the
    powermeter_scanner_mapping DataFrame to create merged DataFrame that includes
    the ScannerID, Serial, and all relevant power data. Enabling the linkage
    of power measurements with their corresponding MRI scanners.

    Parameters:
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.

    Returns:
        pd.DataFrame: A merged DataFrame containing ScannerID, Serial, and power data.

    """
    # Merge the power data from the powerdata_*.csv files with the powermeter_scanner_mapping DataFrame
    scan_power_df = CON.execute(
        f"""
        SELECT
            powermeter_scanner_mapping.ScannerID,
            powermeter_scanner_mapping.Serial,
            power.*
        FROM
            power
        INNER JOIN
            powermeter_scanner_mapping
        ON
            power.FK_PowermeterSerial = powermeter_scanner_mapping.PowermeterID
            
        
        """
    ).df()

    return scan_power_df


def enrich_power(power_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the power data by creating new columns for total energy, apparent power,
    active power, and reactive power based on the existing columns. This involves
    summing up the energy and power values across different phases (L1, L2, L3)
    to create unified columns that represent the total values. A date key is
    created from the Time column to allow for daily aggregation of energy data.
    Additionally, map the power boundaries for different scanner modes
    by joining with the powerboundries_scanner DataFrame.

    Parameters:
        power_df (pd.DataFrame): The original power data DataFrame to be enriched.

    Returns:
        pd.DataFrame: The enriched power data DataFrame with new columns for total energy,
                    apparent power, active power, reactive power, and a date key
                    for daily aggregation.
    """
    # Create TotalEnergy_KWh by adding up the different phases of
    # TotalEnergyL1_KWh, TotalEnergyL2_KWh, TotalEnergyL3_KWh
    power_df["TotalEnergy_KWh"] = (
        power_df["TotalEnergyL1_KWh"].fillna(0)
        + power_df["TotalEnergyL2_KWh"].fillna(0)
        + power_df["TotalEnergyL3_KWh"].fillna(0)
    )

    # Create TotalApparentPower_KVA by adding up the different phases of
    # ApparentPowerL1_VA, ApparentPowerL2_VA, ApparentPowerL3_VA and
    # converting from VA to KVA
    power_df["TotalApparentPower_KVA"] = (
        power_df["ApparentPowerL1_VA"].fillna(0)
        + power_df["ApparentPowerL2_VA"].fillna(0)
        + power_df["ApparentPowerL3_VA"].fillna(0)
    ) / 1000

    # Create TotalActivePower_KW by adding up the different phases of
    # ActivePowerL1_W, ActivePowerL2_W, ActivePowerL3_W and converting from W to KW
    power_df["TotalActivePower_KW"] = (
        power_df["ActivePowerL1_W"].fillna(0)
        + power_df["ActivePowerL2_W"].fillna(0)
        + power_df["ActivePowerL3_W"].fillna(0)
        # Convert from W to KW
    ) / 1000

    # Create TotalReactivePower_KVAR by taking the square root of the difference between the
    # square of TotalApparentPower_KVA and the square of TotalActivePower_KW
    power_df["TotalReactivePower_KVAR"] = (
        power_df["TotalApparentPower_KVA"] ** 2 - power_df["TotalActivePower_KW"] ** 2
    ).pow(0.5)

    # Create a date key from the Time column to allow for daily aggregation of
    # energy data
    power_df["Date"] = power_df["Time"].dt.date

    # Register the powerdata as a temp_power table, since it is needed for the
    # following SQL query
    CON.register("temp_power", power_df)

    # Merge the temp_power table with the powerboundries_scanner table to map
    # the power boundaries for different scanner modes
    power_boundries_df = CON.execute(
        f"""
        SELECT
            temp_power.*,
            powerboundries_scanner.EcoPowerModeBoundary_kW AS EcoPowerModeBoundary_kW,
            powerboundries_scanner.IdleBoundary_kW AS IdleBoundary_kW
        FROM
            temp_power AS temp_power
        Inner JOIN
            powerboundries_scanner
        ON
            temp_power.Serial = powerboundries_scanner.Serial
        ORDER BY
            temp_power.ScannerID asc,
            temp_power.Serial asc,
            temp_power.Time asc
        """
    ).df()

    return power_boundries_df


def agg_power_boundries(
    power_df: pd.DataFrame, agg_cols_energy: list, agg_cols_power: list
) -> pd.DataFrame:
    """
    Some scanner have multiple powermeters and thus multiple measurements
    for energy and power. Thus, aggregate the energy and power data on
    a scanner level by adding up the energy columns and averaging the
    power columns of the different powermeters.


    Parameters:
        power_df (pd.DataFrame): The enriched power data DataFrame containing energy and power columns.
        agg_cols_energy (list): A list of column names corresponding to energy measurements that should be summed.
        agg_cols_power (list): A list of column names corresponding to power measurements that should be

    Returns:
        pd.DataFrame: An aggregated DataFrame with energy columns summed and power columns averaged on a scanner level.
    """
    # Aggregate the power data for each scanner, Date and Time by summing the energy
    # values and averaging the power values.
    agg_power_boundries_df = power_df.groupby(
        ["Serial_energy", "Date_energy", "Time_energy"], as_index=False
    ).agg(
        {
            **{col: "sum" for col in agg_cols_energy},
            **{col: "mean" for col in agg_cols_power},
        }
    )

    return agg_power_boundries_df


def agg_daily_energy(power_boundries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the daily energy consumption for idle mode, eco power mode,
    and scanning mode using pandas.

    Parameters:
        power_boundries_df (pd.DataFrame): The enriched power data DataFrame with mapped power
                                           boundaries for different scanner modes.

    Returns:
        pd.DataFrame: A compact DataFrame containing the daily energy
                      for each FK_PowermeterSerial, Date, and Serial.
    """

    # Create all possible combinations of Date_energy and Serial_energy
    all_combinations = (
        power_boundries_df[["Date_energy", "Serial_energy"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Calculate DailyEcoPowerModeEnergy_KWh. NaN values in EcoPowerModeBoundary_kW_energy
    # will lead to NaN values in DailyEcoPowerModeEnergy_KWh_energy,
    # which will be filled with 0 later since they indicate that there is no
    # eco power mode energy for these rows.
    agg_daily_eco = (
        power_boundries_df[
            (power_boundries_df["LeanExamination_energy"] == "idle")
            & (
                power_boundries_df["TotalActivePower_KW_energy"]
                < power_boundries_df["EcoPowerModeBoundary_kW_energy"]
            )
        ]
        .groupby(["Date_energy", "Serial_energy"], as_index=False)
        .agg(DailyEcoPowerModeEnergy_KWh_energy=("TotalEnergy_KWh_energy", "sum"))
    )

    # Merge with all combinations to ensure all groups are present
    agg_daily_eco = all_combinations.merge(
        agg_daily_eco, on=["Date_energy", "Serial_energy"], how="left"
    )

    # Fill the NaN values in DailyEcoPowerModeEnergy_KWh_energy with 0,
    # since NaN values indicate that there is no eco power mode energy for these rows
    agg_daily_eco["DailyEcoPowerModeEnergy_KWh_energy"] = agg_daily_eco[
        "DailyEcoPowerModeEnergy_KWh_energy"
    ].fillna(0)

    # Calculate the daily idle and eco energy by summing up all idle energy
    # and eco power mode energy
    agg_daily_idle = (
        power_boundries_df[(power_boundries_df["LeanExamination_energy"] == "idle")]
        .groupby(["Date_energy", "Serial_energy"], as_index=False)
        .agg(DailyIdleEcoEnergy_KWh_energy=("TotalEnergy_KWh_energy", "sum"))
    )

    # Calculate DailyIdleEnergy_KWh by subtracting the DailyEcoPowerModeEnergy_KWh
    # from the DailyIdleEcoEnergy_KWh
    agg_daily_idle["DailyIdleEnergy_KWh_energy"] = (
        agg_daily_idle["DailyIdleEcoEnergy_KWh_energy"]
        - agg_daily_eco["DailyEcoPowerModeEnergy_KWh_energy"]
    )

    # Calculate DailyTotalEnergy_KWh
    agg_daily_total = power_boundries_df.groupby(
        ["Date_energy", "Serial_energy"], as_index=False
    ).agg(DailyTotalEnergy_KWh_energy=("TotalEnergy_KWh_energy", "sum"))

    # Merge the aggregated DataFrames
    agg_daily_combined = (
        agg_daily_total.merge(
            agg_daily_eco, on=["Date_energy", "Serial_energy"], how="inner"
        )
        .merge(agg_daily_idle, on=["Date_energy", "Serial_energy"], how="inner")
        .fillna(0)
    )

    # Calculate DailyScanEnergy_KWh
    agg_daily_combined["DailyScanEnergy_KWh_energy"] = (
        agg_daily_combined["DailyTotalEnergy_KWh_energy"]
        - agg_daily_combined["DailyEcoPowerModeEnergy_KWh_energy"]
        - agg_daily_combined["DailyIdleEnergy_KWh_energy"]
    )

    return agg_daily_combined


def get_meas_power(CON: dd.DuckDBPyConnection) -> pd.DataFrame:
    """
    Merge the measurements DataFrame with the aggregated power data
    for each measurement period by joining the scan_exams_meas_params table with
    the agg_power_boundries table based on the scanner serial and the examination interval.
    The query calculates the total energy and average power for each measurement
    period.

    Parameters:
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.

    Returns:
        pd.DataFrame: A DataFrame containing the scanner measurements parameters
                        along with the aggregated energy and power data for each
                        measurement period.
    """

    # Merge the measurements parameters DataFrame with the aggregated power data
    # for each measurement period
    meas_power_df = CON.execute(
        f"""
    WITH aggregated_data AS (
        SELECT
            sem_params.Serial_scan,
            sem_params.MeasurementID_meas,
            SUM(agg_power_boundries.TotalEnergy_KWh_energy) AS TotalEnergy_KWh_meas,
            AVG(agg_power_boundries.TotalActivePower_KW_energy) AS TotalActivePower_KW_meas,
            AVG(agg_power_boundries.TotalApparentPower_KVA_energy) AS TotalApparentPower_KVA_meas,
            AVG(agg_power_boundries.TotalReactivePower_KVAR_energy) AS TotalReactivePower_KVAR_meas
        FROM 
            scan_exams_meas_params AS sem_params
        INNER JOIN 
            agg_power_boundries AS agg_power_boundries
        ON 
            agg_power_boundries.Serial_energy = sem_params.Serial_scan AND
            agg_power_boundries.Time_energy BETWEEN sem_params.MeasurementStart_meas AND sem_params.MeasurementEnd_meas
        GROUP BY 
            sem_params.MeasurementID_meas, 
            sem_params.Serial_scan
    )
    -- Attach the aggregated results back to each measurement keeps rows without power data
    SELECT
        sem_params.Serial_scan,
        sem_params.MeasurementID_meas,   
        aggregated_data.TotalEnergy_KWh_meas AS TotalEnergy_KWh_meas,
        aggregated_data.TotalActivePower_KW_meas AS TotalActivePower_KW_meas,
        aggregated_data.TotalApparentPower_KVA_meas AS TotalApparentPower_KVA_meas,
        aggregated_data.TotalReactivePower_KVAR_meas AS TotalReactivePower_KVAR_meas
    FROM 
        scan_exams_meas_params AS sem_params
    LEFT JOIN 
        aggregated_data AS aggregated_data
    ON 
        aggregated_data.Serial_scan = sem_params.Serial_scan AND
        aggregated_data.MeasurementID_meas = sem_params.MeasurementID_meas
    WHERE
        aggregated_data.TotalEnergy_KWh_meas IS NOT NULL
    ORDER BY
        sem_params.Serial_scan asc,
        sem_params.MeasurementID_meas asc
    """
    ).df()

    return meas_power_df


def get_exams_power(CON: dd.DuckDBPyConnection) -> pd.DataFrame:
    """
    Merge the examinations DataFrame with the aggregated power data
    for each examination period by joining the scan_exams_meas_params table with
    the agg_power_boundries table based on the scanner serial and the examination interval.
    The query calculates the total energy and average power for each examination
    period.

    Parameters:
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.

    Returns:
        pd.DataFrame: A DataFrame containing the scanner examinations parameters
                        along with the aggregated energy and power data for each
                        examination period.
    """

    # Get unique combinations of ScannerID, ExaminationID, and PowermeterID to
    # avoid duplicate rows in the result due to multiple measurements per examination
    unique_semp_df = CON.execute(
        f"""
        SELECT 
            *
        FROM 
            scan_exams_meas_params AS sem_params
        QUALIFY 
            ROW_NUMBER() 
        OVER 
            (PARTITION BY ScannerID_scan, ExaminationID_exam) = 1
        """
    ).df()

    # Register the unique DataFrame as a DuckDB table
    CON.register(
        "unique_sem_params",
        unique_semp_df,
    )

    # Merge the examinations DataFrame with the aggregated power data for each examination period
    exams_power_df = CON.execute(
        f"""
    WITH aggregated_data AS (
        SELECT
            unique_sem_params.Serial_scan,
            unique_sem_params.ExaminationID_exam,
            SUM(agg_power_boundries.TotalEnergy_KWh_energy) AS TotalEnergy_KWh_exam,
            AVG(agg_power_boundries.TotalActivePower_KW_energy) AS TotalActivePower_KW_exam,
            AVG(agg_power_boundries.TotalApparentPower_KVA_energy) AS TotalApparentPower_KVA_exam,
            AVG(agg_power_boundries.TotalReactivePower_KVAR_energy) AS TotalReactivePower_KVAR_exam
        FROM 
            unique_sem_params AS unique_sem_params
        INNER JOIN 
            agg_power_boundries AS agg_power_boundries
        ON 
            agg_power_boundries.Serial_energy = unique_sem_params.Serial_scan AND
            agg_power_boundries.Time_energy BETWEEN unique_sem_params.ExaminationStart_exam AND unique_sem_params.ExaminationEnd_exam
        GROUP BY 
            unique_sem_params.ExaminationID_exam, 
            unique_sem_params.Serial_scan
    )
    -- Attach the aggregated results back to each examination, keeps rows without power data
    SELECT
        unique_sem_params.Serial_scan,
        unique_sem_params.ExaminationID_exam,   
        aggregated_data.TotalEnergy_KWh_exam AS TotalEnergy_KWh_exam,
        aggregated_data.TotalActivePower_KW_exam AS TotalActivePower_KW_exam,
        aggregated_data.TotalApparentPower_KVA_exam AS TotalApparentPower_KVA_exam,
        aggregated_data.TotalReactivePower_KVAR_exam AS TotalReactivePower_KVAR_exam
    FROM 
        unique_sem_params AS unique_sem_params
    LEFT JOIN 
        aggregated_data AS aggregated_data
    ON 
        aggregated_data.Serial_scan = unique_sem_params.Serial_scan AND
        aggregated_data.ExaminationID_exam = unique_sem_params.ExaminationID_exam
    WHERE
        aggregated_data.TotalEnergy_KWh_exam IS NOT NULL
    ORDER BY
        unique_sem_params.Serial_scan asc,
        unique_sem_params.ExaminationID_exam asc
    """
    ).df()

    return exams_power_df


def merge_meas_exams_power(CON: dd.DuckDBPyConnection) -> pd.DataFrame:
    """
    Merge the measurements parameters DataFrame with the examinations DataFrame and the
    aggregated power data for both measurements and examinations.

    Parameters:
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.

    Returns:
        pd.DataFrame: A DataFrame containing the scanner measurements parameters
                      along with the corresponding aggregated power data for both
                      measurements and examinations.
    """

    # Merge the measurements parameters DataFrame with the examinations DataFrame
    # and the aggregated power data for both measurements and examinations
    exams_meas_params_power_df = CON.execute(
        f"""
        SELECT
            sem_params.*,
            exams_power.TotalEnergy_KWh_exam,
            exams_power.TotalActivePower_KW_exam,
            exams_power.TotalApparentPower_KVA_exam,
            exams_power.TotalReactivePower_KVAR_exam,
            meas_power.TotalEnergy_KWh_meas,
            meas_power.TotalActivePower_KW_meas,
            meas_power.TotalApparentPower_KVA_meas,
            meas_power.TotalReactivePower_KVAR_meas
        FROM
            scan_exams_meas_params AS sem_params
        INNER JOIN 
            meas_power
        ON 
            sem_params.Serial_scan = meas_power.Serial_scan AND
            sem_params.MeasurementID_meas = meas_power.MeasurementID_meas
        INNER JOIN 
            exams_power
        ON 
            sem_params.Serial_scan = exams_power.Serial_scan AND
            sem_params.ExaminationID_exam = exams_power.ExaminationID_exam
        ORDER BY
            sem_params.Serial_scan asc,
            sem_params.MeasurementID_meas asc,
            sem_params.ExaminationID_exam asc
        """
    ).df()

    return exams_meas_params_power_df


def merge_meas_exams_power_daily(CON: dd.DuckDBPyConnection) -> pd.DataFrame:
    """
    Merge the daily aggregated power data with the measurements parameters and
    examinations DataFrame.

    Parameters:
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.

    Returns:
        pd.DataFrame: A DataFrame containing the scanner measurements parameters,
                        examinations data, and the corresponding daily aggregated power data.
    """

    # Merge the daily aggregated power data with the measurements parameters
    # and examinations DataFrame
    exams_meas_params_power_daily_df = CON.execute(
        f"""
    SELECT
        em_parmas_power.*,
        agg_power_daily.DailyEcoPowerModeEnergy_KWh_energy,
        agg_power_daily.DailyIdleEnergy_KWh_energy,
        agg_power_daily.DailyScanEnergy_KWh_energy,
        agg_power_daily.DailyTotalEnergy_KWh_energy
    FROM
        exams_meas_params_power as em_parmas_power
    LEFT JOIN 
        agg_power_daily as agg_power_daily
    ON 
        em_parmas_power.Serial_scan = agg_power_daily.Serial_energy AND
        em_parmas_power.Date_energy = agg_power_daily.Date_energy
    ORDER BY
        em_parmas_power.ScannerID_scan asc,
        em_parmas_power.MeasurementID_meas asc,
        em_parmas_power.ExaminationID_exam asc
    """
    ).df()

    return exams_meas_params_power_daily_df


def reorder_columns_by_suffix_priority(df, suffix_priority):
    """
    Reorders the columns of a DataFrame based on a given suffix priority and
    then alphabetically.

    Parameters:
        df (pd.DataFrame): The DataFrame whose columns need to be reordered.
        suffix_priority (dict): A dictionary defining the priority of suffixes.
                                Keys are suffixes, values are priority (lower is higher priority).

    Returns:
        pd.DataFrame: A DataFrame with reordered columns.
    """

    # Sort columns based on the suffix priority and then alphabetically
    cols_sorted = sorted(
        df.columns,
        # Determine the priority of each column based on its suffix and then sort alphabetically
        key=lambda x: (
            suffix_priority.get(
                next(
                    (suffix for suffix in suffix_priority if x.endswith(suffix)), None
                ),
                float("inf"),
            ),
            x,
        ),
    )
    return df[cols_sorted]


# ------------------------------------------------------------------------------
# 🚀 Wrapper Functions
# ------------------------------------------------------------------------------
#
# To streamline the data preparation and merging process for MRI data, the
# following wrapper functions are implemented:
#
# 1. Prepare the Merging 🛠️
#    - Prepares data for merging by:
#        1. Retrieving the data directory path.
#        2. Identifying MRI folders in the data directory.
#        3. Validating MRI folders for required files and data.
#        4. Creating a mapping for scanner data.
#        5. Retrieving power boundaries for scanners.
#        6. Retrieving parameters description and creating a renaming dictionary.
#    - Function: wrapper_prepare_merging
#
# 2. Get and Prepare Data 📂
#    - Retrieves all necessary data from:
#        - Scanner.csv
#        - Measurements.csv
#        - ProtocolParameters.csv
#        - Examinations.csv
#        - Events.csv
#        - powerdata_*.csv
#    - Renames ProtocolParameters for consistency and clarity.
#    - Function: wrapper_get_prepare_data
#
# 3. Merge Scanner, Examinations, Measurements, and Parameters 🔗
#    - Manages the merging process by:
#        - Combining Measurements with renamed Parameters to enrich measurement data.
#        - Merging the result with Scanner and Examinations to create a unified dataset.
#    - Function: wrapper_merge_scan_exams_meas_params_power
#
# 4. Merge and Enrich Power Data ⚡
#    - Merges power data with scanner information using powermeter_scanner_mapping.
#    - Enriches power data by:
#        - Creating new columns for total energy, apparent power, active power,
#          and reactive power.
#        - Aggregating power data at the scanner level (sum energy, average power
#          for multiple powermeters).
#        - Calculating daily energy consumption for Idle, Eco Power, and Scanning modes.
#    - Function: wrapper_merge_enrich_power
#
# 5. Merge Measurements Parameters with Examinations and Power Data 🔗
#    - Merges Measurements Parameters with Examinations and aggregated power data
#      for both measurements and examinations.
#    - Function: wrapper_merge_agg_power
#
# 6. Merge All DataFrames 📊
#    - Combines Examinations, Measurements, Parameters, and Power DataFrames with
#      daily aggregated power data.
#    - Function: wrapper_merge_agg_daily_power


def wrapper_prepare_merging(
    data_dir: str,
    CON: dd.DuckDBPyConnection,
) -> tuple[str, dd.DuckDBPyConnection, list, pd.DataFrame, dict]:
    """
    Wrapper function to prepare the data for merging by performing the following steps:
    1. Get the data directory path.
    2. Get the MRI folders in the data directory.
    3. Check the MRI folders for the required files and data.
    4. Create the mapping for the scanner data.
    5. Get the power boundaries for the scanners.
    6. Get the parameters description and create a renaming dictionary.

    Parameters:
        data_dir (str): The path to the data directory.
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.

    Returns:
        tuple: A tuple containing the following elements:
            - data_dir (str): The path to the data directory.
            - CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.
            - str: The path to the data directory.
            - list: A list of valid MRI folders that contain the required data.
            - pd.DataFrame: A DataFrame containing the parameters description.
            - dict: A dictionary for renaming columns based on the parameters description.

    """
    # Get the data directory path
    data_dir = get_data_dir(data_dir)

    # Get the MRI folders in the data directory and verify them
    mri_folders = get_mri_folders(data_dir)
    valid_mri_folders = check_mri_folder(data_dir, CON, mri_folders)

    # Create the mapping for the scanner and power data
    CON = create_mapping_scanner_mapping(data_dir, CON, mri_folders)
    # Get the power boundaries for the scanners
    CON = get_powerboundries_scanner(data_dir, CON)

    # Get all of the parameters description and create a renaming dictionary
    param_renaming_df, param_renaming_dict = get_parameters_description(data_dir, CON)

    return data_dir, CON, valid_mri_folders, param_renaming_df, param_renaming_dict


def wrapper_get_prepare_data(
    CON: dd.DuckDBPyConnection, date_folder_path: str, param_renaming_dict: dict
) -> dd.DuckDBPyConnection:
    """
    A wrapper function that orchestrates the data loading, preparation, and registration
    process. It loads the data from the specified CSV files, unifies energy measurements,
    renames parameters, and registers the prepared DataFrames as DuckDB tables for further analysis.

    Parameters:
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database.
        date_folder_path (str): The path to the folder containing the CSV files for a specific date.
        param_renaming_dict (dict): A dictionary for renaming columns, mapping ProtocolParameterName to Explanation.

    Returns:
        dd.DuckDBPyConnection: The DuckDB connection object with the prepared DataFrames registered as tables.
    """

    # Load in the data from the CSV files into DataFrames
    scan_df = get_data(date_folder_path, "Scanner.csv")
    meas_df = get_data(date_folder_path, "Measurements.csv")
    params_df = get_data(date_folder_path, "ProtocolParameters.csv")
    exams_df = get_data(date_folder_path, "Examinations.csv")
    events_df = get_data(date_folder_path, "Events.csv")
    power_df = get_data(date_folder_path, "powerdata_*.csv")

    # Unify the energy measurements in the measurements and examinations DataFrame
    meas_df, exams_df = unify_energy_meas_exams(meas_df, exams_df)

    # Add a suffix to the scan DataFrame
    scan_df = scan_df.add_suffix("_scan")
    # Add a suffix to the measurements DataFrame
    meas_df = meas_df.add_suffix("_meas")
    # Add a suffix to the examinations DataFrame
    exams_df = exams_df.add_suffix("_exam")

    # Register the DataFrames as DuckDB tables for further use in SQL queries
    CON.register("scan", scan_df)
    CON.register("meas", meas_df)
    CON.register("params", params_df)
    CON.register("exams", exams_df)
    CON.register("events", events_df)
    CON.register("power", power_df)

    # Rename the columns of the parameters DataFrame using the renaming dictionary
    renamed_params_df = rename_parameter(param_renaming_dict, params_df)

    # Add a suffix to the renamed parameters DataFrame
    renamed_params_df = renamed_params_df.add_suffix("_param")
    CON.register("renamed_params", renamed_params_df)

    return CON


def wrapper_merge_scan_exams_meas_params_power(
    CON: dd.DuckDBPyConnection,
) -> dd.DuckDBPyConnection:
    """
    A wrapper function that orchestrates the merging of the measurements
    DataFrame with the renamed parameters DataFrame, and then merges the resulting
    DataFrame with the scanner and examinations DataFrames.

    Parameters:
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database
                                    with the necessary DataFrames registered as tables.

    Returns:
        dd.DuckDBPyConnection: The DuckDB connection object with the merged DataFrames registered as tables.
    """

    # Merge the measurements DataFrame with the renamed parameters
    # DataFrame using an inner join on MeasurementID and FK_MeasurementID
    meas_params_df = merge_meas_params(CON)
    CON.register("meas_params", meas_params_df)

    # Merge the scanner, examinations, measurements, parameters DataFrames
    scan_exams_meas_params_df = merge_scan_exams_meas_params(CON)
    CON.register("scan_exams_meas_params", scan_exams_meas_params_df)

    return CON


def wrapper_merge_enrich_power(
    CON: dd.DuckDBPyConnection, agg_cols_energy: list, agg_cols_power: list
) -> dd.DuckDBPyConnection:
    """
    A wrapper function that orchestrates the merging of the power data with the
    corrsponding scanner information, by using the powermeter_scanner_mapping DataFrame,
    and then enriches the power data by creating new columns for total energy,
    apparent power, active power, and reactive power. It aggregates the
    power data on a scanner level by adding up the energy columns and
    averaging per powermeter, since scanner could have multiple powermeters. Finally,
    it calculates the daily energy consumption for idle mode, eco power mode,
    and scanning mode.

    Parameters:
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database
                                    with the necessary DataFrames registered as tables.
        agg_cols_energy (list): A list of column names corresponding to energy
                                measurements that should be summed during aggregation.
        agg_cols_power (list): A list of column names corresponding to power
                                measurements that should be averaged during aggregation.

    Returns:
        dd.DuckDBPyConnection: The DuckDB connection object with the enriched
                                and aggregated power DataFrames registered as tables.
    """

    # Merge the power data with the powermeter_scanner_mapping DataFrame
    scan_power_df = merge_power_mapping(CON)
    CON.register("scan_power_df", scan_power_df)

    # Enrich the power data
    power_boundries_df = enrich_power(scan_power_df)
    CON.register("power_boundries", power_boundries_df)

    # Add an suffix to the power_boundries_df to prepare for the following merges
    power_boundries_df = power_boundries_df.add_suffix("_energy")

    # Aggregate the powermeters to get a unified energy and power df
    agg_power_boundries_df = agg_power_boundries(
        power_boundries_df, agg_cols_energy, agg_cols_power
    )
    CON.register("agg_power_boundries", agg_power_boundries_df)

    # Calculate the daily energy consumption in the different scanner modes (EcoPowerMode, Idle, Scanning)
    agg_power_daily_df = agg_daily_energy(power_boundries_df)
    CON.register("agg_power_daily", agg_power_daily_df)

    return CON


def wrapper_merge_agg_power(CON: dd.DuckDBPyConnection) -> dd.DuckDBPyConnection:
    """
    A wrapper function that orchestrates the merging of the measurements
    parameters DataFrame with the examinations DataFrame and the aggregated power
    data for both measurements and examinations DataFrame.


    Parameters:
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database
                                    with the necessary DataFrames registered as tables.

    Returns:
        dd.DuckDBPyConnection: The DuckDB connection object with the merged DataFrame
                                containing the scanner measurements parameters, examinations data,
                                and the corresponding power data registered as a table.
    """

    # Merge the measurements and the power data
    meas_power_df = get_meas_power(CON)
    CON.register("meas_power", meas_power_df)

    # Merge the examinations and the power data
    exams_power_df = get_exams_power(CON)
    CON.register("exams_power", exams_power_df)

    # Merge the measurements, examinations and power data
    exams_meas_params_power_df = merge_meas_exams_power(CON)

    # Create a Date column
    exams_meas_params_power_df["Date_energy"] = exams_meas_params_power_df[
        "MeasurementStart_meas"
    ].dt.date
    CON.register("exams_meas_params_power", exams_meas_params_power_df)

    return CON


def wrapper_merge_agg_daily_power(
    CON: dd.DuckDBPyConnection,
) -> Tuple[dd.DuckDBPyConnection, pd.DataFrame]:
    """
    A wrapper function that orchestrates the merging of the examinations,
    measurements, parameters, power DataFrame with the daily aggregated power data.

    Parameters:
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database
                                    with the necessary DataFrames registered as tables.
    Returns:
        Tuple[dd.DuckDBPyConnection, pd.DataFrame]: The DuckDB connection object with the final
                                                    merged DataFrame containing the scanner measurements
                                                    parameters, examinations, power data,
                                                    and the corresponding daily aggregated power data
                                                    registered as a table, and the final merged DataFrame
                                                    itself for further use in analysis.

    """

    # Merge the daily aggregated power data with the measurements, examinations and parameters data
    exams_meas_params_power_daily_df = merge_meas_exams_power_daily(CON)
    CON.register("exams_meas_params_power_daily", exams_meas_params_power_daily_df)

    return CON, exams_meas_params_power_daily_df


# ------------------------------------------------------------------------------
# 🚀 Main Loop: Processing MRI Data
# ------------------------------------------------------------------------------
#
# - The main loop is responsible for processing MRI data.
# - It calls all predefined wrapper functions for data preparation, merging,
#   enrichment, and aggregation.
#
# Steps executed:
# 1. 📂 Iterate through MRI folders and their respective date folders.
# 2. ✅ Check for the presence of all expected files in each folder.
# 3. 📊 Load the data from the files into DataFrames.
# 4. 🔗 Merge and enrich the data to ensure completeness and consistency.
# 5. 🧩 Concatenate the results into a single, comprehensive DataFrame.
#
# - The loop is robust to missing data and errors, logging any issues for later
#   review.
# - Designed to efficiently process large datasets and can resume from previously
#   processed data if provided.
#
# ---
#
# > ⚠️ **Important:**
# > Before running this main loop, you must first:
# >  - Run the **Idle_EPM_threshhold.ipynb** in the **Determine power threshold_opt folder**
# >    to determine the power thresholds for each scanner mode.
# > - Generate and place the `powerboundries_scanner_mapping.csv` file in the data directory.
# >   These are required for correct mode inference and downstream processing.
#
# ---
#
# 📝 Input:
# - data_dir (str): The directory containing the MRI data folders.
# - CON (dd.DuckDBPyConnection): DuckDB connection object for querying and processing
#    data.
# - preprocessed_df (pd.DataFrame, optional): Pre-existing DataFrame with preprocessed
#   data, if available. Defaults to None.
#
#   > Note: If preprocessed_df is provided, it will be used as the starting point
#           for further processing, and additional data will be merged into it.
#
# ---
# 🎯 Output:
# - mri_df: Complete merged and enriched DataFrame containing all processed MRI data.
# - missing_data_df: Logs of any missing data encountered during processing.
# - problems_processing_df: Records of any issues or problems encountered while
#   processing each MRI and date folder.


def process_mri_data(
    data_dir: str,
    CON: dd.DuckDBPyConnection,
    preprocessed_df: pd.DataFrame = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process the MRI data by iterating over the MRI folders and their date folders,
    checking for the presence of all expected files, loading the data into DataFrames,
    merging and enriching the data, and finally concatenating the results into a single
    DataFrame. Additionaly, if a preprocessed DataFrame is provided, the function will
    continue processing from the last date in this DataFrame for each MRI.

    The function will return the complete merged and enriched DataFrame
    containing the processed MRI data. Two DataFrames logging any missing data
    and problems encountered during the processing of each MRI and date folder.

    Parameters:
        data_dir (str): Path to the data directory.
        CON (dd.DuckDBPyConnection): A connection object to the DuckDB database
        preprocessed_df (pd.DataFrame, optional): A preprocessed DataFrame containing previously processed MRI data.
                                                 If provided, the function will continue processing from the last date in
                                                 this DataFrame.

    Returns:
            pd.DataFrame: A DataFrame containing the processed MRI data with merged and enriched information.
            pd.DataFrame: A DataFrame logging the missing data for each MRI and date folder.
            pd.DataFrame: A DataFrame logging any problems encountered during the processing of each MRI and date folder.
    """

    # Define the columns to aggregate for energy, which will be summed
    agg_cols_energy = [
        "TotalEnergy_KWh_energy",
    ]
    # Define the columns to aggregate for power, which will be averaged
    agg_cols_power = [
        "TotalApparentPower_KVA_energy",
        "TotalActivePower_KW_energy",
        "TotalReactivePower_KVAR_energy",
    ]

    # Wrapper function to prepare the data for merging
    data_dir, CON, valid_mri_folders, param_renaming_df, param_renaming_dict = (
        wrapper_prepare_merging(data_dir, CON)
    )

    # If a preprocessed DataFrame is provided, use it as the starting point for
    # the MRI data processing.
    if preprocessed_df is not None:
        # Use the preprocessed DataFrame as the starting point for the MRI data processing
        mri_df = preprocessed_df

        # For every Machine get the last date for which data is available in the preprocessed DataFrame
        meta_data = mri_df.groupby("Serial_scan")["Date_energy"].max().reset_index()

        # Create the folder names by concatenating "ukt_" with the Serial_scan
        meta_data["MRI_Folder"] = "ukt_" + meta_data["Serial_scan"].astype(str)
    else:
        # Initialize an empty DataFrame to store the concatenated results from all MRI folders
        mri_df = pd.DataFrame()

    # Initialize DataFrames to log missing data and problems during processing
    missing_data_df = pd.DataFrame(columns=["MRI_Folder", "Date_Folder"])
    problems_processing_df = pd.DataFrame(
        columns=["MRI_Folder", "Date_Folder", "Issue"]
    )

    # Iterate over MRI folders and their date folders
    for mri_folder in tqdm(valid_mri_folders, desc="Processing MRI: {mri_folder}"):
        # Construct the path to the MRI folder
        mri_folder_path = os.path.join(data_dir, mri_folder)

        # If the meta_data["MRI_Folder"] is in the current mri_folder,
        # continue after the last date in the preprocessed DataFrame
        if (
            preprocessed_df is not None
            and (meta_data["MRI_Folder"] == mri_folder).any()
        ):
            last_date = meta_data.loc[
                meta_data["MRI_Folder"] == mri_folder, "Date_energy"
            ].values[0]

            print(
                f"Continuing processing for MRI 🧲: {mri_folder} after last date in preprocessed DataFrame: {last_date}"
            )

            # Adapt the list of date folders to only include those that are after
            # the last date in the preprocessed DataFrame
            date_folders = sorted(os.listdir(mri_folder_path))
            valid_date_folders = [
                date_folder
                for date_folder in date_folders
                if pd.to_datetime(date_folder).date() > pd.to_datetime(last_date).date()
            ]
        else:
            valid_date_folders = os.listdir(mri_folder_path)

        # Iterate over the date folders within the MRI folder
        for date_folder in valid_date_folders:

            # Construct the path to the date folder
            date_folder_path = os.path.join(mri_folder_path, date_folder)

            # Check if all expected files are available in the date folder
            all_files_available = check_data(date_folder_path)

            # If not all expected files are available, skip the current date
            # folder and continue with the next one
            if not (all_files_available):
                print(f"Skipping MRI 🧲: {mri_folder}, Date 📅: {date_folder}")
                # Log the missing data in the missing_data_df DataFrame
                missing_data_df = pd.concat(
                    [
                        missing_data_df,
                        pd.DataFrame(
                            [{"MRI_Folder": mri_folder, "Date_Folder": date_folder}]
                        ),
                    ],
                    ignore_index=True,
                )
                continue

            try:

                # Wrapper function to get and prepare the data for merging ###
                CON = wrapper_get_prepare_data(
                    CON, date_folder_path, param_renaming_dict
                )

                # Wrapper function to merge the scan, examinations, measurements,
                # parameters and power data
                CON = wrapper_merge_scan_exams_meas_params_power(CON)

                # Wrapper function to merge and enrich the power data, and aggregate
                # it on a scanner level
                CON = wrapper_merge_enrich_power(CON, agg_cols_energy, agg_cols_power)

                # Wrapper function to merge the measurements parameters DataFrame
                # with the examinations DataFrame and the aggregated power data
                # for both measurements and examinations
                CON = wrapper_merge_agg_power(CON)

                # Wrapper function to merge the examinations, measurements,
                # parameters, power DataFrame with the daily aggregated power data
                CON, exams_meas_params_power_daily_df = wrapper_merge_agg_daily_power(
                    CON
                )

                # Register the final merged DataFrame for the current MRI and date folder,
                CON.register("daily_df", exams_meas_params_power_daily_df)

                # If mri_df is empty, initialize it with the current daily_df
                if mri_df.empty:
                    mri_df = exams_meas_params_power_daily_df.copy()
                    CON.register("mri_df", mri_df)

                # If mri_df is not empty, merge the daily_df with the existing mri_df
                # using UNION BY NAME to ensure that all columns are included,
                # even if they are not present in both DataFrames
                else:
                    CON.register("previous_mri_df", mri_df)
                    mri_df = CON.execute(
                        """
                        SELECT 
                            * 
                        FROM 
                            previous_mri_df
                        UNION BY NAME
                        SELECT 
                            * 
                        FROM 
                            daily_df
                    """
                    ).df()
                    CON.unregister("previous_mri_df")
                    CON.register("mri_df", mri_df)

                # Print the shape of the merged DataFrame for the current MRI and date folder
                print(
                    f"MRI 🧲: {mri_folder}, Date 📅: {date_folder}, 📊 Merged DataFrame shape: {mri_df.shape}"
                )

            except Exception as e:
                # If an error occurs during processing, log the error and continue with the next date folder
                problems_processing_df = pd.concat(
                    [
                        problems_processing_df,
                        pd.DataFrame(
                            [
                                {
                                    "MRI_Folder": mri_folder,
                                    "Date_Folder": date_folder,
                                    "Issue": str(e),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

                print(
                    f"An error occurred while processing MRI 🧲: {mri_folder}, Date 📅: {date_folder}. Error details: {e}"
                )
                continue

    suffix_priority = {"_scan": 0, "_exam": 1, "_meas": 2, "_param": 3, "_energy": 4}
    mri_df = reorder_columns_by_suffix_priority(mri_df, suffix_priority)

    # Sort by Serial_scan, ExaminationStart_exam, MeasuremntStart_meas to have a logical order of the rows
    mri_df.sort_values(
        by=["Serial_scan", "ExaminationStart_exam", "MeasurementStart_meas"],
        inplace=True,
    )

    # Reset the index of the final DataFrame
    mri_df.reset_index(drop=True, inplace=True)

    # Raise an error if the final DataFrame is empty after processing all MRI and date folders
    if mri_df.empty:
        raise ValueError(
            "❌ The final merged DataFrame is empty after processing all MRI and date folders. Please check the logs for missing data and problems during processing."
        )

    return mri_df, missing_data_df, problems_processing_df


# ------------------------------------------------------------------------------
# 💾 Saving DataFrames
# ------------------------------------------------------------------------------
#
# Save the following outputs in both Pickle and Parquet formats to the specified
# output directory:
#
# 1. 📊 The processed MRI DataFrame.
# 2. 📝 The Missing Data Log.
# 3. ⚠️ The Error Log (problems encountered during processing).


def save_dfs(
    mri_df: pd.DataFrame,
    missing_data_df: pd.DataFrame,
    problems_processing_df: pd.DataFrame,
    output_dir: str,
):
    """
    Save the processed MRI DataFrame, missing data log, and problems processing
    df as pickle and parquet files in the specified output directory.

    Parameters:
        mri_df (pd.DataFrame): The DataFrame containing the processed MRI data.
        missing_data_df (pd.DataFrame): The DataFrame logging the missing data for each MRI and date folder.
        problems_processing_df (pd.DataFrame): The DataFrame logging any problems encountered during the processing of each MRI and date folder.
        output_dir (str): The directory where the CSV files will be saved.
    """
    # Ensure the output directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)

    # Save the processed MRI DataFrame as a pickle file
    mri_df.to_pickle(os.path.join(output_dir, "processed_mri_data.pkl"))
    # Save the processed MRI DataFrame as a parquet file
    mri_df.to_parquet(os.path.join(output_dir, "processed_mri_data.parquet"))

    # Save the missing data log as a pickle file
    missing_data_df.to_pickle(os.path.join(output_dir, "missing_data_log.pkl"))
    # Save the missing data log as a parquet file
    missing_data_df.to_parquet(os.path.join(output_dir, "missing_data_log.parquet"))

    # Save the problems processing log as a pickle file
    problems_processing_df.to_pickle(
        os.path.join(output_dir, "problems_processing_log.pkl")
    )
    # Save the problems processing log as a parquet file
    problems_processing_df.to_parquet(
        os.path.join(output_dir, "problems_processing_log.parquet")
    )

    # Load and verify the processed MRI DataFrame
    mri_df_loaded = pd.read_pickle(os.path.join(output_dir, "processed_mri_data.pkl"))
    print(f"✅ Loaded MRI DataFrame successfully. Shape: {mri_df_loaded.shape}")
    if mri_df.equals(mri_df_loaded):
        print("✔️ The original and loaded MRI DataFrames are identical.")
    else:
        raise ValueError("❌ The original and loaded MRI DataFrames are NOT identical.")

    # Load and verify the Missing Data Log DataFrame
    missing_data_df_loaded = pd.read_pickle(
        os.path.join(output_dir, "missing_data_log.pkl")
    )
    print(
        f"✅ Loaded Missing Data Log successfully. Shape: {missing_data_df_loaded.shape}"
    )
    if missing_data_df.equals(missing_data_df_loaded):
        print("✔️ The original and loaded Missing Data Log DataFrames are identical.")
    else:
        raise ValueError(
            "❌ The original and loaded Missing Data Log DataFrames are NOT identical."
        )

    # Load and verify the Problems Processing Log DataFrame
    problems_processing_df_loaded = pd.read_pickle(
        os.path.join(output_dir, "problems_processing_log.pkl")
    )
    print(
        f"✅ Loaded Problems Processing Log successfully. Shape: {problems_processing_df_loaded.shape}"
    )
    if problems_processing_df.equals(problems_processing_df_loaded):
        print(
            "✔️ The original and loaded Problems Processing Log DataFrames are identical."
        )
    else:
        raise ValueError(
            "❌ The original and loaded Problems Processing Log DataFrames are NOT identical."
        )


# Set up the DuckDB connection, which has to be defined at the global level
CON = set_up_duckdb(CWD)


def main():
    """
    Main function to execute the MRI data processing pipeline. This wrapper is
    necessary since else during import of this module the functions would
    be preintialized with the defined global variables, which would cause issues
    during testing.
    """

    # Construct the path to the data directory
    # data_dir = os.path.normpath(
    #     os.path.join(
    #        CWD, "..", "..", "Data storage Siemens", "Pipeline-V2", "Test data"
    #    )
    # )

    # Construct the path to the data directory
    data_dir = os.path.normpath(
        os.path.join(
            CWD, "..", "..",  "Data storage Siemens", "Pipeline-V2", "MRI data"
        )
    )

    mri_df, missing_data_df, problems_processing_df = process_mri_data(
        data_dir=data_dir, CON=CON, preprocessed_df=None
    )

    output_dir = os.path.normpath(
        os.path.join(
            CWD, "..", "..", "saved_datasets", "Pipeline-V2", "preprocessing"
        )
    )
    save_dfs(mri_df, missing_data_df, problems_processing_df, output_dir)

    # Close the DuckDB connection after use
    CON.close()


# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()
