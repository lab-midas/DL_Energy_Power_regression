### 🛠️ General Utils ###########################################################
# These are general utility functions and configurations that are used across
# the entire pipeline and defined in a separate file to facilitate reuse and
# maintainability.

### 📦 Imports #################################################################
import os
import sys

import pandas as pd
import polars as pl
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from great_tables import GT

import itertools

from typing import Tuple, List, Dict

import warnings

# tqdm
try:
    # Try to detect if running in a notebook
    from IPython import get_ipython

    # For the python notebooks use tqdm.notebooks
    if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
        from tqdm.notebook import tqdm
    # For the python scripts use tqdm
    else:
        from tqdm import tqdm
except Exception:
    from tqdm import tqdm

import json

# Get the current working directory
CWD = os.getcwd()

### 🎨 Color List ##############################################################
# Define a global list of colors
COLOR_LIST = [
    "red",
    "blue",
    "green",
    "orange",
    "magenta",
    "brown",
    "grey",
]

### 🗂️ Body Region Grouping Configuration #######################################
# - A dictionary to group specific body regions into broader categories for
#   analysis and visualization.

BODYREGION_GROUPING = {
    # Neuroradiology
    "BRAIN": "Brain",
    # Head & Neck
    "HEAD": "Head & Neck",
    "FACE": "Head & Neck",
    "ORBIT": "Head & Neck",
    "NECK": "Head & Neck",
    "HEADNECK": "Head & Neck",
    "PHARYNX": "Head & Neck",
    # Musculoskeletal
    "TMJ": "Musculoskeletal",
    "KNEE": "Musculoskeletal",
    "ANKLE": "Musculoskeletal",
    "LEG": "Musculoskeletal",
    "HIP": "Musculoskeletal",
    "THIGH": "Musculoskeletal",
    "SHOULDER": "Musculoskeletal",
    "ARM": "Musculoskeletal",
    "HAND": "Musculoskeletal",
    "ELBOW": "Musculoskeletal",
    "EXTREMITY": "Musculoskeletal",
    "FOOT": "Musculoskeletal",
    "FEMUR": "Musculoskeletal",
    "HUMERUS": "Musculoskeletal",
    "CLAVICLE": "Musculoskeletal",
    "WRIST": "Musculoskeletal",
    "CALF": "Musculoskeletal",
    "FINGER": "Musculoskeletal",
    "SPINE": "Musculoskeletal",
    "CSPINE": "Musculoskeletal",
    "TSPINE": "Musculoskeletal",
    "LSPINE": "Musculoskeletal",
    "SSPINE": "Musculoskeletal",
    "TLSPINE": "Musculoskeletal",
    "LSSPINE": "Musculoskeletal",
    "ILIOSACRALJOINT": "Musculoskeletal",
    "ILEOSACRALJOINT": "Musculoskeletal",
    "ILEOSACRAL JOINT": "Musculoskeletal",
    # Thorax
    "CHEST": "Thorax",
    "LUNG": "Thorax",
    "BRONCHUS": "Thorax",
    "STERNUM": "Thorax",
    "NECKCHEST": "Thorax",
    # Breast
    "BREAST": "Breast",
    # Abdomen/Pelvis
    "ABDOMEN": "Abdomen/Pelvis",
    "PELVIS": "Abdomen/Pelvis",
    "ABDOMENPELVIS": "Abdomen/Pelvis",
    "CHESTABDPELVIS": "Abdomen/Pelvis",
    "RECTUM": "Abdomen/Pelvis",
    "COLON": "Abdomen/Pelvis",
    "LIVER": "Abdomen/Pelvis",
    "PANCREAS": "Abdomen/Pelvis",
    "LYMPH": "Abdomen/Pelvis",
    # Urogenital
    "UTERUS": "Urogenital",
    "CERVIX": "Urogenital",
    "VAGINA": "Urogenital",
    "VULVA": "Urogenital",
    "PROSTATE": "Urogenital",
    "SCROTUM": "Urogenital",
    "PENIS": "Urogenital",
    "URETER": "Urogenital",
    "KIDNEY": "Urogenital",
    "BLADDER": "Urogenital",
    # Cardiovascular
    "HEART": "Cardiovascular",
    "VESSEL": "Cardiovascular",
    "AORTA": "Cardiovascular",
    # Whole-body
    "WHOLEBODY": "Whole-body",
}

### 🎨 Stylesheets and Themes ###################################################
# To ensure consistent visuals, the code searches multiple possible paths for style sheets
# (matplotlib and plotly) relative to the current working directory. This makes the pipeline
# robust to different execution locations and always applies the correct styles.

# Add increasing levels of parent directories for the matplotlib stylesheet
MATPLOTLIB_PATHS = []
for i in range(0, 6):
    path = os.path.normpath(
        os.path.join(
            CWD,
            *[".." for _ in range(i)],
            "Stylesheets",
            "style_sheet_matplotlib.mplstyle",
        )
    )
    MATPLOTLIB_PATHS.append(path)

# Add increasing levels of parent directories for the plotly stylesheet
PLOTLY_PATHS = []
for i in range(0, 6):
    path = os.path.normpath(
        os.path.join(
            CWD, *[".." for _ in range(i)], "Stylesheets", "style_sheet_plotly.json"
        )
    )
    PLOTLY_PATHS.append(path)

# Check if the style sheets exist in the specified paths. If not, raise an error.
for PATH in MATPLOTLIB_PATHS:
    if os.path.exists(PATH):
        STYLE_PATH_MATPLOTLIB = PATH
        print(f"✅ Found style_sheet_matplotlib.mplstyle at: {STYLE_PATH_MATPLOTLIB}")
        break
else:
    raise FileNotFoundError(
        "❌ style_sheet_matplotlib.mplstyle not found in expected locations."
    )

# Check if the style sheet exists in the specified paths. If not, raise an error.
for PATH in PLOTLY_PATHS:
    if os.path.exists(PATH):
        THEME_PATH_PLOTLY = PATH
        print(f"✅ Found style_sheet_plotly.json at: {THEME_PATH_PLOTLY}")
        break
else:
    raise FileNotFoundError(
        "❌ style_sheet_plotly.json not found in expected locations."
    )


# Load the custom theme from the JSON file
with open(THEME_PATH_PLOTLY, "r") as f:
    CUSTOM_THEME = json.load(f)

# Register the theme with Plotly
pio.templates["custom_theme"] = CUSTOM_THEME
# Set the custom theme as the default
pio.templates.default = "custom_theme"
# Set the font size of the subplots annotations or titles to 22
SUBPLOT_TITLE_FONTSIZE = 22

# Load the custom style sheet for matplotlib
plt.style.use(STYLE_PATH_MATPLOTLIB)


# Set a variable to fix the bbox_to_anchor
ANCHOR_LEGEND = (1.05, 1)
LEGEND_TITLE_FONTSIZE = 24

# Maximum number of rows to display in a pandas dataframe
ROW_MAX = 5

# Set pandas options to display all columns and 5 rows
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", ROW_MAX)

# Set polars options to display all columns and 5 rows
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(ROW_MAX)

### 🗃️ Load and filter DataFrames ##############################################
# These functions load DataFrames from saved pickle or parquet files, renames
# and creates necessary new columns (e.g., converting scan duration from ms to s),
# perform various filtering steps and plots them (drop NaNs, remove zeros, exclude long
# examinations/measurements)


def data_read_in(save_path: str, df_name: str):
    """
    Read in datasets from the specified path, supporting both pickle and parquet formats.

    Parameters:
        save_path (str): The path to the saved datasets.
        df_name (str): The name of the dataframe to read in, including the file extension.

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: The dataframe read in from the file.
    """
    # Check if the save_path exists
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"❌The specified path {save_path} does not exist.")

    # Check if the df_name exists in the save_path
    file_path = os.path.join(save_path, df_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"❌The specified file {df_name} does not exist in the path {save_path}."
        )

    # Determine the file type based on the extension
    file_extension = df_name.split(".")[-1].lower()

    if file_extension == "pkl":
        # Read in the pickle file
        df = pd.read_pickle(file_path)
    elif file_extension == "parquet":
        # Read in the parquet file
        df = pl.read_parquet(file_path)
    else:
        raise ValueError(
            f"❌Unsupported file format: {file_extension}. Only 'pkl' and 'parquet' are supported."
        )

    return df


def create_rename_columns(
    df: pd.DataFrame | pl.DataFrame,
) -> pd.DataFrame | pl.DataFrame:
    """
    Create the ScanDuration_s_exam from the ScanDuration_ms_exam column and
    drop the ScanDuration_ms_exam column. Rename the ScanningTime_meas column
    to ScanDuration_s_meas.

    Parameters:
        df (pd.DataFrame or pl.DataFrame): The DataFrame to rename columns in.

    Returns:
        The DataFrame with renamed columns.
    """
    if isinstance(df, pd.DataFrame):
        # Create the ScanDuration_s_exam column by converting from ms to s
        df["ScanDuration_s_exam"] = df["ScanDuration_ms_exam"] / 1000
        # Drop the ScanDuration_ms_exam column
        df = df.drop(columns=["ScanDuration_ms_exam"])
        # Rename the ScanningTime_meas column to ScanDuration_s_meas
        df = df.rename(columns={"ScanningTime_meas": "ScanDuration_s_meas"})

    elif isinstance(df, pl.DataFrame):
        # Create the ScanDuration_s_exam column by converting from ms to s
        df = df.with_columns(
            (pl.col("ScanDuration_ms_exam") / 1000).alias("ScanDuration_s_exam")
        )
        # Drop the ScanDuration_ms_exam column
        df = df.drop(["ScanDuration_ms_exam"])
        # Rename the ScanningTime_meas column to ScanDuration_s_meas
        df = df.rename({"ScanningTime_meas": "ScanDuration_s_meas"})

    else:
        raise TypeError("❌ df must be a pandas or polars DataFrame.")

    return df


def filter_dataframes(
    df: pd.DataFrame | pl.DataFrame,
) -> Tuple[pd.DataFrame | pl.DataFrame, dict]:
    """
    Filter a DataFrame (pandas or polars) based on specified columns and values. Removes
    rows with NaN values, 0 values, and excludes examinations longer than 3h or
    measurements longer than 30min. If one measurement of the examination is 
    excluded, the whole examination is excluded.

    Parameters:
        df (pd.DataFrame or pl.DataFrame): The DataFrame to filter.

    Returns:
        Tuple: The filtered DataFrame and a dictionary containing the number of rows before and after each filtering step.
    """
    # Define columns to filter
    subset_cols = [
        "TotalEnergy_KWh_exam",
        "TotalActivePower_KW_exam",
        "TotalApparentPower_KVA_exam",
        "TotalReactivePower_KVAR_exam",
        "TotalEnergy_KWh_meas",
        "TotalActivePower_KW_meas",
        "TotalApparentPower_KVA_meas",
        "TotalReactivePower_KVAR_meas",
        "DailyTotalEnergy_KWh_energy",
    ]

    # Check if the columns exist in the DataFrame
    for col in subset_cols + ["ScanDuration_s_exam", "ScanDuration_s_meas"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    # Dictionary to track row counts at each step
    filtering_dict = {}

    # Get the initial number of rows
    initial_rows = df.shape[0]
    filtering_dict["Initial Rows"] = initial_rows
    print(f"1. Initial number of rows: {initial_rows}")

    if isinstance(df, pd.DataFrame):
        # Pandas DataFrame filtering
        # Drop rows with NaN values
        df = df.dropna(subset=subset_cols)
        after_dropna_rows = df.shape[0]
        filtering_dict["Dropping NaN Values"] = after_dropna_rows
        print(f"2. Dropped {initial_rows - after_dropna_rows} rows with NaN values.")

        # Drop rows with 0 values
        df = df[(df[subset_cols] != 0).all(axis=1)]
        after_dropzero_rows = df.shape[0]
        filtering_dict["Dropping Zero Values"] = after_dropzero_rows
        print(
            f"3. Dropped {after_dropna_rows - after_dropzero_rows} rows with 0 values."
        )

        # Filter rows with examination duration >= 3h (10800s)
        df = df[df["ScanDuration_s_exam"] < 10800]
        after_exam_duration_filter_rows = df.shape[0]
        filtering_dict["Filtering Examination Duration"] = (
            after_exam_duration_filter_rows
        )
        print(
            f"4. Dropped {after_dropzero_rows - after_exam_duration_filter_rows} rows due to examination duration."
        )

        # Filter rows with measurement duration >= 30min (1800s). If one measuremnt
        # is an outlier the whole examination is an outlier and should be dropped.
        df["Include_meas"] = df["ScanDuration_s_meas"] < 1800
        # Group by the examination and drop the whole examination if one measurement
        # is an outlier
        df = df.groupby("ExaminationID_exam").filter(lambda x: x["Include_meas"].all())
        # Drop the Include_meas column
        df = df.drop(columns=["Include_meas"])
        after_meas_duration_filter_rows = df.shape[0]
        filtering_dict["Filtering Measurement Duration"] = (
            after_meas_duration_filter_rows
        )
        print(
            f"5. Dropped {after_exam_duration_filter_rows - after_meas_duration_filter_rows} rows due to measurement duration."
        )

        df = df.reset_index(drop=True)

    elif isinstance(df, pl.DataFrame):
        # Polars DataFrame filtering
        # Drop rows with NaN values
        df = df.drop_nulls(subset=subset_cols)
        after_dropna_rows = df.shape[0]
        filtering_dict["Dropping NaN Values"] = after_dropna_rows
        print(f"2. Dropped {initial_rows - after_dropna_rows} rows with NaN values.")

        # Drop rows with 0 values
        df = df.filter(pl.all_horizontal([pl.col(c) != 0 for c in subset_cols]))
        after_dropzero_rows = df.shape[0]
        filtering_dict["Dropping Zero Values"] = after_dropzero_rows
        print(
            f"3. Dropped {after_dropna_rows - after_dropzero_rows} rows with 0 values."
        )

        # Filter rows with examination duration > 3h (10800s)
        df = df.filter(pl.col("ScanDuration_s_exam") < 10800)
        after_exam_duration_filter_rows = df.shape[0]
        filtering_dict["Filtering Examination Duration"] = (
            after_exam_duration_filter_rows
        )
        print(
            f"4. Dropped {after_dropzero_rows - after_exam_duration_filter_rows} rows due to examination duration."
        )

        # Filter rows with measurement duration >= 30min (1800s). If one measuremnt
        # is an outlier the whole examination is an outlier and should be dropped.
        df = df.with_columns(
            (pl.col("ScanDuration_s_meas") < 1800).alias("Include_meas")
        )
        # Group by the examination and drop the whole examination if one measurement
        # is an outlier
        df = df.filter(
            pl.col("ExaminationID_exam").is_in(
                df.group_by("ExaminationID_exam")
                .agg(pl.col("Include_meas").all().alias("all_valid"))
                .filter(pl.col("all_valid"))
                .get_column("ExaminationID_exam")
            )
        )
        # Drop the Include_meas column
        df = df.drop("Include_meas")
        after_meas_duration_filter_rows = df.shape[0]
        filtering_dict["Filtering Measurement Duration"] = (
            after_meas_duration_filter_rows
        )
        print(
            f"5. Dropped {after_exam_duration_filter_rows - after_meas_duration_filter_rows} rows due to measurement duration."
        )

        df = df.with_row_count(name="index").drop("index")

    else:
        raise TypeError("❌ The DataFrame must be either a pandas or polars DataFrame.")

    # Print the number of rows before and after filtering
    print(
        f"🧹 Filtered {initial_rows - after_meas_duration_filter_rows} rows from the DataFrame."
    )
    print(f"✅ Number of rows after filtering: {after_meas_duration_filter_rows}")

    return df, filtering_dict


def plot_filtering_steps_funnel(
    filtering_dict: dict, save_fig: str = None
) -> go.Figure:
    """
    Plots a funnel chart to visualize the filtering steps and their impact on the
    dataset size.

    Parameters:
        - filtering_dict: A dictionary where keys are the names of the filtering steps
                    and values are the counts of records remaining after each step.
        - save_fig: Optional string specifying the name of the figure to save.

    Returns:
        - fig: A Plotly Figure object representing the funnel chart.
    """

    # Create the funnel chart
    fig = go.Figure()
    fig.add_trace(
        go.Funnel(
            name="Filtering Steps",
            y=list(filtering_dict.keys()),
            x=list(filtering_dict.values()),
            textinfo="value+percent initial",
            textposition="inside",
            texttemplate="%{value:,} (%{percentInitial:.2%})",
            marker=dict(color=COLOR_LIST[: len(filtering_dict)]),
            orientation="h",
        )
    )

    # Update layout for better visualization
    fig.update_layout(
        title="Filtering Steps Funnel",
        funnelmode="stack",
    )

    # Save the figure if a save_fig is provided
    if save_fig:
        save_fig_formats(fig, save_fig)

    return fig


### 🗺️ Mapping Scanner and Grouping Body Regions ##############################
# - 🏥 Apply a mapping to the scanner IDs to create a new column with the general
#   scanner names for easier analysis and visualization.
# - 🧑‍⚕️ Apply a broader grouping to the body regions to create a new column with the
#   grouped body region names for easier analysis and visualization.
# - 🗂️ The defined BODYREGION_GROUPING is defined in the config_utils.py file and can
#   be easily modified to adjust the grouping as needed.


def apply_scanner_mapping(
    df: pd.DataFrame, scanner_mapping: Dict[str, str], column: str
) -> pd.DataFrame:
    """
    Map the scanner IDs in the specified column to their general names using the
    provided mapping dictionary.

    Parameters:
        df (pd.DataFrame or pl.DataFrame): The DataFrame containing the scanner IDs.
        scanner_mapping (Dict[str, str]): A dictionary mapping scanner IDs to their general names.
        column (str): The column name containing the scanner IDs.

    Returns:
        The DataFrame with a new column 'Machine_scan' containing the mapped names.
    """

    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    # Check if the scanner_mapping is a dictionary
    if not isinstance(scanner_mapping, dict):
        raise ValueError("scanner_mapping must be a dictionary.")

    # If the DataFrame is a pandas DataFrame, use pandas mapping.
    if isinstance(df, pd.DataFrame):
        # Create the new column
        df["Machine_scan"] = df[column].map(scanner_mapping).fillna(df[column])

    # If the DataFrame is a polars DataFrame, use polars mapping.
    elif isinstance(df, pl.DataFrame):
        # Create the new column
        df = df.with_columns(
            pl.col(column)
            .map_elements(lambda x: scanner_mapping.get(x, x), return_dtype=pl.String)
            .alias("Machine_scan")
        )

    else:
        raise TypeError("❌ df must be a pandas or polars DataFrame.")

    return df


def apply_bodyregion_grouping(
    df: pl.DataFrame | pd.DataFrame, BODYREGION_GROUPING: dict
) -> pl.DataFrame | pd.DataFrame:
    """
    Maps the 'BodyRegion_meas' column in the DataFrame to a new 'BodyRegion_group' column
    based on the provided BODYREGION_GROUPING dictionary. Unmapped values are categorized as 'Other'.
    This mapping allows to group specific body regions into broader categories for analysis and visualization.

    Parameters:
        df (pl.DataFrame | pd.DataFrame): Input DataFrame containing the 'BodyRegion_meas' column.
        BODYREGION_GROUPING (dict): Dictionary mapping specific body regions to broader groups.

    Returns:
        pl.DataFrame | pd.DataFrame: DataFrame with an added 'BodyRegion_group' column containing the mapped group names.
    """

    # Check if the 'BodyRegion_meas' column exists in the DataFrame
    if "BodyRegion_meas" not in df.columns:
        raise ValueError(
            "❌ The DataFrame must contain a 'BodyRegion_meas' column for mapping."
        )

    # Check if the BODYREGION_GROUPING is a dictionary
    if not isinstance(BODYREGION_GROUPING, dict):
        raise ValueError("❌ BODYREGION_GROUPING must be a dictionary.")

    # Warn about unmapped body regions that will be categorized as 'Other'
    undique_body_regions = df["BodyRegion_meas"].unique()
    unmapped_body_regions = set(undique_body_regions) - set(BODYREGION_GROUPING.keys())
    if unmapped_body_regions:
        print(
            f"⚠️ The following body regions are not mapped in BODYREGION_GROUPING and will be categorized as 'Other': {unmapped_body_regions}"
        )

    # Check if the input DataFrame is a Polars DataFrame
    if isinstance(df, pl.DataFrame):
        # Apply the mapping to create the 'BodyRegionGroup_meas' column, using 'Other' for unmapped values
        return df.with_columns(
            pl.col("BodyRegion_meas")
            .map_elements(
                lambda x: BODYREGION_GROUPING.get(x, "Other"), return_dtype=pl.String
            )
            .alias("BodyRegionGroup_meas")
        )
    # Check if the input DataFrame is a Pandas DataFrame
    elif isinstance(df, pd.DataFrame):
        # Apply the mapping to create the 'BodyRegionGroup_meas' column, using 'Other' for unmapped values
        df["BodyRegionGroup_meas"] = df["BodyRegion_meas"].map(
            lambda x: BODYREGION_GROUPING.get(x, "Other")
        )
        return df


### 🔀 Reorder Columns by Suffix Priority #######################################
# - 🔤 Reorder the columns of a DataFrame based on a specified suffix priority and
# then alphabetically.


def reorder_columns_by_suffix_priority(
    df: pd.DataFrame | pl.DataFrame, suffix_priority: dict
) -> pd.DataFrame | pl.DataFrame:
    """
    Reorders the columns of a DataFrame based on a given suffix priority and
    then alphabetically.

    Parameters:
        df (pd.DataFrame or pl.DataFrame): The DataFrame whose columns need to be reordered.
        suffix_priority (dict): A dictionary defining the priority of suffixes.
                                Keys are suffixes, values are priority (lower is higher priority).

    Returns:
        DataFrame: A DataFrame with reordered columns.
    """

    # Sort columns based on the suffix priority and then alphabetically
    def sort_cols(cols):
        return sorted(
            cols,
            key=lambda x: (
                # Get the suffix of the column name and find its priority.
                # If no suffix matches, assign a default priority of infinity.
                suffix_priority.get(
                    next(
                        (suffix for suffix in suffix_priority if x.endswith(suffix)),
                        None,
                    ),
                    float("inf"),
                ),
                x,
            ),
        )

    # Handle both pandas and polars DataFrames
    if isinstance(df, pd.DataFrame):
        cols_sorted = sort_cols(df.columns)
        return df[cols_sorted]
    elif isinstance(df, pl.DataFrame):
        cols_sorted = sort_cols(df.columns)
        return df.select(cols_sorted)
    else:
        raise TypeError("df must be a pandas or polars DataFrame.")


### 💾 Saving Figures ############################################################
# Save Plotly figures in multiple formats (JPG, PNG, SVG, HTML) to the 📁 "plots" directory
# for easy sharing, publication, and reproducibility.


def save_fig_formats(fig: go.Figure, save_fig: str) -> None:
    """
    Save the figure in different formats.

    Parameters:
        fig (go.Figure): The figure to save.
        save_fig (str): The name of the figure to save.
    """

    # Get the plotting directory
    plot_dir = os.path.normpath(os.path.join(CWD, "plots"))
    # Ensure the directory exists
    os.makedirs(plot_dir, exist_ok=True)
    # Get the save path for the figure
    save_path = os.path.join(plot_dir, save_fig)

    # Save the figure in different formats
    fig.write_html(f"{save_path}.html")
    fig.write_image(f"{save_path}.png", scale = 2)
    fig.write_image(f"{save_path}.svg", scale = 2)
    fig.write_image(f"{save_path}.jpg", scale = 2)


### 📊 Create Summary DataFrame ################################################
# - 📝 Create a summary DataFrame that includes column names, unique values, data types
#  and counts of NaN and non-NaN values for each column in the input DataFrame
# - 💾 The summary is saved as a CSV file at the specified path


def create_summary_df(df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    """
    Create a summary DataFrame that includes column names, unique values, data types,
    and counts of NaN and non-NaN values for each column in the input DataFrame.
    The summary is saved as a CSV file at the specified path.

    Parameters:
        df(pd.DataFrame): The input DataFrame to summarize.
        save_path(str): The file path where the summary CSV will be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the summary of the input DataFrame.
    """
    # Skip if the file already exists
    if os.path.exists(save_path):
        print(f"✅ Summary file already exists at {save_path}. Skipping creation.")
        return pd.read_csv(save_path, sep=";", decimal=",", encoding="utf-8")

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create a list to hold the summary information for each column
    summary = []
    # Iterate through each column in the DataFrame and gather summary information
    for col in df.columns:
        # Get the data of the columns
        col_data = df[col]
        # Get the first 10 unique values of the column (or all if there are less than 10)
        unique_values = col_data.unique()[:10]
        # If the unique values can be converted to string, do so (e.g., for arrays or lists)
        if hasattr(unique_values, "astype"):
            unique_values = unique_values.astype(str)
        # Unpack the unique values and join them as a comma-separated string for CSV
        unique_values_str = ", ".join(map(str, unique_values))
        # Get the maximum and minimum value
        max_value = col_data.max() if pd.api.types.is_numeric_dtype(col_data) else None
        min_value = col_data.min() if pd.api.types.is_numeric_dtype(col_data) else None
        # Replace the "." in the number with "," for better readability in the CSV file.
        # Since the contains None and numeric values, the automatic conversion
        # of the to_csv function does not work and we need to do it manually.
        if isinstance(max_value, (int, float)):
            max_value = str(max_value).replace(".", ",")
        if isinstance(min_value, (int, float)):
            min_value = str(min_value).replace(".", ",")

        # Create the summary information for the column and append it to the summary list
        summary.append(
            {
                "Column Name": col,
                "Unique Values": unique_values_str,
                "Type": col_data.dtype,
                "Non-NaN Count": col_data.notna().sum(),
                "NaN Count": col_data.isna().sum(),
                "Non-NaN Percentage": 100 * col_data.notna().sum() / len(col_data),
                "NaN Percentage": 100 * col_data.isna().sum() / len(col_data),
                "Unique Count": col_data.nunique(dropna=True),
                "Min Value": min_value,
                "Max Value": max_value,
            }
        )

    # Convert it to a DataFrame and save it as a CSV file
    summary_df = pd.DataFrame(summary)
    # Save the summary DataFrame to a CSV file
    summary_df.to_csv(save_path, index=False, sep=";", decimal=",")
    return summary_df


### 📋 Saving Tables ############################################################
# Save tables in multiple formats (BMP, PNG, PDF) to the 📁 "tables" directory
# for easy sharing, publication, and reproducibility.


def save_table_formats(table, save_table: str) -> None:
    """
    Save the table in different formats.

    Parameters:
        table: The table to save.
        save_table (str): The name of the table to save.
    """

    # Get the table directory path
    table_dir = os.path.normpath(os.path.join(cwd, "tables"))
    # Ensure the directory exists
    os.makedirs(table_dir, exist_ok=True)
    # Get the save path for the table
    save_path = os.path.join(table_dir, save_table)

    # Define the formats to save the table in
    formats = [".jpg", ".png", ".pdf"]

    # Save the table in each format with the specified settings
    for fmt in formats:
        GT.gtsave(
            self=table,
            file=save_path + fmt,
            vwidth=4096,
            vheight=4096,
        )
