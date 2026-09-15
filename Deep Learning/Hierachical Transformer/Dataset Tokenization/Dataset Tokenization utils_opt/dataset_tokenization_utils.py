### 🧩 Tokenization Utilities for Deep Learning Pipeline
#
# 🧰 This module provides helper functions for preprocessing text, numerical, and
# categorical data to create a custom dataset  for the Hierarchical Transformer
#
# This section provides utility functions to:
# - 🔤 Encode text columns with tokenizers, calculate max token lengths, and generate padding/mask arrays.
# - 📅 Decompose datetime columns into components for model input.
# - 🔢 Determine and map numerical/categorical features, including forced mappings and integer encoding.
# - 📏 Z-normalize numerical features and support inverse transformation.
# - 🔗 Concatenate processed DataFrames and update feature mappings.
# - 🗂️ Create the CustomDataset class that integrates all preprocessing steps.
# - 📊 Visualize dataset tensors for inspection and debugging.

import sys
import os

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go

from typing import Tuple, List, Dict

import json

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

import numpy as np

from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get the current working directory
CWD = os.getcwd()

# Define the path to the global utils directory
GLOBAL_UTILS_DIR = os.path.normpath(
    os.path.join(CWD, "..", "..", "..", "..", "Global utils_opt")
)

# Append this directory to the system path
sys.path.append(GLOBAL_UTILS_DIR)

from global_utils import *

### 🧩 Tokenization functions for CustomDataset ################################
# 
# - 🔤 Tokenizes text columns using a pretrained tokenizer.
# - 🚫 Replaces NaN values with a special `[NAN]` token.
# - 🧼 Normalizes special characters before tokenization: 
#   `,` and `;` become [SEP] 
#   `_`, `%\`, `%/`, `%` becomes [SUBSEQ] 
#   `/` becomes [SUB] 
#   `+` becomes [AND] 
# - 🔁 Converts numbered bracketed patterns such as `[/12]` or `[3]` into 
#   `[REPEAT]12` and `[REPEAT]3` so repeated protocol fragments are handled consistently.
# - 🧱 Pads sequences to the maximum length of the column.
# - 🎭 Generates attention masks to indicate valid tokens.
# - ✅ Ensures all text data is consistently formatted for transformer models.


def encoding_max_token_length(
    text_df: pd.DataFrame, tokenizer: Tokenizer
) -> Tuple[Dict[str, Dict[str, int]], list, int]:
    """
    Calculate the maximum token length for each column in the DataFrame and store
    the encodings.

    Parameters:
    text_df (pd.DataFrame): The DataFrame containing text columns to be tokenized.
    tokenizer (Tokenizer): The tokenizer to use for encoding the text.

    Returns:
    Tuple[Dict[str, Dict[str, int]], list, int]: A tuple containing:
        - A dictionary with the start and end indices and max length for each column.
        - A list of encodings for each column.
        - The overall maximum tokenized length across all columns.

    """

    # Step 1: Precompute max tokenized length for each column and store encodings

    # Initalize columns_token_idx to store the start and end indices and max length
    # for each column
    cols_token_idx = {}
    # Initialize a list to store the encodings for each column
    cols_encodings = []
    # Initialize start_idx to 0
    start_idx = 0

    print("\nStep 1: Computing maximum lengths and initial encodings...")
    # Iterate through each text column in the DataFrame
    for text_col in tqdm(text_df.columns, desc="Encoding columns"):
        # Get the text values for the column, convert to string and to a list
        text_val = text_df[text_col].astype(str).tolist()

        # Encode the text values in the column
        encodings = tokenizer.encode_batch(text_val)

        # Store the encodings in the list
        cols_encodings.append(encodings)

        # Get the maximum length of the tokenized sequences
        length = max(len(enc.ids) for enc in encodings)
        # Get the end index for the current column
        end_idx = start_idx + length

        # Store the start_idx, end_idx and length in the dictionary
        cols_token_idx[text_col] = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "length": length,
        }

        print(
            f"  {text_col}: length={length}, start_idx={start_idx}, end_idx={end_idx}"
        )

        # Set the start_idx for the next column
        start_idx = end_idx

    text_max_length = start_idx
    print(f"\nOverall maximum length across all columns: {text_max_length}")

    return cols_token_idx, cols_encodings, text_max_length


def get_column_encodings(
    text_df: pd.DataFrame,
    text_cols_mappings: Dict[str, Dict[str, int]],
    cols_encodings: list,
    pad_token_id: int,
    nan_token_id: int,
) -> Tuple[Dict[str, Dict], pd.DataFrame, pd.DataFrame]:
    """
    Process the encodings of text columns in a DataFrame, generating padded token IDs and masks.

    Parameters:
    text_df (pd.DataFrame): The DataFrame containing text columns to be processed.
    text_cols_mappings (Dict[str, Dict[str, int]]): A dictionary containing the
        start and end indices and max length for each column.
    cols_encodings (list): A list of encodings for each text column.
    pad_token_id (int): The token ID used for padding.
    nan_token_id (int): The token ID used for NaN values.

    Returns:
    Tuple[Dict[str, Dict[str, int]], pd.DataFrame, pd.DataFrame]: A tuple containing:
        - A dictionary with encoded IDs, tokens, padding masks, NaN masks,
            non-NaN masks, and combined masks for each column.
        - A DataFrame containing the padded token IDs for each column.
        - A DataFrame containing the combined masks for each column.

    """

    # Step 2: Process encodings with padding and mask generation
    print("\nStep 2: Processing encodings with padding and mask generation...")
    text_tokens_pad_dict = {}

    # Create an empty DataFrame to store the encoded ids and combined masks
    text_tokens_df = pd.DataFrame(index=text_df.index, columns=text_df.columns)
    non_nan_text_df = pd.DataFrame(index=text_df.index, columns=text_df.columns)

    # Iterate through each text column and its encodings
    for idx_col, text_col in enumerate(
        tqdm(text_df.columns, desc="Processing encodings")
    ):
        # Get the encodings for the current column
        encodings = cols_encodings[idx_col]
        # Get the max length for the current column
        max_length = text_cols_mappings[text_col]["length"]

        # Store all encodings for the columns
        col_data = {
            keys: []
            for keys in [
                "ids",
                "tokens",
                "padding_mask",
                "nan_mask",
                "non_nan_mask",
                "combined_mask",
            ]
        }

        # Process each encoding (each row) in this column
        for enc in encodings:
            # Get the ids, tokens, and attention mask for this encoding
            ids = enc.ids[:]
            tokens = enc.tokens[:]
            padding_mask = [0] * len(ids)

            # Calculate the padding length
            pad_len = max_length - len(ids)

            # Raise error if the padding length is negative
            if pad_len < 0:
                raise ValueError(
                    f"Padding length for column {text_col} is negative: {pad_len}. "
                    "This indicates that there is a problem in calculating the maximum token length for the column."
                )

            # If padding is needed, add pad_token_id and pad_token to the lists
            if pad_len > 0:
                ids += [pad_token_id] * pad_len
                tokens += ["[PAD]"] * pad_len
                padding_mask += [1] * pad_len

            # Create different types of masks
            nan_mask = [1 if token_id == nan_token_id else 0 for token_id in ids]
            non_nan_mask = [1 if token_id != nan_token_id else 0 for token_id in ids]
            combined_mask = [
                0 if (nan or pad) else 1 for nan, pad in zip(nan_mask, padding_mask)
            ]

            # Append this row's encoding to the column lists
            col_data["ids"].append(ids)
            col_data["tokens"].append(tokens)
            col_data["padding_mask"].append(padding_mask)
            col_data["nan_mask"].append(nan_mask)
            col_data["non_nan_mask"].append(non_nan_mask)
            col_data["combined_mask"].append(combined_mask)

        # Overhand the column data dictionary
        text_tokens_pad_dict[text_col] = col_data
        # Get the Token IDs and combined masks for the column
        text_tokens_df[text_col] = col_data["ids"]
        non_nan_text_df[text_col] = col_data["combined_mask"]
        print(
            f"  {text_col}: {len(col_data['ids'])} rows encoded with max_length={max_length}"
        )

    return text_tokens_pad_dict, text_tokens_df, non_nan_text_df


def encode_text_columns(
    df: pd.DataFrame, text_columns: List[str], tokenizer: Tokenizer
) -> Tuple[Dict[str, Dict[str, int]], pd.DataFrame, pd.DataFrame]:
    """
    Encode the text columns in the DataFrame using the provided tokenizer.

    Parameters:
    df (pd.DataFrame): The DataFrame containing text columns to be encoded.
    text_columns (List[str]): The list of text columns to be encoded.
    tokenizer (Tokenizer): The tokenizer to use for encoding.

    Returns:
    Tuple[Dict[str, Dict[str, int]], pd.DataFrame, pd.DataFrame]: A tuple containing:
        - A dictionary with encoded IDs, tokens, padding masks, NaN masks,
            non-NaN masks, and combined masks for each column.
        - A DataFrame containing the padded token IDs for each column.
        - A DataFrame containing the combined masks for each column.
    """
    # Get the token IDs for the pad token and NaN token from the tokenizer
    pad_token_id = tokenizer.token_to_id("[PAD]")
    nan_token_id = tokenizer.token_to_id("[NAN]")

    print(f"Pad token ID: {pad_token_id}")
    print(f"NaN token ID: {nan_token_id}")

    # Get the text columns from the DataFrame
    text_df = df[text_columns].copy()

    # Replace the NaN values in the text columns with [NAN] to avoid issues during tokenization
    text_df = text_df.fillna("[NAN]")

    # Replace the , or ; with [SEP], _, %\, %/, % with [SUBSEQ], / with [SUB], and
    # + with [AND] in the text columns to create special tokens that can be used to
    text_df = text_df.replace(
        {r",|;": "[SEP]", r"_|%\\|%/|%": "[SUBSEQ]", "/": "[SUB]", r"\+": "[AND]"},
        regex=True,
    )

    # Replace the "[/Number]" with "[REPEAT] /Number", \d+ captures one or more digits
    # inside the square brackets and appendes it to the string "[REPEAT] "
    text_df = text_df.replace(r"\[(\d+)\]", r"[REPEAT]\1", regex=True)


    # Remove any [SUBSEQ] or [SUB] at the beginning of a string
    text_df = text_df.replace(r"^\[SUBSEQ\]|^\[SUB\]", "", regex=True)

    # Remove duplicate following [SEP], [SUBSEQ], [SUB], and [AND] tokens
    text_df = text_df.replace(
        r"(\[(?:SEP|SUBSEQ|SUB|AND)\])(?:\1)+",
        r"\1",
        regex=True,
    )
    
    # Step 1: Calculate maximum encoding lengths and initial encodings
    text_cols_mappings, cols_encode_dict, text_max_length = encoding_max_token_length(
        text_df, tokenizer
    )

    # Step 2: Process encodings with padding and mask generation
    text_tokens_pad_dict, text_tokens_df, non_nan_text_df = get_column_encodings(
        text_df, text_cols_mappings, cols_encode_dict, pad_token_id, nan_token_id
    )

    return (
        text_tokens_pad_dict,
        text_cols_mappings,
        text_tokens_df,
        non_nan_text_df,
        text_max_length,
    )


### 🧮 Functions for Converting Numerical, Datetime, and Categorical Features ##
# 
# This section provides utility functions to preprocess numerical, datetime, and <br>
# categorical features for the `CustomDataset`:
# 
# - 🗓️ Datetime decomposition:  
#   - Splits datetime columns into components (year, month, day, hour, minute, day 
#   of week, day of year).
# 
# - 🔎 Feature Type Validation:
#     - Dynamically identifies numerical and categorical columns based on data characteristics.
#     - Categorical columns are defined as those containing fewer than 10 unique values.
#     
# - 🗺️ Categorical Mapping:
#     -   Divides categorical columns into two types:
#         - 🏷️ Categorical Columns: Directly used as categorical features.
#         - 🔢 Mapping Categorical Columns: Columns with values mapped to numerical indices.
#             - Columns are designated as "mapping categorical" if their unique 
#             values have a high average distance (over 100), indicating a 
#             need for index-based representation.
# 
# - 📏 Numerical Feature Normalization: Numerical features are z-normalized 
# to have a mean of 0 and a standard deviation of 1.
# - ➕ Categorical Feature Handling: The overall minimum value is added to 
# all categorical columns to ensure that all values are non-negative.



def decompose_datetime_columns(
    df: pd.DataFrame, datetime_columns_decompose: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], pd.DataFrame]:
    """
    Decompose datetime columns into Year, Month, Day, Hour, and Minute components.

    Parameters:
    df (pd.DataFrame): The input dataframe
    datetime_columns_decompose (List[str]): List of column names containing datetime data

    Returns:
    Tuple[pd.DataFrame, Dict[str, Dict[str, int]], pd.DataFrame, pd.DataFrame]: A tuple containing:
        - A DataFrame with decomposed datetime components
        - A dictionary with start and end indices and max length for each new column
        - A DataFrame indicating non-NaN values in the original datetime columns
    """

    # Create a copy of the original DataFrame to avoid modifying it directly
    df = df.copy()
    # Initialize an empty DataFrame to store the decomposed datetime components
    # and a dictionary for mappings and a start index for the new columns
    df_processed = pd.DataFrame()
    datetime_mappings = {}
    start_idx = 0

    for col in datetime_columns_decompose:
        # Check if the column exists in the DataFrame
        if col in df.columns:
            # Convert to datetime if not already
            date_columns = pd.to_datetime(df[col], errors="coerce")

            # Extract datetime components
            df_processed[f"{col}_Year"] = date_columns.dt.year
            df_processed[f"{col}_Month"] = date_columns.dt.month
            df_processed[f"{col}_Day"] = date_columns.dt.day
            df_processed[f"{col}_Hour"] = date_columns.dt.hour
            df_processed[f"{col}_Minute"] = date_columns.dt.minute
            # Decompose into day of week and day of year
            df_processed[f"{col}_DayOfWeek"] = date_columns.dt.dayofweek
            df_processed[f"{col}_DayOfYear"] = date_columns.dt.dayofyear
            print(
                f"Decomposed {col} into year, month, day, hour, minute, day_of_week, day_of_year components"
            )
        else:
            raise ValueError(f"Column {col} not found in the DataFrame.")

    # Iterate over all new columns in the processed DataFrame
    for new_col in df_processed.columns:
        # Create a mapping for the new column
        datetime_mappings[new_col] = {
            "start_idx": start_idx,
            # Each new column is a single value
            "end_idx": start_idx + 1,
            # Each new column has a max length of 1
            "max_length": 1,
        }
        # Update the start index for the next column
        start_idx += 1

    # Get the NaN mask and non-NaN mask for the processed DataFrame
    nan_df = df_processed.isna().astype(int)
    non_nan_df = 1 - nan_df

    return df_processed, datetime_mappings, non_nan_df


def determine_num_cat_features(
    df: pd.DataFrame,
    initial_numerical_columns: list,
    initial_categorical_columns: list,
    datetime_df: list,
    force_numerical_columns: list = [],
    force_categorical_columns: list = [],
) -> tuple:
    """
    Determine the numerical and categorical features in a DataFrame. Inital
    numerical and categorical columns are provided, and the function will
    analyze those and update them. For inferring if the features are
    numerical or categorical, the function will look at the number of unique
    values in the columns. If a column has less than 10 unique values, it is
    considered categorical, otherwise numerical. Additionally, the function
    checks the average distance between unique values in categorical columns
    to determine if the categorical values should be mapped to integer values.
    Further, it can be numerical and categorical columns overhanded which
    can be forced to be numerical or categorical. The datetime columns
    are always treated as numerical columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    initial_numerical_columns (list): List of initial numerical columns.
    initial_categorical_columns (list): List of initial categorical columns.
    datetime_columns (list): List of datetime columns to be treated as numerical.
    datetime_columns_decompose (list): List of datetime columns to decompose.
    force_numerical_columns (list): List of columns to be forced as numerical.
    force_categorical_columns (list): List of columns to be forced as categorical.
    """

    # Define columns that should be forced to be numerical, datetime columns
    # should always be treated as numerical
    force_numerical_date_cols = force_numerical_columns + list(datetime_df.columns)
    # Drop duplicates while preserving order
    force_numerical_date_columns = list(dict.fromkeys(force_numerical_date_cols))

    # Combine the guessed numerical and categorical columns with the forced
    # numerical columns
    combined_columns = list(
        set(initial_numerical_columns + initial_categorical_columns)
    )

    # Exclude any object type columns from the combined columns, as they will be
    # treated as categorical since their values will be replaced by 
    # the respective unique values.
    object_columns = [col for col in combined_columns if pd.api.types.is_object_dtype(df[col])]
    combined_columns = [col for col in combined_columns if col not in object_columns]
    
    # Get the combined DataFrame with the specified columns
    df = df[combined_columns].copy()

    # Get inital categorical columns, which are those with less than 10 unique values
    # and which are not forced numerical/date columns. Further, investigate only
    # the guessed categorical columns
    inferred_categorical_columns = (
        df[combined_columns]
        .columns[
            (df[combined_columns].nunique() < 10)
            & ~df[combined_columns].columns.isin(force_numerical_date_cols)
        ]
        .tolist()
    )

    # Get inital numerical columns, which are those with at least 10 unique values
    # and not in the force numerical columns. Further, investigate only
    # the guessed numerical columns
    inferred_numerical_columns = (
        df[combined_columns].columns[df[combined_columns].nunique() >= 10].tolist()
    )

    # Average distance dictionary
    average_distance_dict = {}
    # Columns which are already categorical
    categorical_columns = []
    # Columns which should be mapped to categorical
    mapping_categorical_colummns = []
    # Columns which are numerical
    numerical_columns = []

    # Loop through the inital categorical columns to determine if they should be
    # treated as categorical or numerical. 
    for col in inferred_categorical_columns:
        # Drop NaN values and get unique values
        unique_values = df[col].dropna().unique()

        # If the unique values are of numeric type, calculate the average distance between them
        if pd.api.types.is_numeric_dtype(unique_values.dtype):
            unique_values = np.sort(unique_values)

            # Calculate the average distance between unique numerical values
            if len(unique_values) > 1:
                avg_distance = np.mean(np.diff(sorted(unique_values)))
            else:
                avg_distance = 0
        else:
            raise ValueError(
                f"Column {col} contains non-numeric unique values, cannot calculate average distance."
            )

        average_distance_dict[col] = avg_distance

        # If the average distance is less than 1000, we consider it as categorical,
        # otherwise we consider it as numerical and map it to categorical integers
        if avg_distance < 1000:
            categorical_columns.append(col)
        else:
            mapping_categorical_colummns.append(col)
            
    
    # Combine the forced numerical columns with the inferred numerical columns
    numerical_columns = list(
        set(force_numerical_date_columns + inferred_numerical_columns)
    )
    
    # Combine the forced categorical columns with the inferred categorical columns
    categorical_columns = list(set(force_categorical_columns + categorical_columns + object_columns))

    return (
        sorted(numerical_columns),
        sorted(categorical_columns),
        sorted(mapping_categorical_colummns),
    )


def map_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    mapping_categorical_colummns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict], pd.DataFrame]:
    """
    Map multiple columns to categorical integers and return mapping dictionaries.
    The overall minimum value across all mapped columns is calculated and added to the
    DataFrame to ensure that all categorical values are positive. This ensures
    that the categorical values can be used as indices for embedding layers without
    encountering negative values.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the columns to be mapped.
    categorical_columns (List[str]): List of columns that are already categorical
                                     and should be kept as they are.
    mapping_categorical_colummns (List[str]): List of columns that should be mapped to
                                               categorical integers based on their unique values.

    Returns:
    Tuple[pd.DataFrame, Dict[str, Dict], pd.DataFrame]: A tuple containing:
        - A DataFrame with the mapped categorical columns.
        - A dictionary with mapping information for each column.
        - A DataFrame indicating non-NaN values in the original categorical columns.
    """
    # Initialize output DataFrame and mappings
    df = df.copy()
    df_processed = pd.DataFrame()
    categorical_mappings = {}
    categorical_cols = {}
    start_idx = 0

    # Iterate through the columns which have to be mapped to categorical integers
    for col in tqdm(
        categorical_columns + mapping_categorical_colummns, desc="Mapping columns"
    ):
        # Reset mapping_cat flag for each column
        mapping_cat = False

        # Get unique values and create mapping
        unique_values = df[col].dropna().unique()

        # If the unique values are of object type, try to convert them to float.
        if pd.api.types.is_object_dtype(unique_values.dtype):
            try:
                unique_values = unique_values.astype(float)
                # Also try to convert the actual column data
                df[col] = df[col].astype(float)
            except ValueError:
                # If conversion fails, it means it contains non-numeric values
                print(f"Column {col} contains non-numeric values, keeping as object.")
                mapping_cat = True

        # If the column should be mapped to categorical integers
        if col in mapping_categorical_colummns or mapping_cat:
            value_mapping = {
                value: idx for idx, value in enumerate(sorted(unique_values))
            }
            # Create reverse mapping for easy lookup
            reverse_mapping = {idx: value for value, idx in value_mapping.items()}

            # The new categorical column includes the mapped values
            categorical_cols[col] = df[col].map(value_mapping)

        # If the column is already categorical, keep the values as they are
        elif col in categorical_columns:
            # For already categorical columns, just copy the values
            value_mapping = {None: None}
            reverse_mapping = {None: None}

            # Copy the current column to the new categorical column
            categorical_cols[col] = df[col].copy()

        # Create mapping info similar to datetime function
        categorical_mappings[col] = {
            "start_idx": start_idx,
            "end_idx": start_idx + 1,
            "length": 1,
            "value_mapping": value_mapping,
            "reverse_mapping": reverse_mapping,
            "num_categories": len(unique_values),
        }

        # Update start_idx for next column
        start_idx += 1

    # Concatenate all new columns at once to avoid fragmentation
    if categorical_cols:
        df_processed = pd.DataFrame(categorical_cols, index=df.index)
    else:
        df_processed = pd.DataFrame(index=df.index)

    # Get the NaN mask and non-NaN mask for the processed DataFrame
    nan_df = df_processed.isna().astype(int)
    non_nan_df = 1 - nan_df

    # Get the overall minimum and add it to the DataFrame that the categorical values
    # are always positive
    if not df_processed.empty:
        categorical_min_val = df_processed.min().min()
        if categorical_min_val < 0:
            df_processed += abs(categorical_min_val)

    return df_processed, categorical_mappings, non_nan_df, categorical_min_val


def z_normalize_numerical_features(
    df: pd.DataFrame,
    datetime_df: pd.DataFrame = pd.DataFrame(),
    numerical_columns: List[str] = [],
    label_columns: List[str] = [],
) -> Tuple[pd.DataFrame, Dict, Dict, pd.DataFrame]:
    """
    Perform z-normalizations on numerical features in the overhanded df
    and datetime_df. The function handles NaN values by preserving them during
    normalization and creating masks. It returns a  DataFrame with normalized
    numerical features, a dictionary of scalers for inverse transformation, a
    dictionary with mapping information, and DataFrames indicating non-NaN
    values for each column.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing numerical features.
    datetime_df (pd.DataFrame): DataFrame containing datetime features to be z-normalized.
    numerical_columns (List[str]): List of numerical columns to be z-normalized.
    label_columns (List[str]): List of label columns to be z-normalized.


    Returns:
    Tuple[pd.DataFrame, Dict, Dict, pd.DataFrame]: A tuple containing:
        - A DataFrame with z-normalized numerical features.
        - A dictionary with scalers for each numerical feature for inverse transformation.
        - A dictionary with mapping information for each numerical feature.
        - A DataFrame indicating non-NaN values in the original numerical features.
    """

    # Concatenate the original DataFrame with the datetime_df
    df_input = pd.concat([df, datetime_df], axis=1)

    # Sort the columns
    numerical_selected_columns = sorted(
        list(set(numerical_columns + label_columns + list(datetime_df.columns)))
    )

    # Sort the df_input according to the numerical_selected_columns
    df_input = df_input[numerical_selected_columns]

    print(f"Selected numerical columns for normalization: {numerical_selected_columns}")

    # Initialize dictionaries to store processed columns, mappings, and scalers
    processed_cols = {}
    numerical_mappings = {}
    feature_scalers = {}
    start_idx = 0

    # Calculate NaN masks for the entire dataframe
    nan_df = df_input.isna().astype(int)
    non_nan_df = 1 - nan_df

    # Z-normalize numerical features
    for col in tqdm(numerical_selected_columns, desc="Normalizing columns"):

        # Ensure the column exists in the DataFrame
        if col not in df_input.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        # Copy the original column to processed dataframe first (preserves NaN values)
        col_data = df_input[col].copy()

        # If the column is of object type, try to convert it to float. If conversion
        # fails, it means it contains non-numeric values
        if pd.api.types.is_object_dtype(col_data.dtype):
            try:
                col_data = col_data.astype(float)
            except ValueError:
                # If conversion fails, it means it contains non-numeric values
                # Thus set the average distance to 0 to keep it as categorical
                print(f"Column {col} contains non-numeric values, keeping as object.")
                continue

        # Always cast to float before normalization to avoid dtype issues
        col_data = col_data.astype(float)
        non_nan_mask = col_data.notna()

        # If there are only NaN values, skip normalization
        if non_nan_mask.sum() == 0:
            scaler = None
            processed_cols[col] = col_data
            warnings.warn(
                f"Column {col} has no non-NaN values, skipping normalization."
            )
        # If there is only one real Value, set it to 0 since n-1 is used
        # in the denominator of the z-normalization formula
        elif non_nan_mask.sum() == 1:
            scaler = None
            col_data.loc[non_nan_mask] = 0.0
            processed_cols[col] = col_data
            warnings.warn(
                f"Column {col} has only one non-NaN value, setting it to 0.0 for normalization."
            )
        else:
            # Create and fit scaler ONLY on non-NaN values
            scaler = StandardScaler()
            non_nan_values = col_data[non_nan_mask].values.reshape(-1, 1)

            # Fit and transform only the non-NaN values
            normalized_values = scaler.fit_transform(non_nan_values)

            # Update ONLY the non-NaN positions in the processed dataframe
            # NaN values remain as NaN in their original positions
            col_data.loc[non_nan_mask] = normalized_values.flatten()
            processed_cols[col] = col_data

            # Store scaler for potential inverse transformation
            feature_scalers[col] = scaler

        # Create mapping info for the normalized column
        numerical_mappings[col] = {
            "start_idx": start_idx,
            "end_idx": start_idx + 1,
            "length": 1,
            "normalized_column": col,
            # Store scaler if available
            "scaler": scaler,
            "has_scaler": scaler is not None,
        }

        # Update start_idx for next column ,only counting the normalized column
        start_idx += 1

    # Sort the processed columns to maintain order
    df_processed = pd.DataFrame(processed_cols)

    return df_processed, feature_scalers, numerical_mappings, non_nan_df


def inverse_normalize_features(
    df: pd.DataFrame, scalers: Dict, columns: List[str] = None
) -> pd.DataFrame:
    """
    Inverse transform normalized features back to original scale.
    Only transforms columns that have scalers (excludes single-value and all-NaN columns).

    Parameters:
    df (pd.DataFrame): DataFrame with normalized features
    scalers (Dict): Dictionary of fitted scalers from z_normalize_numerical_features
    columns (List[str]): Specific columns to inverse transform. If None, transform all available.

    Returns:
    pd.DataFrame: DataFrame with inverse transformed features
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_inverse = df.copy()

    # If no specific columns are provided, attempt to inverse transform all
    # columns that have scalers
    if columns is None:
        columns = list(scalers.keys())

    transformed_count = 0
    skipped_count = 0

    # Iterate through the specified columns and apply inverse transformation where possible
    for col in columns:
        # Check if the column has a corresponding scaler
        if col in scalers:
            # Get the scaler for this column
            scaler = scalers[col]

            # Only attempt inverse transformation if a scaler and column exists
            if scaler is not None and col in df_inverse.columns:
                non_nan_mask = df_inverse[col].notna()

                # Check if there are non-NaN values to inverse transform
                if non_nan_mask.sum() > 0:
                    # Get normalized values for non-NaN positions
                    normalized_values = df_inverse.loc[
                        non_nan_mask, col
                    ].values.reshape(-1, 1)

                    # Inverse transform using the stored scaler
                    original_values = scaler.inverse_transform(normalized_values)

                    # Update only non-NaN positions with inverse transformed values
                    df_inverse.loc[non_nan_mask, col] = original_values.flatten()

                    print(f"Inverse transformed column: {col}")
                    transformed_count += 1
                else:
                    print(f"Skipping {col}: No non-NaN values to inverse transform")
                    skipped_count += 1
            else:
                print(
                    f"Skipping {col}: No scaler available (was single-value or all-NaN)"
                )
                skipped_count += 1
        else:
            print(f"Warning: Column {col} not found in scalers dictionary")
            skipped_count += 1

    return df_inverse

### 🔗 Concatenation of DataFrames ############################################

def concatenate_dataframes(
    df_list: List[pd.DataFrame],
    mappings: List[Dict[str, Dict]],
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Concatenate multiple DataFrames and their corresponding non-NaN DataFrames.
    Also update the mappings to include the new columns and their start and end indices.

    Parameters:
    df_list (List[pd.DataFrame]): List of DataFrames to concatenate
    mappings (List[Dict[str, Dict]]): List of mappings for each DataFrame

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
        - Concatenated DataFrame
        - Updated mappings dictionary
    """
    # Concatenate the DataFrames
    concatenated_df = pd.concat(df_list, axis=1)

    # Initialize updated mappings
    updated_mappings = {}
    start_idx = 0

    for mapping in mappings:
        for col, info in mapping.items():
            # Update the start and end indices
            updated_mappings[col] = {
                "start_idx": start_idx,
                "end_idx": start_idx + info["length"],
                "length": info["length"],
            }
            start_idx += info["length"]

    print(f"Concatenated DataFrame with {len(concatenated_df.columns)} columns.")
    print(f"Updated mappings with {len(updated_mappings)} columns.")
    return concatenated_df, updated_mappings


### 🧩 Custom Dataset #########################################################
# 
# The `CustomDataset` is a PyTorch-compatible dataset class tailored for 
# multimodal MRI data, facilitating transformer-based deep learning workflows. 
# It integrates and preprocesses various feature types to optimize model performance:
# 
# - 🔤 Text Features:  
#     - Text columns are tokenized using a fine-tuned WordPiece tokenizer.
#     - NaN values are replaced with a special [NAN] token, and sequences are 
#       padded for consistent length. 
#     - Special characters are normalized before tokenization: `,` and `;` become [SEP], 
#        `_`, `%\`, `%/`, `%` becomes [SUBSEQ],`/` becomes [SUB], `+` becomes [AND]
#     - Numbered bracketed patterns such as `[/12]` or `[3]` are converted to 
#       `[REPEAT]12` and `[REPEAT]3` so repeated protocol fragments are handled consistently. 
#     - Attention masks distinguish valid tokens from padding.
# 
# - 📅 Datetime Features: Datetime columns are decomposed into components 
# (year, month, day, etc.) to capture temporal patterns.
# 
# - 🔢 Numerical Features: Numerical and datetime-derived features are 
# z-normalized to ensure consistent scaling.
# 
# 
# - 📊 Categorical Features:
#     - Categorical features are divided into:
#         - Categorical Columns: Directly used as categorical features.
#         - Mapping Categorical Columns: Values are mapped to numerical 
#         indices for efficient processing.
#     - The overall minimum value is added to ensure non-negative values.
# 
# - 🎯 Label Features: Label columns are z-normalized to standardize the target variable.
# 
# - 🎭 Attention Masks: Missing or padded values are tracked with masks so 
#   the model can distinguish valid entries from placeholders.
# 
# Each sample returned by the dataset consists of eight tensors:
# 
# 1.  IDX of the original dataframe
# 2.  Numerical features
# 3.  Text features (tokenized ids)
# 4.  Categorical features
# 5.  Label features
# 6.  Numerical mask
# 7.  Text attention mask
# 8.  Categorical Mask
# 9.  Label Mask
# 
# This structure enables flexible batching, masking, and multimodal input <br>
# handling, optimizing transformer model training for MRI data analysis.

class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for handling MRI data preprocessing. Tetxt, numerical,
    and categorical features are encoded. Finally, the dataset can be used
    for training or evaluation of machine learning models. The text features
    are tokenized using a provided tokenizer. Datetime columns are decomposed
    into individual components and are as well as the numerical features
    z-normalized. Categorical features are kept or mapped to integers depending
    on their number of unique values and average distance between them. Label
    columns are z-normalized as well. The dataset returns tensors for numerical,
    categorical, and text features, as well as labels and their corresponding
    masks for training or evaluation.

    Attributes:
        - data (pd.DataFrame): The input DataFrame containing the MRI data.
        - tokenizer (Tokenizer): The tokenizer used for encoding text features.
        - text_columns (List[str]): List of text columns to be tokenized.
        - datetime_columns_decompose (List[str]): List of datetime columns to be decomposed.
        - force_numerical_columns (List[str]): List of columns to be forced as numerical.
        - force_categorical_columns (List[str]): List of columns to be forced as categorical.
        - initial_numerical_columns (List[str]): List of initial numerical columns.
        - initial_categorical_columns (List[str]): List of initial categorical columns.
        - label_columns (List[str]): List of label columns to be z-normalized.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: Tokenizer,
        text_columns: List[str],
        datetime_columns_decompose: List[str],
        force_numerical_columns: List[str] = [],
        force_categorical_columns: List[str] = [],
        initial_numerical_columns: List[str] = [],
        initial_categorical_columns: List[str] = [],
        label_columns: List[str] = [],
    ) -> None:

        self.data = data
        self.text_columns = text_columns
        self.datetime_columns_decompose = datetime_columns_decompose
        self.force_numerical_columns = force_numerical_columns
        self.force_categorical_columns = force_categorical_columns
        self.initial_numerical_columns = initial_numerical_columns
        self.initial_categorical_columns = initial_categorical_columns
        self.label_columns = label_columns

        self.param_columns = self.data.columns[
            self.data.columns.str.endswith("_param")
        ].tolist()

        self.tokenizer = tokenizer

        # Encode the text columns using the provided tokenizer
        (
            self.text_tokens_pad_dict,
            self.text_mappings,
            self.text_tokens_df,
            self.non_nan_text_df,
            self.max_text_length,
        ) = encode_text_columns(
            df=self.data, text_columns=self.text_columns, tokenizer=self.tokenizer
        )

        # Decompose datetime columns
        (
            self.datetime_df,
            self.datetime_mappings,
            self.non_nan_datetime_df,
        ) = decompose_datetime_columns(
            df=self.data,
            datetime_columns_decompose=self.datetime_columns_decompose,
        )

        # Determine numerical and categorical features
        (
            self.numerical_columns,
            self.categorical_columns,
            self.mapping_categorical_columns,
        ) = determine_num_cat_features(
            df=self.data,
            initial_numerical_columns=self.initial_numerical_columns,
            initial_categorical_columns=self.initial_categorical_columns,
            datetime_df=self.datetime_df,
            force_numerical_columns=self.force_numerical_columns,
            force_categorical_columns=self.force_categorical_columns,
        )

        # Map categorical features
        (
            self.categorical_df,
            self.categorical_mappings,
            self.non_nan_categorical_df,
            self.categorical_min_val,
        ) = map_categorical_features(
            df=self.data,
            categorical_columns=self.categorical_columns,
            mapping_categorical_colummns=self.mapping_categorical_columns,
        )

        # Z-normalize numerical features
        (
            self.z_normalized_numerical_df,
            self.numerical_scalers,
            self.numerical_mappings,
            self.non_nan_numerical_df,
        ) = z_normalize_numerical_features(
            df=self.data,
            datetime_df=self.datetime_df,
            numerical_columns=self.numerical_columns,
        )

        # Z-normalize label columns
        (
            self.z_normalized_label_df,
            self.label_scalers,
            self.label_mappings,
            self.non_nan_label_df,
        ) = z_normalize_numerical_features(
            df=self.data, label_columns=self.label_columns
        )

        # Replace NaN values in the z-normalized DataFrames with 0
        self.z_normalized_numerical_df.fillna(0, inplace=True)
        self.z_normalized_label_df.fillna(0, inplace=True)
        self.categorical_df.fillna(0, inplace=True)

        # Get the vocabulary size from the tokenizer
        self.vocab_size_text = self.tokenizer.get_vocab_size()
        # Get the vocabulary size for categorical features, which should be the
        # maximal number of unique values across all categorical columns
        self.vocab_size_categorical = int(self.categorical_df.max().max()) + 1

        # Get the maximum length of the numerical features
        self.max_numerical_length = self.z_normalized_numerical_df.shape[1]
        # Get the maximum length of the categorical features
        self.max_categorical_length = self.categorical_df.shape[1]

    def __len__(self) -> int:
        """
        Get the length of the dataset, which is required for PyTorch Dataset compatibility.
        
        Returns:
            - int: The number of samples in the dataset.
        """
        
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        To make this a valid PyTorch Dataset, we need to implement the __getitem__
        method, which retrieves a single item from the dataset given an index.

        Parameters:
            - idx (int): Index of the item to retrieve.

        Returns:
        Tuple[torch.Tensor, ...]: A tuple containing:
            - idx_df (torch.Tensor): Tensor containing the original index from 
            - numerical_features (torch.Tensor): Tensor of numerical features for the given index.
            - text_tensor (torch.Tensor): Tensor of tokenized text features for the given index.
            - categorical_features (torch.Tensor): Tensor of categorical features for the given index.
            - label (torch.Tensor): Tensor of label values for the given index.
            - numerical_mask (torch.Tensor): Tensor mask indicating non-NaN numerical features.
            - text_attention_mask (torch.Tensor): Tensor mask indicating non-NaN text features.
            - categorical_mask (torch.Tensor): Tensor mask indicating non-NaN categorical features.
            - label_mask (torch.Tensor): Tensor mask indicating non-NaN label features.
        """
        # Get the original index from the DataFrame for the current index
        idx_df = self.data.index[idx]
        
        
        # Z-normalized numerical features
        numerical_features = torch.tensor(
            self.z_normalized_numerical_df.iloc[idx].values, dtype=torch.float32
        )

        # Text features: concatenate all tokenized columns into a single 1D tensor
        text_tokens = []
        non_nan_text_tokens = []
        # Iterate through each text column and append the tokenized ids and masks
        for col in self.text_tokens_df.columns:
            text_tokens.extend(self.text_tokens_df.iloc[idx][col])
            non_nan_text_tokens.extend(self.non_nan_text_df.iloc[idx][col])
        # Convert the list of token ids to a tensor
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        # Attention mask for text features
        text_attention_mask = torch.tensor(non_nan_text_tokens, dtype=torch.long)

        # Categorical features
        categorical_features = torch.tensor(
            self.categorical_df.iloc[idx].values, dtype=torch.long
        )

        # Z-normalized label features
        label = torch.tensor(
            self.z_normalized_label_df.iloc[idx].values, dtype=torch.float32
        )

        # Masks for numerical, categorical, and label features (1 for non-NaN, 0 for NaN)
        numerical_mask = torch.tensor(
            self.non_nan_numerical_df.iloc[idx].values, dtype=torch.long
        )
        categorical_mask = torch.tensor(
            self.non_nan_categorical_df.iloc[idx].values, dtype=torch.long
        )
        label_mask = torch.tensor(
            self.non_nan_label_df.iloc[idx].values, dtype=torch.long
        )

        return (
            idx_df,
            numerical_features,
            text_tensor,
            categorical_features,
            label,
            numerical_mask,
            text_attention_mask,
            categorical_mask,
            label_mask,
        )


### 📊 Visualization of Tensors ################################################
# 
# - 🔤 Text tensors:  
#   - NaN values are replaced with a special [NAN] token, and sequences are 
#     padded for consistent length. 
#   - Special characters are normalized before tokenization: `,` and `;` become [SEP], 
#       `_`, `%\`, `%/`, `%` becomes [SUBSEQ],`/` becomes [SUB], `+` becomes [AND]
#   - Numbered bracketed patterns such as `[/12]` or `[3]` are converted to 
#     `[REPEAT]12` and `[REPEAT]3` so repeated protocol fragments are handled consistently. 
#   - Attention masks distinguish valid tokens from padding.
# 
# - 🔢 Numerical tensors:  
#   NaN values were replaced with `0` to ensure compatibility with tensor  
#   operations and model input requirements.
# 
# - 🎭 Attention masks:  
#   For both text and numerical tensors, attention masks were created to indicate 
#   which values are valid (real) and which are placeholders (NaN or padding). 
#   This allows the transformer model to ignore padded or missing values during 
#   training and inference.


def plot_dataset_tensors(
    dataset: torch.utils.data.Dataset,
    idx: int = 0,
    kind_plot: str = "all",
    save_fig: str = None,
) -> go.Figure:
    """
    Plot the tensors for numerical, text, categorical, and label features
    from a dataset. This function visualizes the numerical features, text and
    and label features against their indices. Allows plotting all values, only
    masked values, or the mask itself.

    Parameters:
    dataset (torch.utils.data.Dataset): The dataset containing the tensors.
    idx (int): The index of the sample to plot.
    kind_plot (str): Type of plot to generate. Options:
        - "all": Plot all values.
        - "masked": Plot only masked values (valid entries).
        - "mask": Plot the mask itself.
    save_fig (str): Optional path to save the figure. If None, the figure is not saved.


    Returns:
    go.Figure: A Plotly figure object containing the subplots for numerical,
               text + categorical, and label features.
    """

    (   idx_df,
        numerical_tensor,
        text_tensor,
        categorical_tensor,
        label_tensor,
        numerical_attention_mask,
        text_attention_mask,
        categorical_attention_mask,
        label_attention_mask,
    ) = dataset[idx]

    # Convert tensors to numpy arrays
    numerical = numerical_tensor.numpy()
    text = text_tensor.numpy()
    categorical = categorical_tensor.numpy()
    label = label_tensor.numpy()
    numerical_mask = numerical_attention_mask.numpy().flatten()
    text_mask = text_attention_mask.numpy()
    categorical_mask = categorical_attention_mask.numpy()
    label_mask = label_attention_mask.numpy()

    # Prepare data based on kind_plot
    if kind_plot == "all":
        # Combine text and categorical features
        text_categorical = np.concatenate((text, categorical), axis=0)

        # Create titles for subplots
        sub_plot_1_title = f"Numerical ({idx} sample)"
        sub_plot_2_title = f"Text + Categorical Tokens ({idx} sample)"
        sub_plot_3_title = f"Label Features ({idx} sample)"

    elif kind_plot == "masked":
        # Filter out masked values
        numerical = numerical[numerical_mask == 1]
        text = text[text_mask == 1]
        categorical = categorical[categorical_mask == 1]
        label = label[label_mask == 1]

        # Combine text and categorical features
        text_categorical = np.concatenate((text, categorical), axis=0)

        # Create titles for subplots
        sub_plot_1_title = f"Numerical (Masked, {idx} sample)"
        sub_plot_2_title = f"Text + Categorical Tokens (Masked, {idx} sample)"
        sub_plot_3_title = f"Label Features (Masked, {idx} sample)"

    elif kind_plot == "mask":
        # Get the massks
        numerical = numerical_mask
        # Combine text and categorical masks
        text_categorical = np.concatenate((text_mask, categorical_mask), axis=0)
        label = label_mask

        # Create titles for subplots
        sub_plot_1_title = f"Numerical Mask ({idx} sample)"
        sub_plot_2_title = f"Text + Categorical Tokens Mask ({idx} sample)"
        sub_plot_3_title = f"Label Features Mask ({idx} sample)"

    else:
        raise ValueError("Invalid kind_plot. Choose from 'all', 'masked', or 'mask'.")

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            sub_plot_1_title,
            sub_plot_2_title,
            sub_plot_3_title,
        ],
        vertical_spacing=0.2,
    )

    # Numerical
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(numerical)),
            y=numerical,
            mode="markers",
            marker=dict(color="blue", size=8, opacity=0.7),
            name="Numerical Features",
            hovertemplate="Index: %{x}<br>Value: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Text + Categorical
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(text_categorical)),
            y=text_categorical,
            mode="markers",
            marker=dict(color="purple", size=8, opacity=0.7),
            name="Text + Categorical Features",
            hovertemplate="Index: %{x}<br>Value: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Label Features
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(label)),
            y=label,
            mode="markers",
            marker=dict(color="orange", size=10, opacity=0.7),
            name="Label",
            hovertemplate="Index: %{x}<br>Value: %{y}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Update layout
    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(title_text="Feature Index", row=1, col=1)
    fig.update_xaxes(title_text="Token Index", row=2, col=1)
    fig.update_xaxes(title_text="Label Index", row=3, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Token ID", row=2, col=1)
    fig.update_yaxes(title_text="Label Value", row=3, col=1)

    if save_fig:
        save_fig_formats(fig, save_fig)

    return fig


def plot_dataset_tensors_combined(
    dataset: torch.utils.data.Dataset,
    idx: int = None,
    kind_plot: str = "all",
    save_fig: str = None,
) -> go.Figure:
    """
    Plot the tensors for the lables int the dastaset with the applied
    mask against thir indices  Allows plotting all values, only
    masked values, or the mask itself.

    Parameters:
    dataset (torch.utils.data.Dataset): The dataset containing the tensors.
    idx (int, optional): The number of samples to process from the dataset. If None,
        process the entire dataset. Defaults to None.
    kind_plot (str): Type of plot to generate. Options:
        - "all": Plot all values.
        - "masked": Plot only masked values (valid entries).
        - "mask": Plot the mask itself.
    save_fig (str): Optional path to save the figure. If None, the figure is not saved.

    Returns:
    go.Figure: A Plotly figure object containing the plot for label features.
    """

    labels = []
    label_masks = []

    idx = idx if idx is not None else len(dataset)

    for i in tqdm(range(idx), desc="Processing dataset"):
        _, _, _, _, label_tensor, _, _, _, label_attention_mask = dataset[i]
        labels.append(label_tensor.numpy())
        label_masks.append(label_attention_mask.numpy())

    labels = np.array(labels)
    label_masks = np.array(label_masks)

    # Plot all labels with and without mask
    if kind_plot == "all":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(labels)),
                y=labels.flatten(),
                mode="markers",
                name="Label Values",
                line=dict(color="blue"),
            )
        )

    elif kind_plot == "masked":
        # Plot only masked values (valid entries)
        fig = go.Figure()
        valid_indices = np.where(label_masks > 0)[0]
        fig.add_trace(
            go.Scatter(
                x=valid_indices,
                y=labels[valid_indices].flatten(),
                mode="markers",
                name="Masked Label Values",
                line=dict(color="green"),
            )
        )

    elif kind_plot == "mask":
        # Plot the mask itself
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(label_masks)),
                y=label_masks.flatten(),
                mode="markers",
                name="Label Mask",
                line=dict(color="red"),
            )
        )

    else:
        raise ValueError("Invalid kind_plot value. Use 'all', 'masked', or 'mask'.")

    # Update layout
    fig.update_layout(
        title=f"Dataset Tensors - {kind_plot.capitalize()}",
        xaxis_title="Index",
        yaxis_title="Value",
        showlegend=True,
    )

    if save_fig:
        save_fig_formats(fig, save_fig)

    # Show the plot
    return fig