# 🧹 Data Preprocessing

This stage transforms the raw data into a structured, analysis-ready format.  
It involves inferring the operational modes of each scanner and merging all <br>
relevant CSV files into a single, coherent hierarchical DataFrame.

---

## 🗂️ Structure of the Merged DataFrame

The resulting DataFrame from the MRI data preprocessing pipeline is **hierarchical**, <br>
reflecting real-world relationships:

- **Scanner** ⟶ **Examinations** ⟶ **Measurements** ⟶ **Parameters**

This means:
- Each **scanner** can have multiple **examinations**.
- Each **examination** can have multiple **measurements**.
- Each **measurement** is associated with multiple **scanning parameters**.
- Additional information about daily energy consumption is included at the scanner level.

### 🔖 Column Suffixes Explained

To keep the merged DataFrame organized and clarify the origin of each column, suffixes are used:

| Suffix      | Description                                      |
|-------------|--------------------------------------------------|
| `_scan`     | Information about the **scanner** (e.g., model, serial number) |
| `_exam`     | Information about the **examinations** (e.g., examination date) |
| `_meas`     | Information about the **measurements** (e.g., measurement time, results) |
| `_param`    | Information about the **scanning parameters** (e.g., protocol settings, technical details) |
| `_energy`   | **Daily energy consumption** and power-related metrics (aggregated per scanner per day) |

This structure enables comprehensive analysis, allowing you to trace each <br>
measurement back to its examination and scanner, while also incorporating <br>
relevant technical and energy data.

---

## 📁 Data Location

- **Data Directory**:  
  `MRI_data_analysis/Data storage Siemens/Pipeline-V2`

- **Raw Data** 🗄️  
  - Contains all data downloaded using the Half Automated Data Pulling Pipeline  
    (`Half automated data pulling pipeline/ukt-dataset-download.sh`).
  - Includes both MRI and CT data, which are manually organized into the <br>
    `MRI data` and `CT data` folders.

- **MRI Data** 🧲  
  - Organized by MRI machine (e.g., `single_MRI_machine_1`, `single_MRI_machine_2`).
  - Each machine folder contains subfolders for each date (e.g., `2024-12-11`), <br>
    which in turn contain the relevant data and CSV files.

- **CT Data** 🩻  
  - Follows a similar structure as the MRI data, but for CT machines.

- **Test Data** 🛠️  
  - Contains a subset of MRI data for testing and debugging purposes.

---

## 🏃‍♂️ Preprocessing Workflow

> *Note: This workflow focuses exclusively on the MRI data in `MRI_data_analysis/Data storage Siemens/Pipeline-V2/MRI data`.*

1. **Determine Power Thresholds**  
   - Before running the main preprocessing pipeline, you must first identify <br>
   the power thresholds that distinguish different MRI scanner modes.
   - Open and run: `MRI_data_analysis/Data preprocessing/Pipeline-V2/Determine power threshold_opt/Determine_power_threshold.ipynb`
   - Manually infer the power boundaries and save them as a CSV file.
   - Place the resulting `powerboundries_scanner_mapping.csv` in: <br>
     `MRI_data_analysis/Data storage Siemens/Pipeline-V2/MRI data/`

2. **Run the Preprocessing**  
   - You can preprocess the data using either the Jupyter notebook or the Python script:
     - **Jupyter Notebook**:  
       Ideal for interactive exploration, debugging, and step-by-step inspection <br>
       of the preprocessing steps.
    - **Python Script**:  
    Best suited for automated, large-scale processing—ideal for use on a computing cluster. <br>  
    To run the script as a batch job, use the provided Slurm file: <br>   
    `MRI_data_analysis/Data preprocessing/Pipeline-V2/Data preprocessing_opt/data_preprocessing.sbatch`


---
### 🧲 Why Infer Scanner Modes from Power Data?

MRI scanners operate in several modes, each with distinct power consumption patterns:
- 🌱 **Eco-Power Mode**: Energy-saving, typically at night.
- 😴 **Idle Mode**: Scanner is on but not scanning, often after a scan.
- 🧲 **Scanning Mode**: Active image acquisition.

The raw data **does not explicitly label** these modes.  
To accurately preprocess and analyze the data, we must infer the current mode by analyzing the **Total Active Power (kW)** over time.

### 🧠 How Are Modes Inferred?

1. **Analyze Power Consumption**  
   - Review the power time series for each scanner over a month.
   - Identify characteristic patterns for each mode.

2. **Assign Modes Based on Thresholds**  
   - 🌙 Eco-Power: Lowest power, usually at night.
   - ⏱️ Idle: Intermediate power, typically after scans.
   - 🧲 Scanning: High power during active scans.

3. **Set Thresholds**  
   - Define power thresholds that reliably separate these modes.
   - These thresholds are critical for all downstream processing and analysis.

