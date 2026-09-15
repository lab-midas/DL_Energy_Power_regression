# ⚙️ Data Preprocessing

- You can perform preprocessing using either the Jupyter notebook or the Python script.
- The Python script is especially suitable for running on a computing cluster via an `sbatch` job file.
- Using the script enables automated, large-scale processing without manual intervention.
> **Note:** Before running the preprocessing, make sure to execute the <br>
   `Determine power threshold_.ipynb` notebook to generate the required power threshold values.


---
## Explanations of Files and Folders

- **data_preprocessing.py**: Main Python script for data preprocessing.
- **data_preprocessing.ipynb**: Main Notebook for data preprocessing
  > Both the script and notebook contain the same content.
- **data_preprocessing.sbatch**: Job submission file for running preprocessing <br>
  on a computing cluster.
- **logs**: Folder containing SLURM log files from cluster jobs.
- **old_files**: Archive of previous versions and deprecated scripts.
- **duckdbf_files**: Directory for DuckDB database setup and storage.
---

## 🗂️ Data Structure

The folder structure for the data is organized as follows:

```plaintext
📂 Data storage Siemens
├── 📂 MRI data
│   ├── 📂 single_MRI_machine_1
│   │   ├── 📂 2024-12-11
│   │   │   ├── 📄 Scanner.csv
│   │   │   ├── 📄 Examinations.csv
│   │   │   ├── 📄 Measurements.csv
│   │   │   ├── 📄 Events.csv
│   │   │   ├── 📄 ProtocolParameters.csv
│   │   │   ├── 📄 ProtocolParametersDescription.csv
│   │   │   ├── 📄 powerdata_1.csv
│   │   │   └── ...  
│   │   ├── 📂 2024-12-12
│   │   │   ├── 📄 Scanner.csv
│   │   │   ├── 📄 Examinations.csv
│   │   │   ├── 📄 Measurements.csv
│   │   │   ├── 📄 Events.csv
│   │   │   ├── 📄 ProtocolParameters.csv
│   │   │   ├── 📄 ProtocolParametersDescription.csv
│   │   │   ├── 📄 powerdata_1.csv
│   │   │   └── ...  
│   ├── 📂 single_MRI_machine_2
│   │   ├── 📂 2024-12-11
│   │   │   ├── 📄 Scanner.csv
│   │   │   ├── 📄 Examinations.csv
│   │   │   ├── 📄 Measurements.csv
│   │   │   ├── 📄 Events.csv
│   │   │   ├── 📄 ProtocolParameters.csv
│   │   │   ├── 📄 ProtocolParametersDescription.csv
│   │   │   ├── 📄 powerdata_2.csv
│   │   │   └── ...  
│   │   └── ...
│   └── ...
├── 📂 CT data
├── 📂 Playground
├── 📂 Raw data
...
```

## 📁 Explanation of Folder Structure

### **Raw Data** 🗄️  
- Contains all the data downloaded by running the *Half Automated Data Pulling Pipeline*.  
- Includes both MRI and CT data. The corresponding data is manually copied into <br>
the `MRI data` and `CT data` folders.  

### **MRI Data** 🧲  
- Contains subfolders for each MRI machine (e.g., `single_MRI_machine_1`, `single_MRI_machine_2`).  
- Each machine folder contains subfolders for each date (e.g., `2024-12-11`, `2024-12-12`).  
- Each date folder contains the corresponding data and CSV files.  

### **CT Data** 🩻  
- Similar structure to MRI data, but for CT machines.  

### **Test data** 🛠️  
- Contains a subsample of data used for debugging and testing the data processing pipeline.  

---

## 📄 Explanation of CSV Files

- **`Scanner.csv`**: General information about the scanner and MRI machines.  
- **`Examinations.csv`**: Details about the examinations performed on the MRI machines.  
- **`Measurements.csv`**: Information about individual measurements during the examinations.  
- **`Events.csv`**: Logs of all events, mapping examinations and measurements.  
- **`ProtocolParameters.csv`**: Parameters used in each measurement (abbreviated format).  
- **`ProtocolParametersDescription.csv`**: Explanation of parameter abbreviations used in `ProtocolParameters.csv`.  
- **`powerdata_*`**: Data collected from the powermeters at a 1 Hz sampling rate.  
  - Each scanner is connected to a specific powermeter, and the filename reflects this association.  
  - The filename always starts with `powerdata_`, followed by the name of the <br>
  corresponding powermeter (e.g., `powerdata_Powermeter1.csv`, `powerdata_Powermeter2.csv`).  
  - These files contain detailed power and energy consumption data for the scanner.
---

## 🗂️ Structure of the Merged DataFrame

The resulting DataFrame from the MRI data preprocessing pipeline is <br>
**hierarchical** in nature, reflecting the real-world relationships between <br>
scanners, examinations, and measurements:

- **Scanner** ⟶ **Examinations** ⟶ **Measurements** ⟶ **Parameters**

This means:
- Each **scanner** can have multiple **examinations**.
- Each **examination** can have multiple **measurements**.
- Each **measurement** is associated with multiple **scanning parameters**.
- Additional information about daily energy consumption is included at the scanner level.

### 🔖 Column Suffixes Explained

To keep the merged DataFrame organized and to clarify the origin of each column, <br>
suffixes are used:

| Suffix      | Description                                      |
|-------------|--------------------------------------------------|
| `_scan`     | Information about the **scanner** (e.g., model, serial number) |
| `_exam`     | Information about the **examinations** (e.g., examination date) |
| `_meas`     | Information about the **measurements** (e.g., measurement time, results) |
| `_param`    | Information about the **scanning parameters** (e.g., protocol settings, technical details) |
| `_energy`   | **Daily energy consumption** and power-related metrics (aggregated per scanner per day) |

This structure allows for comprehensive analysis, enabling you to trace each <br>
measurement back to its examination and scanner, while also incorporating relevant <br>
technical and energy data.