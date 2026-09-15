# 🎯 General

## 🔍 Overview
This repository provides a deep learning framework for analyzing <br>
and predicting the energy and power consumption of MRI scanners. <br>
The main focus is the Hierarchical Multimodal Transformer, which <br>
learns complex relationships between MRI scanner operation, sequence <br>
characteristics, and energy consumption.

The model uses scanner and energy data to predict energy demand and <br>
analyze which features have the greatest influence on energy consumption <br>
using Transformer attention mechanisms.

### 🚀 **Research Impact**
The deep learning approach provides a data-driven method to identify <br>
energy-intensive MRI sequences and scanner operating conditions. <br>
The results can support sequence optimization, workflow management, <br>
and standardized energy-efficiency benchmarking.

### 🔬 **Methodological Innovations**
- **🤖 Hierarchical Multimodal Transformer**: Advanced deep <br>
learning model based on the transformer architecture for identifying feature <br>
importance and complex interactions in energy consumption patterns
    - **📊 Attention Mechanisms**: Enable identification of most critical features <br>
    for energy consumption prediction and analysis
- **🔄 Reverse Analysis Approach**: Instead of pre-selecting factors, we <br>
collected all observable variables and identified which meaningfully influence the system


## 🔄 Workflow  
- This a general instruction which files to execute in which order

1. **📂 Check Raw Data**
    - The raw data should be included in the following folder: <br>
    `~/DL_Energy_Power_regression/Data storage Siemens/Raw data`
    - ✂️ Manually sort the raw data into:
        - 🧲 **MRI data:** `~/DL_Energy_Power_regression/Data storage Siemens/MRI data`
        - 💻 **CT data:** `~/DL_Energy_Power_regression/Data storage Siemens/CT data`
    - 🏷️ **Tip**: CT data folders always start with **ukt_9999<number>** for easy identification.

2. **🔧 Preprocessing**
    - ⚡ **Determine power thresholds** for scanner modes (scanning, idle, ecopowermode): <br>
        - 📝 Run: `~/DL_Energy_Power_regression/Data preprocessing/Determine power threshold_opt/Determine_power_threshold.ipynb`
        - ✍️ Manually set the boundaries based on the power output.
        - 💾 Create a CSV file named `powerboundries_scanner_mapping.csv` and save it in:  <br>
       `~/DL_Energy_Power_regression/Data storage Siemens/MRI data/powerboundries_scanner_mapping.csv`
    - 🐍 **Main preprocessing**:
        - Run the python file: <br>
        `~/DL_Energy_Power_regression/Data preprocessing/Data preprocessing_opt/data_preprocessing.py`
    - 💾 Output: Data will be saved in `~DL_Energy_Power_regression/saved_datasets/preprocessing`

3. **🧠 Run the Transformer**
    - Train the Transformer to analyze the attention scores of the features
    - Train and test the Transformer by running python file preferably on a cluster:<br>
    `~/DL_Energy_Power_regression/Deep Learning/Transformer/Transformer_opt/transformer.py`
    - 💾 After training, `attention_predictions_mappings.pt` will be saved in: <br>
    `~\DL_Energy_Power_regression_MT\Deep Learning\Transformer_opt\model_output`
    - 📁 Move the `attention_predictions_mappings.pt` to `~\DL_Energy_Power_regression_MT\saved_DL_model_outputs\Transformer`

## 📄 License
This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0). <br>
See the LICENSE file for license rights and limitations.