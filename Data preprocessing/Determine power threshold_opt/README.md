# 🧲 Inferring MRI Scanner Modes from Power Data

---

## 🚦 Scanner Modes

MRI scanners operate in several distinct modes:
- 🌱 **Eco-Power Mode**: Energy-saving mode, typically active during the night.
- 😴 **Idle Mode**: The scanner is powered but not actively scanning, often right after a scan.
- 🧲 **Scanning Mode**: The scanner is actively acquiring images.

---

## ❓ Challenge

- The raw data **does not explicitly indicate** the operational mode of the MRI scanners.
- To **infer the current mode**, we must analyze the **Total Active Power (kW)** over time.

---

## 📈 Approach

1. **Analyze the Total Active Power (kW)**  
   - Examine the power consumption time series for each scanner over a month.
   - Identify characteristic power patterns for each mode.

2. **Mode Inference**  
   - 🌙 **Eco-Power Mode**: Expected during nighttime hours (lowest power usage).
   - ⏱️ **Idle Mode**: Typically follows a scanning session (intermediate power usage).
   - 🧲 **Scanning Mode**: Detected by high power consumption during active scans.

3. **Threshold Determination**  
   - Establish power thresholds that reliably distinguish between the different modes.
   - These thresholds are essential for downstream data processing and analysis.

---

## ⚠️ Important

> **This step must be completed _before_ running the main data preprocessing pipeline!**  
> The preprocessing workflow requires a `powerboundries_scanner_mapping.csv` file  
> in the data directory, containing the determined power thresholds for each scanner.


---
## Explanations of Files and Folders

- **Determine power threshold.ipynb**: Main notebook for determining scanner power thresholds.
- **plots**: Directory for saved plots generated during threshold analysis.
- **duckdbf_files**: Directory for DuckDB database setup and storage.
---
