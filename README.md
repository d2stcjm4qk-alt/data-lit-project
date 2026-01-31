# Do Seasonal Patterns Affect Road Fatalities Differently in Germany and the United Kingdom?

## Project Overview
This repository contains the data processing and analysis code for a university project
investigating seasonal patterns in road traffic fatalities in Germany and the United Kingdom.
The study examines whether temporal variations in fatal road accidents differ between the two
countries and explores potential seasonal effects.

This project was completed as part of the **Data Literacy (MSc)** course at the **University of Tübingen**.

---

## Data
The analysis is based on official road traffic accident and fatality data for Germany and the
United Kingdom.

- **Geographic scope:** Germany and the United Kingdom  
- **Temporal scope:** 2024  
- **Data type:** Observational, time-series data  
- **Source:** Official national road safety statistics (see report for full references)

Raw data are stored in `data/raw/`, while cleaned and processed datasets are saved in
`data/preprocessed/`. As several datasets exceed GitHub’s file size limits, find more information about the datasets [here](https://github.com/d2stcjm4qk-alt/data-lit-project/blob/main/data/README.md).

---

## Methods
The methodological details are described fully in the accompanying report. In brief, the analysis includes:

- Data cleaning and harmonization across countries  
- Temporal aggregation and seasonal decomposition  
- Descriptive and comparative statistical analysis  

---

## Repository Structure
```text
data-lit-project/
├── data/                # Raw and processed datasets
├── notebooks/           # Exploratory analysis and visual inspection
├── src/                 # Python scripts for data loading, preprocessing, and analysis
├── results/             # Generated figures and summary tables
├── report/              # Final writte
```
---

## Environment

The analysis was conducted using **Python 3.11**.  
All required packages are listed in `requirements.txt` and can be installed via:

```bash
pip install -r requirements.txt
```
---

## Code Pipeline
1.  Download all datasets to corresponding data folder structure
2.  Run all preprocessing notebooks
3.  Run [regional_merge.py](https://github.com/d2stcjm4qk-alt/data-lit-project/blob/main/src/regional_merge.py)
4.  Run [traffic_exposure_harmonization.py](https://github.com/d2stcjm4qk-alt/data-lit-project/blob/main/src/traffic_exposure_harmonization.py)
5.  Now you can run all anaylsis scripts of the `src` folder. 
