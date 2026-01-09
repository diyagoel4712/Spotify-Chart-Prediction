Built-in required packages/libraries:
- os
- logging
- pickle
- time

Third-party required packages/libraries:
- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- xgboost
- scipy
- tqdm
- pytorch

The dataset (Spotify_Dataset_V3.csv) needs to be in the same folder as the main codebase. Similarly, the modified dataset (Spotify_Dataset_V3_visualizer.csv) needs to be in the same folder as the visualizer programs and spotify_date_extractor.

-------------------------------------
Executions:

Main codebase: start a new server and run "main_codebase.ipynb" in the strict cell order presented in the notebook. Most simulation cells take a long time to execute. That is why we have set up a data file system for some of them (the les heavy ones), thanks to which you will be able to run the plotting cells of all the models without simulating them (except LSTM and MLPReg). For that purpose, the data files need to be in a folder named model_results in the same directory as the notebook (as attached in the .zip file).
For visulaizations:
Place Spotify_Dataset_V3_visualizer.csv alongside the visualizer scripts and spotify_date_extractor.py. Run any visualizer from the project root, e.g. python monthly_visualizer.py to generate month plots, python weekday_visualizer.py for weekdays, python weeks_of_year_visualizer.py for week numbers, python quarterly_visualizer.py for quarters. Each script calls the spotify_date_extractor functions to loads the CSV, cleans dates, computes summary stats, and calls functions to generate PNGs under spotify_final/final_plots/ (feature-specific plots go into the matching subfolder). If you want all single-feature plots too, uncomment the plot_all_scaled_single_features() call near the bottom of the desired script before running.



# Spotify Chart Prediction

This repository contains a machine learning project for predicting Spotify chart performance using temporal and audio feature analysis. The codebase implements multiple models and extensive visualizations of Spotify datasets.

## Overview

The project uses two large datasets:
- `Spotify_Dataset_V3.csv`: Primary dataset with Spotify track features.
- `Spotify_Dataset_V3_visualizer.csv`: Modified dataset for temporal visualizations.

Models include XGBoost, sklearn ensembles, LSTM, and MLPRegressors, with precomputed results for most to avoid long simulations.

## Dependencies

### Built-in
- `os`
- `logging`
- `pickle`
- `time`

### Third-party
pandas
numpy
matplotlib
seaborn
sklearn
xgboost
scipy
tqdm
pytorch

Install with `pip install -r requirements.txt` 

## Setup

1. Place `Spotify_Dataset_V3.csv` in the project root.
2. Place `Spotify_Dataset_V3_visualizer.csv` in the root.
3. Create `model_results/` folder with precomputed simulation outputs (for non‑LSTM/MLPReg plots).

## Usage

### Main analysis
Run `main_codebase.ipynb` in strict cell order. Simulation cells are computationally intensive; use `model_results/` data for plotting where available.

### Visualizations
Run visualizer scripts from project root:

```bash
python monthly_visualizer.py     # Monthly plots
python weekday_visualizer.py     # Weekday plots
python weeks_of_year_visualizer.py  # Week of year plots
python quarterly_visualizer.py   # Quarterly plots


Outputs go to spotify_final/final_plots/ (aggregate) and feature subfolders (single‑feature). Uncomment plot_all_scaled_single_features() for additional plots.

Each script uses spotify_date_extractor.py for date cleaning and summary statistics.

## Project structure

├── main_codebase.ipynb              # Core ML models and analysis
├── Spotify_Dataset_V3.csv           # Primary data (LFS)
├── Spotify_Dataset_V3_visualizer.csv # Visualizer data (LFS)
├── model_results/                   # Precomputed model outputs
├── monthly_visualizer.py
├── weekday_visualizer.py
├── weeks_of_year_visualizer.py
├── quarterly_visualizer.py
├── spotify_date_extractor.py        # Date processing utilities
└── spotify_final/
    └── final_plots/                 # Generated plots


## Notes

LSTM and MLPReg models require full simulation (no precomputed data).

Large CSVs are tracked via Git LFS.

Visualizers generate PNGs automatically.











