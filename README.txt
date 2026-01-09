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
