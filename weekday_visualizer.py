import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from spotify_date_extractor import load_spotify_data, clean_and_prepare_data_weekday


logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

WEEK_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FEATURES = ["Danceability", "Energy", "Loudness","Speechiness", "Acousticness", "Instrumentalness", "Valence"]

def weekday_stats(spotify_data_csv):
    spotify_data_csv = spotify_data_csv.copy()
    for f in FEATURES:
        spotify_data_csv[f] = pd.to_numeric(spotify_data_csv[f], errors="coerce")

    group_days = spotify_data_csv.groupby("day_of_week", observed=True)
    mean_feature_per_day = group_days[FEATURES].mean().reindex(WEEK_ORDER)
    spread_feature_per_day = group_days[FEATURES].std().reindex(WEEK_ORDER, fill_value=0)
    count_songs_per_day = group_days.size().reindex(WEEK_ORDER, fill_value=0)

    return mean_feature_per_day, spread_feature_per_day, count_songs_per_day

#min-max-scaled for all features
def plot_scaled_all_features(mean_feature_per_day):
    os.makedirs("spotify_final/final_plots", exist_ok=True)

    scaler_object = MinMaxScaler()
    scaled_features = pd.DataFrame(scaler_object.fit_transform(mean_feature_per_day[FEATURES]),columns=FEATURES,index=WEEK_ORDER)
    week_order_range = np.arange(len(WEEK_ORDER))
    colors = sns.color_palette("husl", len(FEATURES))
    plt.figure(figsize=(18, 6), dpi=300)
    
    for each_color, each_feature in zip(colors, FEATURES):
        plt.plot(week_order_range, scaled_features[each_feature], marker="o", linewidth=2.5, color=each_color, label=each_feature)

    plt.xticks(week_order_range, WEEK_ORDER, rotation=45, fontsize=16)
    plt.yticks(fontsize=18)
    plt.title("Relative Trends of all features by day", fontsize=18)
    plt.ylabel("Scaled Value (0 = Min, 1 = Max)", fontsize=18)
    plt.axhline(0.5, linestyle="--", color="black")
    plt.grid(alpha=0.4)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=18, ncols=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    plt.tight_layout()
    plt.savefig("spotify_final/final_plots/all_features_scaled_by_day.png")
    plt.close()
    
    logging.info("Saved: scaled all-features graph")

def plot_scaled_single_feature(feature_name, mean_feature_per_day , spread_feature_per_day):
    os.makedirs("spotify_final/final_plots/features", exist_ok=True)
    feature_vals = mean_feature_per_day[feature_name].values.reshape(-1, 1)
    scaler_obj = MinMaxScaler()
    #is ravel beter or flatten?
    scaled_features = scaler_obj.fit_transform(feature_vals).flatten()  
    week_order_range = np.arange(len(WEEK_ORDER))
    fig, scaled_plots = plt.subplots(2, 1, figsize=(16, 12), dpi=300)
    scaled_plots[0].plot(week_order_range, scaled_features, marker="o", color="hotpink", linewidth=3)
    scaled_plots[0].axhline(0.5, linestyle="--", color="gray")





    scaled_plots[0].set_xticks(week_order_range)
    scaled_plots[0].set_xticklabels(WEEK_ORDER, rotation=45)
    scaled_plots[0].set_ylabel("Scaled Value (0 = Min, 1 = Max)")
    scaled_plots[0].set_title(f"{feature_name} – Min-Max Scaled Trend by Day", fontsize=18)
#    # Annotate peak###### good for visibility
    peak_plot = np.argmax(scaled_features)
    peak_day = WEEK_ORDER[peak_plot]
    peak_val = scaled_features[peak_plot]

    scaled_plots[0].annotate(
        f"Peak: {peak_val:.2f}\n{peak_day}",
        xy=(peak_plot, peak_val),
        xytext=(peak_plot + 0.2, peak_val + 0.15),  #####find a good place to fit the text properly #trial and error 
        arrowprops=dict(arrowstyle="->", color="hotpink"),
        color="hotpink",
        fontsize=12)

    avg_vals = mean_feature_per_day[feature_name]
    errors = spread_feature_per_day[feature_name]

    scaled_plots[1].bar(WEEK_ORDER,avg_vals, yerr=errors, color="hotpink", edgecolor="black",capsize=6)
    scaled_plots[1].set_title(f"{feature_name}Average by Play Day", fontsize=18)
    scaled_plots[1].set_ylabel("Average Value")

    for i, val in enumerate(avg_vals):
        scaled_plots[1].text(i, val + errors.iloc[i] + 0.01, f"{val:.4f}", ha="center", fontsize=10)


    plt.tight_layout()
    out = f"spotify_final/final_plots/features/{feature_name.lower()}_by_play_day.png"
    plt.savefig(out)
    plt.close()

    logging.info(f"Saved: {out}")



#z-score for all features - not using this as min-max scaler gives a better visual
# def plot_all_zscores(mean_feature_per_day ):
#     os.makedirs("spotify_final/final_plots", exist_ok=True)

#     z_scores = mean_feature_per_day.apply(zscore)
#     week_order_range = np.arange(len(WEEK_ORDER))
#     palette = sns.color_palette("husl", len(FEATURES))

#     plt.figure(figsize=(16, 8), dpi=300)
#     for color, f in zip(palette, FEATURES):
#         plt.plot(week_order_range, z_scores[f], marker="o", linewidth=2.5, color=color, label=f)

#     plt.xticks(week_order_range, WEEK_ORDER, rotation=45)
#     plt.title("Z-scores of all features by day")
#     plt.ylabel("Average Z-Score")
#     plt.axhline(0, linestyle="--", color="gray")
#     plt.grid(alpha=0.4)
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig("spotify_final/final_plots/all_features_zscore_by_day.png")
#     plt.close()
#     logging.info("Saved: all-features zscore graph")

#plot single feature z-score for each day
# def plot_single_feature(feature_name, mean_feature_per_day , spread_feature_per_day ):
#     os.makedirs("spotify_final/final_plots/features", exist_ok=True)

#     week_order_range = np.arange(len(WEEK_ORDER))
#     z_scores = zscore(mean_feature_per_day [feature_name])

#     fig, z_score_graph = plt.subplots(2, 1, figsize=(16, 12), dpi=300)

#     # ---- Top: Z-SCORE ----
#     z_score_graph[0].plot(week_order_range, z_scores, marker="o", color="hotpink", linewidth=3)
#     z_score_graph[0].axhline(0, linestyle="--", color="gray")
#     z_score_graph[0].set_xticks(week_order_range)
#     z_score_graph[0].set_xticklabels(WEEK_ORDER, rotation=45)
#     z_score_graph[0].set_ylabel("Z-Score")
#     z_score_graph[0].set_title(f"{feature_name} – Z-Score by Play Day")

#     # Annotate Peak
#     peak_i = int(np.argmax(np.asarray(z_scores)))
#     peak_day = WEEK_ORDER[peak_i]
#     peak_val = z_scores[peak_i]

#     z_score_graph[0].annotate(
#         f"Peak: +{peak_val:.2f}σ\n{peak_day}",
#         xy=(peak_i, peak_val),
#         xytext=(peak_i+0.3, peak_val+0.5),
#         arrowprops=dict(arrowstyle="->", color="hotpink"),
#         color="hotpink"
#     )

#     # ---- Bottom: Average Bar ----
#     avg_vals = mean_feature_per_day [feature_name]
#     errors = spread_feature_per_day [feature_name]

#     z_score_graph[1].bar(WEEK_ORDER, avg_vals, yerr=errors, color="#ff99aa", edgecolor="black", capsize=6)
#     z_score_graph[1].set_title(f"{feature_name} – Average by Play Day")
#     z_score_graph[1].set_ylabel("Average Value")

#     for i, val in enumerate(avg_vals):
#         z_score_graph[1].text(i, val + errors.iloc[i] + 0.01, f"{val:.4f}", ha="center")
#     plt.tight_layout()


#     out = f"spotify_final/final_plots/features/{feature_name.lower()}_by_play_day.png"
#     plt.savefig(out)
#     plt.close()

#     logging.info(f"Saved: {out}")

#plot all single features z-score 
# def plot_all_single_features(mean_feature_per_day , spread_feature_per_day ):
#     for f in FEATURES:
#         plot_single_feature(f, mean_feature_per_day , spread_feature_per_day )

def plot_all_scaled_single_features(mean_feature_per_day , spread_feature_per_day):
        for feature in FEATURES:
            plot_scaled_single_feature(feature, mean_feature_per_day , spread_feature_per_day)

if __name__ == "__main__":
    spotify_data_csv = load_spotify_data("spotify_final/Spotify_Dataset_V3_visualizer.csv")
    spotify_data_csv = clean_and_prepare_data_weekday(spotify_data_csv)

    mean_feature_per_day , spread_feature_per_day , count_songs_per_day  = weekday_stats(spotify_data_csv)

    plot_scaled_all_features(mean_feature_per_day )
    #plot_all_scaled_single_features(mean_feature_per_day , spread_feature_per_day)

#    plot_all_zscores(mean_feature_per_day )
    #plot_all_single_features(mean_feature_per_day , spread_feature_per_day ) 

