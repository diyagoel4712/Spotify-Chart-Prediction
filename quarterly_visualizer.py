import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from spotify_date_extractor import load_spotify_data, clean_and_prepare_data_quarter

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]

FEATURES = ["Danceability", "Energy", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Valence"]


def quarter_stats(spotify_data_csv):
    spotify_data_csv = spotify_data_csv.copy()

    spotify_data_csv["Quarter"] = spotify_data_csv["Quarter"].astype(str).str[-2:]
    spotify_data_csv["Quarter"] = pd.Categorical(spotify_data_csv["Quarter"], categories=QUARTER_ORDER, ordered=True)
    for feature in FEATURES:
        spotify_data_csv[feature] = pd.to_numeric(spotify_data_csv[feature], errors="coerce")

    spotify_data_csv = spotify_data_csv.dropna(subset=FEATURES + ["Quarter"])

    grouped = spotify_data_csv.groupby("Quarter", observed=True)

    mean_feature_per_quarter = grouped[FEATURES].mean().reindex(QUARTER_ORDER)
    spread_feature_per_quarter = grouped[FEATURES].std().reindex(QUARTER_ORDER, fill_value=0)
    count_songs_per_quarter = grouped.size().reindex(QUARTER_ORDER, fill_value=0)

    return mean_feature_per_quarter, spread_feature_per_quarter, count_songs_per_quarter


def plot_scaled_all_features(mean_feature_per_quarter):
    os.makedirs("spotify_final/final_plots/quarter", exist_ok=True)

    scaler_object = MinMaxScaler()
    scaled_features = pd.DataFrame(scaler_object.fit_transform(mean_feature_per_quarter[FEATURES]),columns=FEATURES,index=QUARTER_ORDER,)

    quarter_order_range = np.arange(len(QUARTER_ORDER))
    palette = sns.color_palette("husl", len(FEATURES))
    plt.figure(figsize=(18, 6), dpi=300)

    for each_color, each_feature in zip(palette, FEATURES):
        plt.plot(quarter_order_range,scaled_features[each_feature].to_numpy(),marker="o",linewidth=2.5,color=each_color,label=each_feature,)

    plt.xticks(quarter_order_range, QUARTER_ORDER, fontsize=16)
    plt.yticks(fontsize=18)
    plt.ylabel("Scaled Value (0 = Min, 1 = Max)", fontsize=18)
    plt.title("Relative Trends of all features by quarter", fontsize=18)
    plt.axhline(0.5, linestyle="--", color="gray", alpha=0.6)
    plt.grid(alpha=0.4)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=18, ncols=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)

    plt.tight_layout()
    out = "spotify_final/final_plots/quarter/all_features_scaled_by_quarter.png"
    plt.savefig(out)
    plt.close()

    logging.info(f"Saved: {out}")


def plot_scaled_single_feature(feature_name, mean_feature_per_quarter, spread_feature_per_quarter):
    os.makedirs("spotify_final/final_plots/quarter/features", exist_ok=True)

    # min-max scale this one feature
    feature_values = mean_feature_per_quarter[feature_name].values.reshape(-1, 1)
    scaler_object = MinMaxScaler()
    scaled_features = scaler_object.fit_transform(feature_values).flatten()

    quarter_order_range = np.arange(len(QUARTER_ORDER))

    fig, scaled_plots = plt.subplots(2, 1, figsize=(12, 10), dpi=300)

    scaled_plots[0].plot(quarter_order_range,scaled_features,marker="o",linewidth=3,color="hotpink",)
    scaled_plots[0].axhline(0.5, linestyle="--", color="gray", alpha=0.7)
    scaled_plots[0].set_xticks(quarter_order_range)
    scaled_plots[0].set_xticklabels(QUARTER_ORDER)
    scaled_plots[0].set_ylabel("Scaled Value (0 = Min, 1 = Max)")
    scaled_plots[0].set_title(f"{feature_name} – Min-Max Scaled by Quarter")

    peak_plot = np.argmax(scaled_features)
    peak_quarter = QUARTER_ORDER[peak_plot]
    peak_val = scaled_features[peak_plot]

    scaled_plots[0].annotate(f"Peak: {peak_val:.2f}\n{peak_quarter}",xy=(peak_plot, peak_val),xytext=(peak_plot + 0.3, peak_val + 0.15),arrowprops=dict(arrowstyle="->", color="hotpink"),color="hotpink",fontsize=11)
    avg_vals = mean_feature_per_quarter[feature_name]
    errors = spread_feature_per_quarter[feature_name]

    scaled_plots[1].bar(QUARTER_ORDER,avg_vals,yerr=errors,color="hotpink",edgecolor="black",capsize=6,)
    scaled_plots[1].set_title(f"{feature_name} – Average by Quarter")
    scaled_plots[1].set_ylabel("Average Value")

    for index, value in enumerate(avg_vals):
        scaled_plots[1].text(index,value + errors.iloc[index] + 0.01,f"{value:.4f}",ha="center",fontsize=9,)

    plt.tight_layout()
    out = f"spotify_final/final_plots/quarter/features/{feature_name.lower()}_by_quarter.png"
    plt.savefig(out)
    plt.close()

    logging.info(f"Saved: {out}")


def plot_all_scaled_single_features(mean_feature_per_quarter, spread_feature_per_quarter):
    for feature in FEATURES:
        plot_scaled_single_feature(feature, mean_feature_per_quarter, spread_feature_per_quarter)


if __name__ == "__main__":
    spotify_data_csv = load_spotify_data("spotify_final/Spotify_Dataset_V3_visualizer.csv")
    spotify_data_csv = clean_and_prepare_data_quarter(spotify_data_csv)

    mean_feature_per_quarter, spread_feature_per_quarter, count_songs_per_quarter = quarter_stats(spotify_data_csv)

    plot_scaled_all_features(mean_feature_per_quarter)
    plot_all_scaled_single_features(mean_feature_per_quarter, spread_feature_per_quarter)
