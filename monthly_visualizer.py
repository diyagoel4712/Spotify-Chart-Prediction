import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from spotify_date_extractor import load_spotify_data, clean_and_prepare_data_month

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

MONTH_ORDER = ["January", "February", "March", "April", "May", "June","July", "August", "September", "October", "November", "December"]

FEATURES = ["Danceability","Energy","Loudness","Speechiness", "Acousticness", "Instrumentalness", "Valence"]


def month_stats(spotify_data_csv):
    spotify_data_csv = spotify_data_csv.copy()
    spotify_data_csv["Month"] = spotify_data_csv["Month"].astype(str).str.title()
    spotify_data_csv["Month"] = pd.Categorical(spotify_data_csv["Month"], categories=MONTH_ORDER, ordered=True)
    for f in FEATURES:
        spotify_data_csv[f] = pd.to_numeric(spotify_data_csv[f], errors="coerce")

    spotify_data_csv = spotify_data_csv.dropna(subset=FEATURES + ["Month"])
    group_months = spotify_data_csv.groupby("Month", observed=True)
    mean_feature_per_month = group_months[FEATURES].mean().reindex(MONTH_ORDER)
    spread_feature_per_month = group_months[FEATURES].std().reindex(MONTH_ORDER, fill_value=0)
    count_songs_per_month = group_months.size().reindex(MONTH_ORDER, fill_value=0)

    return mean_feature_per_month, spread_feature_per_month, count_songs_per_month

def plot_scaled_all_features(mean_feature_per_month):
    os.makedirs("spotify_final/final_plots/month", exist_ok=True)

    scaler_obj = MinMaxScaler()
    scaled_features = pd.DataFrame(scaler_obj.fit_transform(mean_feature_per_month[FEATURES]),columns=FEATURES,index=MONTH_ORDER,)

    month_order_range = np.arange(len(MONTH_ORDER))
    colors = sns.color_palette("husl", len(FEATURES))

    plt.figure(figsize=(18, 6), dpi=300)

    for each_color, each_feature in zip(colors, FEATURES):
        plt.plot(month_order_range,scaled_features[each_feature].to_numpy(), marker="o", linewidth=2.5, color=each_color, label=each_feature,)

    plt.xticks(month_order_range, MONTH_ORDER, rotation=45, fontsize=16)
    plt.yticks(fontsize=18)
    plt.ylabel("Scaled Value (0 = Min, 1 = Max)", fontsize=18)
    plt.title("Relative Trends of all features by month", fontsize=18)
    plt.axhline(0.5, linestyle="--", color="grey")
    plt.grid(alpha=0.4)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=18, ncols=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)

    plt.tight_layout()
    out = "spotify_final/final_plots/month/all_features_scaled_by_month.png"
    plt.savefig(out)
    plt.close()

    logging.info(f"Saved: {out}")


def plot_scaled_single_feature(feature_name,mean_feature_per_month, spread_feature_per_month):
    #functions need to be standardized across files 
    os.makedirs("spotify_final/final_plots/month/features", exist_ok=True)
    feature_vals = mean_feature_per_month[feature_name].values.reshape(-1, 1)
    scaler_obj = MinMaxScaler()

    scaled_features = scaler_obj.fit_transform(feature_vals).flatten()

    month_order_range = np.arange(len(MONTH_ORDER))

    fig, scaled_plots = plt.subplots(2, 1, figsize=(18, 12), dpi=300) ## is the resilution ok for the final document??

    scaled_plots[0].plot(
        month_order_range,
        scaled_features,
        marker="o",
        linewidth=3,
        color="hotpink",
    )
    scaled_plots[0].axhline(0.5, linestyle="--", color="gray", alpha=0.7)
    scaled_plots[0].set_xticks(month_order_range)
    scaled_plots[0].set_xticklabels(MONTH_ORDER, rotation=45)
    scaled_plots[0].set_ylabel("Scaled Value (0 = Min, 1 = Max)")
    scaled_plots[0].set_title(f"{feature_name} – Min-Max Scaled by Month")

    peak_plot = np.argmax(scaled_features)
    peak_month = MONTH_ORDER[peak_plot]
    peak_val = scaled_features[peak_plot]

    scaled_plots[0].annotate(
        f"Peak: {peak_val:.2f}\n{peak_month}", #.2f for 2 decimal places preicision but do we need that much?
        xy=(peak_plot, peak_val),
        xytext=(peak_plot + 0.3, peak_val + 0.15),
        arrowprops=dict(arrowstyle="->", color="hotpink"),
        color="hotpink",
        fontsize=11,
    )

    avg_vals = mean_feature_per_month[feature_name]
    errors = spread_feature_per_month[feature_name]

    scaled_plots[1].bar(
        MONTH_ORDER,
        avg_vals,
        yerr=errors,
        color="hotpink",
        edgecolor="black",
        capsize=6,
    )
    scaled_plots[1].set_title(f"{feature_name} – Average by Month")
    scaled_plots[1].set_ylabel("Average Value")

    for i, v in enumerate(avg_vals):
        scaled_plots[1].text(
            i,
            v + errors.iloc[i] + 0.01,
            f"{v:.4f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    out = f"spotify_final/final_plots/month/features/{feature_name.lower()}_by_month.png"
    plt.savefig(out)
    plt.close()

    logging.info(f"Saved: {out}")

def plot_all_scaled_single_features(mean_feature_per_month,spread_feature_per_month):
    for feature in FEATURES:
        plot_scaled_single_feature(feature,mean_feature_per_month, spread_feature_per_month)

if __name__ == "__main__":
    spotify_data_csv = load_spotify_data("spotify_final/Spotify_Dataset_V3_visualizer.csv")
    spotify_data_csv = clean_and_prepare_data_month(spotify_data_csv)

    mean_feature_per_month, spread_feature_per_month, count_songs_per_month = month_stats(spotify_data_csv)

    plot_scaled_all_features(mean_feature_per_month)
    #plot_all_scaled_single_features(mean_feature_per_month, spread_feature_per_month)
