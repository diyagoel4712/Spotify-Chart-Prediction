import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D
import datetime as dt
from spotify_date_extractor import load_spotify_data,clean_and_prepare_data_weekofyear

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")



WEEK_OF_YEAR_ORDER = list(range(1, 54))
month_start_weeks = ["Jan-1", "Feb-1", "Mar-1", "Apr-1", "May-1", "Jun-1", "Jul-1", "Aug-1", "Sep-1", "Oct-1", "Nov-1", "Dec-1"]

FEATURES = ["Danceability","Energy","Loudness","Speechiness","Acousticness","Instrumentalness", "Valence"]
def weekofyear_stats(spotify_data_csv):
    spotify_data_csv = spotify_data_csv.copy()
    spotify_data_csv["Date"] = pd.to_datetime(spotify_data_csv["Date"], format="%d/%m/%Y")
    spotify_data_csv["WeekOfYear"] = (spotify_data_csv["Date"].dt.isocalendar().week.astype(int))
    spotify_data_csv = spotify_data_csv[spotify_data_csv["WeekOfYear"].between(1, 53)]
    spotify_data_csv["WeekOfYear"] = pd.Categorical(spotify_data_csv["WeekOfYear"],categories=WEEK_OF_YEAR_ORDER,ordered=True,)
    spotify_data_csv["Month"] = spotify_data_csv["Date"].dt.month
    for feature in FEATURES:
        spotify_data_csv[feature] = pd.to_numeric(spotify_data_csv[feature], errors="coerce")
    spotify_data_csv = spotify_data_csv.dropna(subset=FEATURES + ["WeekOfYear"])
    group_weeks = spotify_data_csv.groupby("WeekOfYear", observed=True)
    mean_feature_per_week = group_weeks[FEATURES].mean().reindex(WEEK_OF_YEAR_ORDER)
    spread_feature_per_week = group_weeks[FEATURES].std().reindex(WEEK_OF_YEAR_ORDER, fill_value=0)
    count_songs_per_week = group_weeks.size().reindex(WEEK_OF_YEAR_ORDER, fill_value=0)
    month_start_weeks = (spotify_data_csv.groupby("Month")["WeekOfYear"].min().dropna().astype(int).to_dict())
    return (mean_feature_per_week,spread_feature_per_week,count_songs_per_week,month_start_weeks,)


def plot_scaled_all_features(mean_feature_per_week, month_start_weeks=None):
    """
    Plot min–max scaled trends of all FEATURES by week-of-year,
    with vertical lines and x-axis ticks at calendar month boundaries
    (Jan-1, Feb-1, ..., Dec-1).
    """
    out_dir = "spotify_final/final_plots/weekofyear"
    os.makedirs(out_dir, exist_ok=True)
    scaler_obj = MinMaxScaler()
    scaled_features = pd.DataFrame(
        scaler_obj.fit_transform(mean_feature_per_week[FEATURES]),
        columns=FEATURES,
        index=WEEK_OF_YEAR_ORDER,
    )

    week_range = np.arange(len(WEEK_OF_YEAR_ORDER))  
    colors = sns.color_palette("husl", len(FEATURES))

    plt.figure(figsize=(18, 6), dpi=300)
    ax = plt.gca()

    # --- plot each feature
    feature_handles = []
    for each_color, each_feature in zip(colors, FEATURES):
        line, = ax.plot(
            week_range,
            scaled_features[each_feature].to_numpy(),
            marker="o",
            linewidth=2.0,
            color=each_color,
            label=each_feature,
        )
        feature_handles.append(line)

    # --- y-axis
    ax.set_ylabel("Scaled Value (0 = Min, 1 = Max)", fontsize=18)
    ax.set_title("Relative Trends of all features by Week-of-Year", fontsize=18)
    ax.axhline(0.5, linestyle="--", color="grey", alpha=0.7)
    ax.grid(alpha=0.3)
    # Force Jan-1 vertical line to align with x=0 border
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="y", labelsize=18)
    # --- EXACT month boundaries from Jan 1, Feb 1, ... Dec 1 ---
    base = dt.date(2019, 1, 1)  # arbitrary non-leap year
    month_boundary_positions = []
    month_labels = []

    for m in range(1, 13):
        d = dt.date(2019, m, 1)
        days_from_jan1 = (d - base).days
        week_pos = days_from_jan1 / 7.0  
        month_boundary_positions.append(week_pos)
        month_labels.append(d.strftime("%b-1"))  # "Jan-1", "Feb-1", ...

    # --- draw vertical month lines across full plot height ---
    for pos in month_boundary_positions:
        ax.axvline(
            pos,
            linestyle="--",
            color="gray",
            alpha=0.7,
            linewidth=2.2,
        )

    ax.set_xticks(month_boundary_positions)
    ax.set_xticklabels(month_labels, fontsize=14, rotation=45, ha="right")

    month_proxy = Line2D(
        [0], [0],
        linestyle="--",
        color="gray",
        linewidth=2.2,
        alpha=0.7,
        label="Month boundary",
    )

    handles = feature_handles + [month_proxy]
    labels = FEATURES + ["Month boundary"]

    ax.legend(
        handles=handles,
        labels=labels,
        fontsize=14,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        frameon=False,
    )
    # Force Jan-1 vertical line to align with x=0 border
    ax.set_xlim(left=0)
    plt.tight_layout()
    out = os.path.join(out_dir, "all_features_scaled_by_weekofyear.png")
    plt.savefig(out)
    plt.close()
    logging.info(f"Saved: {out}")

#too clutterted with 53 weeks
# def plot_scaled_all_features(mean_feature_per_week, month_start_weeks):
#     out_dir = "spotify_final/final_plots/weekofyear"
#     os.makedirs(out_dir, exist_ok=True)
#     scaler_obj = MinMaxScaler()
#     scaled_features = pd.DataFrame(scaler_obj.fit_transform(mean_feature_per_week[FEATURES]),columns=FEATURES,index=WEEK_OF_YEAR_ORDER,)
#     week_range = np.arange(len(WEEK_OF_YEAR_ORDER))
#     colors = sns.color_palette("husl", len(FEATURES))
#     plt.figure(figsize=(18, 6), dpi=300)
#     for each_color, each_feature in zip(colors, FEATURES):
#         plt.plot(week_range,scaled_features[each_feature].to_numpy(),marker="o",linewidth=2.0,color=each_color,label=each_feature,)

#     plt.xticks(week_range,[str(w) for w in WEEK_OF_YEAR_ORDER],rotation=90,fontsize=16)
#     plt.yticks(fontsize=18)
#     plt.ylabel("Scaled Value (0 = Min, 1 = Max)", fontsize=18)
#     plt.title("Relative Trends of all features by Week-of-Year", fontsize=18)
#     plt.axhline(0.5, linestyle="--", color="grey", alpha=0.7)
#     plt.grid(alpha=0.3)
#     plt.ylim(-0.05, 1.05)
#     plt.legend(fontsize=18, ncols=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
#     for month, start_week in month_start_weeks.items():
#         if start_week in WEEK_OF_YEAR_ORDER:
#             month_number = WEEK_OF_YEAR_ORDER.index(start_week)
#             plt.axvline(month_number, linestyle="--", color="gray", alpha=0.4, linewidth=1)
#             plt.text(month_number,1.05,f"M{month}",ha="center",va="bottom",fontsize=7,color="gray")

#     plt.tight_layout()
#     out = os.path.join(out_dir, "all_features_scaled_by_weekofyear.png")
#     plt.savefig(out)
#     plt.close()
#     logging.info(f"Saved: {out}")

def plot_scaled_single_feature(feature_name, mean_feature_per_week, spread_feature_per_week, month_start_weeks):
    out_dir = "spotify_final/final_plots/weekofyear/features"
    os.makedirs(out_dir, exist_ok=True)
    feature_vals = mean_feature_per_week[feature_name].to_numpy().reshape(-1, 1)
    scaler_obj = MinMaxScaler()
    scaled_vals = scaler_obj.fit_transform(feature_vals).flatten()
    week_range = np.arange(len(WEEK_OF_YEAR_ORDER))
    
    
    fig, scaled_plots = plt.subplots(2, 1, figsize=(20, 10), dpi=300)
    scaled_plots[0].plot(week_range,scaled_vals,marker="o",linewidth=2.5,color="hotpink")
    scaled_plots[0].axhline(0.5, linestyle="--", color="gray", alpha=0.7)
    scaled_plots[0].set_xticks(week_range)
    scaled_plots[0].set_xticklabels([str(w) for w in WEEK_OF_YEAR_ORDER], rotation=90, fontsize=7)
    scaled_plots[0].set_ylabel("Scaled Value (0 = Min, 1 = Max)")
    scaled_plots[0].set_title(f"{feature_name} – Min-Max Scaled by Week-of-Year")

    # Vertical month lines here too
    for month, start_week in month_start_weeks.items():
        if start_week in WEEK_OF_YEAR_ORDER:
            month_number = WEEK_OF_YEAR_ORDER.index(start_week)
            scaled_plots[0].axvline(month_number, linestyle="--", color="gray", alpha=0.4, linewidth=1)
            scaled_plots[0].text(month_number,1.05,f"M{month}",ha="center",va="bottom",fontsize=7,color="gray")

    # Peak annotation..
    peak_plot = int(np.argmax(scaled_vals))
    peak_week = WEEK_OF_YEAR_ORDER[peak_plot]
    peak_value = scaled_vals[peak_plot]

    scaled_plots[0].annotate(f"Peak: {peak_value:.2f}\nWeek {peak_week}",xy=(peak_plot, peak_value),xytext=(peak_plot + 1, peak_value + 0.15),arrowprops=dict(arrowstyle="->", color="hotpink"),color="hotpink",fontsize=10)

    avg_vals = mean_feature_per_week[feature_name]
    errors = spread_feature_per_week[feature_name]

    scaled_plots[1].bar(WEEK_OF_YEAR_ORDER,avg_vals,yerr=errors,color="#ff99aa",edgecolor="black",capsize=3)
    scaled_plots[1].set_title(f"{feature_name} – Average by Week-of-Year")
    scaled_plots[1].set_ylabel("Average Value")
    scaled_plots[1].set_xticks(WEEK_OF_YEAR_ORDER)
    scaled_plots[1].set_xticklabels([str(w) for w in WEEK_OF_YEAR_ORDER], rotation=90, fontsize=7)

    for i, val in enumerate(avg_vals):
        scaled_plots[1].text(i,val + errors.iloc[i] + 0.01,f"{val:.4f}",ha="center",fontsize=6)

    plt.tight_layout()
    out = os.path.join(out_dir, f"{feature_name.lower()}_by_weekofyear.png")
    plt.savefig(out)
    plt.close()

    logging.info(f"Saved: {out}")







def plot_all_scaled_single_features(mean_feature_per_week, spread_feature_per_week, month_start_weeks):
    for each_feature in FEATURES:
        plot_scaled_single_feature(each_feature,mean_feature_per_week,spread_feature_per_week,month_start_weeks)










if __name__ == "__main__":
    #load -> clean -> stats -> plot -> save -> accuracy????????????????
    spotify_data_csv = load_spotify_data("spotify_final/Spotify_Dataset_V3_visualizer.csv")
    spotify_data_csv = clean_and_prepare_data_weekofyear(spotify_data_csv)

    mean_feature_per_week,spread_feature_per_week,count_songs_per_week,month_start_weeks=  weekofyear_stats(spotify_data_csv)




    plot_scaled_all_features(mean_feature_per_week, month_start_weeks)

    #plot_all_scaled_single_features(mean_feature_per_week,spread_feature_per_week,month_start_weeks)