import pandas as pd

input_path = "Spotify_Dataset_V3.csv" 
output_day = "spotify_processed_data_day.parquet"
output_month = "spotify_processed_data_month.parquet"
output_quarter = "spotify_processed_data_quarter.parquet"  
output_year = "spotify_processed_data_year.parquet"

def load_spotify_data(path):
    spotify_data_csv = pd.read_csv(path, sep=",", engine="python") # 
    return spotify_data_csv

def clean_and_prepare_data_weekday(spotify_data_csv):
    spotify_data_csv['Date'] = pd.to_datetime(spotify_data_csv['Date'], format='%d/%m/%Y')
    spotify_data_csv['day_of_week'] = spotify_data_csv['Date'].dt.day_name()
    spotify_data_csv['Date_Formatted'] = spotify_data_csv['Date'].dt.strftime('%d/%m/%Y')
    spotify_data_csv.to_parquet(output_day, index=False)
    print(spotify_data_csv[['Date_Formatted', 'day_of_week']].head())
    spotify_data_with_weekday = spotify_data_csv
    return spotify_data_with_weekday

def clean_and_prepare_data_month(spotify_data_csv):
    spotify_data_csv['Date'] = pd.to_datetime(spotify_data_csv['Date'], format='%d/%m/%Y')
    spotify_data_csv['Month'] = spotify_data_csv['Date'].dt.month_name()
    spotify_data_csv['Date_Formatted'] = spotify_data_csv['Date'].dt.strftime('%d/%m/%Y')
    spotify_data_csv.to_parquet(output_month, index=False)
    print(spotify_data_csv[['Date_Formatted', 'Month']].head())
    spotify_data_with_month = spotify_data_csv
    return spotify_data_with_month

def clean_and_prepare_data_quarter(spotify_data_csv):
    spotify_data_csv['Date'] = pd.to_datetime(spotify_data_csv['Date'], format='%d/%m/%Y')
    #spotify_data_csv['Quarter'] = spotify_data_csv['Date'].dt.quarter ### displays only quarter number, i need to display year as well?
    spotify_data_csv['Quarter'] = spotify_data_csv['Date'].dt.to_period('Q').astype(str)
    spotify_data_csv['Date_Formatted'] = spotify_data_csv['Date'].dt.strftime('%d/%m/%Y')
    spotify_data_csv.to_parquet(output_quarter, index=False)
    print(spotify_data_csv[['Date_Formatted', 'Quarter']].head())
    spotify_data_with_quarter = spotify_data_csv
    return spotify_data_with_quarter

def clean_and_prepare_data_year(spotify_data_csv):
    spotify_data_csv['Date'] = pd.to_datetime(spotify_data_csv['Date'], format='%d/%m/%Y')
    spotify_data_csv['Year'] = spotify_data_csv['Date'].dt.year
    spotify_data_csv['Date_Formatted'] = spotify_data_csv['Date'].dt.strftime('%d/%m/%Y')
    spotify_data_csv.to_parquet(output_year, index=False)
    print(spotify_data_csv[['Date_Formatted', 'Year']].head())
    spotify_data_with_year = spotify_data_csv
    return spotify_data_with_year

def clean_and_prepare_data_weekofyear(spotify_data_csv):
    spotify_data_csv = spotify_data_csv.copy()
    spotify_data_csv["Date"] = pd.to_datetime(spotify_data_csv["Date"], format="%d/%m/%Y")
    spotify_data_csv["WeekOfYear"] = spotify_data_csv["Date"].dt.isocalendar().week.astype(int)
    spotify_data_with_weeks_of_year = spotify_data_csv[spotify_data_csv["WeekOfYear"].between(1, 53)]

    return spotify_data_with_weeks_of_year


if __name__ == "__main__":
    pass
    #spotify_data_csv = load_spotify_data(input_path) #harcodeed path for now
    #clean_and_prepare_data_weekday(spotify_data_csv)
    #clean_and_prepare_data_month(spotify_data_csv)
    #clean_and_prepare_data_quarter(spotify_data_csv)
    #clean_and_prepare_data_year(spotify_data_csv)