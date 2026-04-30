import pandas as pd
from geopy.distance import geodesic
import numpy as np

def step2_feature_engineering():
    print("Loading data from Step 1...")
    df_games = pd.read_csv('nba_games_raw.csv')
    
    # Geographic Data for all 30 NBA Teams
    arena_data = {
        'Team_Abbreviation': ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 
                              'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 
                              'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'],
        'Latitude': [33.7573, 42.3662, 40.6826, 35.2251, 41.8806, 41.4965, 32.7905, 39.7486, 42.3410, 37.7680, 
                     29.7508, 39.7639, 34.0430, 34.0430, 35.1381, 25.7814, 43.0451, 44.9795, 29.9490, 40.7505, 
                     35.4634, 28.5392, 39.9012, 33.4458, 45.5316, 38.5802, 29.4270, 43.6435, 40.7683, 38.8981],
        'Longitude': [-84.3963, -71.0621, -73.9754, -80.8392, -87.6742, -81.6881, -96.8103, -105.0075, -83.0550, -122.3877, 
                      -95.3621, -86.1555, -118.2673, -118.2673, -90.0505, -80.1870, -87.9172, -93.2761, -90.0821, -73.9934, 
                      -97.5151, -81.3839, -75.1720, -112.0712, -122.6668, -121.4997, -98.4375, -79.3791, -111.9011, -77.0209],
        'Elevation_m': [320, 5, 15, 229, 181, 199, 131, 1609, 180, 5, 
                        12, 218, 71, 71, 103, 2, 179, 256, -1, 10, 
                        366, 25, 12, 331, 15, 9, 198, 75, 1288, 12]
    }
    df_arenas = pd.DataFrame(arena_data)
    # Save the arenas dataset as well
    df_arenas.to_csv('nba_arenas_geodata.csv', index=False)
    
    # Determine the location of the game (Home or Away)
    df_games['Is_Home'] = df_games['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df_games['Opponent'] = df_games['MATCHUP'].str[-3:]
    df_games['Location_Abbr'] = df_games.apply(lambda row: row['TEAM_ABBREVIATION'] if row['Is_Home'] == 1 else row['Opponent'], axis=1)
    
    # Merge game logs with geographic data
    df_games = df_games.merge(df_arenas, left_on='Location_Abbr', right_on='Team_Abbreviation', how='left')
    
    # Sort data by team and date
    df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
    df_games.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE'], inplace=True)
    
    print("Calculating rest days and back-to-back status...")
    # 1. Days of Rest
    df_games['Previous_Game_Date'] = df_games.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(1)
    df_games['Days_of_Rest'] = (df_games['GAME_DATE'] - df_games['Previous_Game_Date']).dt.days - 1
    df_games['Days_of_Rest'] = df_games['Days_of_Rest'].fillna(5) # Assume 5 days of rest for the first game of the season
    df_games['Is_B2B'] = df_games['Days_of_Rest'].apply(lambda x: 1 if x == 0 else 0)
    
    print("Calculating flight distances (This may take 10-15 seconds)...")
    # 2. Distance Traveled & Altitude Difference
    df_games['Prev_Lat'] = df_games.groupby('TEAM_ABBREVIATION')['Latitude'].shift(1)
    df_games['Prev_Lon'] = df_games.groupby('TEAM_ABBREVIATION')['Longitude'].shift(1)
    df_games['Prev_Elevation'] = df_games.groupby('TEAM_ABBREVIATION')['Elevation_m'].shift(1)
    
    def calculate_distance(row):
        if pd.isna(row['Prev_Lat']):
            return 0 
        return geodesic((row['Prev_Lat'], row['Prev_Lon']), (row['Latitude'], row['Longitude'])).kilometers

    df_games['Distance_Traveled_km'] = df_games.apply(calculate_distance, axis=1)
    
    # 3. Altitude Difference
    df_games['Altitude_Difference'] = df_games['Elevation_m'] - df_games['Prev_Elevation']
    df_games['Altitude_Difference'] = df_games['Altitude_Difference'].fillna(0)
    
    # Drop temporary columns
    columns_to_drop = ['Team_Abbreviation', 'Prev_Lat', 'Prev_Lon', 'Prev_Elevation', 'Previous_Game_Date']
    df_games.drop(columns=columns_to_drop, inplace=True)
    
    # Save the final enriched dataset
    df_games.to_csv('nba_games_enriched.csv', index=False)
    print("Step 2 Complete! All calculations saved to 'nba_games_enriched.csv'.")

if __name__ == "__main__":
    step2_feature_engineering()