import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time

def fetch_nba_data():
    # --- PART 1: Fetching NBA Game Data ---
    # Excluding COVID-19 "Bubble" season, fetching the last 3 regular seasons.
    seasons = ['2021-22', '2022-23', '2023-24'] 
    all_games = []

    print("Fetching game data via nba_api...")
    for season in seasons:
        try:
            print(f"Fetching season {season}...")
            game_log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season')
            df_season = game_log.get_data_frames()[0]
            all_games.append(df_season)
            # Sleep for 2 seconds to avoid being blocked by the API
            time.sleep(2) 
        except Exception as e:
            print(f"Error fetching season {season}: {e}")

    # Combine all seasons into a single DataFrame
    df_games = pd.concat(all_games, ignore_index=True)
    df_games.to_csv('nba_games_raw.csv', index=False)
    print(f"Successfully fetched {len(df_games)} games and saved as 'nba_games_raw.csv'.")

if __name__ == "__main__":
    fetch_nba_data()