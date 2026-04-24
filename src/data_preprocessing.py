import os
import glob
import pandas as pd
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from utils import standardize_teams

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(data_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_delivery_files = glob.glob(os.path.join(data_dir, "*.csv"))
    # Exclude the info files
    delivery_files = [f for f in all_delivery_files if not f.endswith("_info.csv")]
    info_files = [f for f in all_delivery_files if f.endswith("_info.csv")]

    logging.info(f"Found {len(delivery_files)} delivery files and {len(info_files)} info files.")

    # 1. Process Deliveries
    logging.info("Processing deliveries... this may take a minute.")
    df_list = []
    for f in delivery_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")
            
    if df_list:
        deliveries_df = pd.concat(df_list, ignore_index=True)
        # Standardize team names
        deliveries_df = standardize_teams(deliveries_df, ['batting_team', 'bowling_team'])
        
        deliveries_path = os.path.join(output_dir, "deliveries.csv")
        deliveries_df.to_csv(deliveries_path, index=False)
        logging.info(f"Deliveries dataset saved at {deliveries_path} with shape {deliveries_df.shape}")
    else:
        logging.warning("No delivery files combined.")

    # 2. Process Matches Info
    logging.info("Processing match info...")
    match_list = []
    for f in info_files:
        match_id = os.path.basename(f).split('_')[0]
        try:
            info_df = pd.read_csv(f, names=['type', 'key', 'val1', 'val2', 'val3', 'val4'], engine='python', on_bad_lines='skip')
            # Extract basic info
            match_data = {'id': match_id}
            
            # Helper to extract first value for a key
            def get_val(key):
                res = info_df[info_df['key'] == key]['val1'].values
                return res[0] if len(res) > 0 else None
            
            # For teams, there are two entries usually
            teams = info_df[info_df['key'] == 'team']['val1'].values
            match_data['team1'] = teams[0] if len(teams) > 0 else None
            match_data['team2'] = teams[1] if len(teams) > 1 else None
            
            match_data['season'] = get_val('season')
            match_data['date'] = get_val('date')
            match_data['venue'] = get_val('venue')
            match_data['city'] = get_val('city')
            match_data['toss_winner'] = get_val('toss_winner')
            match_data['toss_decision'] = get_val('toss_decision')
            match_data['player_of_match'] = get_val('player_of_match')
            match_data['winner'] = get_val('winner')
            match_data['winner_runs'] = get_val('winner_runs')
            match_data['winner_wickets'] = get_val('winner_wickets')
            
            match_list.append(match_data)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")

    if match_list:
        matches_df = pd.DataFrame(match_list)
        
        # Standardize teams
        matches_df = standardize_teams(matches_df, ['team1', 'team2', 'toss_winner', 'winner'])
        
        # Parse Dates
        matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
        matches_df = matches_df.sort_values(by='date').reset_index(drop=True)
        
        matches_path = os.path.join(output_dir, "matches.csv")
        matches_df.to_csv(matches_path, index=False)
        logging.info(f"Matches dataset saved at {matches_path} with shape {matches_df.shape}")
    else:
        logging.warning("No matches extracted.")

if __name__ == "__main__":
    process_data(config.DATA_RAW, config.DATA_PROCESSED)
