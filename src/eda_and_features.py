import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")

if not os.path.exists(VISUALS_DIR):
    os.makedirs(VISUALS_DIR)

def load_data():
    matches_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "matches.csv"))
    deliveries_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "deliveries.csv"))
    return matches_df, deliveries_df

def run_eda(matches_df, deliveries_df):
    logging.info("Running Exploratory Data Analysis...")
    
    # 1. Most Successful Teams
    plt.figure(figsize=(10,6))
    win_counts = matches_df['winner'].value_counts()
    sns.barplot(x=win_counts.values, y=win_counts.index, palette='viridis')
    plt.title('Most Successful IPL Teams (Most Match Wins)')
    plt.xlabel('Number of Wins')
    plt.ylabel('Team')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "most_successful_teams.png"))
    plt.close()
    
    # 2. Top Run Scorers
    plt.figure(figsize=(10,6))
    top_scorers = deliveries_df.groupby('striker')['runs_off_bat'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_scorers.values, y=top_scorers.index, palette='magma')
    plt.title('Top 10 Run Scorers in IPL History')
    plt.xlabel('Total Runs')
    plt.ylabel('Batsman')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "top_run_scorers.png"))
    plt.close()

    # 3. Top Wicket Takers
    plt.figure(figsize=(10,6))
    # Exclude run outs, retired hurts from bowler's wickets
    dismissals = deliveries_df.dropna(subset=['wicket_type'])
    bowler_wickets = dismissals[~dismissals['wicket_type'].isin(['run out', 'retired hurt', 'obstructing the field'])]
    top_bowlers = bowler_wickets.groupby('bowler')['wicket_type'].count().sort_values(ascending=False).head(10)
    sns.barplot(x=top_bowlers.values, y=top_bowlers.index, palette='ocean')
    plt.title('Top 10 Wicket Takers in IPL History')
    plt.xlabel('Total Wickets')
    plt.ylabel('Bowler')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "top_wicket_takers.png"))
    plt.close()
    
    # 4. Toss Decision Impact
    plt.figure(figsize=(6,6))
    toss_decision = matches_df['toss_decision'].value_counts()
    plt.pie(toss_decision, labels=toss_decision.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
    plt.title('Toss Decision Percentage')
    plt.savefig(os.path.join(VISUALS_DIR, "toss_decision_impact.png"))
    plt.close()
    
def feature_engineering(matches_df):
    logging.info("Engineering Features for Modeling...")
    
    # We will build a dataset where each row is a match with features known BEFORE the match starts.
    # We predict the 'winner'. If winner is not present (draw/no result), we drop.
    df = matches_df.copy()
    df = df.dropna(subset=['winner', 'team1', 'team2', 'venue', 'toss_winner', 'toss_decision'])
    
    # Target Variable: 1 if team1 wins, 0 if team2 wins (arbitrary binary target for binary classification)
    # We will formulate it as: Target is 1 if 'team1' wins, else 0.
    df['target'] = np.where(df['winner'] == df['team1'], 1, 0)
    
    # Feature 1: is_toss_winner (Did team1 win the toss?)
    df['team1_won_toss'] = np.where(df['toss_winner'] == df['team1'], 1, 0)
    
    # Feature 2: toss_decision_bat (1 if the toss winner decided to bat)
    df['toss_decision_bat'] = np.where(df['toss_decision'] == 'bat', 1, 0)
    
    # Encoding categorical variables
    # We will need city, venue, team1, team2
    # Ensure they are strings
    df['city'] = df['city'].fillna('Unknown')
    
    features_df = df[['id', 'season', 'team1', 'team2', 'venue', 'city', 'team1_won_toss', 'toss_decision_bat', 'target']].copy()
    
    model_data_path = os.path.join(PROCESSED_DATA_DIR, "model_features.csv")
    features_df.to_csv(model_data_path, index=False)
    logging.info(f"Feature dataset saved to {model_data_path} with shape {features_df.shape}")

if __name__ == "__main__":
    matches, deliveries = load_data()
    run_eda(matches, deliveries)
    feature_engineering(matches)
    logging.info("EDA and Feature Engineering complete!")
