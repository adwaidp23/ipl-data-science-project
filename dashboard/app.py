import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

model_path = os.path.join(MODELS_DIR, "match_predictor.pkl")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .team-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .prediction-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #000;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Layout
st.set_page_config(page_title="IPL Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# Title with custom styling
st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>🏏 IPL Data Science Analytics Dashboard</h1>
    <p style='text-align: center; font-size: 16px;'>Advanced Analytics & ML-Powered Match Prediction</p>
""", unsafe_allow_html=True)

st.markdown("---")


# Load data once and cache it
@st.cache_data
def load_matches_data():
    dataset_path = os.path.join(DATA_DIR, "matches.csv")
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    return None

@st.cache_data
def load_deliveries_data():
    dataset_path = os.path.join(DATA_DIR, "deliveries.csv")
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    return None

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

# Load all data
matches_df = load_matches_data()
deliveries_df = load_deliveries_data()
model = load_model()

# Sidebar filters
st.sidebar.header("🔍 Dashboard Filters")
if matches_df is not None:
    teams = sorted(matches_df['team1'].dropna().unique())
    selected_team = st.sidebar.multiselect("Filter by Team", teams, default=None)
    
    years = sorted(matches_df['date'].dt.year.unique()) if 'date' in matches_df.columns and matches_df['date'].dtype == 'datetime64[ns]' else []
    if years:
        year_range = st.sidebar.slider("Select Year Range", min(years), max(years), (min(years), max(years)))

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview & KPIs", 
    "🔍 Data Explorer", 
    "⚡ Team Analytics",
    "🌟 Player Stats",
    "🔮 Match Predictor"
])

# --- TAB 1: Overview & KPIs ---
with tab1:
    st.header("📊 Dashboard Overview & Key Insights")
    
    if matches_df is not None:
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📈 Total Matches", len(matches_df), delta=None)
        
        with col2:
            unique_teams = matches_df['team1'].nunique() if 'team1' in matches_df.columns else 0
            st.metric("🏆 Unique Teams", unique_teams, delta=None)
        
        with col3:
            unique_venues = matches_df['venue'].nunique() if 'venue' in matches_df.columns else 0
            st.metric("🏟️ Venues", unique_venues, delta=None)
        
        with col4:
            if 'date' in matches_df.columns:
                try:
                    matches_df['date'] = pd.to_datetime(matches_df['date'])
                    years_span = matches_df['date'].dt.year.max() - matches_df['date'].dt.year.min() + 1
                    st.metric("📅 Years Covered", f"{years_span}", delta=None)
                except:
                    st.metric("📅 Years Covered", "N/A", delta=None)
        
        with col5:
            toss_winners = matches_df['toss_winner'].nunique() if 'toss_winner' in matches_df.columns else 0
            st.metric("⭐ Team Toss Wins", toss_winners, delta=None)
        
        st.markdown("---")
        
        # Display visualizations
        st.subheader("📈 Historical Analysis Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            img_path = os.path.join(VISUALS_DIR, "most_successful_teams.png")
            if os.path.exists(img_path):
                st.image(Image.open(img_path), caption="Most Successful Teams", use_container_width=True)
            else:
                st.info("Visualization not found")
                
            img_path_2 = os.path.join(VISUALS_DIR, "toss_decision_impact.png")
            if os.path.exists(img_path_2):
                st.image(Image.open(img_path_2), caption="Toss Decision Impact", use_container_width=True)
            else:
                st.info("Visualization not found")
                
        with col2:
            img_path_3 = os.path.join(VISUALS_DIR, "top_run_scorers.png")
            if os.path.exists(img_path_3):
                st.image(Image.open(img_path_3), caption="Top Run Scorers", use_container_width=True)
            else:
                st.info("Visualization not found")
                
            img_path_4 = os.path.join(VISUALS_DIR, "top_wicket_takers.png")
            if os.path.exists(img_path_4):
                st.image(Image.open(img_path_4), caption="Top Wicket Takers", use_container_width=True)
            else:
                st.info("Visualization not found")

# --- TAB 2: Data Explorer ---
with tab2:
    st.header("🔍 Interactive Matches Data Explorer")
    
    if matches_df is not None:
        # Display summary statistics
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Matches:** {len(matches_df)}")
        with col2:
            st.write(f"**Date Range:** {matches_df['date'].min() if 'date' in matches_df.columns else 'N/A'} to {matches_df['date'].max() if 'date' in matches_df.columns else 'N/A'}")
        
        st.markdown("---")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_team1 = st.selectbox("Filter Team 1", ["All"] + sorted(matches_df['team1'].unique().tolist()))
        with col2:
            filter_team2 = st.selectbox("Filter Team 2", ["All"] + sorted(matches_df['team2'].unique().tolist()))
        with col3:
            filter_venue = st.selectbox("Filter Venue", ["All"] + sorted(matches_df['venue'].dropna().unique().tolist()))
        
        # Apply filters
        filtered_df = matches_df.copy()
        if filter_team1 != "All":
            filtered_df = filtered_df[(filtered_df['team1'] == filter_team1) | (filtered_df['team2'] == filter_team1)]
        if filter_team2 != "All":
            filtered_df = filtered_df[(filtered_df['team1'] == filter_team2) | (filtered_df['team2'] == filter_team2)]
        if filter_venue != "All":
            filtered_df = filtered_df[filtered_df['venue'] == filter_venue]
        
        st.dataframe(filtered_df.head(100), use_container_width=True)
        st.write(f"**Filtered Results:** {len(filtered_df)} matches")
        
        # Download option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="ipl_matches_filtered.csv",
            mime="text/csv"
        )
    else:
        st.error("Unable to load matches data")

# --- TAB 3: Team Analytics ---
with tab3:
    st.header("⚡ Team Performance Analytics")
    
    if matches_df is not None:
        # Team selection
        all_teams = sorted(matches_df['team1'].unique().tolist())
        selected_team = st.selectbox("Select Team for Analysis", all_teams)
        
        if selected_team:
            # Team statistics
            team_matches = matches_df[
                (matches_df['team1'] == selected_team) | (matches_df['team2'] == selected_team)
            ]
            
            wins = len(team_matches[team_matches['winner'] == selected_team]) if 'winner' in matches_df.columns else 0
            total = len(team_matches)
            win_rate = (wins / total * 100) if total > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"🏆 {selected_team} - Wins", wins)
            with col2:
                st.metric("📊 Total Matches", total)
            with col3:
                st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
            with col4:
                st.metric("❌ Losses", total - wins)
            
            st.markdown("---")
            
            # Team vs Team head-to-head
            st.subheader("Head-to-Head Records")
            opponent_stats = []
            for opponent in all_teams:
                if opponent != selected_team:
                    h2h_matches = team_matches[
                        ((team_matches['team1'] == opponent) | (team_matches['team2'] == opponent))
                    ]
                    if len(h2h_matches) > 0:
                        h2h_wins = len(h2h_matches[h2h_matches['winner'] == selected_team]) if 'winner' in matches_df.columns else 0
                        opponent_stats.append({
                            "Opponent": opponent,
                            "Matches": len(h2h_matches),
                            "Wins": h2h_wins,
                            "Losses": len(h2h_matches) - h2h_wins,
                            "Win %": round(h2h_wins / len(h2h_matches) * 100, 1) if len(h2h_matches) > 0 else 0
                        })
            
            if opponent_stats:
                h2h_df = pd.DataFrame(opponent_stats)
                st.dataframe(h2h_df.sort_values("Win %", ascending=False), use_container_width=True)
    else:
        st.error("Unable to load team data")

# --- TAB 4: Player Stats (if deliveries data available) ---
with tab4:
    st.header("🌟 Player Performance Statistics")
    
    if deliveries_df is not None:
        # Basic player stats from deliveries
        st.info("📌 Note: Player statistics are derived from ball-by-ball delivery data.")
        
        # Player types selection
        stat_type = st.radio("Select Statistic Type", ["Run Scorers", "Wicket Takers"])
        
        if stat_type == "Run Scorers":
            # Top run scorers - using striker and runs_off_bat columns
            if 'striker' in deliveries_df.columns and 'runs_off_bat' in deliveries_df.columns:
                batter_runs = deliveries_df.groupby('striker')['runs_off_bat'].sum().sort_values(ascending=False).head(15)
                
                fig = px.bar(
                    x=batter_runs.values, 
                    y=batter_runs.index,
                    orientation='h',
                    labels={'x': 'Total Runs', 'y': 'Player'},
                    title="Top 15 Run Scorers in IPL",
                    color=batter_runs.values,
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Required columns for run scoring stats not found in data.")
            
        else:  # Wicket Takers
            # Top wicket takers - using wicket_type and player_dismissed columns
            if 'player_dismissed' in deliveries_df.columns and 'wicket_type' in deliveries_df.columns:
                # Filter for actual wickets (where player_dismissed is not NaN)
                wickets_df = deliveries_df[deliveries_df['player_dismissed'].notna()]
                
                if 'bowler' in deliveries_df.columns and len(wickets_df) > 0:
                    bowler_wickets = wickets_df.groupby('bowler').size().sort_values(ascending=False).head(15)
                    
                    fig = px.bar(
                        x=bowler_wickets.values,
                        y=bowler_wickets.index,
                        orientation='h',
                        labels={'x': 'Wickets', 'y': 'Player'},
                        title="Top 15 Wicket Takers in IPL",
                        color=bowler_wickets.values,
                        color_continuous_scale="Reds"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Bowler data not found or no wickets recorded.")
            else:
                st.warning("⚠️ Required columns for wicket stats not found in data.")
    else:
        st.warning("⚠️ Deliveries data not available. Unable to display player statistics.")

# --- TAB 5: Enhanced Match Predictor ---
with tab5:
    st.header("🔮 ML-Powered Match Outcome Prediction")
    st.markdown("""
    ### How It Works:
    - Enter match details (teams, venue, toss result)
    - Our trained XGBoost model predicts the winner with probability
    - The model considers historical data, venue impact, and team performance
    """)
    
    if model is not None and matches_df is not None:
        try:
            # Load feature columns
            try:
                feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
            except:
                feature_cols = ['team1', 'team2', 'venue', 'city', 'team1_won_toss', 'toss_decision_bat']
            
            teams = sorted(matches_df['team1'].dropna().unique())
            venues = sorted(matches_df['venue'].dropna().unique())
            cities = sorted(matches_df['city'].dropna().unique())
            
            st.markdown("---")
            st.subheader("⚙️ Match Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Team Selection**")
                team1 = st.selectbox("🏏 Select Team 1", teams, key="team1_select")
                team2_options = [t for t in teams if t != team1]
                team2 = st.selectbox("🏏 Select Team 2", team2_options, key="team2_select")
                toss_winner = st.selectbox("🪙 Toss Winner", [team1, team2], key="toss_winner")
                
            with col2:
                st.markdown("**Match Details**")
                venue = st.selectbox("🏟️ Venue", venues)
                city = st.selectbox("🌆 City", cities)
                toss_decision = st.selectbox("⚡ Toss Decision", ['bat', 'field'])
            
            st.markdown("---")
            
            # Prediction
            if st.button("🎯 Predict Match Winner", type="primary", use_container_width=True):
                team1_won_toss = 1 if toss_winner == team1 else 0
                toss_decision_bat = 1 if toss_decision == 'bat' else 0
                
                input_df = pd.DataFrame([{
                    'team1': team1, 
                    'team2': team2, 
                    'venue': venue, 
                    'city': city,
                    'team1_won_toss': team1_won_toss, 
                    'toss_decision_bat': toss_decision_bat
                }])
                
                try:
                    # Predict
                    prob = model.predict_proba(input_df)[0]
                    prediction = model.predict(input_df)[0]
                    
                    predicted_team = team1 if prediction == 1 else team2
                    team1_prob = prob[1] * 100
                    team2_prob = prob[0] * 100
                    
                    # Display result with enhanced styling
                    st.markdown("---")
                    st.markdown("<h2 style='text-align: center; color: #2ecc71;'>✅ Prediction Complete</h2>", unsafe_allow_html=True)
                    
                    # Winner card
                    winner_col1, winner_col2, winner_col3 = st.columns([1, 2, 1])
                    with winner_col2:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 30px; border-radius: 15px; text-align: center;'>
                            <h1>🏆 {predicted_team}</h1>
                            <p style='font-size: 18px;'>Most Likely Winner</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Probability visualization
                    st.subheader("📊 Win Probability Analysis")
                    
                    # Create probability chart
                    prob_data = pd.DataFrame({
                        'Team': [team1, team2],
                        'Win Probability': [team1_prob, team2_prob]
                    })
                    
                    fig = px.bar(
                        prob_data,
                        x='Team',
                        y='Win Probability',
                        color='Team',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                        labels={'Win Probability': 'Win Probability (%)'},
                        title="Predicted Win Probabilities",
                        text='Win Probability'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"🎯 {team1} Win Probability", f"{team1_prob:.1f}%")
                    with col2:
                        st.metric(f"🎯 {team2} Win Probability", f"{team2_prob:.1f}%")
                    
                    st.markdown("---")
                    
                    # Match Details Summary
                    st.subheader("📋 Match Configuration Summary")
                    summary_data = {
                        "Attribute": ["Team 1", "Team 2", "Venue", "City", "Toss Winner", "Toss Decision"],
                        "Value": [team1, team2, venue, city, toss_winner, toss_decision.capitalize()]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"⚠️ Prediction Error: {e}")
                    st.info("Please ensure all fields are correctly filled.")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("There was an error loading the model or data. Please check the project files.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if model is None:
                st.warning("⚠️ Model not found. Please train the model first using `src/train_model.py`")
        with col2:
            if matches_df is None:
                st.warning("⚠️ Matches data not found. Please run data preprocessing first.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p>🚀 <strong>IPL Data Science Analytics Dashboard</strong></p>
    <p style='font-size: 12px; color: #666;'>Built with Streamlit | Powered by XGBoost ML Model | Data Source: Cricsheet</p>
</div>
""", unsafe_allow_html=True)
