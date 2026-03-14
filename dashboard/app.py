import os
import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

model_path = os.path.join(MODELS_DIR, "match_predictor.pkl")

# Layout
st.set_page_config(page_title="IPL Analytics Dashboard", layout="wide")

st.title("🏏 IPL Data Science Project")
st.markdown("Explore historical IPL stats and predict match outcomes using Machine Learning!")

tab1, tab2, tab3 = st.tabs(["Overview & EDA", "Data Explorer", "Match Predictor"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Historical Analysis")
    st.markdown("Visualizing IPL history directly from the exploratory analysis step.")
    
    col1, col2 = st.columns(2)
    with col1:
        img_path = os.path.join(VISUALS_DIR, "most_successful_teams.png")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), use_container_width=True)
            
        img_path_2 = os.path.join(VISUALS_DIR, "toss_decision_impact.png")
        if os.path.exists(img_path_2):
            st.image(Image.open(img_path_2), use_container_width=True)
            
    with col2:
        img_path_3 = os.path.join(VISUALS_DIR, "top_run_scorers.png")
        if os.path.exists(img_path_3):
            st.image(Image.open(img_path_3), use_container_width=True)
            
        img_path_4 = os.path.join(VISUALS_DIR, "top_wicket_takers.png")
        if os.path.exists(img_path_4):
            st.image(Image.open(img_path_4), use_container_width=True)

# --- TAB 2: Data Explorer ---
with tab2:
    st.header("Raw Matches Data")
    dataset_path = os.path.join(DATA_DIR, "matches.csv")
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        st.dataframe(df.head(100))
        st.write(f"Total Matches in Dataset: {len(df)}")
    else:
        st.write("Matches dataset not found.")

# --- TAB 3: Predictor ---
with tab3:
    st.header("🔮 ML Match Predictor")
    st.write("Enter match details to predict the winner using our trained classifier.")
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            # Load metadata for dropdowns
            matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
            teams = sorted(matches['team1'].dropna().unique())
            venues = sorted(matches['venue'].dropna().unique())
            cities = sorted(matches['city'].dropna().unique())
            
            c1, c2 = st.columns(2)
            with c1:
                team1 = st.selectbox("Select Team 1", teams)
                team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])
                toss_winner = st.selectbox("Toss Winner", [team1, team2])
                
            with c2:
                venue = st.selectbox("Venue", venues)
                city = st.selectbox("City", cities)
                toss_decision = st.selectbox("Toss Decision", ['bat', 'field'])
                
            if st.button("Predict Winner"):
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
                
                # Predict
                prob = model.predict_proba(input_df)[0]
                prediction = model.predict(input_df)[0]
                
                predicted_team = team1 if prediction == 1 else team2
                team1_prob = prob[1] * 100
                team2_prob = prob[0] * 100
                
                st.success(f"🏆 Predicted Winner: **{predicted_team}**")
                
                st.subheader("Win Probabilities")
                st.write(f"- {team1}: {team1_prob:.1f}%")
                st.write(f"- {team2}: {team2_prob:.1f}%")
                
        except Exception as e:
            st.error(f"Error loading model or making prediction: {e}")
    else:
        st.warning("Model not found. Please train the model first.")
