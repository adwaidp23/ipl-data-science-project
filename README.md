# End-to-End IPL Data Science Project Report

## Executive Summary
This project presents an end-to-end data science lifecycle applied to historical Indian Premier League (IPL) cricket data. By scraping, cleaning, preprocessing, and analyzing ball-by-ball data, we extracted high-value insights regarding team performance, toss impact, and stellar player statistics. In addition, we successfully deployed a predictive tool that estimates match-win probabilities.

## 1. Project Objectives
1. **Analyze historical matches** to identify patterns.
2. **Build predictive classifiers** to predict the match winner.
3. **Generate visualizations** to showcase insights.
4. **Develop an interactive dashboard** to present these outcomes.

## 2. Methodology & Workflow
### Data Collection
Data was sourced openly from **Cricsheet**, which provides structured CSV archives mapping over a decade of match history and delivery-by-delivery logs.

### Data Cleaning
Using `pandas`, data silos (thousands of individual CSVs) were merged into highly structured tabular sets: `matches.csv` and `deliveries.csv`. Legacy team names (e.g., Delhi Daredevils) were correctly re-mapped to modern names (Delhi Capitals) to maintain analytical continuity.

### Exploratory Data Analysis (EDA)
- By grouping actions by striker/bowler, we identified the **Top Run Scorers** and **Top Wicket Takers** over IPL history.
- Toss dynamics favored teams choosing to field significantly, highlighting chasing as a prominent modern strategy.
- We charted match wins by team, exposing dominant legacy blocks.

### Feature Engineering & Modeling
- **Features Extracted**: Team names, Playing Venue, Match City, Toss Winner Flag, Toss Decision Batting Flag.
- **Models Used**: Logistic Regression, Random Forest, XGBoost.
- The categorical features were encoded via a robust `ColumnTransformer`, ensuring unseen data handles safely without terminating prediction pipelines. The best-performing pipeline was exported utilizing `joblib`.

### Dashboarding
We implemented an interactive app using local `Streamlit`. The dashboard consumes saved visual artifacts directly into UI tabs and loads the persistent `.pkl` machine-learning model to allow users dynamically input custom team/venue scenarios and evaluate predicted probabilities on the fly.

## Conclusion
The data pipeline built is robust to ongoing additions. Any new season added to Cricsheet can easily trace the `src/` modular pipeline, re-processing raw CSVs, updating visualizations in `visuals/`, outputting a retrained XGBoost model to `models/`, and reflecting transparent updates throughout the `dashboard/app.py` stream.
