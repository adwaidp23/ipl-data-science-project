# IPL Data Science End-to-End Workflow

This notebook summarizes the workflow of collecting, cleaning, analyzing, and modeling historical IPL dataset.

## 1. Data Collection
We fetched ball-by-ball and match metadata datasets for the IPL from **Cricsheet** (using CSV zip archive `ipl_csv2.zip`). The raw dataset contains thousands of individual CSV files.

## 2. Data Cleaning & Feature Engineering
We merged the `.csv` and `_info.csv` files into two distinct, structured global datasets:
- **`matches.csv`**: Contains match-level metadata (toss winners, match winners, venues).
- **`deliveries.csv`**: Contains ball-by-ball actions, runs, wickets, extras.

We also unified team names across different seasons (e.g., Delhi Daredevils was merged with Delhi Capitals).

## 3. Exploratory Data Analysis (EDA)
Using Pandas, Matplotlib, and Seaborn, we investigated:
- **Most Successful Teams**: Sorted teams by total match wins.
- **Top Performers**: Summed `runs_off_bat` and counted `wicket_type` exclusions to find the highest run-scorers and wicket-takers.
- **Toss Decisions**: Analyzed the proportion of teams choosing to bat vs. field.

## 4. Machine Learning
We preprocessed features like `team1`, `team2`, `venue`, `city`, `toss_decision`, and `team1_won_toss` using `OneHotEncoder` and `StandardScaler`.

We trained 3 models:
- Logistic Regression
- Random Forest
- XGBoost

The best model was saved to `models/match_predictor.pkl` to power the Streamlit dashboard.