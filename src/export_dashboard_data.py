import json
import pandas as pd

MATCHES_CSV = "data/processed/matches.csv"
DELIVERIES_CSV = "data/processed/deliveries.csv"
OUTPUT_JSON = "static/data.json"

# Load data
matches_df = pd.read_csv(MATCHES_CSV, dtype=str)
deliveries_df = pd.read_csv(DELIVERIES_CSV, low_memory=False, dtype=str)

# Convert numeric columns
for col in ["runs_off_bat"]:
    if col in deliveries_df.columns:
        deliveries_df[col] = pd.to_numeric(deliveries_df[col], errors="coerce").fillna(0).astype(int)

# Prepare match-level data
matches = matches_df[[
    "id",
    "team1",
    "team2",
    "venue",
    "city",
    "toss_winner",
    "toss_decision",
    "winner"
]].fillna("").to_dict("records")

teams = sorted({t for row in matches for t in (row["team1"], row["team2"]) if t})
venues = sorted({row["venue"] for row in matches if row["venue"]})
cities = sorted({row["city"] for row in matches if row["city"]})

toss_summary = matches_df["toss_decision"].fillna("Unknown").value_counts().to_dict()

# Top teams by wins
team_wins = matches_df["winner"].fillna("Unknown").value_counts().head(10)
top_teams = [{"team": team, "wins": int(wins)} for team, wins in team_wins.items() if team != "Unknown"]

# Top scorers
scorer_runs = deliveries_df.groupby("striker")["runs_off_bat"].sum().sort_values(ascending=False).head(10)
top_scorers = [{"player": player, "runs": int(runs)} for player, runs in scorer_runs.items() if player and player != "nan"]

# Top bowlers
valid_wickets = deliveries_df[
    deliveries_df["wicket_type"].notna() &
    ~deliveries_df["wicket_type"].isin(["run out", "retired hurt", "obstructing the field"])
]
bowler_wickets = valid_wickets["bowler"].value_counts().head(10)
top_bowlers = [{"player": player, "wickets": int(wickets)} for player, wickets in bowler_wickets.items() if player and player != "nan"]

# Player stats
runs_by_player = deliveries_df.groupby("striker")["runs_off_bat"].sum().rename("runs")
wickets_by_player = valid_wickets["bowler"].value_counts().rename("wickets")
player_stats = (pd.concat([runs_by_player, wickets_by_player], axis=1).fillna(0)
                .rename_axis("player").reset_index())
player_stats["runs"] = player_stats["runs"].astype(int)
player_stats["wickets"] = player_stats["wickets"].astype(int)
player_stats = player_stats.sort_values(["runs", "wickets"], ascending=False).head(500)
player_stats = player_stats.to_dict("records")

# Aggregate metrics
total_runs = int(deliveries_df["runs_off_bat"].sum())

output = {
    "matches": matches,
    "teams": teams,
    "venues": venues,
    "cities": cities,
    "tossSummary": toss_summary,
    "topTeams": top_teams,
    "topScorers": top_scorers,
    "topBowlers": top_bowlers,
    "playerStats": player_stats,
    "totalRuns": total_runs,
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

print(f"Exported compact dashboard data to {OUTPUT_JSON}")
