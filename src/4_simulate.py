import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = "data-cleaned"  # or data-features if using engineered features
YEARS = range(2010, 2025)
TOTAL_WEEKS = 18

def load_season(year):
    season = {}
    # Find all files for this year
    files = glob.glob(os.path.join(DATA_DIR, f"{year}_*.csv"))

    if not files:
        print(f"⚠️ No files found for year {year}")
        return season

    for f in files:
        # Extract week from filename
        basename = os.path.basename(f)
        week_str = basename.replace(f"{year}_", "").replace(".csv", "")
        week = int(week_str.lstrip("0"))  # handle zero-padded weeks like 01, 02
        df = pd.read_csv(f)

        # Normalize 'win' column
        df["win"] = df["win"].astype(str).str.upper().str.strip()
        
        season[week] = df

    return season


def pick_team(week_df, used_teams, future_weeks):
    """
    Pick the team that maximizes survival probability considering future weeks.
    
    week_df: DataFrame for current week
    used_teams: set of already picked teams
    future_weeks: list of DataFrames for remaining weeks
    """
    df = week_df[~week_df["team"].isin(used_teams)].copy()
    if df.empty:
        return None, None

    # Base score: current week win probability
    df["score"] = df["win_probability"] * df["future_val"]

    # Adjust score based on future availability
    for i, row in df.iterrows():
        team = row["team"]
        # Count how many future weeks this team would be strong
        future_value = 0
        for fweek in future_weeks:
            if team in fweek["team"].values:
                future_value += fweek.loc[fweek["team"] == team, "win_probability"].values[0]
        # Reduce score if team has high future value (save strong teams for later)
        df.at[i, "score"] = df.at[i, "score"] * (1 / (1 + future_value / 100))

    best_row = df.loc[df["score"].idxmax()]
    return best_row["team"], best_row["win_probability"]


def simulate_season(season):
    used_teams = set()
    survival_history = []

    if not season:
        return survival_history

    available_weeks = sorted(season.keys())

    for idx, week in enumerate(available_weeks):
        df = season[week]

        # Remaining weeks for future-aware scoring
        future_weeks = [season[w] for w in available_weeks[idx + 1:]]

        team, prob = pick_team(df, used_teams, future_weeks)
        if team is None:
            print(f"No available picks in week {week}")
            break

        outcome = df.loc[df["team"] == team, "win"].values[0]
        s = str(outcome).upper().strip()
        survived = s in {"W", "WIN", "WON", "TRUE", "T", "1", "YES", "Y"}

        survival_history.append({
            "week": week,
            "team": team,
            "win_probability": prob,
            "survived": survived
        })

        used_teams.add(team)
        if not survived:
            break  # eliminated

    return survival_history



def simulate_all_seasons():
    results = {}
    for year in YEARS:
        print(f"Simulating {year}...")
        season = load_season(year)
        history = simulate_season(season)
        results[year] = history
        print(f"➡ Survived {len(history)} weeks")
    return results

if __name__ == "__main__":
    all_results = simulate_all_seasons()
