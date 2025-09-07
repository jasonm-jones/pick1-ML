import os
import pandas as pd

DATA_DIR = "data"
CLEAN_DIR = "data-cleaned"

def clean_week(year: int, week: int, path: str) -> pd.DataFrame:
    """
    Clean a single week CSV and enrich with opponent data.
    Returns a dataframe with both team + opponent features.
    """
    df = pd.read_csv(path)

    # Standardize column names
    df = df.rename(columns={
        "EV": "ev",
        "W_pct": "win_probability",
        "P_pct": "pick_percentage",
        "Team": "team",
        "Team_Result": "result",
        "Future_Val": "future_val"
    })

    # Add year/week columns
    df["year"] = year
    df["week"] = week

    # Keep only the columns we care about now
    base_cols = ["year", "week", "team", "win_probability",
                 "pick_percentage", "ev", "future_val", "result"]
    df = df[base_cols]

    # Opponent merge:
    # SurvivorGrid rows are team-centric, so each matchup has 2 rows (one per team).
    # We self-join on (year, week) but exclude the same team.
    df_opp = df.copy()
    df_opp = df_opp.rename(columns={c: f"opp_{c}" for c in df.columns if c not in ["year", "week"]})

    # Merge back on same year/week, but ensure different team
    merged = pd.merge(df, df_opp,
                      on=["year", "week"],
                      suffixes=("", "_drop"))

    # Drop self-joins (team == opp_team)
    merged = merged[merged["team"] != merged["opp_team"]]

    # Each matchup will appear twice (Team A vs Team B and Team B vs Team A),
    # which is what we want for survivor (team-centric).
    return merged


def clean_all():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".csv"):
            continue

        year, week = filename.replace(".csv", "").split("_")
        path = os.path.join(DATA_DIR, filename)

        df = clean_week(int(year), int(week), path)

        out_path = os.path.join(CLEAN_DIR, filename)
        df.to_csv(out_path, index=False)
        print(f"✅ Cleaned {filename} → {out_path}")


if __name__ == "__main__":
    clean_all()
