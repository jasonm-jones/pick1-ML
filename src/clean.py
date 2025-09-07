import os
import pandas as pd

DATA_DIR = "data"
CLEAN_DIR = "data-cleaned"

def clean_week(year: int, week: int, path: str) -> pd.DataFrame:
    """
    Clean a single week CSV and enrich with opponent data.
    """
    df = pd.read_csv(path)

    # Rename common columns
    rename_map = {
        "EV": "ev",
        "W_pct": "win_probability",
        "P_pct": "pick_percentage",
        "Team": "team",
        "Team_Result": "result",
        "Future_Val": "future_val"
    }

    # Only rename "1" and "1_spread" if present
    if "1" in df.columns:
        rename_map["1"] = "opponent"
    else:
        df["opponent"] = None

    if "1_spread" in df.columns:
        rename_map["1_spread"] = "spread"
    else:
        df["spread"] = None

    df = df.rename(columns=rename_map)

    # Keep only relevant columns
    base_cols = ["team", "opponent", "spread", "win_probability",
                 "pick_percentage", "ev", "future_val", "result"]
    df = df[[c for c in base_cols if c in df.columns]].copy()

    # Add year/week
    df["year"] = year
    df["week"] = week

    # ✅ Add numeric win target (1=win, 0=loss, NaN=bye/missing)
    df["win"] = df["result"].map(lambda x: 1 if str(x).upper().startswith("W") else (0 if str(x).upper().startswith("L") else None))

    # Build opponent table for self-join
    df_opp = df.rename(columns={
        "team": "opp_team",
        "opponent": "opp_opponent",
        "spread": "opp_spread",
        "win_probability": "opp_win_probability",
        "pick_percentage": "opp_pick_percentage",
        "ev": "opp_ev",
        "future_val": "opp_future_val",
        "result": "opp_result",
        "win": "opp_win"
    })

    # Merge on opponent == opp_team
    merged = pd.merge(
        df,
        df_opp,
        left_on=["year", "week", "opponent"],
        right_on=["year", "week", "opp_team"],
        how="inner"
    )

    # Drop duplicate join helper
    merged = merged.drop(columns=["opp_opponent"])

    return merged


def clean_all():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".csv"):
            continue

        year, week = filename.replace(".csv", "").split("_")
        path = os.path.join(DATA_DIR, filename)

        try:
            df = clean_week(int(year), int(week), path)

            out_path = os.path.join(CLEAN_DIR, filename)
            df.to_csv(out_path, index=False)
            print(f"✅ Cleaned {filename} → {out_path}")
        except Exception as e:
            print(f"⚠️ Skipping {filename} due to error: {e}")


if __name__ == "__main__":
    clean_all()
