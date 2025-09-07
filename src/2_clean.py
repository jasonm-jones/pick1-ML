import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
CLEANED_DIR = Path("data-cleaned")
CLEANED_DIR.mkdir(exist_ok=True)

def clean_week(file_path: Path):
    """
    Clean a single weekly CSV.
    Adds opponent stats so each row has both team and opponent data.
    """
    df = pd.read_csv(file_path)

    # Standardize/normalize column names and map common aliases
    df.columns = [c.strip().lower() for c in df.columns]
    alias_map = {
        "team_name": "team",
        "school": "team",
        "opponent_team": "opponent",
        "opp": "opponent",
        "w_pct": "win_probability",
        "p_pct": "pick_percentage",
        "team_result": "result",
    }
    df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns}, inplace=True)

    # Add year and week from filename
    filename = file_path.stem  # e.g., "2024_09"
    parts = filename.split("_")
    if len(parts) != 2:
        raise ValueError(f"Filename {filename} does not match expected format YEAR_WEEK.csv")
    year_str, week_str = parts
    df['year'] = int(year_str)
    week_num = int(week_str)  # handles leading zeros
    df['week'] = week_num

    # If 'opponent' missing, derive from wide week columns (e.g., '9', '9_spread')
    if 'opponent' not in df.columns:
        wk_col = str(week_num)
        spr_col = f"{wk_col}_spread"
        if wk_col not in df.columns:
            raise ValueError(
                f"Missing 'opponent' column and week column '{wk_col}' not found in {file_path}. "
                f"Found columns: {list(df.columns)}"
            )
        df['opponent'] = df[wk_col].astype(str).str.strip()
        if 'spread' not in df.columns and spr_col in df.columns:
            df['spread'] = pd.to_numeric(df[spr_col], errors='coerce')

    # Ensure 'team' and 'opponent' exist
    if 'team' not in df.columns or 'opponent' not in df.columns:
        raise ValueError(
            f"Missing 'team' or 'opponent' columns in {file_path}. "
            f"Found columns: {list(df.columns)}"
        )

    # Normalize opponent values (remove '@', drop BYE rows)
    df['opponent'] = df['opponent'].astype(str).str.replace('@', '', regex=False).str.strip()
    df = df[df['opponent'].str.upper() != 'BYE']

    # Convert/derive helpful numerics
    if 'spread' in df.columns:
        df['spread'] = pd.to_numeric(df['spread'], errors='coerce')
    if 'win' in df.columns:
        df['win'] = pd.to_numeric(df['win'], errors='coerce')
    elif 'result' in df.columns:
        df['win'] = df['result'].astype(str).str.upper().str[0].map({'W': 1, 'L': 0})

    # Build opponent lookup from whatever exists
    opp_src_cols = ['team', 'spread', 'ev', 'win_probability', 'pick_percentage', 'future_val', 'result', 'win']
    present_opp_cols = [c for c in opp_src_cols if c in df.columns]
    df_opp = df[present_opp_cols].copy()
    rename_map = {
        'team': 'opp_team',
        'spread': 'opp_spread',
        'ev': 'opp_ev',
        'win_probability': 'opp_win_probability',
        'pick_percentage': 'opp_pick_percentage',
        'future_val': 'opp_future_val',
        'result': 'opp_result',
        'win': 'opp_win'
    }
    df_opp.rename(columns={k: v for k, v in rename_map.items() if k in df_opp.columns}, inplace=True)

    # Merge opponent stats into main df
    df = df.merge(df_opp, left_on='opponent', right_on='opp_team', how='left')

    # Reorder columns
    final_cols = [
        'team', 'opponent', 'spread', 'win_probability', 'pick_percentage',
        'ev', 'future_val', 'result', 'year', 'week', 'win',
        'opp_team', 'opp_spread', 'opp_win_probability', 'opp_pick_percentage',
        'opp_ev', 'opp_future_val', 'opp_result', 'opp_win'
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    # Save cleaned CSV
    cleaned_path = CLEANED_DIR / file_path.name
    df.to_csv(cleaned_path, index=False)
    print(f"✅ Cleaned {file_path.name} → {cleaned_path}")

def clean_all():
    for file_path in DATA_DIR.glob("*.csv"):
        clean_week(file_path)

if __name__ == "__main__":
    clean_all()
