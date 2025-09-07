import os
import time
import csv
import re
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.survivorgrid.com/{year}/{week}"
DATA_DIR = "../data"

NBSP = "\xa0"

def normalize_header(text: str) -> str:
    if not text:
        return ""
    text = text.replace(NBSP, " ").strip()
    text = text.replace("EV▼", "EV").replace("EV▲", "EV")
    text = text.replace("W%", "W_pct")
    text = text.replace("P%", "P_pct")
    text = text.replace("Future Val", "Future_Val")
    return text

def parse_pct(text: str):
    if not text:
        return None
    t = text.strip().replace("%", "")
    try:
        return float(t)
    except ValueError:
        return None

SPREAD_RE = re.compile(
    r"""
    ^\s*
    (?P<opp>@?[A-Z]{2,3})     # Opponent like SF, @SF, NYG, KCCan vary but usually 2-3 letters
    (?:
        \s*
        (?:
            (?P<sign>[+-])(?P<num>\d+(?:\.\d+)?)  # +/-number
            |
            (?P<pk>PK)                            # Pick'em
        )
    )?
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

def parse_week_cell(td):
    """
    Extract opponent and spread from a week cell.
    Supports:
      - Text like '@SF-5.5', 'KCPK', 'DET+3', 'BYE'
      - Subscript/superscript spreads inside tags
    Returns (opp, spread) where spread is float or None; opp may be 'BYE' or None.
    """
    if td is None:
        return (None, None)

    # Attempt to isolate any sub/sup content first
    sub = td.find(["sub", "sup", "small", "span"], string=re.compile(r"(PK|^[+-]?\d)"))
    raw_spread_text = sub.get_text(strip=True) if sub else None

    # Clean text without child sub/sup
    for tag in td.find_all(["sub", "sup", "small", "span"]):
        tag.extract()
    main_text = td.get_text("", strip=True)

    # If we extracted spread in a tag and main_text is the opponent code
    if raw_spread_text:
        if raw_spread_text.upper() == "PK":
            spread = 0.0
        else:
            try:
                spread = float(raw_spread_text)
            except ValueError:
                # Sometimes spread is like "+3.5" or "-7"; fallback remove sign handling below
                mnum = re.search(r"[+-]?\d+(?:\.\d+)?", raw_spread_text)
                spread = float(mnum.group(0)) if mnum else None
        opp = main_text or None
        return (opp if opp else None, spread)

    # Fallback: parse from combined text like '@SF-5.5' or 'KCPK'
    text = main_text or td.get_text("", strip=True)
    if not text:
        return (None, None)

    # Handle BYE explicitly
    if text.upper() in ("BYE",):
        return ("BYE", None)

    m = SPREAD_RE.match(text)
    if m:
        opp = m.group("opp")
        if m.group("pk"):
            spread = 0.0
        elif m.group("sign") and m.group("num"):
            sign = -1.0 if m.group("sign") == "-" else 1.0
            spread = sign * float(m.group("num"))
        else:
            spread = None
        return (opp, spread)

    # Last fallback: try to split trailing +/-number
    m2 = re.search(r"([+-]\d+(?:\.\d+)?)\s*$", text)
    spread = float(m2.group(1)) if m2 else None
    opp = re.sub(r"([+-]\d+(?:\.\d+)?)\s*$", "", text).strip() or None
    return (opp, spread)

def parse_team_cell(text: str):
    # e.g., "NYG(W)" -> ("NYG", "W") or "DAL(L)" -> ("DAL", "L")
    if not text:
        return (None, None)
    m = re.match(r"^\s*([A-Z]{2,3})\s*\((W|L)\)\s*$", text.strip(), re.IGNORECASE)
    if m:
        return (m.group(1), m.group(2).upper())
    return (text.strip(), None)

def extract_future_val(td):
    if td is None:
        return None
    # Prefer explicit attributes (SurvivorGrid uses data-sort-value on the fv cell)
    for attr in ("data-sort-value", "data-future", "data-value", "data-val", "data-fv"):
        if td.has_attr(attr):
            try:
                return float(td[attr])
            except (ValueError, TypeError):
                pass
    # If a child holds the attribute
    child = td.find(attrs={"data-sort-value": True})
    if child:
        try:
            return float(child["data-sort-value"])
        except (ValueError, TypeError, KeyError):
            pass
    # Last fallback: parse any numeric text
    txt = td.get_text(strip=True)
    if not txt:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", txt.replace(",", ""))
    return float(m.group(0)) if m else None

def fetch_week(year: int, week: int):
    """
    Fetch SurvivorGrid weekly page and parse matchups.
    Returns a list of dictionaries with parsed data:
      - EV, W_pct, P_pct, Team, Team_Result
      - For weeks 1..17: <week> (opponent string), <week>_spread (float)
      - Future_Val (float if present)
    """
    url = BASE_URL.format(year=year, week=week)
    print(f"Fetching {url} ...")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the grid table by looking for the "Team" and a digit week header
    table = None
    for t in soup.find_all("table"):
        th_texts = [normalize_header(th.get_text(strip=True)) for th in t.find_all("th")]
        if "Team" in th_texts and any(h.isdigit() for h in th_texts):
            table = t
            break

    if not table:
        print(f"⚠️ No suitable table found for {year} week {week}")
        return []

    raw_headers = [normalize_header(th.get_text(strip=True)) for th in table.find("tr").find_all("th")]

    # Map header index for special columns
    header_idx = {h: i for i, h in enumerate(raw_headers)}
    week_headers = [h for h in raw_headers if h.isdigit()]
    has_future = "Future_Val" in header_idx

    games = []
    for row in table.find_all("tr")[1:]:
        tds = row.find_all("td")
        if not tds:
            continue

        # Build a dict carefully to parse special cells
        row_data = {}

        # EV
        ev_td = tds[header_idx["EV"]] if "EV" in header_idx and header_idx["EV"] < len(tds) else None
        ev_txt = ev_td.get_text(strip=True) if ev_td else ""
        try:
            row_data["EV"] = float(ev_txt) if ev_txt else None
        except ValueError:
            row_data["EV"] = None

        # W% and P%
        w_td = tds[header_idx["W_pct"]] if "W_pct" in header_idx and header_idx["W_pct"] < len(tds) else None
        p_td = tds[header_idx["P_pct"]] if "P_pct" in header_idx and header_idx["P_pct"] < len(tds) else None
        row_data["W_pct"] = parse_pct(w_td.get_text(strip=True) if w_td else None)
        row_data["P_pct"] = parse_pct(p_td.get_text(strip=True) if p_td else None)

        # Team
        team_td = tds[header_idx["Team"]] if "Team" in header_idx and header_idx["Team"] < len(tds) else None
        team_txt = team_td.get_text(strip=True) if team_td else ""
        team, team_res = parse_team_cell(team_txt)
        row_data["Team"] = team
        row_data["Team_Result"] = team_res  # W/L indicator if present

        # Each week: split into opponent and spread
        for h in week_headers:
            idx = header_idx[h]
            td = tds[idx] if idx < len(tds) else None
            opp, spread = parse_week_cell(td)
            row_data[h] = opp  # keep the opponent text (e.g., '@SF', 'BYE', 'DET')
            row_data[f"{h}_spread"] = spread  # numeric spread (PK -> 0.0, else float)

        # Future_Val: use header column if present, else fall back to td.fv
        fv_td = None
        if "Future_Val" in header_idx and header_idx["Future_Val"] < len(tds):
            fv_td = tds[header_idx["Future_Val"]]
        else:
            fv_td = row.find("td", class_=re.compile(r"\bfv\b"))
        row_data["Future_Val"] = extract_future_val(fv_td)

        games.append(row_data)

    return games


def save_week(year: int, week: int, games: list):
    """
    Save weekly games to CSV file with separate opponent and spread per week.
    """
    if not games:
        return

    # Deterministic column order
    base_cols = ["EV", "W_pct", "P_pct", "Team", "Team_Result"]
    # Derive week columns from first row
    week_nums = sorted([int(k) for k in games[0].keys() if k.isdigit()])
    week_cols = []
    for w in week_nums:
        week_cols.append(str(w))
        week_cols.append(f"{w}_spread")
    cols = base_cols + week_cols + ["Future_Val"]

    os.makedirs(DATA_DIR, exist_ok=True)
    filename = os.path.join(DATA_DIR, f"{year}_{week:02}.csv")

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for g in games:
            writer.writerow({k: g.get(k) for k in cols})

    print(f"✅ Saved {filename}")


def scrape_range(start_year=2010, end_year=2024, weeks=18, delay=1.0):
    for year in range(start_year, end_year + 1):
        for week in range(1, weeks + 1):
            try:
                games = fetch_week(year, week)
                save_week(year, week, games)
                time.sleep(delay)
            except Exception as e:
                print(f"❌ Error fetching {year} week {week}: {e}")


if __name__ == "__main__":
    scrape_range()