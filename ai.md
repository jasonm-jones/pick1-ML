AI Project: NFL Eliminator Pick 1 Survivor Challenge
Objective

The ultimate goal of this project is to win the 2025 ESPN NFL Eliminator Pick 1 fantasy football challenge by building a Python-based machine learning system that recommends the optimal weekly NFL pick.

This game is a single-elimination survivor pool:

Each week, you pick one NFL team to win.

If that team wins (or ties), you advance to the next week.

If that team loses, you are eliminated.

You cannot pick the same team more than once per season.

Survive all 18 weeks of the NFL season to win.

Because of these rules, choosing the best available team each week requires balancing win probability with future availability.

Data Source

We will use historical survivor pool data from SurvivorGrid
, which provides:

NFL game matchups and results (back to 2010)

Betting odds, spreads, and win probabilities

Elimination pool pick trends

URL pattern for weekly data:

https://www.survivorgrid.com/YYYY/WW


Where YYYY = year (2010–2024) and WW = week number (1–18).

Phase 1: Data Collection

Task: Extract all historical data into CSV files for training.

For each year (2010–2024) and each week (1–18):

Scrape SurvivorGrid page

Parse matchup, teams, odds, spreads, pick percentages, and outcomes

Store results in /data/YYYY_WW.csv

File format example (2019_03.csv):

Home Team	Away Team	Spread	Win Probability	Survivor Pick %	Result	Winner
NE	NYJ	-14	85%	25%	30-14	NE
Phase 2: Data Engineering

Clean and normalize data (teams, outcomes, spreads).

Create features:

Team win probability

Point spread

Survivor pick popularity

Historical performance (team strength, home/away, injuries if available).

Build season-level state: tracking which teams have been used.

Phase 3: Modeling Approach

We will experiment with multiple ML/optimization methods:

Supervised Learning: Predict probability of each team winning (classification/regression).

Survivor Strategy Optimization:

Constraint: a team can only be used once.

Use dynamic programming, reinforcement learning, or Monte Carlo simulation to simulate seasons and maximize survival probability.

Potential Models:

Logistic Regression / Random Forest (baseline)

Gradient Boosted Trees (XGBoost/LightGBM)

Neural Networks for sequence prediction (LSTM/Transformer).

Phase 4: Training & Evaluation

Train models on 2010–2022 data.

Validate on 2023 season.

Test on 2024 season (final before 2025 play).

Evaluation metric: probability of survival per week and across full season.

Phase 5: Deployment (2025 Season)

Each week:

Input current matchups and spreads from SurvivorGrid.

Model outputs best pick(s), factoring in future weeks.

Track picks and results throughout the season.

Stretch Goals

Integrate betting market data (Vegas lines, implied win probability).

Scrape injury reports and advanced analytics (e.g., DVOA, EPA).

Build a weekly dashboard to visualize survival odds.

Folder Structure
nfl-survivor-ai/
│
├── data/                # historical csv files (YYYY_WW.csv)
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Python modules
│   ├── scrape.py        # data extraction from SurvivorGrid
│   ├── clean.py         # data cleaning & preprocessing
│   ├── model.py         # ML training and optimization
│   └── simulate.py      # survivor strategy simulations
├── ai.md                # project overview (this file)
└── requirements.txt     # Python dependencies
