🧩 Suggested Project Setup
Step 1: Modeling game outcomes

Train a classification model to predict win (0/1) from your cleaned dataset.

Inputs:

spread, win_probability, pick_percentage, ev, future_val,

opponent features (opp_*),

year/week for context.

Output: predicted probability of winning.

This gives you an ML-based version of what Vegas odds and historical patterns say.

Step 2: Weekly survival model

Each week of the season, simulate:

Which teams are left (not picked yet)?

What are their predicted win probabilities this week?

What is their future value (how useful are they in future weeks)?

Step 3: Optimization policy

This is where Eliminator-specific strategy comes in. Options:

Greedy strategy: Pick the team with the highest win probability each week.

Future-aware heuristic: Use future_val to avoid burning teams that are uniquely strong later.

Dynamic programming / simulation:

Simulate thousands of possible season paths.

Use expected survival probability as the optimization target.

Choose the path (and next pick) that maximizes odds of surviving to Week 18.

This is where your project differs from a “normal” win predictor. The optimization policy is the special sauce.

Step 4: Validation

Use past seasons (2010–2024) to “play” Survivor pools.

Track how far your strategy survives compared to baselines:

Always pick the highest Vegas favorite.

Random pick among top 3 teams.

Popular pick percentages.

Measure survival rate (median week reached, % reaching Week 18).

🔮 End Goal

By Week 1 of the 2025 NFL season, you’ll have:

A trained ML classifier for per-game win probabilities.

A season simulator/optimizer that accounts for future constraints.

A decision policy that recommends the optimal pick each week to maximize your survival odds.