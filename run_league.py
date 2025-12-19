from src.load_data import load_matches
from src.league_formats import get_pre_split_matches, encode_teams
from src.likelihood import MultiSeasonFootballLikelihood, get_priors
import bilby
import pandas as pd
import os

# ---------- Pass 1: load + discover teams ----------
season_csvs = os.listdir('./data')
season_csvs = [os.path.join('./data', f) for f in season_csvs if f.endswith('.csv')]

all_pre_split = []
for path in season_csvs:
    print('Loading data from:', path)

    matches = load_matches(path)

    print('Loaded from:', path)
    pre = get_pre_split_matches(matches)
    all_pre_split.append(pre)

combined = pd.concat(all_pre_split, ignore_index=True)
teams, _, team_to_idx, idx_to_team = encode_teams(combined)
print('Unique teams found:', teams)
n_teams = len(team_to_idx)

# ---------- Pass 2: encode per season ----------
season_data = []
for pre in all_pre_split:
    enc = pre.copy()
    enc['home_idx'] = enc['home'].map(team_to_idx)
    enc['away_idx'] = enc['away'].map(team_to_idx)

    season_data.append(dict(
        home_idx=enc['home_idx'].values,
        away_idx=enc['away_idx'].values,
        home_goals=enc['home_goals'].values,
        away_goals=enc['away_goals'].values,
    ))

# ---------- Likelihood & priors ----------
likelihood = MultiSeasonFootballLikelihood(
    season_data=season_data,
    n_teams=n_teams
)

priors = get_priors(n_teams)

# ---------- Sampling ----------
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler='nessai',
    nlive=700,
    outdir='out_all_data/',
    label='football_multiseason',
    npool=16
)

result.plot_corner()
