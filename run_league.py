from src.load_data import load_matches
from src.league_formats import get_pre_split_matches, encode_teams
from src.likelihood import FootballLikelihood, get_priors
import bilby

matches = load_matches("data/scotland_201617.csv")
pre_split = get_pre_split_matches(matches)
encoded, team_to_idx, idx_to_team = encode_teams(pre_split)

home_idx = encoded['home_idx'].values
away_idx = encoded['away_idx'].values
home_goals = encoded['home_goals'].values
away_goals = encoded['away_goals'].values
n_teams = len(team_to_idx)

likelihood = FootballLikelihood(home_idx, away_idx, home_goals, away_goals, n_teams)
priors = get_priors(n_teams)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler='dynesty',
    nlive=500,  
    outdir='out/',
    label='football_pre_split'
)

result.plot_corner()
