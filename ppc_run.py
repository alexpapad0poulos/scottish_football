import bilby
import numpy as np
from src.league_format_sim import visualise_metrics, run_comparison, dictlist_to_metric_dict
import os
import pandas as pd
from csv import QUOTE_NONE # Import constant for quoting


def encode_teams_multi_season(csv_dir):
    all_matches = []
    for fname in os.listdir(csv_dir):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(csv_dir, fname),engine ='python', encoding='utf-8', 
                     encoding_errors='replace', on_bad_lines='skip',
                     quoting=QUOTE_NONE)
            print('Loaded:', fname)
            all_matches.append(df)
    combined = pd.concat(all_matches, ignore_index=True)
    
    # global team index
    teams = sorted(pd.concat([combined['HomeTeam'].astype(str), combined['AwayTeam'].astype(str)]).unique())
    team_to_idx = {team: i for i, team in enumerate(teams)}
    idx_to_team = {i: team for team, i in team_to_idx.items()}

    # encode each season individually
    season_matches_encoded = []
    for df in all_matches:
        df['home_idx'] = df['HomeTeam'].map(team_to_idx)
        df['away_idx'] = df['AwayTeam'].map(team_to_idx)
        season_matches_encoded.append(df)
    
    return season_matches_encoded, team_to_idx, idx_to_team


if __name__ == "__main__":
    # Load posterior result

    csv_dir = "/home/2386233p/scottish_football/data"
    season_matches_encoded, team_to_idx, idx_to_team = encode_teams_multi_season(csv_dir)

    # Load posterior result
    result = bilby.result.read_in_result("out_all_data/football_multiseason_result.json")

    # Old Firm indices (assumes Celtic and Rangers are present in global team list)
    OLD_FIRM = [team_to_idx['Celtic'], team_to_idx['Rangers']]

    n_teams = len(team_to_idx)-1

    print(n_teams, idx_to_team)

    # Run posterior predictive comparison
    gaps_nosplit, gaps_split = run_comparison(result, n_teams, OLD_FIRM, n_sims=2500)

    # Convert list-of-dicts to numeric arrays
    gaps_ns_dict = dictlist_to_metric_dict(gaps_nosplit)
    gaps_sp_dict = dictlist_to_metric_dict(gaps_split)

    # Visualize
    visualise_metrics(gaps_ns_dict, gaps_sp_dict)