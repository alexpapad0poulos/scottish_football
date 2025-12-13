import pandas as pd

def apply_split(table_df, top_n=6):
    """
    Split table into top and bottom groups after Phase 1.
    Returns two lists of teams.
    """
    top = table_df.iloc[:top_n].index.tolist()
    bottom = table_df.iloc[top_n:].index.tolist()
    return top, bottom

def get_pre_split_matches(matches_df, matches_per_team=33):
    n_teams = len(set(matches_df['home']) | set(matches_df['away']))
    total_pre_split_matches = (n_teams * matches_per_team) // 2
    return matches_df.iloc[:total_pre_split_matches].copy()


def encode_teams(matches_df):
    teams = sorted(set(matches_df['home']) | set(matches_df['away']))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    idx_to_team = {i: team for team, i in team_to_idx.items()}

    matches_df['home_idx'] = matches_df['home'].map(team_to_idx)
    matches_df['away_idx'] = matches_df['away'].map(team_to_idx)

    return matches_df, team_to_idx, idx_to_team