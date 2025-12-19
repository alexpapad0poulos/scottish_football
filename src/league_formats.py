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

    home_teams = matches_df['home'].dropna().astype(str)
    away_teams = matches_df['away'].dropna().astype(str)
    n_teams = len(set(home_teams) | set(away_teams))
    total_pre_split_matches = (n_teams * matches_per_team) // 2
    return matches_df.iloc[:total_pre_split_matches].copy()


def encode_teams(matches_df):
    home_teams = matches_df['home'].dropna().astype(str)
    away_teams = matches_df['away'].dropna().astype(str)
    teams = sorted(set(home_teams) | set(away_teams))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    idx_to_team = {i: team for team, i in team_to_idx.items()}

    matches_df['home_idx'] = matches_df['home'].map(team_to_idx)
    matches_df['away_idx'] = matches_df['away'].map(team_to_idx)

    return teams, matches_df, team_to_idx, idx_to_team

def encode_with_existing_mapping(df, team_to_idx):
    df = df.copy()
    df['home_idx'] = df['home'].map(team_to_idx)
    df['away_idx'] = df['away'].map(team_to_idx)
    return df
