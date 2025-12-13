import pandas as pd

def load_matches(csv_path):

    df = pd.read_csv(csv_path)
    df = df.rename(columns={
    'HomeTeam': 'home',
    'AwayTeam': 'away',
    'FTHG': 'home_goals',
    'FTAG': 'away_goals'
    })


    required = ['home', 'away', 'home_goals', 'away_goals']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")


    return df[required].copy()