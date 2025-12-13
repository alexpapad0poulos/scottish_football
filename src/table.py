

import pandas as pd
import numpy as np
from collections import defaultdict

def initialise_table(teams):
    table = {}
    for t in teams:
        table[t] = {
            'P': 0,
            'W': 0,
            'D': 0,
            'L': 0,
            'GF': 0,
            'GA': 0,
            'GD': 0,
            'Pts': 0
        }
    return table



def update_table(table, home, away, hg, ag):
    table[home]['P'] += 1
    table[away]['P'] += 1


    table[home]['GF'] += hg
    table[home]['GA'] += ag
    table[away]['GF'] += ag
    table[away]['GA'] += hg

    if hg > ag:
        table[home]['W'] += 1
        table[away]['L'] += 1
        table[home]['Pts'] += 3
    elif hg < ag:
        table[away]['W'] += 1
        table[home]['L'] += 1
        table[away]['Pts'] += 3
    else:
        table[home]['D'] += 1
        table[away]['D'] += 1
        table[home]['Pts'] += 1
        table[away]['Pts'] += 1


    table[home]['GD'] = table[home]['GF'] - table[home]['GA']
    table[away]['GD'] = table[away]['GF'] - table[away]['GA']






def compute_table(matches_df):
    teams = sorted(set(matches_df['home']) | set(matches_df['away']))
    table = initialise_table(teams)


    for _, row in matches_df.iterrows():
        update_table(
        table,
        row['home'],
        row['away'],
        int(row['home_goals']),
        int(row['away_goals'])
        )


    table_df = pd.DataFrame.from_dict(table, orient='index')
    table_df = table_df.sort_values(
    by=['Pts', 'GD', 'GF'],
    ascending=False
    )


    table_df['Rank'] = np.arange(1, len(table_df) + 1)
    return table_df