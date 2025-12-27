from src.league_format_sim import unpack_params, fixtures_no_split, simulate_match
import numpy as np
import pandas as pd
import random

def run_simulation(team_pool, injected_params_row, num_seasons=5, teams_per_season=12):
    n_total_teams = len(team_pool)
    idx_to_team = {i: name for i, name in enumerate(team_pool)}
    
    # Unpack the global parameters for all 15 teams once
    params = unpack_params(injected_params_row, n_total_teams)
    save_hyperparameters(params, n_total_teams)


    for s in range(1, num_seasons + 1):
        # 1. Select a unique subset of teams for this specific season
        season_team_indices = random.sample(range(n_total_teams), teams_per_season)
        print(season_team_indices)
        print([idx_to_team[i] for i in season_team_indices]) 
        # 2. Generate fixtures
        fixtures = fixtures_no_split(len(season_team_indices), rounds = 2)
        
        # 3. Simulate matches
        season_results = []
        for h_idx, a_idx in fixtures:
            hg, ag = simulate_match(params, h_idx, a_idx)
            season_results.append({
                'HomeTeam': idx_to_team[h_idx],
                'AwayTeam': idx_to_team[a_idx],
                'FTHG': hg,
                'FTAG': ag
            })
        
        # 4. Create DataFrame and Export to unique CSV
        df = pd.DataFrame(season_results)
        file_name = f'simulated_season_{s}.csv'
        df.to_csv(file_name, index=False)
        print(f"Created {file_name} with {len(df)} matches.")

def save_hyperparameters(params, team_pool, filename="injected_parameters.txt"):
    """Saves global and team-specific parameters to a formatted text file."""
    with open(filename, 'w') as f:
        f.write("=== GLOBAL HYPERPARAMETERS ===\n")
        f.write(f"mu: {params['mu']}\n")
        f.write(f"home_adv: {params['home_adv']}\n")
        f.write(f"r (over-dispersion): {params['r']}\n\n")
        
        f.write("=== TEAM SPECIFIC PARAMETERS ===\n")
        f.write(f"{'Team Name':<20} | {'Attack':<10} | {'Defence':<10}\n")
        f.write("-" * 45 + "\n")
        
        for i in range(team_pool):
            att = params['attack'][i]
            dfn = params['defence'][i]
            f.write(f"{i:<20} | {att:>10.4f} | {dfn:>10.4f}\n")
            
    print(f"Detailed parameters saved to {filename}")

all_teams = [
    'Aberdeen', 'Celtic', 'Dundee', 'Dundee United', 'Dunfermline', 
    'Hearts', 'Hibernian', 'Inverness C', 'Kilmarnock', 'Livingston', 
    'Motherwell', 'Rangers', 'St Mirren', 'Ross County', 'St Johnstone'
]


example_injected_row = {
    'mu': 0.08, 'home_adv': 0.22, 'r': 20., 'sigma_attack': 1., 'sigma_defence': 1.,
    **{f'attack_raw_{i}': np.random.normal(0, 0.25) for i in range(15)},
    **{f'defence_raw_{i}': np.random.normal(0, 0.25) for i in range(15)}
}

    

run_simulation(all_teams, example_injected_row)
