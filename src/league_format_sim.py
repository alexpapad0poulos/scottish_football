import numpy as np
from src.table import initialise_table, update_table
from tqdm import tqdm as _tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def simulate_match(params, home_idx, away_idx):
    mu = params['mu']
    home_adv = params['home_adv']
    attack = params['attack']
    defence = params['defence']

    lam_home = np.exp(mu + home_adv + attack[home_idx] - defence[away_idx])
    lam_away = np.exp(mu + attack[away_idx] - defence[home_idx])

    hg = np.random.poisson(lam_home)
    ag = np.random.poisson(lam_away)

    return hg, ag

def unpack_params(row, n_teams):
    attack = np.array([row[f'attack_raw_{i}'] for i in range(n_teams)]) * row['sigma_attack']
    defence = np.array([row[f'defence_raw_{i}'] for i in range(n_teams)])

    attack -= attack.mean()
    defence -= defence.mean()

    return dict(
        mu=row['mu'],
        home_adv=row['home_adv'],
        attack=attack,
        defence=defence,
    )

def simulate_season(fixtures, params):
    table = initialise_table()

    for home, away in fixtures:
        hg, ag = simulate_match(params, home, away)
        update_table(table, home, away, hg, ag)

    return table

def fixtures_no_split(n_teams):
    fixtures = []
    for i in range(n_teams):
        for j in range(n_teams):
            if i != j:
                fixtures.append((i, j))
    return fixtures

def fixtures_split(params, n_teams, all_team_indices):
    # First 33 games: 3x round robin
    fixtures = []
    for _ in range(3):
        for i in range(n_teams):
            for j in range(n_teams):
                if i != j:
                    fixtures.append((i, j))

    # Simulate first phase
    table = initialise_table(all_team_indices)
    for home, away in fixtures:
        hg, ag = simulate_match(params, home, away)
        update_table(table, home, away, hg, ag)

    # Rank teams
    ranked = sorted(
        table.items(),
        key=lambda x: (x[1]['Pts'], x[1]['GF'] - x[1]['GA']),
        reverse=True
    )
    top6 = [t[0] for t in ranked[:6]]
    bot6 = [t[0] for t in ranked[6:]]

    # Split fixtures (single round robin in each half)
    split_fixtures = []
    for group in [top6, bot6]:
        for i in group:
            for j in group:
                if i != j:
                    split_fixtures.append((i, j))

    return fixtures + split_fixtures

def simulate_season(fixtures, params, teams):
    table = initialise_table(teams)

    for home, away in fixtures:
        hg, ag = simulate_match(params, home, away)
        update_table(table, home, away, hg, ag)

    return table

def gap_metrics(table, old_firm_idxs):
    pts = np.array([table[i]['Pts'] for i in table])

    of_pts = pts[old_firm_idxs]
    non_of_pts = np.array(
        [pts[i] for i in table if i not in old_firm_idxs]
    )

    ranking = np.argsort(pts)[::-1]
    top2 = set(ranking[:2])

    p_top2 = int(top2.issubset(set(old_firm_idxs)))

    return {
        'gap_mean': of_pts.mean() - non_of_pts.mean(),
        'gap_median': np.median(of_pts) - np.median(non_of_pts),
        'p_top2': p_top2,
    }

def run_comparison(result, n_teams, old_firm_idxs, n_sims=1000):
    gaps_split = []
    gaps_nosplit = []

    posterior = result.posterior.sample(n_sims)

    all_team_indices = list(range(n_teams))

    for _, row in _tqdm(posterior.iterrows()):
        params = unpack_params(row, n_teams)

        # No split
        f_ns = fixtures_no_split(n_teams)
        table_ns = simulate_season(f_ns, params, all_team_indices)
        gaps_nosplit.append(gap_metrics(table_ns, old_firm_idxs))

        # Split
        f_sp = fixtures_split(params, n_teams,  all_team_indices)
        table_sp = simulate_season(f_sp, params, all_team_indices)
        gaps_split.append(gap_metrics(table_sp, old_firm_idxs))

    return np.array(gaps_nosplit), np.array(gaps_split)

def dictlist_to_metric_dict(dict_list):
    metrics = dict_list[0].keys()
    return {m: np.array([d[m] for d in dict_list]) for m in metrics}

def visualise_metrics(gaps_ns_dict, gaps_sp_dict, metrics=None):
    if metrics is None:
        metrics = list(gaps_ns_dict.keys())

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ns = gaps_ns_dict[metric]
        sp = gaps_sp_dict[metric]

        if metric == 'p_top2':
            # Bar plot for probabilities
            ax.bar(
                ['No split', 'Split'],
                [ns.mean(), sp.mean()],
                yerr=[
                    ns.std() / np.sqrt(len(ns)),
                    sp.std() / np.sqrt(len(sp))
                ],
                alpha=0.7
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("P(Old Firm finish top 2)")
            continue

        # Density plots
        sns.kdeplot(ns, fill=True, ax=ax, label="No split")
        sns.kdeplot(sp, fill=True, ax=ax, label="Split")

        # Median and 68% CI
        def median_ci(arr):
            med = np.median(arr)
            ci68 = np.percentile(arr, [16, 84])
            return med, ci68

        med_ns, ci68_ns = median_ci(ns)
        med_sp, ci68_sp = median_ci(sp)

        ax.axvline(med_ns, color='blue', linestyle='--')
        ax.fill_betweenx(
            [0, ax.get_ylim()[1]],
            ci68_ns[0], ci68_ns[1],
            color='blue', alpha=0.2
        )

        ax.axvline(med_sp, color='orange', linestyle='--')
        ax.fill_betweenx(
            [0, ax.get_ylim()[1]],
            ci68_sp[0], ci68_sp[1],
            color='orange', alpha=0.2
        )

        # Posterior difference
        diff = sp - ns
        sns.kdeplot(diff, color='green', linestyle=':', ax=ax, label='Split - No split')

        ax.set_title(metric)
        ax.set_xlabel("Points gap")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.savefig("gap_metrics_comparison.png")
    plt.show()
