import bilby
import numpy as np
from scipy.stats import poisson


def get_priors(n_teams):
    priors = dict()
    priors['mu'] = bilby.core.prior.Uniform(-5, 5)
    priors['home_adv'] = bilby.core.prior.Uniform(-2, 2)
    priors['sigma_attack'] = bilby.core.prior.Uniform(0.01, 1)
    priors['sigma_defence'] = bilby.core.prior.Uniform(0.01, 1)
    
    for i in range(n_teams):
        priors[f'attack_raw_{i}'] = bilby.core.prior.Uniform(-20, 20)
        priors[f'defence_raw_{i}'] = bilby.core.prior.Uniform(-20, 20)
    return priors

class MultiSeasonFootballLikelihood(bilby.Likelihood):
    def __init__(self, season_data, n_teams):
        self.season_data = season_data
        self.n_teams = n_teams

        parameters = dict(
            mu=0.0,
            home_adv=0.0,
            sigma_attack=0.2,
            sigma_defence=0.2,
        )

        for i in range(n_teams):
            parameters[f'attack_raw_{i}'] = 0.0
            parameters[f'defence_raw_{i}'] = 0.0

        super().__init__(parameters=parameters)

    def log_likelihood(self):
        mu = self.parameters['mu']
        home_adv = self.parameters['home_adv']
        sigma_a = self.parameters['sigma_attack']
        sigma_d = self.parameters['sigma_defence']

        attack = np.array([
            self.parameters[f'attack_raw_{i}'] for i in range(self.n_teams)
        ]) * sigma_a
        defence = np.array([
            self.parameters[f'defence_raw_{i}'] for i in range(self.n_teams)
        ]) * sigma_d

        attack -= attack.mean()
        defence -= defence.mean()

        # Concatenate all seasons
        home_idx = np.concatenate([s['home_idx'] for s in self.season_data])
        away_idx = np.concatenate([s['away_idx'] for s in self.season_data])
        home_goals = np.concatenate([s['home_goals'] for s in self.season_data])
        away_goals = np.concatenate([s['away_goals'] for s in self.season_data])

        # Compute Poisson rates
        lam_home = np.exp(mu + home_adv + attack[home_idx] - defence[away_idx])
        lam_away = np.exp(mu + attack[away_idx] - defence[home_idx])

        # Vectorized log-likelihood
        ll = np.sum(poisson.logpmf(home_goals, lam_home)) + np.sum(poisson.logpmf(away_goals, lam_away))
        return ll
