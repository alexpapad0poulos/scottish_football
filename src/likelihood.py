import bilby
import numpy as np
from scipy.stats import poisson


def get_priors(n_teams):
    priors = dict()
    priors['mu'] = bilby.core.prior.Normal(0, 1)
    priors['home_adv'] = bilby.core.prior.Normal(0, 0.5)
    priors['sigma_attack'] = bilby.core.prior.HalfNormal(0.5)
    priors['sigma_defence'] = bilby.core.prior.HalfNormal(0.5)

    for i in range(n_teams):
        priors[f'attack_raw_{i}'] = bilby.core.prior.Normal(0, 1)
        priors[f'defence_raw_{i}'] = bilby.core.prior.Normal(0, 1)

    return priors

class FootballLikelihood(bilby.Likelihood):
    def __init__(self, home_idx, away_idx, home_goals, away_goals, n_teams):
        self.home_idx = np.array(home_idx)
        self.away_idx = np.array(away_idx)
        self.home_goals = np.array(home_goals)
        self.away_goals = np.array(away_goals)
        self.n_teams = n_teams

        parameters = {
            'mu': 0.0, 'home_adv': 0.0,
            'sigma_attack': 0.2, 'sigma_defence': 0.2
        }
        for i in range(n_teams):
            parameters[f'attack_raw_{i}'] = 0.0
            parameters[f'defence_raw_{i}'] = 0.0
        super().__init__(parameters=parameters)

    def log_likelihood(self):
        mu = self.parameters['mu']
        home_adv = self.parameters['home_adv']
        sigma_attack = self.parameters['sigma_attack']
        sigma_defence = self.parameters['sigma_defence']

        attack = np.array([self.parameters[f'attack_raw_{i}'] for i in range(self.n_teams)]) * sigma_attack
        defence = np.array([self.parameters[f'defence_raw_{i}'] for i in range(self.n_teams)]) * sigma_defence
        attack -= np.mean(attack)
        defence -= np.mean(defence)

        log_lambda_home = mu + home_adv + attack[self.home_idx] - defence[self.away_idx]
        log_lambda_away = mu + attack[self.away_idx] - defence[self.home_idx]

        ll_home = poisson.logpmf(self.home_goals, np.exp(log_lambda_home))
        ll_away = poisson.logpmf(self.away_goals, np.exp(log_lambda_away))
        return np.sum(ll_home) + np.sum(ll_away)
