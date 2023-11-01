"""Hyperparameter optimization using optuna."""
import optuna
from run_train import run_train


def objective(trial):
    lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)

    # train for a fixed number of epochs and get the lowest loss as the obj

study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params
