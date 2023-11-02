"""Hyperparameter optimization using optuna."""
import optuna
from optuna.samplers import TPESampler
import optax
import numpy as np
import joblib

from run_train import prep_train, run_train, slurm_job_id, procid
from pmwd.sto.so import soft_len
from pmwd.sto.mlp import init_mlp_params


def objective(trial, sobol_ids, gsdata, snap_ids):
    n_epochs = 3
    shuffle_epoch = True
    so_type = 2

    if so_type == 2:
        n_input = [soft_len(k_fac=3), soft_len()]

    if so_type == 3:
        n_input = [soft_len()] * 3

    # hypars to search
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
    optimizer = optax.adam(learning_rate)

    n_layers = trial.suggest_int('n_layers', 2, 5)
    so_nodes = [[n] * n_layers + [1] for n in n_input]
    so_params = init_mlp_params(n_input, so_nodes, scheme='last_ws_b1')

    # train for a fixed number of epochs and get the lowest loss as the obj
    losses = run_train(n_epochs, sobol_ids, gsdata, snap_ids, shuffle_epoch,
                       learning_rate, optimizer, so_type, so_nodes, so_params,
                       ret=True, log_id=trial.number)
    losses = np.array(losses)

    return losses.min()


def optune(n_trials):
    sampler = TPESampler(seed=0)  # to make the pars suggested by Optuna the same for all procs
    study = optuna.create_study(sampler=sampler)

    sobol_ids_global = np.arange(0, 16)
    snap_ids = np.arange(0, 121, 1)
    sobol_ids, gsdata = prep_train(sobol_ids_global, snap_ids)
    study.optimize(lambda trial: objective(trial, sobol_ids, gsdata, snap_ids),
                   n_trials=n_trials)

    return study


if __name__ == "__main__":
    study = optune(5)
    if procid == 0:
        joblib.dump(study, f'study/{slurm_job_id}.pkl')
