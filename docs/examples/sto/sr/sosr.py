# Symbolic regression of the SO neural nets data.

import os
import pickle
import numpy as np
from pysr import PySRRegressor
from pmwd.sto.so import soft_names_tex


def save_tex(model, eq_file, var_names, net):
    tex_str = model.latex_table()
    # replace variables with provided names
    for i, vn in enumerate(var_names):
        tex_str = tex_str.replace(f'x_{{{i}}}', '{'+vn+'}')
    tex_str = tex_str.replace('y = ', f'{net} = ')
    # save the equations to latex file
    with open(f'{eq_file}.tex', 'w') as f:
        f.write(tex_str)


def load_data(jobid, epoch, soft_i):
    with open(f'nn_data/{jobid}_e{epoch}.pickle', 'rb') as f:
        data = pickle.load(f)
    var_names_dic = {
        'f': soft_names_tex(soft_i, 'f'),
        'g': soft_names_tex(soft_i, 'g'),
    }
    return data, var_names_dic


def run_pysr(X, y, eq_file, var_names):
    """Run PySR on the data and get the equations."""
    model = PySRRegressor(
        # search size
        niterations = 100,
        populations = 3 * os.cpu_count(),
        ncyclesperiteration = 10000,

        # search space
        binary_operators = ['+', '*', '^', '/'],
        unary_operators = ['neg', 'exp', 'log'],
        maxsize = 35,

        # complexities
        parsimony = 0.0001,
        adaptive_parsimony_scaling = 20.0,
        constraints = {'^': (-1, 1)},
        nested_constraints = {
            'exp': {'exp': 1},
            'log': {'log': 0},
        },

        # mutations
        weight_optimize = 0.001,

        # objective
        loss = 'loss(prediction, target) = (prediction - target)^2',

        # performance and parallelization
        batching = True,
        batch_size = 128,

        # exporting the results
        equation_file = eq_file + '.csv',
    )
    model.fit(X, y)

    save_tex(model, eq_file, var_names)

    return model


if __name__ == "__main__":
    jobid = 3099093
    epoch = 591

    eq_file = f'eq_files/{jobid}_e{epoch}'
    data, var_names_dic = load_data(jobid, epoch)
    run_pysr(data['f']['X'], data['f']['y'], eq_file+'_f', var_names_dic['f'])
    run_pysr(data['g']['X'], data['g']['y'], eq_file+'_g', var_names_dic['g'])
