# Symbolic regression of the SO neural nets data.

import os
import pickle
import numpy as np
from pysr import PySRRegressor
from pmwd.sto.so import soft_names_tex


def make_pdf(tex_str, eq_path, eq_file):
    tex_file = "\\documentclass[5pt]{article}\n"
    tex_file += "\\usepackage[a3paper, margin=2cm]{geometry}\n"
    tex_file += "\\usepackage{xcolor}\n"
    tex_file += "\\usepackage{breqn}\n"
    tex_file += "\\usepackage{booktabs}\n"
    tex_file += "\n\\begin{document}\n"

    tex_file += "\n".join(tex_str.splitlines()[5:])

    tex_file += "\n\\end{document}\n"

    with open(f'{eq_path}{eq_file}_table.tex', 'w') as f:
        f.write(tex_file)

    os.system(f'cd {eq_path}' +
              f'&& latexmk -pdf -silent {eq_file}_table.tex' +
              '&& rm  *.fdb_latexmk  *.log  *.fls  *.aux')


def save_tex(model, eq_path, eq_file, var_names, net):
    tex_str = model.latex_table()
    # mark the chosen equation red
    tex_str = tex_str.replace(model.latex(), '{\\color{red} '+model.latex()+'}')
    # replace variables with provided names
    for i, vn in enumerate(var_names):
        tex_str = tex_str.replace(f'x_{{{i}}}', '{'+vn+'}')
    tex_str = tex_str.replace('y = ', f'{net} = ')
    # save the equations to latex file
    with open(f'{eq_path}{eq_file}.tex', 'w') as f:
        f.write(tex_str)
    make_pdf(tex_str, eq_path, eq_file)


def load_data(jobid, epoch, soft_i):
    with open(f'nn_data/j{jobid}_e{epoch}.pickle', 'rb') as f:
        data = pickle.load(f)
    var_names_dic = {
        'f': soft_names_tex(soft_i, 'f'),
        'g': soft_names_tex(soft_i, 'g'),
    }
    return data, var_names_dic


def run_pysr(data, net, eq_path, eq_file, var_names_dic):
    """Run PySR on the data and get the equations."""
    X, y = data[net]['X'], data[net]['y']
    var_names = var_names_dic[net]
    eq_file += f'_{net}'

    model = PySRRegressor(
        # search size
        niterations = 30,
        populations = 3 * os.cpu_count(),
        ncyclesperiteration = 20000,

        # search space
        binary_operators = ['+', '*', '^', '/'],
        unary_operators = ['neg', 'exp', 'log'],
        maxsize = 35,
        maxdepth = 7,

        # complexities
        parsimony = 0.001,
        adaptive_parsimony_scaling = 2000.0,
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
        # batching = True,
        # batch_size = 128,

        # exporting the results
        equation_file = eq_path + eq_file + '.csv',
        output_jax_format = True,
        extra_jax_mappings = None,
    )
    model.fit(X, y)

    save_tex(model, eq_path, eq_file, var_names, net)

    return model


if __name__ == "__main__":
    jobid = 3177874
    epoch = 3000
    soft_i = 'soft_v2'

    eq_path = 'eq_files/'
    eq_file = f'{jobid}_e{epoch}'
    data, var_names_dic = load_data(jobid, epoch, soft_i)
    model_f = run_pysr(data, 'f', eq_path, eq_file, var_names_dic)
    model_g = run_pysr(data, 'g', eq_path, eq_file, var_names_dic)
