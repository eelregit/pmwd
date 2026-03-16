import os
import argparse
import pickle
import numpy as np
from pysr import PySRRegressor


def make_pdf(tex_str, eq_path, eq_file):
    tex_file = "\\documentclass[5pt]{article}\n"
    tex_file += "\\usepackage[a3paper, margin=2cm]{geometry}\n"
    tex_file += "\\usepackage{xcolor}\n"
    tex_file += "\\usepackage{breqn}\n"
    tex_file += "\\usepackage{booktabs}\n"
    tex_file += "\n\\begin{document}\n"

    tex_file += "\n".join(tex_str.splitlines()[5:])

    tex_file += "\n\\end{document}\n"

    tex_full_path = os.path.join(eq_path, f'{eq_file}_table.tex')
    with open(tex_full_path, 'w') as f:
        f.write(tex_file)

    os.system(f'cd {eq_path} ' +
              f'&& latexmk -pdf -silent {eq_file}_table.tex ' +
              '&& rm *.fdb_latexmk *.log *.fls *.aux')


def save_tex(model, eq_path, eq_file, var_names, net):
    tex_str = model.latex_table()
    # mark the chosen equation red
    try:
        tex_str = tex_str.replace(model.latex(), '{\\color{red} '+model.latex()+'}')
    except Exception as e:
        print(f"Warning: Could not highlight best equation in LaTeX table: {e}")
    
    # replace variables with provided names
    for i, vn in enumerate(var_names):
        tex_str = tex_str.replace(f'x_{{{i}}}', '{'+vn+'}')
    
    tex_str = tex_str.replace('y = ', f'{net} = ')
    
    # save the equations to latex file
    os.makedirs(eq_path, exist_ok=True)
    tex_full_path = os.path.join(eq_path, f'{eq_file}.tex')
    with open(tex_full_path, 'w') as f:
        f.write(tex_str)
    
    try:
        make_pdf(tex_str, eq_path, eq_file)
    except Exception as e:
        print(f"Warning: Failed to generate PDF for {eq_file}: {e}")


def load_data(jobid, epoch, sr_features='simple', data_dir='nn_data'):
    fn = os.path.join(data_dir, f'j{jobid}_e{epoch}.{sr_features}.npz')
    if not os.path.exists(fn):
        # Fallback to old format just in case
        old_fn = os.path.join(data_dir, f'j{jobid}_e{epoch}.npz')
        if os.path.exists(old_fn):
            fn = old_fn
        else:
            raise FileNotFoundError(f"Data file {fn} not found. Run sample_nn.py first.")
    
    data = np.load(fn, allow_pickle=True)
    var_names_dic = {
        'f': data['f_names_tex'],
        'g': data['g_names_tex'],
    }
    return data, var_names_dic


def run_pysr(data, net, eq_path, eq_file, var_names_dic, niterations=100):
    """Run PySR on the data and get the equations."""
    X, y = data[f'{net}_X'], data[f'{net}_y']
    var_names = var_names_dic[net]
    eq_file_net = f'{eq_file}_{net}'

    # ensure the output directory exists before fitting
    os.makedirs(eq_path, exist_ok=True)

    model = PySRRegressor(
        # search size
        niterations = niterations,
        populations = 3 * (os.cpu_count() or 1),
        ncyclesperiteration = 50000,

        # search space
        binary_operators = ['+', '*', '^', '/'],
        unary_operators = ['neg', 'exp', 'log'],
        maxsize = 40,
        maxdepth = 10,

        # complexities
        parsimony = 0.001,
        adaptive_parsimony_scaling = 1000.0,
        constraints = {'^': (-1, 1)},
        nested_constraints = {
            'exp': {'exp': 1},
            'log': {'log': 0},
        },

        # mutations
        weight_optimize = 0.001,

        # objective
        loss = 'loss(prediction, target) = (prediction - target)^2',

        # exporting the results
        equation_file = os.path.join(eq_path, f'{eq_file_net}.csv'),
        output_jax_format = True,
    )
    model.fit(X, y)

    save_tex(model, eq_path, eq_file_net, var_names, net)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PySR fitting on SO NN data.')
    parser.add_argument('--jobid', type=int, required=True, help='Slurm job ID')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number')
    parser.add_argument('--net', type=str, choices=['f', 'g', 'both'], default='both', 
                        help='Which network to fit (f, g, or both)')
    parser.add_argument('--sr_features', type=str, choices=['simple', 'physical'], default='simple',
                        help='Which feature set was used for sampling')
    parser.add_argument('--niterations', type=int, default=100, help='Number of PySR iterations')
    parser.add_argument('--data_dir', type=str, default='nn_data', help='Directory where NN data is stored')
    parser.add_argument('--eq_path', type=str, default='eq_files', help='Directory to save equations')
    
    args = parser.parse_args()

    eq_file = f'{args.jobid}_e{args.epoch}_{args.sr_features}'
    data, var_names_dic = load_data(args.jobid, args.epoch, sr_features=args.sr_features, 
                                   data_dir=args.data_dir)
    
    nets = ['f', 'g'] if args.net == 'both' else [args.net]
    
    for net in nets:
        print(f"Running PySR for network: {net} using {args.sr_features} features")
        run_pysr(data, net, args.eq_path, eq_file, var_names_dic, niterations=args.niterations)

