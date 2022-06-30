import shutil, os, json, sys
import argparse
import logging
import time

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch

from utils import permute_matrix, match_permutation
import randPoincGauss, randSPDGauss
from geomhmm import PoincareDiskGaussianHMM, SPDGaussianHMM 
from extensions import SPD_EM_GaussianHMM
from func_tests import compute_loss


logger = logging.getLogger(__name__) 
logger.setLevel(logging.INFO)

# Comment out below if you don't want to see progress outputs:
console_handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s %(asctime)s %(module)s] %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler) 

def gen_chain_Salem2021(num_ex=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    mean1 = torch.tensor([0, 0])
    disp1 = .1

    mean2 = torch.tensor([.29, .82])
    disp2 = .4

    mean3 = torch.tensor([-.29, .82,])
    disp3 = .4


    A = np.array([[.4, .3, .3],
                  [.2, .6, .2],
                  [.1, .1, .8]])
    phi = [0.18181818, 0.27272727, 0.54545455] # This is the approximate solution.

    # Construct the Markov chain:
    x = []
    init_dist = [1, 0, 0]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, A[curr_state], size=1)
        curr_state = np.argmax(curr_state, axis=1)[0]
        x.append(curr_state)

    # Construct the observations:
    y1 = randPoincGauss.randPoincGauss(mean1, disp1, num_ex, rng=rng)
    y2 = randPoincGauss.randPoincGauss(mean2, disp2, num_ex, rng=rng)
    y3 = randPoincGauss.randPoincGauss(mean3, disp3, num_ex, rng=rng)

    y = []
    for xi in x:
        if xi == 0:
            y.append(y1.pop())
        elif xi == 1:
            y.append(y2.pop())
        else:
            y.append(y3.pop())

    return x, y, {'B': [[mean1, disp1], [mean2, disp2], [mean3, disp3]],
            'A': A, 'phi': phi}

def gen_chain_Tupker2021(num_ex=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    mean1 = torch.tensor([0, 0])
    disp1 = .2

    mean2 = torch.tensor([.29, .82])
    disp2 = 1

    mean3 = torch.tensor([-.29, .82])
    disp3 = 1

    A = np.array([[.4, .3, .3],
                  [.2, .6, .2],
                  [.1, .1, .8]])
    phi = [0.18181818, 0.27272727, 0.54545455] # This is the approximate solution. 

    # Construct the Markov chain:
    x = []
    init_dist = [1, 0, 0]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, A[curr_state], size=1)
        curr_state = np.argmax(curr_state, axis=1)[0]
        x.append(curr_state)

    # Construct the observations:
    y1 = randPoincGauss.randPoincGauss(mean1, disp1, num_ex, rng=rng)
    y2 = randPoincGauss.randPoincGauss(mean2, disp2, num_ex, rng=rng)
    y3 = randPoincGauss.randPoincGauss(mean3, disp3, num_ex, rng=rng)

    y = []
    for xi in x:
        if xi == 0:
            y.append(y1.pop())
        elif xi == 1:
            y.append(y2.pop())
        else:
            y.append(y3.pop())

    return x, y, {'B': [[mean1, disp1], [mean2, disp2], [mean3, disp3]],
            'A': A, 'phi': phi}

def gen_chain_2by2_N5(num_ex=400, rng=None):
    '''
    Five mixture components, with nontrivial
    transition matrix. Values on the 2 by 2 SPD manifolds.
    '''
    if rng is None:
        rng = np.random.default_rng()

    mean1 = np.array([[1.646, 0.056],
                      [0.056, 2.379]])
    mean2 = np.array([[2.294,0.744],
                      [0.744,1.415]])
    mean3 = np.array([[2.631,-0.127],
                      [-0.127,1.277]])
    mean4 = np.array([[0.674,0.454],
                      [0.454,2.056]])
    mean5 = np.array([[1.829,-0.919],
                      [-0.919,1.602]])

    disp1 = disp2 = disp3 = disp4 = disp5 = .1

    A = np.array([[.3, .1, .2, .1, .3],
                  [.1, .4, .2, .2, .1],
                  [.2, .2, .3, .1, .2],
                  [.1, .1, .2, .5, .1],
                  [.4, .1, .1, .1, .3]])
    phi = [0.22744361, 0.17132116, 0.19924812, 0.19522019, 0.20676692] # This is the approximate solution.

    # Construct the Markov chain:
    x = []
    init_dist = [1/5, 1/5, 1/5, 1/5, 1/5]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, A[curr_state], size=1)
        curr_state = np.argmax(curr_state, axis=1)[0]
        x.append(curr_state)

    # Construct the observations:
    y1 = randSPDGauss.randSPDGauss(mean1, disp1, num_ex, rng=rng)
    y1 = [y1[:,:,i] for i in range(y1.shape[2])]
    y2 = randSPDGauss.randSPDGauss(mean2, disp2, num_ex, rng=rng)
    y2 = [y2[:,:,i] for i in range(y2.shape[2])]
    y3 = randSPDGauss.randSPDGauss(mean3, disp3, num_ex, rng=rng)
    y3 = [y3[:,:,i] for i in range(y3.shape[2])]
    y4 = randSPDGauss.randSPDGauss(mean4, disp4, num_ex, rng=rng)
    y4 = [y4[:,:,i] for i in range(y4.shape[2])]
    y5 = randSPDGauss.randSPDGauss(mean5, disp5, num_ex, rng=rng)
    y5 = [y5[:,:,i] for i in range(y5.shape[2])]
        
    y = []
    for xi in x:
        if xi == 0:
            y.append(y1.pop())
        elif xi == 1:
            y.append(y2.pop())
        elif xi == 2:
            y.append(y3.pop())
        elif xi == 3:
            y.append(y4.pop())
        else:
            assert xi == 4
            y.append(y5.pop())

    return y, {'B': [[mean1, disp1], [mean2, disp2], [mean3, disp3], [mean4, disp4], [mean5, disp5]],
            'A': A, 'phi': phi}

def gen_chain_3by3_N2(num_ex=400, rng=None):
    '''
    Two mixture components on the SPD manifold, with non-trivial
    transition matrix.
    '''
    if rng is None:
        rng = np.random.default_rng()

    true_mean1 = np.eye(3)*2
    true_mean2 = np.array([[1, .8, .64], 
                          [.8, 1, .8],
                          [.64, .8, 1]])
    true_disp1 = true_disp2 = .1

    y1 = randSPDGauss.randSPDGauss(true_mean1, true_disp1, num_ex, rng=rng)
    y1 = [y1[:,:,i] for i in range(y1.shape[2])]
    y2 = randSPDGauss.randSPDGauss(true_mean2, true_disp2, num_ex, rng=rng)
    y2 = [y2[:,:,i] for i in range(y2.shape[2])]

    A = np.array([[.4, .6],
                  [.2, .8]])

    phi = [.25, .75]

    # Construct the Markov chain:
    x = []
    init_dist = [1, 0]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, A[curr_state], size=1)
        curr_state = np.argmax(curr_state, axis=1)[0]
        x.append(curr_state)

    y = []
    for xi in x:
        if xi == 0:
            y.append(y1.pop())
        else:
            y.append(y2.pop())

    return y, {'B': [[true_mean1, true_disp1], [true_mean2, true_disp2]],
            'A': A, 'phi': phi}

def run_Salem2021_exp(given_true=False, output_name='output', output_path='./',
                      max_lag=3, num_samples_K=500, num_runs=20, seed=None):

    rng = np.random.default_rng(seed)

    os.makedirs(output_path, exist_ok=True)
    file_handler = logging.FileHandler(output_path + '/' + output_name + '.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s %(asctime)s %(module)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    mean_centroids = np.zeros((3, 2))
    mean_dispersions = np.zeros(3)
    mean_trans_mat = np.zeros((3, 3))
    mean_runtime = 0

    ret_centroids = np.zeros((num_runs, 3, 2))
    ret_dispersions = np.zeros((num_runs, 3))
    ret_trans_mat = np.zeros((num_runs, 3, 3))
    ret_runtime = np.zeros(num_runs)

    for it in range(num_runs):
        x, y, label = gen_chain_Salem2021(num_ex=10000, rng=rng)
        y_np = np.array([np.array(yi) for yi in y])
        np.save(output_path + '/' + output_name + '_inp_x{}.npy'.format(it), np.array(x))
        np.save(output_path + '/' + output_name + '_inp_y{}.npy'.format(it), y_np)

        m = PoincareDiskGaussianHMM(S=3, max_lag=max_lag, num_samples_K=num_samples_K, rng=rng)
        if given_true:
            m.B_params, m.phi = label['B'], np.array(label['phi'])
        start = time.time()
        m.partial_fit(y, fit_B_phi=(not given_true))
        run_time = time.time() - start

        true_centroids = np.array([l[0] for l in label['B']])
        pred_centroids = np.array([p[0] for p in m.B_params])
        true_disp = np.array([l[1] for l in label['B']])
        pred_disp = np.array([p[1] for p in m.B_params])
        true_trans_mat = label['A']
        pred_trans_mat = m.A_hat

        perm = match_permutation(true_centroids, pred_centroids, 3, m.compute_dist)
        pred_centroids = pred_centroids[perm]
        pred_centroids = np.array([np.array(centr) for centr in pred_centroids])
        pred_disp = pred_disp[perm]
        pred_trans_mat = permute_matrix(pred_trans_mat, perm)

        ret_centroids[it, :, :] = pred_centroids
        ret_dispersions[it, :] = pred_disp
        ret_trans_mat[it, :, :] = pred_trans_mat
        ret_runtime[it] = run_time

        mean_centroids += pred_centroids
        mean_dispersions += pred_disp
        mean_trans_mat += pred_trans_mat
        mean_runtime += run_time

    mean_centroids = mean_centroids/num_runs
    mean_dispersions = mean_dispersions/num_runs
    mean_trans_mat = mean_trans_mat/num_runs
    mean_runtime = mean_runtime/num_runs


    logger.info('Mean centroid are {}'.format(mean_centroids))
    c_diff = np.zeros(3)
    for i in range(3):
        c_diff[i] = m.compute_dist(torch.tensor(mean_centroids[i]), torch.tensor(true_centroids[i]))
    logger.info('sqrt(sum_i d(y-y_i_hat)^2) is {}'.format(np.linalg.norm(c_diff)))

    logger.info('Mean dispersions are {}'.format(mean_dispersions))
    logger.info('2norm(sigma-sigma_hat) is {}'.format(np.linalg.norm(mean_dispersions - true_disp)))
    logger.info('Mean transition matrix is {}'.format(mean_trans_mat))
    logger.info('Frob(A-Ahat) is {}'.format(np.linalg.norm(mean_trans_mat - true_trans_mat)))
    logger.info('Mean runtime is {}'.format(mean_runtime))

    np.save(output_path + '/' + output_name + '_centroids.npy', ret_centroids)
    np.save(output_path + '/' + output_name + '_disp.npy', ret_dispersions)
    np.save(output_path + '/' + output_name + '_trans_mat.npy', ret_trans_mat)
    np.save(output_path + '/' + output_name + '_runtime.npy', ret_runtime)

    return np.linalg.norm(mean_trans_mat - true_trans_mat)

def run_Tupker2021_exp(given_true=False, output_name='output', output_path='./',
                       num_runs = 20, seed=None):

    rng = np.random.default_rng(seed)

    os.makedirs(output_path, exist_ok=True)

    file_handler = logging.FileHandler(output_path + '/' + output_name + '.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s %(asctime)s %(module)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    mean_centroids = np.zeros((3, 2))
    mean_dispersions = np.zeros(3)
    mean_trans_mat = np.zeros((3, 3))
    mean_runtime = 0

    ret_centroids = np.zeros((num_runs, 3, 2))
    ret_dispersions = np.zeros((num_runs, 3))
    ret_trans_mat = np.zeros((num_runs, 3, 3))
    ret_trans_mat_frob_diff = np.zeros(num_runs)
    ret_runtime = np.zeros(num_runs)

    for it in range(num_runs):
        x, y, label = gen_chain_Tupker2021(num_ex=10000)
        y_np = np.array([np.array(yi) for yi in y])
        np.save(output_path + '/' + output_name + '_inp_x{}.npy'.format(it), np.array(x))
        np.save(output_path + '/' + output_name + '_inp_y{}.npy'.format(it), y_np)

        m = PoincareDiskGaussianHMM(S=3, max_lag=5, num_samples_K=10000)
        if given_true:
            m.B_params, m.phi = label['B'], np.array(label['phi'])
        start = time.time()
        m.partial_fit(y, fit_B_phi=(not given_true))
        run_time = time.time() - start

        true_centroids = np.array([l[0] for l in label['B']])
        pred_centroids = np.array([p[0] for p in m.B_params])
        true_disp = np.array([l[1] for l in label['B']])
        pred_disp = np.array([p[1] for p in m.B_params])
        true_trans_mat = label['A']
        pred_trans_mat = m.A_hat
        ret_trans_mat_frob_diff[it] = np.linalg.norm(pred_trans_mat - true_trans_mat)

        perm = match_permutation(true_centroids, pred_centroids, 3, m.compute_dist)
        pred_centroids = pred_centroids[perm]
        pred_centroids = np.array([np.array(centr) for centr in pred_centroids])
        pred_disp = pred_disp[perm]
        pred_trans_mat = permute_matrix(pred_trans_mat, perm)

        ret_centroids[it, :, :] = pred_centroids
        ret_dispersions[it, :] = pred_disp
        ret_trans_mat[it, :, :] = pred_trans_mat
        ret_runtime[it] = run_time

        mean_centroids += pred_centroids
        mean_dispersions += pred_disp
        mean_trans_mat += pred_trans_mat
        mean_runtime += run_time

    logger.info('Mean centroid are {}'.format(mean_centroids/num_runs))
    logger.info('Mean dispersions are {}'.format(mean_dispersions/num_runs))
    logger.info('Mean transition matrix is {}'.format(mean_trans_mat/num_runs))
    logger.info('Transition RMSE is {}'.format(np.linalg.norm(ret_trans_mat_frob_diff) / np.sqrt(num_runs)))
    logger.info('Mean runtime is {}'.format(mean_runtime/num_runs))

    np.save(output_path + '/' + output_name + '_centroids.npy', ret_centroids)
    np.save(output_path + '/' + output_name + '_disp.npy', ret_dispersions)
    np.save(output_path + '/' + output_name + '_trans_mat.npy', ret_trans_mat)
    np.save(output_path + '/' + output_name + '_trans_mat_frob_diff.npy', ret_trans_mat_frob_diff)
    np.save(output_path + '/' + output_name + '_runtime.npy', ret_runtime)

def train_evaluate(trial_index, output_path, output_name, parameters):
    output_path = output_path + '/' + str(trial_index)
    os.makedirs(output_path, exist_ok=True)

    return run_Salem2021_exp(given_true=False, output_name=output_name, output_path=output_path,
                      max_lag=parameters['max_lag'], num_samples_K=parameters['num_samples_K'], num_runs=20)

def tune_hyperparams(input_name, input_path, output_name, output_path, total_trials):
    from ax.service.ax_client import AxClient

    input_file = input_path + '/' + input_name
    os.makedirs(output_path, exist_ok=True)
    shutil.copy(input_file, output_path + '/hyperparams.json')
    parameters = json.load(open(input_file))

    # Set up client to manage hyperparameter tuning:
    ax_client = AxClient()
    ax_client.create_experiment(
        name = "geomhmm_hyperparam_tuning",
        parameters = parameters,
        objective_name = 'Frob(A-Ahat)',
        minimize=True
    )

    for i in range(total_trials):
        parameters, trial_index = ax_client.get_next_trial()
        try:
            result = ax_client.complete_trial(trial_index=trial_index, raw_data=train_evaluate(trial_index, output_path, output_name, parameters))
        except:
            # Should any optimization iterations fail during evaluation, 
            # log_trial_failure will ensure that the same trial is not proposed again
            ax_client.log_trial_failure(trial_index=trial_index)
        # Save current trial progress:
        ax_client.save_to_json_file(output_path + '/ax_client_checkpoint' + str(i) + '.json')

def save_to_csv(content, fpath):
    df = pd.DataFrame(data=content)
    df.to_csv(fpath)

def generate_evolving_estimates(input_config='in.json', output_path='./'):
    """
    Will save:
    - The true parameters of the HMM.
    - The realization of HMM.
    - Estimated Gaussian parameters (if "given_true": true).
    - Estimated transition matrix at each observation.
    - Estimated H and K matrix at each observation (if applicable).
    - Log outputs for everything.
    - A copy of input_config.
    """
    config = json.load(open(input_config))
    out_dirname = os.path.join(output_path, config['experiment'])
    if not os.path.exists(out_dirname):
        os.mkdir(out_dirname)
    shutil.copy(input_config, out_dirname)
    rng = np.random.default_rng(config['seed'])

    logging.basicConfig(filename=f"{out_dirname}/log.log", filemode='w', format='%(levelname)s %(asctime)s %(module)s] %(message)s')


    # Generate the example and save true states to a JSON:
    if "PoincareDisk" in config['learner']['name']:
        _, y, label = gen_chain_Salem2021(num_ex=config['num_obs'], rng=rng)
    elif config['SPD_dim'] == 2 and config['num_hidden_states'] == 5:
        y, label = gen_chain_2by2_N5(num_ex=config['num_obs'], rng=rng)
    elif config['SPD_dim'] == 3 and config['num_hidden_states'] == 2:
        y, label = gen_chain_3by3_N2(num_ex=config['num_obs'], rng=rng)
    else:
        raise ValueError(
            f"SPD_dim = {config['SPD_dim']} and num_hidden_states = {config['num_hidden_states']} not supported.")

    true_params_fname = f"{out_dirname}/true_params.json"
    true_params_content = {}
    true_params_content['trans_matrix'] = label['A'].copy().flatten().tolist()
    true_params_content['gaussian_params'] = {}
    for index, true_param in enumerate(label['B']):
        if "PoincareDisk" in config['learner']['name']:
            true_params_content['gaussian_params'][str(index)] = {
                'mean': true_param[0].tolist(),
                'dispersion': true_param[1]
            }
        else:
            true_params_content['gaussian_params'][str(index)] = {
                'mean': true_param[0].copy().flatten().tolist(),
                'dispersion': true_param[1]
            }
    true_params_content['stat_dist'] = label['phi']
    with open(true_params_fname, 'w') as f:
        json.dump(true_params_content, f, indent=4)

    # Initialize observations/estimates to be updated during the learning step:
    obs_fname = f"{out_dirname}/obs.csv"
    obs_content = {}
    if "PoincareDisk" in config['learner']['name']:
        obs_content["N_1"], obs_content["N_2"] = [], []
    else:
        for i in range(config['SPD_dim']):
            for j in range(config['SPD_dim']):
                obs_content[f"N_{i}{j}"] = []

    est_trans_mat_fname = f"{out_dirname}/est_trans_mat.csv"
    est_trans_mat_content = {}
    for i in range(config['num_hidden_states']):
        for j in range(config['num_hidden_states']):
            est_trans_mat_content[f"P_{i}{j}"] = []

    if 'max_lag' in config['learner']:
        est_H_fname = f"{out_dirname}/est_H.csv"
        est_K_fname = f"{out_dirname}/est_K.csv"
        est_H_content = {}
        est_K_content = {}
        for i in range(config['num_hidden_states']):
            for j in range(config['num_hidden_states']):
                est_K_content[f"K_{i}{j}"] = []
                for lag_ind in range(config['learner']['max_lag']+1):
                    est_H_content[f"H{lag_ind}_{i}{j}"] = []

    if not config['given_true']:
        est_guassian_params_fname = f"{out_dirname}/est_guassian_params.csv"
        est_params_content = {}
        for k in range(config['num_hidden_states']):
            est_params_content[f"d{k}"] = []
            if "PoincareDisk" in config['learner']['name']:
                est_params_content[f"c{k}_1"] = []
                est_params_content[f"c{k}_2"] = []
            else:
                for i in range(config['SPD_dim']):
                    for j in range(config['SPD_dim']):
                        est_params_content[f"c{k}_{i}{j}"] = []

    if config['learner']['name'] == 'SPDGaussianHMM':
        m = SPDGaussianHMM(
            S=config['num_hidden_states'], 
            p=config['SPD_dim'], 
            max_lag=config['learner']['max_lag'], 
            num_samples_K=config['learner']['num_samples_K'], 
            rng=rng)
        if config['given_true']:
            m.B_params, m.phi = [[B_i[0].copy(), B_i[1]] for B_i in label['B']], np.array(label['phi']).copy()
    elif config['learner']['name'] == 'SPD_EM_GaussianHMM':
        m = SPD_EM_GaussianHMM(
            S=config['num_hidden_states'], 
            p=config['SPD_dim'], 
            max_lag=config['learner']['max_lag'], 
            num_samples_K=config['learner']['num_samples_K'], 
            num_omit_MCMC=config['learner']['num_omit_MCMC'],
            rng=rng,
            )

        if not config['given_true']:
            # Since EM can't fit incrementally, we fit the parameters right away here
            # and store the results:
            m.partial_fit(y, fit_B_phi=True)
            for k in range(config['num_hidden_states']):
                est_params_content[f"d{k}"].append(m.B_params[k][1])
                for i in range(config['SPD_dim']):
                    for j in range(config['SPD_dim']):
                        est_params_content[f"c{k}_{i}{j}"].append(m.B_params[k][0][i, j])

            save_to_csv(est_params_content, est_guassian_params_fname)

            # Reset the internal states of m:
            m.N = 0
            m.H_hat = np.zeros([m.max_lag + 1, m.S, m.S])
            m.H_N = np.zeros(m.max_lag + 1) # Number of samples used to estimate H_hat
            m.obs_cache = []

            config['given_true'] = True
        else:
            m.B_params, m.phi = [[B_i[0].copy(), B_i[1]] for B_i in label['B']], np.array(label['phi']).copy()
    elif config['learner']['name'] == 'PoincareDiskGaussianHMM':
        m = PoincareDiskGaussianHMM(
            S=config['num_hidden_states'], 
            max_lag=config['learner']['max_lag'], 
            num_samples_K=config['learner']['num_samples_K'], 
            rng=rng)
        if config['given_true']:
            m.B_params, m.phi = [[B_i[0].clone(), B_i[1]] for B_i in label['B']], np.array(label['phi']).copy()
    else:
        raise ValueError('Invalid name for the geometric HMM learner.')


    # Learning step:
    for y_index, y_i in enumerate(y):
        logger.info(f"Learning on the {y_index} th observation.")

        m.partial_fit([y_i], not config['given_true'])

        # Store all data pertaining to the current state: 

        # Dispersion:
        if not config['given_true']:
            for k in range(config['num_hidden_states']):
                est_params_content[f"d{k}"].append(m.B_params[k][1])

        # Observation and mean:
        if "PoincareDisk" in config['learner']['name']:
            obs_content["N_1"].append(y_i[0].item())
            obs_content["N_2"].append(y_i[1].item())
            if not config['given_true']:
                for k in range(config['num_hidden_states']):
                    if m.B_params[k][0] is not None:
                        est_params_content[f"c{k}_1"].append(m.B_params[k][0][0].item())
                        est_params_content[f"c{k}_2"].append(m.B_params[k][0][1].item())
                    else:
                        est_params_content[f"c{k}_1"].append(None)
                        est_params_content[f"c{k}_2"].append(None)
        else:
            for i in range(config['SPD_dim']):
                for j in range(config['SPD_dim']):
                    obs_content[f"N_{i}{j}"].append(y_i[i, j])
                    if not config['given_true']:
                        for k in range(config['num_hidden_states']):
                            est_params_content[f"c{k}_{i}{j}"].append(m.B_params[k][0][i, j])
        
        # Estimated transition, H, and K:
        for i in range(config['num_hidden_states']):
            for j in range(config['num_hidden_states']):
                est_trans_mat_content[f"P_{i}{j}"].append(m.A_hat[i, j])
                if 'max_lag' in config['learner']:
                    est_K_content[f"K_{i}{j}"].append(m.K_hat[i, j])
                    for lag_ind in range(config['learner']['max_lag']+1):
                        est_H_content[f"H{lag_ind}_{i}{j}"].append(m.H_hat[lag_ind, i, j])

    compute_loss(m, label)

    save_to_csv(obs_content, obs_fname)
    save_to_csv(est_trans_mat_content, est_trans_mat_fname)
    save_to_csv(est_H_content, est_H_fname)
    save_to_csv(est_K_content, est_K_fname)
    if not config['given_true']:       
        save_to_csv(est_params_content, est_guassian_params_fname)

def generate_nonevolving_estimates(input_config='in.json', output_path='./'):
    """
    Will save:
    - The true parameters of the HMM.
    - The realization of HMM.
    - Estimated Gaussian parameters (if "given_true": true).
    - Estimated transition matrix.
    - Estimated H and K matrix (if applicable).
    - Log outputs for everything.
    - A copy of input_config.
    """
    config = json.load(open(input_config))
    out_dirname = os.path.join(output_path, config['experiment'])
    if not os.path.exists(out_dirname):
        os.mkdir(out_dirname)
    shutil.copy(input_config, out_dirname)
    rng = np.random.default_rng(config['seed'])

    logging.basicConfig(filename=f"{out_dirname}/log.log", filemode='w', format='%(levelname)s %(asctime)s %(module)s] %(message)s')


    # Generate the example and save true states to a JSON:
    if "PoincareDisk" in config['learner']['name']:
        _, y, label = gen_chain_Salem2021(num_ex=config['num_obs'], rng=rng)
    elif config['SPD_dim'] == 2 and config['num_hidden_states'] == 5:
        y, label = gen_chain_2by2_N5(num_ex=config['num_obs'], rng=rng)
    elif config['SPD_dim'] == 3 and config['num_hidden_states'] == 2:
        y, label = gen_chain_3by3_N2(num_ex=config['num_obs'], rng=rng)
    else:
        raise ValueError(
            f"SPD_dim = {config['SPD_dim']} and num_hidden_states = {config['num_hidden_states']} not supported.")

    true_params_fname = f"{out_dirname}/true_params.json"
    true_params_content = {}
    true_params_content['trans_matrix'] = label['A'].copy().flatten().tolist()
    true_params_content['gaussian_params'] = {}
    for index, true_param in enumerate(label['B']):
        if "PoincareDisk" in config['learner']['name']:
            true_params_content['gaussian_params'][str(index)] = {
                'mean': true_param[0].tolist(),
                'dispersion': true_param[1]
            }
        else:
            true_params_content['gaussian_params'][str(index)] = {
                'mean': true_param[0].copy().flatten().tolist(),
                'dispersion': true_param[1]
            }
    true_params_content['stat_dist'] = label['phi']
    with open(true_params_fname, 'w') as f:
        json.dump(true_params_content, f, indent=4)

    # Initialize observations/estimates to be updated during the learning step:
    obs_fname = f"{out_dirname}/obs.csv"
    obs_content = {}
    if "PoincareDisk" in config['learner']['name']:
        obs_content["N_1"], obs_content["N_2"] = [], []
    else:
        for i in range(config['SPD_dim']):
            for j in range(config['SPD_dim']):
                obs_content[f"N_{i}{j}"] = []

    est_trans_mat_fname = f"{out_dirname}/est_trans_mat.csv"
    est_trans_mat_content = {}
    for i in range(config['num_hidden_states']):
        for j in range(config['num_hidden_states']):
            est_trans_mat_content[f"P_{i}{j}"] = []

    if 'max_lag' in config['learner']:
        est_H_fname = f"{out_dirname}/est_H.csv"
        est_K_fname = f"{out_dirname}/est_K.csv"
        est_H_content = {}
        est_K_content = {}
        for i in range(config['num_hidden_states']):
            for j in range(config['num_hidden_states']):
                est_K_content[f"K_{i}{j}"] = []
                for lag_ind in range(config['learner']['max_lag']+1):
                    est_H_content[f"H{lag_ind}_{i}{j}"] = []

    if not config['given_true']:
        est_guassian_params_fname = f"{out_dirname}/est_guassian_params.csv"
        est_params_content = {}
        for k in range(config['num_hidden_states']):
            est_params_content[f"d{k}"] = []
            if "PoincareDisk" in config['learner']['name']:
                est_params_content[f"c{k}_1"] = []
                est_params_content[f"c{k}_2"] = []
            else:
                for i in range(config['SPD_dim']):
                    for j in range(config['SPD_dim']):
                        est_params_content[f"c{k}_{i}{j}"] = []

    if config['learner']['name'] == 'SPDGaussianHMM':
        m = SPDGaussianHMM(
            S=config['num_hidden_states'], 
            p=config['SPD_dim'], 
            max_lag=config['learner']['max_lag'], 
            num_samples_K=config['learner']['num_samples_K'], 
            rng=rng)
        if config['given_true']:
            m.B_params, m.phi = [[B_i[0].copy(), B_i[1]] for B_i in label['B']], np.array(label['phi']).copy()
    elif config['learner']['name'] == 'SPD_EM_GaussianHMM':
        m = SPD_EM_GaussianHMM(
            S=config['num_hidden_states'], 
            p=config['SPD_dim'], 
            max_lag=config['learner']['max_lag'], 
            num_samples_K=config['learner']['num_samples_K'], 
            rng=rng)

        if not config['given_true']:
            # Since EM can't fit incrementally, we fit the parameters right away here
            # and store the results:
            m.partial_fit(y, fit_B_phi=True)
            for k in range(config['num_hidden_states']):
                est_params_content[f"d{k}"].append(m.B_params[k][1])
                for i in range(config['SPD_dim']):
                    for j in range(config['SPD_dim']):
                        est_params_content[f"c{k}_{i}{j}"].append(m.B_params[k][0][i, j])

            save_to_csv(est_params_content, est_guassian_params_fname)

            # Reset the internal states of m:
            m.N = 0
            m.H_hat = np.zeros([m.max_lag + 1, m.S, m.S])
            m.H_N = np.zeros(m.max_lag + 1) # Number of samples used to estimate H_hat
            m.obs_cache = []

            config['given_true'] = True
        else:
            m.B_params, m.phi = [[B_i[0].copy(), B_i[1]] for B_i in label['B']], np.array(label['phi']).copy()
    elif config['learner']['name'] == 'PoincareDiskGaussianHMM':
        m = PoincareDiskGaussianHMM(
            S=config['num_hidden_states'], 
            max_lag=config['learner']['max_lag'], 
            num_samples_K=config['learner']['num_samples_K'], 
            rng=rng)
        if config['given_true']:
            m.B_params, m.phi = [[B_i[0].clone(), B_i[1]] for B_i in label['B']], np.array(label['phi']).copy()
    else:
        raise ValueError('Invalid name for the geometric HMM learner.')


    # Learning step:
    m.partial_fit(y, not config['given_true'])

    # Store all data pertaining to the current state: 

    # Dispersion:
    if not config['given_true']:
        for k in range(config['num_hidden_states']):
            est_params_content[f"d{k}"].append(m.B_params[k][1])

    # Observation and mean:
    for y_index, y_i in enumerate(y):
        if "PoincareDisk" in config['learner']['name']:
            obs_content["N_1"].append(y_i[0].item())
            obs_content["N_2"].append(y_i[1].item())
            if not config['given_true'] and y_index == 0:
                for k in range(config['num_hidden_states']):
                    if m.B_params[k][0] is not None:
                        est_params_content[f"c{k}_1"].append(m.B_params[k][0][0].item())
                        est_params_content[f"c{k}_2"].append(m.B_params[k][0][1].item())
                    else:
                        est_params_content[f"c{k}_1"].append(None)
                        est_params_content[f"c{k}_2"].append(None)
        else:
            for i in range(config['SPD_dim']):
                for j in range(config['SPD_dim']):
                    obs_content[f"N_{i}{j}"].append(y_i[i, j])
                    if not config['given_true'] and y_index == 0:
                        for k in range(config['num_hidden_states']):
                            est_params_content[f"c{k}_{i}{j}"].append(m.B_params[k][0][i, j])
    
    # Estimated transition, H, and K:
    for i in range(config['num_hidden_states']):
        for j in range(config['num_hidden_states']):
            est_trans_mat_content[f"P_{i}{j}"].append(m.A_hat[i, j])
            if 'max_lag' in config['learner']:
                est_K_content[f"K_{i}{j}"].append(m.K_hat[i, j])
                for lag_ind in range(config['learner']['max_lag']+1):
                    est_H_content[f"H{lag_ind}_{i}{j}"].append(m.H_hat[lag_ind, i, j])

    compute_loss(m, label)

    save_to_csv(obs_content, obs_fname)
    save_to_csv(est_trans_mat_content, est_trans_mat_fname)
    save_to_csv(est_H_content, est_H_fname)
    save_to_csv(est_K_content, est_K_fname)
    if not config['given_true']:       
        save_to_csv(est_params_content, est_guassian_params_fname)

def sensitivity_analysis(input_config='in.json', output_path='./'):
    """
    Will save:
    - The true parameters of the HMM.
    - Log outputs for everything.
    - A copy of input_config.
    - A csv called 'perturb_dispersion', containing est transition matrix
    with its corresponding loss value.
    - A csv called 'perturb_mean', containing est transition matrix
    with its corresponding loss value.
    """
    config = json.load(open(input_config))
    out_dirname = os.path.join(output_path, config['experiment'])
    if not os.path.exists(out_dirname):
        os.mkdir(out_dirname)
    shutil.copy(input_config, out_dirname)
    rng = np.random.default_rng(config['seed'])

    logging.basicConfig(filename=f"{out_dirname}/log.log", filemode='w', format='%(levelname)s %(asctime)s %(module)s] %(message)s')

    # Generate the example and save true states to a JSON:
    if "PoincareDisk" in config['learner']['name']:
        _, y, label = gen_chain_Salem2021(num_ex=config['num_obs'], rng=rng)
    elif config['SPD_dim'] == 2 and config['num_hidden_states'] == 5:
        y, label = gen_chain_2by2_N5(num_ex=config['num_obs'], rng=rng)
    elif config['SPD_dim'] == 3 and config['num_hidden_states'] == 2:
        y, label = gen_chain_3by3_N2(num_ex=config['num_obs'], rng=rng)
    else:
        raise ValueError(
            f"SPD_dim = {config['SPD_dim']} and num_hidden_states = {config['num_hidden_states']} not supported.")

    true_params_fname = f"{out_dirname}/true_params.json"
    true_params_content = {}
    true_params_content['trans_matrix'] = label['A'].copy().flatten().tolist()
    true_params_content['gaussian_params'] = {}
    for index, true_param in enumerate(label['B']):
        if "PoincareDisk" in config['learner']['name']:
            true_params_content['gaussian_params'][str(index)] = {
                'mean': true_param[0].tolist(),
                'dispersion': true_param[1]
            }
        else:
            true_params_content['gaussian_params'][str(index)] = {
                'mean': true_param[0].copy().flatten().tolist(),
                'dispersion': true_param[1]
            }
    true_params_content['stat_dist'] = label['phi']
    with open(true_params_fname, 'w') as f:
        json.dump(true_params_content, f, indent=4)

    mean_perturbs = config['mean_perturbs']
    dispersion_perturbs = config['dispersion_perturbs']
    mean_pert_fname = f"{out_dirname}/perturb_mean.csv"
    dispersion_pert_fname = f"{out_dirname}/perturb_dispersion.csv"
    if 'SPD_dim' in config:
        mean_pert_content = { 
            f"est_P_{i}{j}": [] for i, j in product(np.arange(config['SPD_dim']), np.arange(config['SPD_dim']))
        }
        mean_pert_content['perturb_val'] = []
        mean_pert_content['loss'] = [] 

        disp_pert_content = {
            'est_disp': [],
            'perturb_val': [],
            'loss': [],
        }   

    else:
        raise NotImplementedError

    if config['SPD_dim'] == 2 and config['num_hidden_states'] == 5:
        set_perturb_val = lambda err: 2*np.exp(err)
    else:
        raise NotImplementedError

    for mean_perturb in mean_perturbs:
        if config['SPD_dim'] == 2 and config['num_hidden_states'] == 5:
            m = SPD_EM_GaussianHMM(
                S=config['num_hidden_states'], 
                p=config['SPD_dim'], 
                max_lag=config['learner']['max_lag'], 
                num_samples_K=config['learner']['num_samples_K'], 
                rng=rng)
            m.B_params, m.phi = [[B_i[0].copy(), B_i[1]] for B_i in label14['B']], np.array(label14['phi']).copy()
            m.B_params[0][0][0, 0] = set_perturb_val(mean_perturb)
            m.partial_fit(y, True)
            # Compute Loss
            # Update mean_pert_content

    # Analogous for loop for dispersion_perturbs.

    # Save the dict as csv files by involking save_to_csv().

    raise NotImplementedError


def parse_args():
    # TODO: instead of CLI, use yaml config files instead.
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        help='Mode of the experiment.')
    parser.add_argument('--givenTrue', type=bool,
                        help='Whether to train emission probability and stationary distribution.',
                        default=False)
    parser.add_argument('--iname', type=str,
                        help='Name of input file.', default='input')
    parser.add_argument('--ipath', type=str,
                        help='Path of input.', default='./in')
    parser.add_argument('--oname', type=str,
                        help='Name of output file.', default='output')
    parser.add_argument('--opath', type=str,
                        help='Path of output.', default='./out')
    parser.add_argument('--totT', type=int,
                        help='Maximum num of tune_hyperparams trials.', default=30)
    parser.add_argument('--seed', type=int,
                        help='Seed for the random number generator.', default=2022)

    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == 'Salem2021':
        run_Salem2021_exp(given_true=args.givenTrue, output_name=args.oname, output_path=args.opath, seed=args.seed)
    elif args.mode == 'Tupker2021':
        run_Tupker2021_exp(given_true=args.givenTrue, output_name=args.oname, output_path=args.opath, seed=args.seed)
    elif args.mode == 'hyp_tuning':
        tune_hyperparams(input_name=args.iname, input_path=args.ipath,
                         output_path=args.opath, output_name=args.oname, total_trials=args.totT)
    elif args.mode == 'evolv_est':
        generate_evolving_estimates(input_config=args.iname, output_path=args.opath)
    elif args.mode == 'nonevolv_est':
        generate_nonevolving_estimates(input_config=args.iname, output_path=args.opath)
    else:
        raise ValueError('The experiment task {} is invalid. Exiting.'.format(args.mode))

if __name__ == "__main__":
    main()

