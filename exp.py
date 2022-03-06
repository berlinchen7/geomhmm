from geomhmm import PoincareDiskGaussianHMM
import randPoincGauss
import torch
import numpy as np
import time
import itertools
import argparse


def gen_chain_Salem2021(num_ex=1000, seed=2):
    np.random.seed(seed)

    mean1 = torch.tensor([0, 0])
    disp1 = .1

    mean2 = torch.tensor([.82, .29])
    disp2 = .4

    mean3 = torch.tensor([.82, -.29])
    disp3 = .4


    A = np.array([[.4, .3, .3],
                  [.2, .6, .2],
                  [.1, .1, .8]])
    phi = [0.18181818, 0.27272727, 0.54545455] # This is the approximate solution.

    # Construct the Markov chain:
    x = []
    init_dist = [1, 0, 0]
    curr_state = np.random.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = np.random.multinomial(1, A[curr_state], size=1)
        curr_state = np.argmax(curr_state, axis=1)[0]
        x.append(curr_state)

    # Construct the observations:
    y1 = randPoincGauss.randPoincGauss(mean1, disp1, num_ex)
    y2 = randPoincGauss.randPoincGauss(mean2, disp2, num_ex)
    y3 = randPoincGauss.randPoincGauss(mean3, disp3, num_ex)

    y = []
    for xi in x:
        if xi == 0:
            y.append(y1.pop())
        elif xi == 1:
            y.append(y2.pop())
        else:
            y.append(y3.pop())

    return y, {'B': [[mean1, disp1], [mean2, disp2], [mean3, disp3]],
            'A': A, 'phi': phi}

def gen_chain_Tupker2021(num_ex=10000, seed=2):
    np.random.seed(seed)

    mean1 = torch.tensor([0, 0])
    disp1 = .2

    mean2 = torch.tensor([.82, .29])
    disp2 = 1

    mean3 = torch.tensor([.82, -.29])
    disp3 = 1

    A = np.array([[.4, .3, .3],
                  [.2, .6, .2],
                  [.1, .1, .8]])
    phi = [0.18181818, 0.27272727, 0.54545455] # This is the approximate solution. 

    # Construct the Markov chain:
    x = []
    init_dist = [1, 0, 0]
    curr_state = np.random.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = np.random.multinomial(1, A[curr_state], size=1)
        curr_state = np.argmax(curr_state, axis=1)[0]
        x.append(curr_state)

    # Construct the observations:
    y1 = randPoincGauss.randPoincGauss(mean1, disp1, num_ex)
    y2 = randPoincGauss.randPoincGauss(mean2, disp2, num_ex)
    y3 = randPoincGauss.randPoincGauss(mean3, disp3, num_ex)

    y = []
    for xi in x:
        if xi == 0:
            y.append(y1.pop())
        elif xi == 1:
            y.append(y2.pop())
        else:
            y.append(y3.pop())

    return y, {'B': [[mean1, disp1], [mean2, disp2], [mean3, disp3]],
            'A': A, 'phi': phi}


def match_permutation(true, predicted, num_states, dist):
    ''' Find the permutation that minimizes the average distance between true and predicted
    '''
    curr_min_cost, curr_min_permutation = float('inf'), None
    for perm in itertools.permutations(np.arange(num_states)):
        perm = list(perm)
        curr_cost = 0
        for s in range(num_states):
            curr_perm_pred = predicted[perm]
            curr_cost += dist(true[s], curr_perm_pred[s])
        curr_cost /= num_states
        if curr_cost < curr_min_cost:
            curr_min_cost = curr_cost
            curr_min_permutation = perm
    return list(curr_min_permutation)

def permute_matrix(matrix, permutation):
    ret = matrix.copy()
    dim = matrix.shape[0]
    for i in range(dim):
        ret[:, i] = ret[permutation, i]
    for i in range(dim):
        ret[i, :] = ret[i, permutation]
    return ret

def run_Salem2021_exp(given_true=args.givenTrue, output_name=args.out, output_path='./'):

    num_runs = 20

    mean_centroids = np.zeros((3, 2))
    mean_dispersions = np.zeros(3)
    mean_trans_mat = np.zeros((3, 3))
    mean_runtime = 0

    ret_centroids = np.zeros((num_runs, 3, 2))
    ret_dispersions = np.zeros((num_runs, 3))
    ret_trans_mat = np.zeros((num_runs, 3, 3))
    ret_runtime = np.zeros(num_runs)

    for it in range(num_runs):
        y, label = gen_chain_Salem2021(num_ex=10000, seed=it+3)
        m = PoincareDiskGaussianHMM(S=3, max_lag=3, num_samples_K=500)
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

    print('Mean centroid are {}'.format(mean_centroids/num_runs))
    print('Mean dispersions are {}'.format(mean_dispersions/num_runs))
    print('Mean transition matrix is {}'.format(mean_trans_mat/num_runs))
    print('Mean runtime is {}'.format(mean_runtime/num_runs))

    np.save(output_path + output_name + '_centroids.npy', ret_centroids)
    np.save(output_path + output_name + '_disp.npy', ret_dispersions)
    np.save(output_path + output_name + '_trans_mat.npy', ret_trans_mat)
    np.save(output_path + output_name + '_runtime.npy', ret_runtime)

def run_Tupker2021_exp(given_true, output_name, output_path='./'):

    num_runs = 20

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
        y, label = gen_chain_Tupker2021(num_ex=10000, seed=it+3)
        m = PoincareDiskGaussianHMM(S=3, max_lag=5, num_samples_K=500)
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

    print('Mean centroid are {}'.format(mean_centroids/num_runs))
    print('Mean dispersions are {}'.format(mean_dispersions/num_runs))
    print('Mean transition matrix is {}'.format(mean_trans_mat/num_runs))
    print('Transition RMSE is {}'.format(np.linalg.norm(ret_trans_mat_frob_diff) / np.sqrt(num_runs)))
    print('Mean runtime is {}'.format(mean_runtime/num_runs))

    np.save(output_path + output_name + '_centroids.npy', ret_centroids)
    np.save(output_path + output_name + '_disp.npy', ret_dispersions)
    np.save(output_path + output_name + '_trans_mat.npy', ret_trans_mat)
    np.save(output_path + output_name + '_trans_mat_frob_diff.npy', ret_trans_mat_frob_diff)
    np.save(output_path + output_name + '_runtime.npy', ret_runtime)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        help='Mode of the experiment.')
    parser.add_argument('--givenTrue', type=bool,
                        help='Whether to train emission probability and stationary distribution.',
                        default=False)
    parser.add_argument('--out', type=str,
                        help='Name of output npy files.', default='output')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == 'Salem2021':
        run_Salem2021_exp(given_true=args.givenTrue, output_name=args.out)
    if args.mode == 'Tupker2021':
        run_Tupker2021_exp(given_true=args.givenTrue, output_name=args.out)
    else:
        raise ValueError('The experiment task {} is invalid. Exiting.'.format(config['task'] ))

if __name__ == "__main__":
    main()

