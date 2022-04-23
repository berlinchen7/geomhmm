from geomhmm import SPDGaussianHMM, PoincareDiskGaussianHMM, EuclideanGaussianHMM
from utils import match_permutation, permute_matrix
import randSPDGauss
import randPoincGauss
import torch
import numpy as np
import time

import logging

logger = logging.getLogger(__name__) 
logger.setLevel(logging.INFO)

# Comment out below if you don't want to see progress outputs:
console_handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s %(asctime)s %(module)s] %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler) 

def gen_ex1(num_ex=1000, rng=None):
    '''
    Simple case with one mixture component.
    Values on the SPD manifold.
    This is one of the examples tested in
    the Zanini et al., 2017 paper.
    '''
    true_mean = np.array([[1, .8, .64], 
                          [.8, 1, .8],
                          [.64, .8, 1]])
    true_disp = .125
    true_trans_mat = np.array([[1]])

    y = randSPDGauss.randSPDGauss(true_mean, true_disp, num_ex, rng=rng)
    y = [y[:,:,i] for i in range(y.shape[2])]

    phi = [1]

    return y, {'B': [[true_mean, true_disp]], 'A': true_trans_mat, 'phi': phi}

def gen_ex2(num_ex=200, rng=None):
    '''
    Two mixture components on the SPD manifold, with non-trivial
    transition matrix.
    '''
    if rng is None:
        rng = np.random.default_rng()

    true_mean1 = np.eye(3)
    true_mean2 = np.array([[1, .7, .49],                                        
                          [.7, 1, .7],                                         
                          [.49, .7, 1]])
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

def gen_ex3(num_ex=400, num_truncated_ex=100, rng=None):
    '''
    Three mixture components, with nontrivial
    transition matrix. Values on the SPD manifold.
    The (geometrically ergodic) markov chain is taken
    from the following source:
    www.stat.berkeley.edu/~mgoldman/Section0220.pdf
    '''
    if rng is None:
        rng = np.random.default_rng()

    mean1 = np.eye(3)
    mean2 = np.array([[1, .7, .49],
                      [.7, 1, .7],
                      [.49, .7, 1]])
    mean3 = np.array([[1, .3, .09],
                      [.3, 1, .3],
                      [.09, .3, 1]])

    disp1 = disp2 = disp3 = .05

    A = np.array([[.6, .1, .3],
                  [.1, .7, .2],
                  [.2, .2, .6]])
    phi = [.2759, .3448, .3793] # This is the approximate solution.

    # Construct the Markov chain:
    x = []
    init_dist = [1/3, 1/3, 1/3]
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
        
    y = []
    for xi in x:
        if xi == 0:
            y.append(y1.pop())
        elif xi == 1:
            y.append(y2.pop())
        else:
            y.append(y3.pop())

    y = y[num_truncated_ex:]

    return y, {'B': [[mean1, disp1], [mean2, disp2], [mean3, disp3]],
            'A': A, 'phi': phi}

def gen_ex4(num_ex=200, rng=None):
    '''
    Two mixture components on the Euclidean space, with non-trivial
    transition matrix.
    '''
    if rng is None:
        rng = np.random.default_rng()

    mean1 = np.array([0])
    disp1 = np.eye(1)*.5

    mean2 = np.array([-1])
    disp2 = np.eye(1)*.5


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

    # Construct the observations:
    y1 = rng.multivariate_normal(mean1, disp1, num_ex)
    y2 = rng.multivariate_normal(mean2, disp2, num_ex)
 
    y1, y2 = list(y1), list(y2)

    y = []
    for xi in x:
        if xi == 0:
            y.append(y1.pop())
        else:
            y.append(y2.pop())

    return y, {'B': [[mean1, disp1], [mean2, disp2]],
            'A': A, 'phi': phi}

def gen_ex5(num_ex=10000, rng=None):
    '''
    Three mixture components, with nontrivial
    transition matrix. Values on Poincare disk.
    The set up was taken
    from Tupker et al., 2021.
    '''
    if rng is None:
        rng = np.random.default_rng()

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

    return y, {'B': [[mean1, disp1], [mean2, disp2], [mean3, disp3]],
            'A': A, 'phi': phi}

def gen_ex6(num_ex=10000, rng=None):
    '''
    Three mixture components, with nontrivial
    transition matrix. Values on Euclidean spaces.
    '''
    if rng is None:
        rng = np.random.default_rng()

    mean1 = np.array([0])
    disp1 = np.eye(1)*.5

    mean2 = np.array([-1])
    disp2 = np.eye(1)*.5

    mean3 = np.array([1])
    disp3 = np.eye(1)*.5

    A = np.array([[.6, .1, .3],
                  [.1, .7, .2],
                  [.2, .2, .6]])
    phi = [.2759, .3448, .3793] # This is the approximate solution.

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
    y1 = rng.multivariate_normal(mean1, disp1, num_ex)
    y2 = rng.multivariate_normal(mean2, disp2, num_ex)
    y3 = rng.multivariate_normal(mean3, disp3, num_ex)
 
    y1, y2, y3 = list(y1), list(y2), list(y3)

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

def gen_ex7(num_ex=1000, rng=None):
    '''
    Also three mixture components, with nontrivial
    transition matrix. Values on Poincare disk.
    The set up was taken
    from Salem et al., 2021.
    '''
    if rng is None:
        rng = np.random.default_rng()

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

    return y, {'B': [[mean1, disp1], [mean2, disp2], [mean3, disp3]],
            'A': A, 'phi': phi}


def evaluate(m, y, label, fit_B_phi=True):
    m.partial_fit(y, fit_B_phi)


    true_centroids = np.array([l[0] for l in label['B']])
    pred_centroids = np.array([p[0] for p in m.B_params])
    true_disp = np.array([l[1] for l in label['B']])
    pred_disp = np.array([p[1] for p in m.B_params])
    true_trans_mat = label['A']
    pred_trans_mat = m.A_hat

    perm = match_permutation(true_centroids, pred_centroids, m.A_hat.shape[0], m.compute_dist)
    pred_centroids = pred_centroids[perm]
    pred_centroids = np.array(pred_centroids)
    true_centroids = np.array(true_centroids)
    pred_disp = pred_disp[perm]
    pred_trans_mat = permute_matrix(pred_trans_mat, perm)
    pred_phi = np.array(m.phi)[perm]
 
    logger.info('The true centroids are : \n{}'.format(true_centroids))
    logger.info('The fitted centroids are : \n{}'.format(pred_centroids))
    logger.info('The true dispersions are : \n{}'.format(true_disp))
    logger.info('The fitted dispersions are : \n{}'.format(pred_disp))
    logger.info('The true transition matrix is : \n{}'.format(label['A']))
    logger.info('The fitted transition matrix is : \n{}'.format(m.A_hat))
    logger.info('The true stationary distribution is : \n{}'.format(label['phi']))
    logger.info('The fitted stationary distribution is : \n{}'.format(pred_phi))


def main():
    rng = np.random.default_rng(2022)

    # logger.info('Start testing on ex 1.')
    # # y1 is the generated samples, while
    # # label1 is a dictionary containing the labels:
    # y1, label1 = gen_ex1(rng=rng)
    # m1 = SPDGaussianHMM(S=1, p=3, rng=rng)
    # evaluate(m1, y1, label1)

    # logger.info('Start testing on ex 2.')
    # y2, label2 = gen_ex2(num_ex=10000, rng=rng)
    # m2 = SPDGaussianHMM(S=2, p=3, max_lag=4, num_samples_sigma=100, num_samples_K=10000, rng=rng)
    # evaluate(m2, y2, label2)

    # logger.info('Start testing on ex 3.')
    # y3, label3 = gen_ex3(num_ex=1000, rng=rng)
    # m3 = SPDGaussianHMM(S=3, p=3, max_lag=3, num_samples_K=1000, rng=rng)
    # evaluate(m3, y3, label3)

    # logger.info('Start testing on ex 4.')
    # y4, label4 = gen_ex4(num_ex=2000, rng=rng)
    # m4 = EuclideanGaussianHMM(S=2, p=2, max_lag=1, rng=rng)
    # evaluate(m4, y4, label4)

    # logger.info('Start testing on ex 5.')
    # y5, label5 = gen_ex5(num_ex=10000, rng=rng)
    # m5 = PoincareDiskGaussianHMM(S=3, max_lag=3, num_samples_K=10000, rng=rng)
    # evaluate(m5, y5, label5)

    # logger.info('Start testing on ex 6.')
    # y6, label6 = gen_ex6(rng=rng)
    # m6 = EuclideanGaussianHMM(p=2, max_lag=3)
    # evaluate(m6, y6, label6)

    logger.info('Start testing on ex 7.')
    y7, label7 = gen_ex7(num_ex=10000, rng=rng)
    m7 = PoincareDiskGaussianHMM(S=3, max_lag=3, num_samples_K=10000, rng=rng)
    evaluate(m7, y7, label7, fit_B_phi=True)

if __name__ == "__main__":
    main()
