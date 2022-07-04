import logging
import time

import torch
import numpy as np

from geomhmm import SPDGaussianHMM, PoincareDiskGaussianHMM, EuclideanGaussianHMM
from extensions import *
from utils import match_permutation, permute_matrix
import randSPDGauss
import randPoincGauss

logger = logging.getLogger(__name__) 
logger.setLevel(logging.INFO)

# Comment out below if you don't want to see logging at this level 
# (you may still see logging at the subprocess level):
console_handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s %(asctime)s %(module)s] %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler) 

# Comment out below if you want to assess deprecation/future warnings:
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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

    pi_inf_hat = [1]

    return y, {'B': [[true_mean, true_disp]], 'P': true_trans_mat, 'pi_inf_hat': pi_inf_hat}

def gen_ex2(num_ex=200, rng=None):
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
    # true_mean2 = np.array([[1, .7, .49],                                        
    #                       [.7, 1, .7],                                         
    #                       [.49, .7, 1]])
    true_disp1 = true_disp2 = .05

    y1 = randSPDGauss.randSPDGauss(true_mean1, true_disp1, num_ex, rng=rng)
    y1 = [y1[:,:,i] for i in range(y1.shape[2])]
    y2 = randSPDGauss.randSPDGauss(true_mean2, true_disp2, num_ex, rng=rng)
    y2 = [y2[:,:,i] for i in range(y2.shape[2])]

    P = np.array([[.4, .6],
                  [.2, .8]])

    pi_inf_hat = [.25, .75]

    # Construct the Markov chain:
    x = []
    init_dist = [1, 0]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, P[curr_state], size=1)
        curr_state = np.argmax(curr_state, axis=1)[0]
        x.append(curr_state)

    y = []
    for xi in x:
        if xi == 0:
            y.append(y1.pop())
        else:
            y.append(y2.pop())

    return y, {'B': [[true_mean1, true_disp1], [true_mean2, true_disp2]],
            'P': P, 'pi_inf_hat': pi_inf_hat}

def gen_ex3(num_ex=400, rng=None):
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

    P = np.array([[.6, .1, .3],
                  [.1, .7, .2],
                  [.2, .2, .6]])
    pi_inf_hat = [.2759, .3448, .3793] # This is the approximate solution.

    # Construct the Markov chain:
    x = []
    init_dist = [1/3, 1/3, 1/3]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, P[curr_state], size=1)
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

    return y, {'B': [[mean1, disp1], [mean2, disp2], [mean3, disp3]],
            'P': P, 'pi_inf_hat': pi_inf_hat}

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


    P = np.array([[.4, .6],
                  [.2, .8]])

    pi_inf_hat = [.25, .75]

    # Construct the Markov chain:
    x = []
    init_dist = [1, 0]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, P[curr_state], size=1)
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
            'P': P, 'pi_inf_hat': pi_inf_hat}

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

    P = np.array([[.4, .3, .3],
                  [.2, .6, .2],
                  [.1, .1, .8]])
    pi_inf_hat = [0.18181818, 0.27272727, 0.54545455] # This is the approximate solution. 

    # Construct the Markov chain:
    x = []
    init_dist = [1, 0, 0]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, P[curr_state], size=1)
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
            'P': P, 'pi_inf_hat': pi_inf_hat}

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

    P = np.array([[.6, .1, .3],
                  [.1, .7, .2],
                  [.2, .2, .6]])
    pi_inf_hat = [.2759, .3448, .3793] # This is the approximate solution.

    # Construct the Markov chain:
    x = []
    init_dist = [1, 0, 0]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, P[curr_state], size=1)
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
            'P': P, 'pi_inf_hat': pi_inf_hat}

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


    P = np.array([[.4, .3, .3],
                  [.2, .6, .2],
                  [.1, .1, .8]])
    pi_inf_hat = [0.18181818, 0.27272727, 0.54545455] # This is the approximate solution.

    # Construct the Markov chain:
    x = []
    init_dist = [1, 0, 0]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, P[curr_state], size=1)
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
            'P': P, 'pi_inf_hat': pi_inf_hat}

def gen_ex8(num_ex=400, rng=None):
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

    P = np.array([[.3, .1, .2, .1, .3],
                  [.1, .4, .2, .2, .1],
                  [.2, .2, .3, .1, .2],
                  [.1, .1, .2, .5, .1],
                  [.4, .1, .1, .1, .3]])
    pi_inf_hat = [0.22744361, 0.17132116, 0.19924812, 0.19522019, 0.20676692] # This is the approximate solution.

    # Construct the Markov chain:
    x = []
    init_dist = [1/5, 1/5, 1/5, 1/5, 1/5]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, P[curr_state], size=1)
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
            'P': P, 'pi_inf_hat': pi_inf_hat}

def gen_ex9(num_ex=400, rng=None):
    '''
    Five mixture components, with nontrivial
    transition matrix. Values on the Euclidean line.
    '''
    if rng is None:
        rng = np.random.default_rng()

    mean1 = np.array([0])
    disp1 = np.eye(1)*.1

    mean2 = np.array([-3])
    disp2 = np.eye(1)*.1

    mean3 = np.array([3])
    disp3 = np.eye(1)*.1

    mean4 = np.array([-10])
    disp4 = np.eye(1)*.1

    mean5 = np.array([10])
    disp5 = np.eye(1)*.1


    P = np.array([[.3, .1, .2, .1, .3],
                  [.1, .4, .2, .2, .1],
                  [.2, .2, .3, .1, .2],
                  [.1, .1, .2, .5, .1],
                  [.4, .1, .1, .1, .3]])
    pi_inf_hat = [0.22744361, 0.17132116, 0.19924812, 0.19522019, 0.20676692] # This is the approximate solution.

    # Construct the Markov chain:
    x = []
    init_dist = [1/5, 1/5, 1/5, 1/5, 1/5]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, P[curr_state], size=1)
        curr_state = np.argmax(curr_state, axis=1)[0]
        x.append(curr_state)

    # Construct the observations:
    y1 = rng.multivariate_normal(mean1, disp1, num_ex)
    y2 = rng.multivariate_normal(mean2, disp2, num_ex)
    y3 = rng.multivariate_normal(mean3, disp3, num_ex)
    y4 = rng.multivariate_normal(mean4, disp4, num_ex)
    y5 = rng.multivariate_normal(mean5, disp5, num_ex)
 
    y1, y2, y3, y4, y5 = list(y1), list(y2), list(y3), list(y4), list(y5)
        
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
            'P': P, 'pi_inf_hat': pi_inf_hat}

def gen_ex10(num_ex=400, rng=None):
    '''
    Five mixture components, with nontrivial
    transition matrix. Values on the Poincare disk.
    '''
    if rng is None:
        rng = np.random.default_rng()

    mean1 = torch.tensor([0, 0])
    disp1 = .1

    mean2 = torch.tensor([.82, .29])
    disp2 = .1

    mean3 = torch.tensor([.82, -.29])
    disp3 = .1

    mean4 = torch.tensor([-.82, .29])
    disp4 = .1

    mean5 = torch.tensor([-.82, -.29])
    disp5 = .1

    P = np.array([[.3, .1, .2, .1, .3],
                  [.1, .4, .2, .2, .1],
                  [.2, .2, .3, .1, .2],
                  [.1, .1, .2, .5, .1],
                  [.4, .1, .1, .1, .3]])
    pi_inf_hat = [0.22744361, 0.17132116, 0.19924812, 0.19522019, 0.20676692] # This is the approximate solution.

    # Construct the Markov chain:
    x = []
    init_dist = [1/5, 1/5, 1/5, 1/5, 1/5]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, P[curr_state], size=1)
        curr_state = np.argmax(curr_state, axis=1)[0]
        x.append(curr_state)

    # Construct the observations:
    y1 = randPoincGauss.randPoincGauss(mean1, disp1, num_ex, rng=rng)
    y2 = randPoincGauss.randPoincGauss(mean2, disp2, num_ex, rng=rng)
    y3 = randPoincGauss.randPoincGauss(mean3, disp3, num_ex, rng=rng)
    y4 = randPoincGauss.randPoincGauss(mean4, disp4, num_ex, rng=rng)
    y5 = randPoincGauss.randPoincGauss(mean5, disp5, num_ex, rng=rng)
        
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
            'P': P, 'pi_inf_hat': pi_inf_hat}

def gen_ex11(num_ex=400, rng=None):
    '''
    Three mixture components, with nontrivial
    transition matrix. Values on the 3 by 3 SPD manifold.
    The (geometrically ergodic) markov chain is taken
    from Salem et al 2021.
    '''
    if rng is None:
        rng = np.random.default_rng()

    mean1 = np.eye(3)*2
    mean2 = np.array([[1, .8, .64], 
                          [.8, 1, .8],
                          [.64, .8, 1]])
    mean3 = np.array([[1, .7, .49],                                        
                          [.7, 1, .7],                                         
                          [.49, .7, 1]])

    disp1 = disp2 = disp3 = 0.1

    P = np.array([[.4, .3, .3],
                  [.2, .6, .2],
                  [.1, .1, .8]])
    pi_inf_hat = [0.18181818, 0.27272727, 0.54545455] # This is the approximate solution.

    # Construct the Markov chain:
    x = []
    init_dist = [1, 0, 0]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, P[curr_state], size=1)
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

    return y, {'B': [[mean1, disp1], [mean2, disp2], [mean3, disp3]],
            'P': P, 'pi_inf_hat': pi_inf_hat}

def gen_ex12(num_ex=400, rng=None):
    '''
    Five mixture components, with nontrivial
    transition matrix. Values on the 2 by 2 SPD manifolds.
    '''
    if rng is None:
        rng = np.random.default_rng()

    mean1 = np.array([[1., 0.],
                      [0., 1.]])
    mean2 = np.array([[1., 0.],
                      [0., 2.]])
    mean3 = np.array([[2., 0.],
                      [0., 1.]])
    mean4 = np.array([[2., 0.],
                      [0., 2.]])
    mean5 = np.array([[3., 0.],
                      [0., 3.]])

    disp1 = disp2 = disp3 = disp4 = disp5 = 1

    P = np.array([[.3, .1, .2, .1, .3],
                  [.1, .4, .2, .2, .1],
                  [.2, .2, .3, .1, .2],
                  [.1, .1, .2, .5, .1],
                  [.4, .1, .1, .1, .3]])
    pi_inf_hat = [0.22744361, 0.17132116, 0.19924812, 0.19522019, 0.20676692] # This is the approximate solution.

    # Construct the Markov chain:
    x = []
    init_dist = [1/5, 1/5, 1/5, 1/5, 1/5]
    curr_state = rng.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = rng.multinomial(1, P[curr_state], size=1)
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
            'P': P, 'pi_inf_hat': pi_inf_hat}


def compute_loss(m, label):
    true_centroids = np.array([l[0] for l in label['B']])
    pred_centroids = np.array([p[0] for p in m.B_params])
    true_disp = np.array([l[1] for l in label['B']])
    pred_disp = np.array([p[1] for p in m.B_params])
    true_trans_mat = label['P']
    pred_trans_mat = m.P_hat.copy()

    perm = match_permutation(true_centroids, pred_centroids, m.P_hat.shape[0], m.compute_dist)
    pred_centroids = pred_centroids[perm]
    pred_centroids = np.array(pred_centroids)
    true_centroids = np.array(true_centroids)
    pred_disp = pred_disp[perm]
    pred_trans_mat = permute_matrix(pred_trans_mat, perm)
    pred_pi_inf = np.array(m.pi_inf_hat)[perm]
 
    logger.info('The true centroids are : \n{}'.format(true_centroids))
    logger.info('The fitted centroids are : \n{}'.format(pred_centroids))
    logger.info('The true dispersions are : \n{}'.format(true_disp))
    logger.info('The fitted dispersions are : \n{}'.format(pred_disp))
    logger.info('The true transition matrix is : \n{}'.format(true_trans_mat))
    logger.info('The fitted transition matrix is : \n{}'.format(pred_trans_mat))
    logger.info(f"The Frob diff of true - pred transition matrices is {np.linalg.norm(true_trans_mat - pred_trans_mat, 'fro')}")
    logger.info('The true stationary distribution is : \n{}'.format(label['pi_inf_hat']))
    logger.info('The fitted stationary distribution is : \n{}'.format(pred_pi_inf))

def evaluate(m, y, label, fit_pi_inf_B=True):
    m.partial_fit(y, fit_pi_inf_B)
    compute_loss(m, label)
    

def main():
    rng = np.random.default_rng(2022)

    # logger.info('Start testing on ex 1.')
    # # y1 is the generated samples, while
    # # label1 is a dictionary containing the labels:
    # y1, label1 = gen_ex1(rng=rng)
    # m1 = SPDGaussianHMM(S=1, p=3, rng=rng)
    # evaluate(m1, y1, label1)

    # logger.info('Start testing on ex 2, using extensions.')
    # y2, label2 = gen_ex2(num_ex=10000, rng=rng)
    # m2 = SPD_EM_GaussianHMM(S=2, max_lag=1, num_samples_K=10000, p=3, rng=rng) 
    # m2.B_params, m2.pi_inf_hat = [[B_i[0].copy(), B_i[1]] for B_i in label2['B']], np.array(label2['pi_inf_hat']).copy()
    # evaluate(m2, y2, label2, fit_pi_inf_B=False)

    # logger.info('Start testing on ex 3.')
    # y3, label3 = gen_ex3(num_ex=1000, rng=rng)
    # m3 = SPDGaussianHMM(S=3, p=3, max_lag=3, num_samples_K=1000, rng=rng)
    # evaluate(m3, y3, label3)

    # logger.info('Start testing on ex 4.')
    # y4, label4 = gen_ex4(num_ex=10000, rng=rng)
    # m4 = EuclideanGaussianHMM(S=2, p=1, max_lag=2)
    # evaluate(m4, y4, label4)

    # logger.info('Start testing on ex 5.')
    # y5, label5 = gen_ex5(num_ex=10000, rng=rng)
    # m5 = PoincareDiskGaussianHMM(S=3, max_lag=3, num_samples_K=10000, rng=rng)
    # evaluate(m5, y5, label5)

    # logger.info('Start testing on ex 6.')
    # y6, label6 = gen_ex6(num_ex=20000, rng=rng)
    # m6 = EuclideanGaussianHMM(S=3, p=1, max_lag=3)
    # evaluate(m6, y6, label6)

    # logger.info('Start testing on ex 7.')
    # y7, label7 = gen_ex7(num_ex=10000, rng=rng)
    # m7 = PoincareDiskGaussianHMM(S=3, max_lag=2, num_samples_K=10000, rng=rng)
    # evaluate(m7, y7, label7, fit_pi_inf_B=True)

    # logger.info('Start testing on ex 8, using extensions.')
    # y8, label8 = gen_ex8(num_ex=10000, rng=rng) #700
    # m8 = SPD_EM_GaussianHMM(S=5, p=2, max_lag=1, num_samples_K=10000, rng=rng, num_omit_MCMC=100)
    # # m8.B_params, m8.pi_inf_hat = [[B_i[0].copy(), B_i[1]] for B_i in label8['B']], np.array(label8['pi_inf_hat']).copy()
    # evaluate(m8, y8, label8, fit_pi_inf_B=True)

    logger.info('Start testing on ex 9.')
    y9, label9 = gen_ex9(num_ex=1000, rng=rng)
    m9 = EuclideanGaussianHMM(S=5, max_lag=3)
    evaluate(m9, y9, label9)

    logger.info('Start testing on ex 10.')
    y10, label10 = gen_ex10(num_ex=100, rng=rng)
    m10 = PoincareDiskGaussianHMM(S=5, max_lag=2, num_samples_K=100, rng=rng)
    evaluate(m10, y10, label10)

    logger.info('Start testing on ex 11.')
    y11, label11 = gen_ex11(num_ex=100, rng=rng)
    m11 = SPDGaussianHMM(S=3, p=3, max_lag=3, num_samples_K=30, rng=rng, num_omit_MCMC=100)
    # m11.B_params, m11.pi_inf_hat = [[B_i[0].copy(), B_i[1]] for B_i in label11['B']], np.array(label11['pi_inf_hat'])
    evaluate(m11, y11, label11, fit_pi_inf_B=True)

    logger.info('Start testing on ex 12.')
    y12, label12 = gen_ex12(num_ex=100, rng=rng)
    m12 = SPDGaussianHMM(S=5, p=2, max_lag=1, num_samples_K=30, rng=rng, num_omit_MCMC=100)
    m12.B_params, m12.pi_inf_hat = [[B_i[0].copy(), B_i[1]] for B_i in label12['B']], np.array(label12['pi_inf_hat']).copy()
    evaluate(m12, y12, label12, fit_pi_inf_B=True)

    logger.info('Start testing on ex 2, using extensions.')
    y2, label2 = gen_ex2(num_ex=100, rng=rng)
    m2 = SPD_Zanini_Simple_GaussianHMM(S=2, p=3, rng=rng) 
    m2.B_params, m2.pi_inf_hat = [[B_i[0].copy(), B_i[1]] for B_i in label2['B']], np.array(label2['pi_inf_hat']).copy()
    evaluate(m2, y2, label2, fit_pi_inf_B=True)

    logger.info('Start testing on ex 2, using extensions.')
    y2, label2 = gen_ex2(num_ex=100, rng=rng)
    m2 = SPD_EM_Simple_GaussianHMM(S=2, p=3, rng=rng) 
    m2.B_params, m2.pi_inf_hat = [[B_i[0].copy(), B_i[1]] for B_i in label2['B']], np.array(label2['pi_inf_hat']).copy()
    evaluate(m2, y2, label2, fit_pi_inf_B=True)


if __name__ == "__main__":
    main()
