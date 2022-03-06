from geomhmm import SPDGaussianHMM, PoincareDiskGaussianHMM, EuclideanGaussianHMM
import randSPDGauss
import randPoincGauss
import torch
import numpy as np
import time

def gen_ex1(num_ex=1000):
    '''
    Simple case with one mixture component.
    This is one of the examples tested in
    the Zanini et al., 2017 paper.
    '''
    true_mean = np.array([[1, .8, .64], 
                          [.8, 1, .8],
                          [.64, .8, 1]])
    true_disp = .125
    true_trans_mat = np.array([[1]])

    y = randSPDGauss.randSPDGauss(true_mean, true_disp, num_ex, seed=1)
    y = [y[:,:,i] for i in range(y.shape[2])]

    phi = [1]

    return y, {'B': [[true_mean, true_disp]], 'A': true_trans_mat, 'phi': phi}

def gen_ex2(num_ex=100):
    '''
    Two mixture components, with identity
    transition matrix.
    This is one of the examples tested in
    the Zanini et al., 2017 paper.
    '''

    true_mean1 = np.eye(3)
    true_mean2 = np.array([[1, .7, .49],                                        
                          [.7, 1, .7],                                         
                          [.49, .7, 1]])
    true_disp1 = true_disp2 = .1

    y1 = randSPDGauss.randSPDGauss(true_mean1, true_disp1, num_ex)
    y1 = [y1[:,:,i] for i in range(y1.shape[2])]
    y2 = randSPDGauss.randSPDGauss(true_mean2, true_disp2, num_ex)
    y2 = [y2[:,:,i] for i in range(y2.shape[2])]

    phi = [.4, .6]

    y = []
    np.random.seed(1)
    latent = np.random.binomial(1, phi[1], num_ex) 
    # Note: the second param is the prob of 1s.
    for l in latent:
        if l == 0:
            y.append(y1.pop())
        else:
            y.append(y2.pop())

    return y, {'B': [[true_mean1, true_disp1], [true_mean2, true_disp2]],
            'A': np.eye(2), 'phi': phi}

def gen_ex3(num_ex=400, num_truncated_ex=100):
    '''
    Three mixture components, with nontrivial
    transition matrix.
    The (geometrically ergodic) markov chain is taken
    from the following source:
    www.stat.berkeley.edu/~mgoldman/Section0220.pdf
    '''
    np.random.seed(2)

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
    curr_state = np.random.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = np.random.multinomial(1, A[curr_state], size=1)
        curr_state = np.argmax(curr_state, axis=1)[0]
        x.append(curr_state)

    # Construct the observations:
    y1 = randSPDGauss.randSPDGauss(mean1, disp1, num_ex)
    y1 = [y1[:,:,i] for i in range(y1.shape[2])]
    y2 = randSPDGauss.randSPDGauss(mean2, disp2, num_ex)
    y2 = [y2[:,:,i] for i in range(y2.shape[2])]
    y3 = randSPDGauss.randSPDGauss(mean3, disp3, num_ex)
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

def gen_ex5(num_ex=10000, seed=2):
    '''
    Also three mixture components, with nontrivial
    transition matrix. Values on Poincare disk, instead of SPD
    manifolds.
    The set up was taken
    from Tupker et al., 2021.
    '''
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

def gen_ex6(num_ex=10000):
    '''
    Also three mixture components, with nontrivial
    transition matrix. Values on Euclidean spaces.
    '''

    np.random.seed(2)

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
    curr_state = np.random.multinomial(1, init_dist, size=1)
    curr_state = np.argmax(curr_state, axis=1)[0]
    x.append(curr_state)
    for i in range(num_ex-1):
        curr_state = np.random.multinomial(1, A[curr_state], size=1)
        curr_state = np.argmax(curr_state, axis=1)[0]
        x.append(curr_state)

    # Construct the observations:
    y1 = np.random.multivariate_normal(mean1, disp1, num_ex)
    y2 = np.random.multivariate_normal(mean2, disp2, num_ex)
    y3 = np.random.multivariate_normal(mean3, disp3, num_ex)
 
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

def gen_ex7(num_ex=1000, seed=2):
    '''
    Also three mixture components, with nontrivial
    transition matrix. Values on Poincare disk, instead of SPD
    manifolds.
    The set up was taken
    from Salem et al., 2021.
    '''
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


def evaluate(m, y, label, fit_B_phi=True):
    m.partial_fit(y, fit_B_phi)
    print('The true B parameters are : \n{}'.format(label['B']))
    print('The fitted B parameters are : \n{}'.format(m.B_params))
    print('The true transition matrix is : \n{}'.format(label['A']))
    print('The fitted transition matrix is : \n{}'.format(m.A_hat))
    print('The true stationary distribution is : \n{}'.format(label['phi']))
    print('The fitted stationary distribution is : \n{}'.format(m.phi))


def main():
    # print('\n\nProcessing test case 1...')
    # # y1 is the generated samples, while
    # # label1 is a dictionary containing the labels:
    # y1, label1 = gen_ex1()
    # m1 = SPDGaussianHMM(S=1, p=3)
    # evaluate(m1, y1, label1)

    # print('\n\nProcessing test case 2...')
    # y2, label2 = gen_ex2()
    # m2 = SPDGaussianHMM(S=2, p=3)
    # evaluate(m2, y2, label2)

    # print('\n\nProcessing test case 3...')
    # y3, label3 = gen_ex3()
    # m3 = SPDGaussianHMM(S=3, p=3)
    # evaluate(m3, y3, label3)

    # print('\n\nProcessing test case 4...')
    # y4, label4 = gen_ex3(num_ex=1000)
    # m4 = SPDGaussianHMM(S=3, p=3, num_samples_sigma=1000, num_samples_K=1000)
    # m4.B_params, m4.phi = label4['B'], np.array(label4['phi'])
    # evaluate(m4, y4, label4, fit_B_phi=False)

    # print('\n\nProcessing test case 5...')
    # y5, label5 = gen_ex5(num_ex=10000)
    # m5 = PoincareDiskGaussianHMM(S=3, max_lag=3, num_samples_K=10000)
    # m5.B_params, m5.phi = label5['B'], np.array(label5['phi'])
    # evaluate(m5, y5, label5, fit_B_phi=False)

    # print('\n\nProcessing test case 6...')
    # y6, label6 = gen_ex6()
    # m6 = EuclideanGaussianHMM(p=2, max_lag=3)
    # m6.B_params, m6.phi = label6['B'], np.array(label6['phi'])
    # evaluate(m6, y6, label6, fit_B_phi=False)

    # print('\n\nProcessing test case 7...')
    # y7, label7 = gen_ex6(num_ex=6000)
    # m7 = EuclideanGaussianHMM(p=2, max_lag=3)
    # evaluate(m7, y7, label7, fit_B_phi=True)

    print('\n\nProcessing test case 8...')
    y8, label8 = gen_ex7(num_ex=4000, seed=10)
    m8 = PoincareDiskGaussianHMM(S=3, max_lag=3, num_samples_K=100)
    evaluate(m8, y8, label8, fit_B_phi=True)


    # print('\n\nProcessing test case 9...')
    # y9, label9 = gen_ex7(seed=10, num_ex=10000)
    # m9 = PoincareDiskGaussianHMM(S=3, max_lag=3, num_samples_K=10000)
    # evaluate(m9, y9, label9, fit_B_phi=True)

    # print('\n\nProcessing test case 10...')
    # y10, label10 = gen_ex7(num_ex=10000, seed=4)
    # m10 = PoincareDiskGaussianHMM(S=3, max_lag=6, num_samples_K=100)
    # m10.B_params, m10.phi = label10['B'], np.array(label10['phi'])
    # evaluate(m10, y10, label10, fit_B_phi=False)


if __name__ == "__main__":
    main()
