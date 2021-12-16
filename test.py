from geomhmm import SPDGaussianHMM
import randSPDGauss
import numpy as np

def gen_ex1(num_ex=100):
    '''
    Simple case with one mixture component.
    This is one of the examples tested in
    the Zanini et al., 2017 paper.
    '''
    true_mean = np.array([[1, .8, .64], 
                          [.8, 1, .8],
                          [.64, .8, 1]])
    true_disp = .1
    true_trans_mat = np.array([[1]])

    y = randSPDGauss.randSPDGauss(true_mean, true_disp, num_ex)
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
            'A': np.eye(3), 'phi': phi}

def gen_ex3(num_ex=200):
    '''
    Three mixture components, with nontrivial
    transition matrix.
    The (geometrically ergodic) markov chain is taken
    from the following source:
    www.stat.berkeley.edu/~mgoldman/Section0220.pdf
    '''
    mean1 = np.eye(3)
    mean2 = np.array([[1, .7, .49],
                      [.7, 1, .7],
                      [.49, .7, 1]])
    mean3 = np.array([[1, .3, .09],
                      [.3, 1, .3],
                      [.09, .3, 1]])

    disp1 = disp2 = disp3 = .1

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

    return y, {'B': [[mean1, disp1], [mean2, disp2], [mean3, disp3]],
            'A': A, 'phi': phi}


def evaluate(m, y, label):
    m.partial_fit(y)
    print('The true B parameters are : \n{}'.format(label['B']))
    print('The fitted B parameters are : \n{}'.format(m.B_params))
    print('The true transition matrix is : \n{}'.format(label['A']))
    print('The fitted transition matrix is : \n{}'.format(m.A_hat))
    print('The true stationary distribution is : \n{}'.format(label['phi']))
    print('The fitted stationary distribution is : \n{}'.format(m.phi))


def main():
#    print('Processing test case 1...')
#    # y1 is the generated samples, while
#    # target is a dictionary containing the labels:
#    y1, label1 = gen_ex1()
#    m1 = SPDGaussianHMM(S=1, p=3)
#    evaluate(m1, y1, label1)

#    print('Processing test case 2...')
#    y2, label2 = gen_ex2()
#    m2 = SPDGaussianHMM(S=2, p=3)
#    evaluate(m2, y2, label2)

    print('Processing test case 3...')
    y3, label3 = gen_ex3()
    m3 = SPDGaussianHMM(S=3, p=3)
    evaluate(m3, y3, label3)

if __name__ == "__main__":
    main()
