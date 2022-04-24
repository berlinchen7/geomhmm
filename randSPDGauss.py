import numpy as np
import math
import utils
# Somehow one cannot just do import scipy and call scipy.linalg.*
import scipy.linalg as la 


def prod_sinh(x, d):
    """Compute \Pi_{i<j} \sinh(|r_i - r_j|/2). """
    p = 1
    for i in range(1, d): # Indexing kinda weird bc adapted from Matlab.
        for j in range(i+1, d+1):
            p = p * math.sinh(abs(x[i-1] - x[j-1])/2)
    return p

def unifpdf_dim_p(x, a, b):
    n = x.shape[0]
    if (x < a).all() or (x > b).all():
        return 0
    return (1/(b-a))**n

def generate_ri(gamma, d, N, rng, omit):
    """ Generate random r_i samples (see equation (29) of Said et al., 2017)
    """

    delta = gamma

    def pdf(x): 
        return (math.exp(-1*np.sum(np.power(x, 2))/(2*(gamma**2))))*prod_sinh(x, d)
    def proppdf(x, y): return unifpdf_dim_p(y-x, -delta, delta)
    def proprnd(x): return x + rng.random(d)*2*delta - delta

    r, _ = utils.mhsample(rng.random(d), N+omit, pdf, proppdf, proprnd, rng=rng)
    return r[omit:] # Chop off the ommited samples.

def generate_r(sigma, num_samples, rng, omit):
    ''' Generate the radius component of random samples.
    For more details, see, e.g., equation (24) of Said et al., 2017.
    '''

    delta = sigma
    pdf = lambda x: np.exp(-.5*np.square(x)/(sigma**2)) * np.sinh(x)
    proppdf = lambda x, y: utils.unifpdf(y - x, -delta, delta)
    proprnd = lambda x: x + rng.random()*2*delta - delta

    r, _ = utils.mhsample(1, num_samples+omit, pdf, proppdf, proprnd, rng=rng)
    return r

def randSPDGauss_p2(Ybar, gamma, N, rng=None, omit=100):
    if rng is None:
        rng = np.random.default_rng()
    Y = np.zeros((2, 2, N))

    # Generate uniformly distributed sample, O,
    # of orthogonal matrices:
    Z = rng.standard_normal(size=(2, 2, N))
    O = np.zeros((2, 2, N))
    T = np.zeros((2, 2, N))
    for i in range(N):
        O[:,:,i], T[:,:,i] = np.linalg.qr(Z[:,:,i], mode='complete')

    # Generate log determinant, t, of
    # Y. This has normal distribution with 
    # mean 0 and variance 2gamma^2:
    t = (2*gamma**2)**.5*rng.standard_normal(N)

    # Generate log of ratio of eigen
    # values of Y, r. This uses Metropolis 
    # algorithm:
    r = generate_r(gamma, N, rng, omit)

    # Recover a Id-centered samples from t and r: 
    for i in range(N):
        theta = O[:,:,i]

        r1 = (t[i] + r[i])/2
        r2 = (t[i] - r[i])/2
        D = np.diag([np.exp(r1), np.exp(r2)])
        Y[:,:,i] = theta.T @ D @ theta

    # Translate to center at Ybar:
    # g = la.sqrtm(Ybar) #NOTE: It is unclear which square root scipy chooses.
    g = utils.SPD_sqrt(Ybar)

    for i in range(N):
        Temp = Y[:,:,i].copy()
        Y[:,:,i] = g.T @ Temp @ g

    return T


def randSPDGauss(Ybar, gamma, N, rng=None, omit=100):
    """ Generate N samples from a Gaussian SPD manifold with mean Ybar and dispersion gamma.

        Ybar: numpy array
    """
    if rng is None:
        rng = np.random.default_rng()

    p = Ybar.shape[0]
    if p == 2:
        return randSPDGauss_p2(Ybar, gamma, N, rng, omit)

    Y = np.zeros((p, p, N))

    Z = rng.standard_normal(size=(p, p, N))
    O = np.zeros((p, p, N))
    T = np.zeros((p, p, N))
    for i in range(N):
        O[:,:,i], T[:,:,i] = np.linalg.qr(Z[:,:,i], mode='complete')

    r = generate_ri(gamma, p, N, rng, omit)

    for i in range(N):
        theta = np.copy(O[:,:,i])

        ri = np.array(r[i])
        D = np.diag(np.exp(ri))
        Temp = np.matmul(theta.T, D)
        Y[:,:,i] = np.matmul(Temp, theta)

    # g = la.sqrtm(Ybar) #NOTE: It is unclear which square root scipy chooses.
    g = utils.SPD_sqrt(Ybar)

    for i in range(N):
        Temp1 = np.copy(Y[:,:,i])
        Temp2 = np.matmul(g.T, Temp1)
        Y[:,:,i] = np.matmul(Temp2, g)
        
    return Y

def main():
    # Some sanity check:
    from numpy.linalg import matrix_rank
    for n in range(2, 6):
        a = np.eye(n)
        ret = randSPDGauss(a, 1, 3)
        for i in range(3):
            assert matrix_rank(ret[:,:,i]) == n

if __name__ == "__main__":
    main()
