import numpy as np
import math
# Somehow one cannot just do import scipy and call scipy.linalg.*
import scipy.linalg as la 

def mhsample(start, nsamples, pdf, proppdf, proprnd):
    """
    Metropolis-Hastings with independence sampler.

    """
    x = start
    accepted = []
    rejected = []   
    while len(accepted) <= nsamples:
        x_ =  proprnd(x)    
        A = min(1, pdf(x_)*proppdf(x, x_)/(pdf(x)*proppdf(x_, x)))
        u = np.random.uniform(low=0.0, high=1.0)
        if u <= A:
            x = x_
            accepted.append(x)
        else:
            rejected.append(x)            
                
    return np.array(accepted), np.array(rejected)


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

def generate_ri(gamma, d, N):
    """ Generate random r_i samples (see equation (29) of Said et al., 2017)
    """

    X = 1000 # Number of initial samples we want to omit.
    delta = gamma

    def pdf(x): 
        return (math.exp(-1*np.sum(np.power(x, 2))/(2*(gamma**2))))*prod_sinh(x, d)
    def proppdf(x, y): return unifpdf_dim_p(y-x, -delta, delta)
    def proprnd(x): return x + np.random.rand(d)*2*delta - delta

    r, _ = mhsample(np.random.rand(d), N+X, pdf, proppdf, proprnd)
    return r[X:] # Chop off the ommited samples.

def randSPDGauss(Ybar, gamma, N):
    """ Generate N samples from a Gaussian SPD manifold with mean Ybar and dispersion gamma.

        Ybar: numpy array
    """
    p = Ybar.shape[0]
    if p == 2:
        pass #TODO: implement closed-form expression for when p=2.

    Y = np.zeros((p, p, N))

    Z = np.random.rand(p, p, N)
    O = np.zeros((p, p, N))
    T = np.zeros((p, p, N))
    for i in range(N):
        O[:,:,i], T[:,:,i] = np.linalg.qr(Z[:,:,i], mode='complete')

    r = generate_ri(gamma, p, N)

    for i in range(N):
        theta = np.copy(O[:,:,i])

        ri = np.array(r[i])
        D = np.diag(np.exp(ri))
        Temp = np.matmul(theta.T, D)
        Y[:,:,i] = np.matmul(Temp, theta)

    g = la.sqrtm(Ybar) #NOTE: It is unclear which square root scipy chooses.

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
