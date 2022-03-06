import numpy as np
import torch, math
import geoopt

def mhsample(start, nsamples, pdf, proprnd):                                                      
    """                                                                                                    
    Metropolis-Hastings with random sampler.                                                         
                                                                                                           
    """                                                                                                    
    x = start                                                                                              
    accepted = []                                                                                          
    rejected = []                                                                                          
    while len(accepted) <= nsamples:                                                                       
        x_ =  proprnd(x)                                                                                   
        A = min(1, pdf(x_)/(pdf(x)))                                         
        u = np.random.uniform(low=0.0, high=1.0)                                                           
        if u <= A:                                                                                         
            x = x_                                                                                         
            accepted.append(x)                                                                             
        else:                                                                                              
            rejected.append(x)                                                                             
    return np.array(accepted), np.array(rejected), x

def compute_dist(X, Y):
    """Compute the hyperbolic distance on the Poincare Disk, using the geoopt package.
    """
    X, Y = torch.tensor(X), torch.tensor(Y)
    PD = geoopt.PoincareBall()
    return PD.dist(x=X, y=Y).item()

def compute_norm_factor(sigma):
    """
    Formula taken from:
    S. Said, L. Bombrun, and Y. Berthoumieu. New riemannian priors on the univariate normal model."""
    const = (2*math.pi)*((math.pi*.5)**.5)
    exp = math.exp(.5*(sigma**2))
    erf = math.erf(sigma/(2**(.5)))
    return const*sigma*exp*erf

def compute_PDGauss_pdf(x, centroid, sigma):
    norm_factor = compute_norm_factor(sigma)

    dist = compute_dist(x, centroid)

    return math.exp(-(dist**2)/(2*(sigma**2)))/norm_factor


def randPoincGauss(Ybar, sigma, N, seed=1, omit=50):
    """ Generate N samples from a Gaussian SPD manifold with mean Ybar and dispersion gamma.

    Use MCMC with Random Walk, along with rejection sampling.
    """
    np.random.seed(seed)

    Ybar = np.array(Ybar)
    proprnd = lambda x: np.random.multivariate_normal(x, np.eye(2)*.001)
    pdf = lambda x: compute_PDGauss_pdf(x, Ybar, sigma)

    ret = []
    curr_start = Ybar
    while len(ret) < N + omit:
        curr_samples, _, new_start = mhsample(curr_start, N + omit, pdf, proprnd)
        for curr_sample in curr_samples:
            # Rejection sampling: throw away if not in the unit circle:
            if np.linalg.norm(curr_sample) < 1:
                ret.append(torch.tensor(curr_sample))
        curr_start = new_start
    return ret[omit:omit + N]


def main():
    print(randPoincGauss(torch.tensor([.2, .4]), .01, 10))

if __name__ == "__main__":
    main()
