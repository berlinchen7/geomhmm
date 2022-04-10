import numpy as np
import math

def mhsample_rs(start, nsamples, pdf, proprnd, rng=None):                                                      
    """                                                                                                    
    Metropolis-Hastings with random sampler.                                                         
                                                                                                           
    """      
    if rng is None:
    	rng = np.random.default_rng()

    x = start                                                                                              
    accepted = []                                                                                          
    rejected = []                                                                                          
    while len(accepted) <= nsamples:                                                                       
        x_ =  proprnd(x)     
        A = min(1, pdf(x_)/(pdf(x)))                                         
        u = rng.random()                                                           
        if u <= A:                                                                                         
            x = x_                                                                                         
            accepted.append(x)                                                                             
        else:                                                                                              
            rejected.append(x)                                                                             
    return np.array(accepted), np.array(rejected), x

def mhsample_is(start, nsamples, pdf, proppdf, proprnd, rng=None, max_tries=1e10):
    """
    Metropolis-Hastings with independence sampler.

    """
    if rng is None:
    	rng = np.random.default_rng()

    x = start
    accepted = []
    rejected = []   
    while len(rejected) + len(accepted) < max_tries:
        x_ = proprnd(x)    
        A = min(1, pdf(x_)*proppdf(x, x_)/(pdf(x)*proppdf(x_, x)))
        u = rng.random()
        if u <= A:
            x = x_
            accepted.append(x)

        else:
            rejected.append(x)

        if len(accepted) == nsamples:
            	break
    
    if len(rejected) + len(accepted) >= max_tries:
    	raise RuntimeError('Maximal tries exceeded in MCMC.')
                
    return np.array(accepted), np.array(rejected)

def compute_PD_dist(X, Y):
    """ Compute the hyperbolic distance on the Poincare Disk.
    """
    ret = 2*(np.linalg.norm(X - Y)**2)/ ((1 - np.linalg.norm(X)**2)*(1 - np.linalg.norm(Y)**2))
    ret = np.arccosh(1 + ret)
    return ret

def compute_PD_norm_factor(sigma):
    """
    Formula taken from:
    S. Said, L. Bombrun, and Y. Berthoumieu. New riemannian priors on the univariate normal model."""
    const = (2*math.pi)*((math.pi*.5)**.5)
    exp = math.exp(.5*(sigma**2))
    erf = math.erf(sigma/(2**(.5)))
    return const*sigma*exp*erf

def compute_PDGauss_pdf(x, centroid, sigma):
    norm_factor = compute_PD_norm_factor(sigma)
    dist = compute_PD_dist(x, centroid)
    return math.exp(-(dist**2)/(2*(sigma**2)))/norm_factor

def unifpdf_vect(x, lower, upper):
    too_low = x < lower
    too_high = x > upper
    just_right = (x >= lower) & (x <= upper)

    ret = x.copy()
    ret[too_low] = 0
    ret[too_high] = 0
    ret[just_right] = 1/(upper - lower)
    return ret

def unifpdf(x, lower, upper):
    if x < lower or x > upper:
        return 0
    else:
        return 1/(upper - lower)