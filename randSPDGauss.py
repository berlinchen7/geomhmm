import numpy as np
import math
import utils
from tqdm import tqdm
# Somehow one cannot just do import scipy and call scipy.linalg.*
import scipy.linalg as la 
from scipy.stats import ortho_group
from scipy.optimize import minimize_scalar



def prod_sinh_alternative(x, d):
    """Compute \Pi_{i<j} \sinh(|r_i - r_j|/2).

    An alternative method relying entirely on numpy 
    manipulations, as opposed to python for loops.
    """
    repeated_r = np.repeat(x, d, axis=0)
    r_diff_matrix = (repeated_r.reshape((d, d), order="F") 
                   - repeated_r.reshape((d, d), order="C"))
    r_diff_vec = r_diff_matrix[np.triu_indices(d, k=1)]
    r_vec = np.absolute(r_diff_vec)/2
    r_vec = np.sinh(r_vec)
    return np.prod(r_vec)

def prod_sinh(x, d):
    """Compute \Pi_{i<j} \sinh(|r_i - r_j|/2). 
    """
    p = 1
    for i in range(d-1):
        for j in range(i+1, d):
            p = p * math.sinh(abs(x[i] - x[j])/2)
    return p

def unifpdf_dim_p(x, a, b):
    n = x.shape[0]
    if (x < a).all() or (x > b).all():
        return 0
    return (1/(b-a))**n



def generate_ri_RW(gamma, d, N, rng, omit, delta=.01):
    """ Generate random r_i samples (see equation (29) of Said et al., 2017)

    Uses MH with random walk.
    """
    def pdf(x): 
        return (math.exp(-1*np.sum(np.power(x, 2))/(2*(gamma**2))))*prod_sinh(x, d)
    def proppdf(x, y): return unifpdf_dim_p(x - y, -delta, delta)
    def proprnd(x): return x + rng.random(d)*2*delta - delta

    r, _ = utils.mhsample(rng.random(d), N+omit, pdf, proppdf, proprnd, rng=rng)
    return r[omit:] # Chop off the ommited samples.

def generate_ri_MALA(gamma, d, N, rng, tau, omit=1000):
    """ Generate random r_i samples (see equation (29) of Said et al., 2017).

    Uses Metropolis-Adjusted Langevin Algorithm.
    """

    def compute_log_grad_pdf(x):
        ret = -1*x.copy()/(gamma**2)
        for k in range(d):
            curr_diff = 0
            for i in range(d):
                if i < k and x[i] >= x[k]:
                    curr_diff += (-.5)*np.cosh((x[i] - x[k])/2)/np.sinh((x[i] - x[k])/2)
                elif k < i and x[k] >= x[i]:
                    curr_diff += (.5)*np.cosh((x[k] - x[i])/2)/np.sinh((x[k] - x[i])/2)
                elif i < k and x[i] < x[k]:
                    curr_diff += (-.5)*np.cosh((x[i] - x[k])/2)/np.sinh((x[i] - x[k])/2)
                elif k < i and x[k] < x[i]:
                    curr_diff += (.5)*np.cosh((x[k] - x[i])/2)/np.sinh((x[k] - x[i])/2)
            ret[k] += curr_diff
        return ret

    def pdf(x): 
        return (math.exp(-1*np.sum(np.power(x, 2))/(2*(gamma**2))))*prod_sinh(x, d)
    
    def proppdf(x, y): return np.exp((-1/(4*tau))*np.linalg.norm(x - y - tau*compute_log_grad_pdf(y))**2) 
    def proprnd(x): 
        loggradient = compute_log_grad_pdf(x)
        return x + tau*loggradient + ((2*tau)**.5)*rng.multivariate_normal(np.zeros(d), np.eye(d))

    r, _ = utils.mhsample(rng.random(d), N+omit, pdf, proppdf, proprnd, rng=rng)
    return r[omit:] # Chop off the ommited samples.

def generate_ri_ULA(gamma, d, N, rng, tau, omit=1000):
    """ Generate random r_i samples (see equation (29) of Said et al., 2017).

    Uses Unconstrained Langevin Algorithm.
    """

    def compute_log_grad_pdf(x):
        ret = -1*x.copy()/(gamma**2)
        for k in range(d):
            curr_diff = 0
            for i in range(d):
                if i < k and x[i] >= x[k]:
                    curr_diff += (-.5)*np.cosh((x[i] - x[k])/2)/np.sinh((x[i] - x[k])/2)
                elif k < i and x[k] >= x[i]:
                    curr_diff += (.5)*np.cosh((x[k] - x[i])/2)/np.sinh((x[k] - x[i])/2)
                elif i < k and x[i] < x[k]:
                    curr_diff += (-.5)*np.cosh((x[i] - x[k])/2)/np.sinh((x[i] - x[k])/2)
                elif k < i and x[k] < x[i]:
                    curr_diff += (.5)*np.cosh((x[k] - x[i])/2)/np.sinh((x[k] - x[i])/2)
            ret[k] += curr_diff
        return ret

    samples = [rng.random(d)]
    for i in range(1, N+omit):
        prev_step = samples[i-1]
        next_step = prev_step + tau*compute_log_grad_pdf(prev_step) + ((2*tau)**.5)*rng.standard_normal(d)
        samples.append(next_step)
    
    return samples[omit:] # Chop off the ommited samples.

def generate_ri_Gibbs_MALA(gamma, d, N, rng, tau, omit, marginal_omit=100):
    """ Generate random r_i samples (see equation (29) of Said et al., 2017).

    Uses Gibbs sampler, where marginal distributions sampled with MALA.

    marginal_omit: number of samples to omit when sampling the marginal distributions.
    """
    
    def compute_log_grad_pdf(r, conds):
        ret = -1*r/(gamma**2)
        for cond in conds:
            ret += (r-cond)/(2*abs(r - cond)*np.tanh(abs(r-cond)/2))
        return ret

    def pdf(r, conds): 
        ret = np.exp((-1*r**2)/(2*gamma**2))
        for cond in conds:
            ret *= np.sinh(abs(r - cond)/2)
        return ret
    
    def proppdf(x, y, conds): return np.exp((-1/(4*tau))*np.linalg.norm(x - y - tau*compute_log_grad_pdf(y, conds))**2) 

    def proprnd(x, conds): 
        loggradient = compute_log_grad_pdf(x, conds)
        return x + tau*loggradient + ((2*tau)**.5)*rng.standard_normal()

    samples = [rng.random(d).tolist()]
    # Set disable=False if you want to see the progress of generating the
    # ri samples:
    for i in tqdm(range(1, N+omit), desc="  Sampling ri", disable=True):
        prev_step = samples[i-1]
        curr_sample = []
        for j in range(d):
            cond_before = curr_sample
            cond_after = [] if j + 1 >= d else prev_step[j+1:]
            conds = cond_before + cond_after

            curr_pdf = lambda x: pdf(x, conds)
            curr_proppdf = lambda x, y: proppdf(x, y, conds)
            curr_proprnd = lambda x: proprnd(x, conds)

            # Start at the mode of the marginal dist.,
            # otherwise MCMC difficult to converge:
            neg_pdf = lambda x: -1*curr_pdf(x)
            mode_res = minimize_scalar(neg_pdf)

            curr_s, _ = utils.mhsample(mode_res.x, 1+marginal_omit, curr_pdf, curr_proppdf, curr_proprnd, rng=rng)
            curr_sample.append(curr_s[-1])
        samples.append(curr_sample)

    return samples[omit:] # Chop off the ommited samples.

def generate_ri_Gibbs_RW(gamma, d, N, rng, omit, delta=.01, marginal_omit=100):
    """ Generate random r_i samples (see equation (29) of Said et al., 2017).

    Uses Gibbs sampler, where marginal distributions sampled with Random Walk MCMC.

    marginal_omit: number of samples to omit when sampling the marginal distributions.
    """    

    def pdf(r, conds): 
        ret = np.exp((-1*r**2)/(2*gamma**2))
        for cond in conds:
            ret *= np.sinh(abs(r - cond)/2)
        return ret
    
    def proppdf(x, y): return utils.unifpdf(y - x, -delta, delta)
    def proprnd(x): return x + rng.random()*2*delta - delta

    samples = [rng.random(d).tolist()]
    # Set disable=False if you want to see the progress of generating the
    # ri samples:
    for i in tqdm(range(1, N+omit), desc="  Sampling ri", disable=True):
        prev_step = samples[i-1]
        curr_sample = []
        for j in range(d):
            cond_before = curr_sample
            cond_after = [] if j + 1 >= d else prev_step[j+1:]
            conds = cond_before + cond_after

            curr_pdf = lambda x: pdf(x, conds)

            # Start at the mode of the marginal dist.,
            # otherwise MCMC difficult to converge:
            neg_pdf = lambda x: -1*curr_pdf(x)
            mode_res = minimize_scalar(neg_pdf)

            curr_s, _ = utils.mhsample(mode_res.x, 1+marginal_omit, curr_pdf, proppdf, proprnd, rng=rng)
            curr_sample.append(curr_s[-1])
        samples.append(curr_sample)

    return samples[omit:] # Chop off the ommited samples.



def generate_r_RW(sigma, num_samples, rng, omit):
    ''' Generate the radius component of random samples.
    For more details, see, e.g., equation (24) of Said et al., 2017.

    Uses MH with random walk.
    '''

    delta = .05
    pdf = lambda x: np.exp(-.25*np.square(x)/(sigma**2)) * np.sinh(.5*np.abs(x))
    proppdf = lambda x, y: utils.unifpdf(y - x, -delta, delta)
    proprnd = lambda x: x + rng.random()*2*delta - delta

    # The pdf of r is symmetric and bimodal, and hence generate sample starting from the 2 modes:

    neg_pdf = lambda x: -1*pdf(x)
    mode_res = minimize_scalar(neg_pdf)
    starting_point = abs(mode_res.x) # Global maxima 1.

    half_num_samples = int(num_samples/2)
    r1, _ = utils.mhsample(starting_point, half_num_samples+omit, pdf, proppdf, proprnd, rng=rng)
    r1 = r1[omit:]

    starting_point = -1*starting_point # Global maxima 2.
    r2, _ = utils.mhsample(starting_point, num_samples+half_num_samples+omit, pdf, proppdf, proprnd, rng=rng)
    r2 = r2[omit:]

    ret = np.concatenate((r1, r2))
    rng.shuffle(ret)
    return ret

    # The following is a simpler approach where the RW starts from 1:
    # r, _ = utils.mhsample(1, num_samples+omit, pdf, proppdf, proprnd, rng=rng)
    # return r[omit:]

def generate_r_MALA(gamma, num_samples, rng, tau, omit):
    ''' Generate the radius component of random samples.
    For more details, see, e.g., equation (24) of Said et al., 2017.

    Uses Metropolis-Adjusted Langevin Algorithm
    '''

    def compute_log_grad_pdf(x):
        return -1*x/(2*gamma**2) + .5*np.cosh(.5*x)/np.sinh(.5*x)

    def proppdf(x, y): return np.exp((-1/(4*tau))*np.linalg.norm(x - y - tau*compute_log_grad_pdf(y))**2) 
    def proprnd(x): 
        loggradient = compute_log_grad_pdf(x)
        return x + tau*loggradient + ((2*tau)**.5)*rng.standard_normal()
    def pdf(x): return np.exp(-.25*np.square(x)/(gamma**2)) * np.sinh(.5*np.abs(x))

    # The pdf of r is symmetric and bimodal, and hence generate sample starting from the 2 modes:
    
    neg_pdf = lambda x: -1*pdf(x)
    mode_res = minimize_scalar(neg_pdf)
    starting_point = abs(mode_res.x) # Global maxima 1.

    half_num_samples = int(num_samples/2)
    r1, _ = utils.mhsample(starting_point, half_num_samples+omit, pdf, proppdf, proprnd, rng=rng)
    r1 = r1[omit:]

    starting_point = -1*starting_point # Global maxima 2.
    r2, _ = utils.mhsample(starting_point, num_samples+half_num_samples+omit, pdf, proppdf, proprnd, rng=rng)
    r2 = r2[omit:]

    ret = np.concatenate((r1, r2))
    rng.shuffle(ret)
    return ret

    # The following is a simpler approach where the Markov chain starts from 1:
    # r, _ = utils.mhsample(1, num_samples+omit, pdf, proppdf, proprnd, rng=rng)
    # r = r[omit:]
    # return r



def randSPDGauss_p2(Ybar, gamma, N, rng=None, omit=None):
    if rng is None:
        rng = np.random.default_rng()
    Y = np.zeros((2, 2, N))

    # Generate log determinant, t, of
    # Y. This has normal distribution with 
    # mean 0 and variance 2gamma^2:
    t = (2*gamma**2)**.5*rng.standard_normal(N)

    # Generate log of ratio of eigen
    # values of Y, r. This uses MCMC:
    omit = omit or 100000
    r = generate_r_RW(gamma, N, rng, omit)
    # r = generate_r_MALA(gamma, N, rng, .0001, omit)

    # Recover a Id-centered samples from t and r: 
    for i in range(N):
        theta = ortho_group.rvs(2, random_state=rng)

        # t and r are the "t" and "rho" in the remarks preceding (31) of
        # "Riemannian Gaussian Distributions on the Space of Symmetric Positive Definite Matrices"
        # by Said et al..
        r1 = (t[i] + r[i])/2
        r2 = (t[i] - r[i])/2
        D = np.diag([np.exp(r1), np.exp(r2)])
        Y[:,:,i] = theta.T @ D @ theta

    # Translate to center at Ybar:
    # g = la.sqrtm(Ybar) #NOTE: It is unclear which square root scipy chooses.
    g = utils.SPD_sqrt(Ybar)

    for i in range(N):
        Y[:,:,i] = g.T @ Y[:,:,i] @ g

    return Y

def randSPDGauss(Ybar, gamma, N, rng=None, omit=None):
    """ Generate N samples from a Gaussian SPD manifold with mean Ybar and dispersion gamma.

        Ybar: numpy array
    """
    if rng is None:
        rng = np.random.default_rng()

    p = Ybar.shape[0]
    if p == 2:
        return randSPDGauss_p2(Ybar, gamma, N, rng, omit)

    Y = np.zeros((p, p, N))

    omit = omit or 1000
    # r = generate_ri_RW(gamma, p, N, rng, omit, delta=.006)
    # r = generate_ri_MALA(gamma, p, N, rng, .0001, omit=1000)
    # r = generate_ri_Gibbs_MALA(gamma, p, N, rng, .001, omit=1000)
    r = generate_ri_Gibbs_RW(gamma, p, N, rng, omit)

    for i in range(N):
        theta = ortho_group.rvs(p, random_state=rng)

        ri = np.array(r[i])
        D = np.diag(np.exp(ri))
        Y[:,:,i] = theta.T @ D @ theta

    # g = la.sqrtm(Ybar) #NOTE: It is unclear which square root scipy chooses.
    g = utils.SPD_sqrt(Ybar)

    for i in range(N):
        Y[:,:,i] = g.T @ Y[:,:,i] @ g
        
    return Y


def main():
    # Some sanity check:
    from numpy.linalg import matrix_rank
    for n in range(2, 6):
        a = np.eye(n)
        ret = randSPDGauss(a, 1, 3)
        for i in range(3):
            assert matrix_rank(ret[:,:,i]) == n

    # Print some examples:
    Ybar = np.eye(2)
    gamma = .1
    N = 10
    samples = randSPDGauss(Ybar, gamma, N)
    for i in range(N):
        print(samples[:,:,i])

if __name__ == "__main__":
    main()
