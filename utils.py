import itertools
import math
import numpy as np


def mhsample(start, nsamples, pdf, proppdf, proprnd, rng=None, max_tries=5e5):
    """
    Metropolis-Hastings sampler.

    Note:
    - proppdf(x, y) = P(x | y), NOT P(y | x).
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
    S. Said, L. Bombrun, and Y. Berthoumieu. New riemannian priors on the univariate normal model.
    """
    const = (2*math.pi)*((math.pi*.5)**.5)
    exp = math.exp(.5*(sigma**2))
    erf = math.erf(sigma/(2**(.5)))
    return const*sigma*exp*erf

def compute_PDGauss_pdf(x, centroid, sigma):
    norm_factor = compute_PD_norm_factor(sigma)
    dist = compute_PD_dist(x, centroid)
    return math.exp(-(dist**2)/(2*(sigma**2)))/norm_factor

def is_SPD(X):
    """Check if a given matrix is (very nearly) an SPD matrix.
    
    NOTE:
    	1. Check symmetric only up to 2 decimal places.
    """
    is_symmetric = (np.round(X, 2) == np.round(X.T, 2)).all()    
    is_positive = np.all(np.linalg.eigvals(X) > 0)

    if not (is_positive and is_symmetric):
        print(X)
        print(np.linalg.eigvals(X))
        print(is_symmetric)
        print(is_positive)
    return is_positive and is_symmetric

def SPD_ize(M):
	"""Return a copy of M that has been projected to the SPD manifold."""

	ret = M.copy()
	N = M.shape[0]

	# Force symmetrize the matrix:
	ret[np.tril_indices(N, k=-1)] = ret.T[np.tril_indices(N, k=-1)]

	# Set negative eigenvalues to a small positive value: 
	eigvals, eigvecs = np.linalg.eig(ret)
	eigvals = eigvals.real
	eigvals[eigvals <= 0] = .01
	ret = eigvecs @ np.diag(eigvals) @ eigvecs.T

	return ret


def SPD_sqrt(M, check_SPD=False, proj_to_SPD=True):
	"""Compute the square root of a SPD matrix. 
	"""
	if check_SPD:
		assert(is_SPD(M))
	if proj_to_SPD:
		M = SPD_ize(M)
	u, s, vh = np.linalg.svd(M)
	# print("M is {}, u is {}, s is {}. vh is {}".format(M, u, s, vh))
	ret = u @ (np.diag(s)**.5) @ vh
	# print("{} has shape {}".format(ret, ret.shape))
	return ret

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

def match_permutation(true, predicted, num_states, dist):
    ''' Find the permutation that minimizes the average distance between true and predicted.
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
