import numpy as np
import math
import scipy

from sklearn.base import BaseEstimator
import cvxpy as cp

import randSPDGauss

class _BaseGeomHMM(BaseEstimator):
    """Abstract base class for geometric HMM."""
    def __init__(self):
        pass

    def partial_fit(self):
        pass

class SPDGaussianHMM(_BaseGeomHMM):
    """SPD-valued HMM with Gaussian emission prob

    Notes
    -----
    Only implemented for 2-by-2 SPDs so far.
    Dist hard coded as Riemannian affine-invariant.
    """
    def __init__(self): # Hard code the hyperparameters for now.
        self.max_lag = 4
        # Cached observations used to compute H hat:
        self.obs_cache = []
        self.S = 3 # Number of hidden states.
        self.N = 0 # Total number of examples seen.
        self.p = 2 # Dimension of the SPD matrices.

        self.phi = np.empty(self.S) # phi denotes the stationary distribution.
        self.A_hat = np.empty([self.S, self.S]) # Transition matrix
        self.B_params = [[None, None] for i in range(self.S)]
        self.H_hat = np.empty([self.max_lag, self.S, self.S])
        self.K_hat = np.empty([self.S, self.S])

    def update_phi_B(self, y):
        """Update current estimate for phi and B."""
        #pass #TODO
        for ind, y_i in enumerate(y):
            # TODO: Need to clarify the appropraite choice of gamma;
            #       for now follow eqn (9) of Titterington 1984
            gammaNp1 = (self.N+ind+1)**(-1)

            # Compute update of phi. See Zanini et al. 2017, eqn (6).
            sN = np.sqrt(self.phi)
            # Compute h_k:
            h = [(sN[i]**2)*self.compute_B(y_i, i) for i in range(self.S)]
            h = np.array(h)
            h = h/h.sum()
            # Compute xi^N+1:
            xiNp1 = h*gammaNp1/(sN*2)
            xiNp1 = xiNp1 - sN
            # Compute s^N+1:
            norm_xi = np.linalg.norm(xiNp1, ord='fro')
            sNp1 = sN*math.cos(norm_xi) + (h/sN - sN)*gammaNp1*math.sin(norm_xi)/(norm_xi*2)
            # Update phi:
            self.phi = sNp1**2



    def update_hat_A(self):
        """Update estimate for A using convex opt."""

        C_hat = np.empty([self.max_lag+1, self.S, self.S])

        self.phi = np.ones(self.S)/self.S # TODO Temporary substitute.
        np.fill_diagonal(C_hat[0, :, :], self.phi)
        Id = np.identity(self.S)
        
        for t in range(self.max_lag):
            curr_tau = t+1 # While C is indexed by curr_tau, H is indexed
                           #  by t. May need to change since confusing.

            # Define and solve a convex problem:
            X = cp.Variable((self.S, self.S))
            self.K_hat = Id #TODO Temporary substitute.
            cost = cp.norm(self.H_hat[t,: ,:] - self.K_hat.T @ C_hat[curr_tau-1, :, :] @ X @ self.K_hat, p='fro')**2
            constraints = [X >= 0, X@Id == Id]
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve()

            C_hat[curr_tau, :, :] = np.matmul(C_hat[curr_tau-1, :, :], X.value)

        # Compute estimate for A by solving yet another convex problem:
        A_hat_stack = cp.Variable((self.S, self.S))
        curr_C_hat_a = np.reshape(C_hat[0:self.max_lag,:,:], (self.S*self.max_lag, self.S))
        curr_C_hat_b = np.reshape(C_hat[1:self.max_lag+1,:,:], (self.S*self.max_lag, self.S))
        cost = cp.norm(curr_C_hat_a@A_hat_stack - curr_C_hat_b, p='fro')**2
        constraints = [A_hat_stack >= 0, A_hat_stack@Id == Id]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        self.A_hat = A_hat_stack.value


    def compute_B(self, y, i):
        """Compute the conditional density B(y | i)."""
        mean = self.B_params[i][0]
        sigma = self.B_params[i][1]

        # NOTE: Below is needed since update_phi_B not implemented yet.
        if mean is None:
            mean = np.eye(self.p)
        if sigma is None:
            sigma = 1

        # TODO Need to modify the following to extend to general m x m matrices:
        assert y.shape == (2, 2)
        norm_factor = ((2*math.pi)**(3/2)) * (sigma**2) * (math.exp(sigma**2/4)) * math.erf(sigma/2)

        dist = d_AIRD(y, mean)

        return math.exp(-(dist**2)/(2*(sigma**2)))/norm_factor

    def update_hat_K(self, numSamples=100):
        """Update current estimate for hat K.
        
        Currently using Standard Monte Carlo, but may derive close form of
        the integral in specific cases.
        """
        for j in range(self.S):
            mean = self.B_params[j][0]
            sigma = self.B_params[j][1]
    
            # NOTE: Below is needed since update_phi_B not implemented yet.
            if mean is None:
                mean = np.eye(self.p)
            if sigma is None:
                sigma = 1

            samples = randSPDGauss.randSPDGauss(mean, sigma, numSamples)
            samples = [samples[:,:,k] for k in range(numSamples)]

            for i in range(self.S):
                vals = []
                for sample in samples:
                    vals.append(self.compute_B(sample, i))
                vals = np.array(vals)
                self.K_hat[i, j] = np.mean(vals)

    def update_hat_H(self, y):
        """Update current estimate for hat H."""

        len_new_obs = len(y)
        y = self.obs_cache + y
        offset = len(self.obs_cache)
        for t in range(self.max_lag):
            curr_tau = t+1
            for i in range(self.S):
                for j in range(self.S):
                    for k in range(len_new_obs):
                        if offset + k - curr_tau < 0:
                            continue
                        y_t = y[offset + k]
                        y_t_minus_tau = y[offset + k - curr_tau]

                        self.H_hat[t, i, j] *= (self.N-curr_tau)
                        self.H_hat[t, i, j] += self.compute_B(y_t, i)*self.compute_B(y_t_minus_tau, j)
                        self.H_hat[t, i, j] /= (self.N+len_new_obs-curr_tau)

    def partial_fit(self, y):
        """Incremental fit on a batch of samples.

        Parameters
        ----------
        y : list of numpy array of shape (self.S, self.S)
            Observed values.

        """
        # TODO: Update B hat and phi hat.


        # Update H hat:
        self.update_hat_H(y)

        # Update K hat:
        self.update_hat_K()

        # Update A hat:
        self.update_hat_A()

        # Finally, update obs_cache and N:
        for y_i in y:
            self.obs_cache.append(y_i)
        self.obs_cache = self.obs_cache[-self.max_lag:]

        self.N += len(y)


def is_SPD(X):
    """Check if a given matrix is (very nearly) an SPD matrix."""
    is_positive = np.all(np.linalg.eigvals(X) > 0)
    # NOTE: Sometimes the matrix is not perfectly symmetric for some reason.
    #       Might be just an artifact of the sampling fxn (randSPDGauss)
    is_symmetric = (np.round(X, 5) == np.round(X.T, 5)).all()
    if not (is_positive and is_symmetric):
        print('%.32f' % X[0, 1])
        print('%.32f' % X[1, 0])
        print(is_positive)
        print(is_symmetric)
        print(X[0, 1] == X[1, 0])
    return is_positive and is_symmetric

def d_AIRD(X, Y):
    """Compute affine-invariant Riemannian distance."""
    assert is_SPD(X) and is_SPD(Y) # Check if X and Y are SPD matrices
    dist = scipy.linalg.logm(np.matmul(np.linalg.inv(X), Y))
    dist = np.linalg.norm(dist, ord='fro')
    return dist


def main():
    m = SPDGaussianHMM()
    a = np.eye(2)
    y = randSPDGauss.randSPDGauss(a, 1, 10)
    obs = [y[:,:,i] for i in range(y.shape[2])]
    m.partial_fit(obs)

if __name__ == "__main__":
    main()
