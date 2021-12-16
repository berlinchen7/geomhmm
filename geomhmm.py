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
    """SPD-valued HMM with emission probabilities being Riemm. Gaussians.

    Notes
    -----
    - Currently does not support p-by-p SPD manifolds where p > 3,
      since haven't figured out how to systematically compute
      an orthonormal basis, where the first basis has trace zero.
    - Will also generalize this implementation to a more abstract
      Gaussian HMM that can have observations that are necc.
      SPD-valued.
    - The normalization constant is computed by MC estimation,
      which is slow. Can perhaps speed up using, e.g., caching.
    """
    def __init__(self, max_lag=4, S=3, N=0, p=2, alpha=.25): 
        ## Meta variables:
        self.max_lag = max_lag
        self.S = S # Number of hidden states.
        self.N = N # Total number of examples seen.
        self.p = p # Dimension of the SPD matrices.

        ## Variables to be learned:
        # phi denotes the stationary distribution; initialized to uniform:
        self.phi = np.ones(self.S)/self.S
        self.A_hat = np.empty([self.S, self.S]) # Transition matrix
        # B_params parametrizes the Gaussian mixture, represented as a list
        # of [p by p numpy array, int], which corresponds to centroid/dispersion:
        self.B_params = [[np.eye(self.p), 2] for i in range(self.S)]
        # NOTE: We choose to initialize the centroids to the first self.S
        #       observations and dispersion to 1. But set them to the identity
        #       and 1 for now. 

        ## Variables needed to compute A:
        self.H_hat = np.empty([self.max_lag, self.S, self.S])
        self.K_hat = np.empty([self.S, self.S])
        # Cached observations used to compute H hat:
        self.obs_cache = []

        ## Variables needed to compute phi and B:
        self.h = np.ones(self.S)/self.S # The "h" in Zanini et al., 2017; updated in the method update_phi().
        self.on_basis = [] # Orthonormal basis of some tangent space.
        self.alpha = alpha # Parameter for Tikhonov regularization, which is used in Tikhonov_inv().

    def compute_updated_centroid(self, y_i, gamma_Np1):
        # Initialize the orthonormal frame with respect to the identity matrix: 
        if self.p == 1:
            raise NotImplementedError
        elif self.p == 2:
            E1 = np.eye(2)/(2**(1/2))
            E2 = np.eye(2)/(2**(1/2))
            E2[1][1] *= -1
            E3 = np.array([[0, 1], [1, 0]])/(2**(1/2))
            self.on_basis = [E1, E2, E3]
        elif self.p == 3:
            E1 = np.eye(3)/(3**(1/2))
            E2 = np.array([[-1/(3**(1/2)), 0, 0], 
                           [0, .5+1/(2*3**(1/2)), 0],
                           [0, 0, -.5+1/(2*3**(1/2))]])
            E3 = np.array([[-1/(3**(1/2)), 0, 0], 
                           [0, -.5+1/(2*3**(1/2)), 0], 
                           [0, 0, .5+1/(2*3**(1/2))]])
            E4 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])/(2**(1/2))
            E5 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])/(2**(1/2))
            E6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])/(2**(1/2))
            self.on_basis = [E1, E2, E3, E4, E5, E6]
        elif self.p > 3:
            raise NotImplementedError('Currently does not support p-by-p SPD matrices with p>3.')
        else:
            raise RuntimeError('self.p is not a positive integer.')

        ret = []
        sigmas = np.array([B[1] for B in self.B_params])
        for k in range(self.S):
            # Translate the orthonormal frame to the tangent space of y_i:
            # (See p. 90 of Medical Image Analysis by Pennec et al.)
            P_sqrt = scipy.linalg.sqrtm(self.B_params[k][0])
            on_basis_transported = [P_sqrt @ Ei @ P_sqrt for Ei in self.on_basis]
    
            # Compute the Fisher information matrices:
            # (See expression (8) of Zanini et al., 2017)
            sigma_k = sigmas[k] # This denotes $\hat\sigma_k^{(N)}$.
            omega_k = self.phi[k] # This denotes $\hat\omega_k^{(N)}$.
            I_1 = omega_k/(sigma_k**2)
            # Note that we don't need to compute <., .> since the Fisher
            # Info matrix is with respect to an orthonormal basis.
    
            
            I_2 = np.eye(len(self.on_basis)-1)
            norm_factor = self.compute_norm_factor(sigma_k)
            # The following computation is used to express
            # $\psi'_2(\eta_k)$ in terms of $\zeta(\sigma_k)$ and $\zeta'(\sigma_k)$,
            # in the notation of Zanini et al., 2017:
            psi_2_prime_eta_k = 3*(sigma_k**2)*math.log(norm_factor)
            psi_2_prime_eta_k += (sigma_k**3)*self.compute_norm_factor_first_derivative(sigma_k)/norm_factor
            dim_M_2 = len(self.on_basis) - 1
            
            I_2 *= (omega_k*psi_2_prime_eta_k)/(dim_M_2*(sigma_k**4))
    
            # Compute the Riemannian score u (note that here we use e.g., y_k_N_1,
            # instead of x_k_N_1, which is different from the notation used by
            # Zanini et al., 2017):
            y_k_N_1, y_k_N_2 = self.compute_irr_decomp(self.B_params[k][0])
            y_Np1_1, y_Np1_2 = self.compute_irr_decomp(y_i)
            
            u_1 = self.h[k]/(sigma_k**2)
            u_1 *= y_k_N_1 - y_Np1_1

            P_sqrt = scipy.linalg.sqrtm(y_k_N_2)
            # We use Tikhonov regularized inverse b/c P_sqrt may have
            # very small eigenvalues:
            P_sqrt_inv = self.Tikhonov_inv(P_sqrt, self.alpha)
            u_2_mat = P_sqrt @ scipy.linalg.logm(P_sqrt_inv @ y_Np1_2 @ P_sqrt_inv) @ P_sqrt
            u_2_mat *= self.h[k]/(sigma_k**2)
            
            def g_y_k_N(V, W): # Computes <V, W>_Riem at y_k_N_2
                P_inv = self.Tikhonov_inv(y_k_N_2, self.alpha)
                return np.trace(V @ P_inv @ W @ P_inv)

            u_2 = np.zeros(dim_M_2)
            for i in range(dim_M_2):
                # Compute u_2[i] = <u_2_mat, E_{i+1}>_Riem:
                u_2[i] = g_y_k_N(u_2_mat, on_basis_transported[i+1])

            xi_1_Np1 = gamma_Np1*u_1/I_1
            xi_2_Np1 = gamma_Np1*(self.Tikhonov_inv(I_2, self.alpha) @ u_2)

            # Transform vector representation into matrix form:
            xi_2_Np1_mat = np.zeros([self.p, self.p])
            for i in range(dim_M_2):
                xi_2_Np1_mat += xi_2_Np1[i]*on_basis_transported[i+1]
            
            # Finally, store the updated centroid (c.f. eqns (9) (10) of Zanini et al., 2017):
            y_k_Np1_1 = y_k_N_1 - xi_1_Np1
            y_k_Np1_2 = P_sqrt @ scipy.linalg.expm(P_sqrt_inv @ xi_2_Np1_mat @ P_sqrt_inv) @ P_sqrt

            # The debug prints below was helpful:
            #print('P_sqrt; YkN2; Y_k_2:')
            #print(P_sqrt)
            #print(y_k_N_2)
            #print(self.B_params[k][0])

            y_k_Np1 = self.compute_irr_decomp_inv(y_k_N_1 - xi_1_Np1, 
                                    P_sqrt @ scipy.linalg.expm(P_sqrt_inv @ xi_2_Np1_mat @ P_sqrt_inv) @ P_sqrt)
            ret.append(y_k_Np1)
        return ret

    def Tikhonov_inv(self, A, alpha=.25):
        ''' Computes Tikohov-regularized inverse.
        
        Note that the regularized inverse is such that
        all eigenvalues <= 1/(2*(alpha**2)).
        '''
        u, s, vh = np.linalg.svd(A)
        s = [s_i/(s_i**2 + alpha) for s_i in s]
        return u @ np.diag(s) @ vh

    def compute_irr_decomp(self, x):
        ''' 
        Compute the decomposition of a SPD matrix into
        isomorphic product $\mathbb{R} times SP_m$, where
        $SP_m$ represents the manifold of SPD matrices with
        unitary determinant, while $\mathbb{R}$ takes into account the 
        part relative to the determinant (see the comments following
        equation (7) of Zanini et al., 2017).
        '''
        sign, logdet = np.linalg.slogdet(x)
        x1 = sign*logdet
        x2 = math.exp(-x1/self.p)*x
        return x1, x2

    def compute_irr_decomp_inv(self, x1, x2):
        return math.exp(x1/self.p)*x2

    def compute_updated_phi(self, y_i, gamma_Np1):
        # Compute update of phi. See Zanini et al. 2017, eqn (6).
        # (phi is denoted as omega in Zanini et al., but since we
        # are also using phi as the stationary distribution of the
        # HMM we denote it as phi).
        sN = np.sqrt(self.phi)
        # Compute h_k:
        h = [(sN[i]**2)*self.compute_B(y_i, i) for i in range(self.S)]
        h = np.array(h)
        h = h/h.sum()
        self.h = h # Update h for later use (e.g., update_centroid()).
        # Compute xi^N+1:
        xiNp1 = (gamma_Np1/2)*((h/sN)-sN)
        # Compute s^N+1:
        norm_xi = np.linalg.norm(xiNp1)
        if norm_xi == 0: # When xiNp1 is a vector of zeros:
            sNp1 = sN
        else:
            sNp1 = sN*math.cos(norm_xi) + (h/sN - sN)*gamma_Np1*math.sin(norm_xi)/(norm_xi*2)
        # Return updated phi:
        return sNp1**2

    def compute_updated_sigma(self, y_i, gamma_Np1):
        ret = []
        for k in range(self.S):
            sigma_k_N = self.B_params[k][1]
            eta_k_N = -1/(2*(sigma_k_N**2))

            zeta = self.compute_norm_factor(sigma_k_N)
            zeta_prime = self.compute_norm_factor_first_derivative(sigma_k_N)
            zeta_prime_prime = self.compute_norm_factor_second_derivative(sigma_k_N)

            psi_prime_eta_k_N = 3*(sigma_k_N**2)*math.log(zeta) + (sigma_k_N**3)*zeta_prime/zeta
            psi_prime_prime_eta_k_N = 15*(sigma_k_N**4)*math.log(zeta)
            psi_prime_prime_eta_k_N += 9*(sigma_k_N**5)*zeta_prime/zeta
            psi_prime_prime_eta_k_N += (sigma_k_N**6)*(zeta_prime_prime*zeta - (zeta_prime**2))/(zeta**2)

            dist_squared = self.compute_dist(self.B_params[k][0], y_i)**2
            omega_k = self.phi[k]

            eta_k_Np1 = dist_squared - psi_prime_eta_k_N
            eta_k_Np1 *= gamma_Np1*self.h[k]/(omega_k*psi_prime_prime_eta_k_N)
            eta_k_Np1 += eta_k_N
            
            # TODO: Sometimes eta_k_Np1 is not negative; not sure how to handle such a case, so set to
            #       abs(eta_k_Np1) for now:
            if eta_k_Np1 >= 0:
                eta_k_Np1 = -1*eta_k_Np1
#                raise RuntimeError('hat eta_{}^(N+1) is {}, which is not negative.'.format(k, eta_k_Np1))
            sigma_k_Np1 = np.sqrt(-1/(2*eta_k_Np1))
            ret.append(sigma_k_Np1)
        return ret


    def update_phi_B(self, y):
        """Update current estimate for phi and B."""
        for ind, y_i in enumerate(y):
            # TODO: Need to clarify the appropraite choice of gamma;
            #       for now follow eqn (9) of Titterington 1984
            gamma_Np1 = (self.N+ind+1)**(-1) # gammaNp1 denotes $\gamma^{(N+1)}$.

            # Compute update of the weights:
            phi_Np1 = self.compute_updated_phi(y_i, gamma_Np1)

            # Compute update of a given centroid:
            y_Np1 = self.compute_updated_centroid(y_i, gamma_Np1)

            # Compute update of the dispersion parameter eta:
            sigma_Np1 = self.compute_updated_sigma(y_i, gamma_Np1)

            self.phi = phi_Np1
            for k in range(self.S):
                self.B_params[k] = [y_Np1[k], sigma_Np1[k]]
#                self.B_params[k] = [y_Np1[k], 1]

    def update_hat_A(self):
        """Update estimate for A using convex opt."""

        C_hat = np.empty([self.max_lag+1, self.S, self.S])

        np.fill_diagonal(C_hat[0, :, :], self.phi)
        Id = np.identity(self.S)
        
        for t in range(self.max_lag):
            curr_tau = t+1 # While C is indexed by curr_tau, H is indexed
                           #  by t. May need to change since confusing.

            # Define and solve a convex problem:
            X = cp.Variable((self.S, self.S))
            cost = cp.norm(self.H_hat[t,: ,:] - self.K_hat.T @ C_hat[curr_tau-1, :, :] @ X @ self.K_hat, p='fro')**2
            constraints = [X >= 0, X@Id == Id]
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve()

            # The cvxpy package is unstable, so given the same input,
            # sometimes X.value is a proper matrix and sometimes X.value
            # is None, in which case there will be an error:
            #print(X.value)
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

    def compute_norm_factor_second_derivative(self, sigma, num_samples=10000):
        return -1
        raise NotImplementedError #TODO

    def compute_norm_factor_first_derivative(self, sigma, num_samples=10000):
        return 0 
        raise NotImplementedError #TODO

    def compute_norm_factor(self, sigma, num_samples=300): 
        if self.p == 1:
            return sigma*((2*math.pi)**1/2)
        elif self.p == 2:
            return ((2*math.pi)**(3/2)) * (sigma**2) * (math.exp(sigma**2/4)) * math.erf(sigma/2)
        elif self.p > 2:
            # Use Monte Carlo simulation to estimate the integral found in (24b) of Said et al..
            # Can optimize by caching these these values.
            int_est = 0
            mean = np.zeros(self.p)
            cov = np.eye(self.p)*(sigma**2)
            # Create MC samples that have shape (num_samples, self.p):
            r = np.random.multivariate_normal(mean, cov, num_samples)
            int_samples = np.zeros(num_samples)
            norm_const = ((2*math.pi)**self.p/2)*(sigma**self.p)
            for i in range(num_samples):
                curr_r = r[i]
                curr_int_est = 1
                for j in range(self.p):
                    for k in range(i+1, self.p):
                        curr_int_est *= math.sinh(abs(curr_r[j] - curr_r[k])/2)

                int_samples[i] = curr_int_est*norm_const
            
            ret = np.mean(int_samples)

            # Finally, multiply the estimate by the constant expressed in (24b) of Said et al.:
            Gamma_m = math.exp(scipy.special.multigammaln(self.p/2, self.p))
            omega_m = (2**self.p)*(math.pi**(.5*(self.p**2)))/Gamma_m
            const = (math.factorial(self.p)*(2**self.p))**-1
            const *= omega_m
            const *= 8**(self.p*(self.p-1)*.25)
            return const*ret
        else:
            raise RuntimeError('self.p is not a positive integer.')

    def compute_dist(self, X, Y):
        """Compute affine-invariant Riemannian distance."""
        dist = scipy.linalg.logm(np.matmul(np.linalg.inv(X), Y))
        dist = np.linalg.norm(dist, ord='fro')
        return dist

    def compute_B(self, y, i):
        """Compute the conditional density B(y | i)."""
        mean = self.B_params[i][0]
        sigma = self.B_params[i][1]

        norm_factor = self.compute_norm_factor(sigma)

        dist = self.compute_dist(y, mean)

        return math.exp(-(dist**2)/(2*(sigma**2)))/norm_factor

    def update_hat_K(self, numSamples=100):
        """Update current estimate for hat K.
        
        Currently using Standard Monte Carlo, but may derive close form of
        the integral in specific cases.
        """
        for j in range(self.S):
            mean = self.B_params[j][0]
            sigma = self.B_params[j][1]
    
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

        # In the case when self.N < self.S, we want to
        # initialize the centroids and leave the other parameters intact:
        if self.N < self.S:
            offset = self.N
            while self.N < self.S:
                curr_y = y[self.N - offset]
                self.B_params[self.N][0] = curr_y
                self.obs_cache.append(curr_y)
                self.N += 1
            self.obs_cache = self.obs_cache[-self.max_lag:]
            y = y[self.N - offset:]
            if len(y) == 0:
                return

        # Update B hat and phi hat.
        self.update_phi_B(y)

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


def main():
    m = SPDGaussianHMM()
    a = np.eye(2)*2
    y = randSPDGauss.randSPDGauss(a, 1, 100)
    obs = [y[:,:,i] for i in range(y.shape[2])]
    m.partial_fit(obs)

if __name__ == "__main__":
    main()
