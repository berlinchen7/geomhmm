import logging

import numpy as np
from tqdm import tqdm

from geomhmm import _BaseGaussianHMM, SPDGaussianHMM


logger = logging.getLogger(__name__) 
logger.setLevel(logging.INFO)

# Comment out below if you don't want to see logging at this level 
# (you may still see logging at the subprocess level):
console_handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s %(asctime)s %(module)s] %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler) 


class _Base_StoEM_GaussianHMM(_BaseGaussianHMM):
    """ 
    Base class for the geometric HMM with Gaussian emission probabilities,
    where the Gaussian parameters are learned using the Stochastic EM algorithm
    described in 
    Stochastic EM Algorithm for Mixture Estimation on Manifolds 
    by Zanini et al., 2017, and the transition matrix is learned using
    Matila et al., 2020.
    """
    def __init__(self, 
                 max_lag=3, 
                 S=3, 
                 N=0, 
                 init_B_params=None,
                 rng=None,
                 ): 
        super().__init__(max_lag=max_lag, S=S, N=N, init_B_params=init_B_params, rng=rng)

    def update_phi_B(self, y):
        raise NotImplementedError


class _Base_Simple_GaussianHMM(_BaseGaussianHMM):
    """ 
    Base class for the geometric HMM with Gaussian emission probabilities,
    where the transition matrix is learned using simple counting of transitions
    between hidden states which are inferred via nearest neighbor.
    """
    def __init__(self, 
                 S=3, 
                 N=0, 
                 init_B_params=None,
                 rng=None,

                 curr_sum_A=None,
                 prev_state=None,
                 ): 
        super().__init__(max_lag=1, S=S, N=N, init_B_params=init_B_params, rng=rng)

        # Initializing the current sum of the transition probabilities:
        self.curr_sum_A = np.zeros((self.S, self.S)) if not curr_sum_A else curr_sum_A
        self.prev_state = prev_state # The most recent hidden state.
        self.curr_obs = [] # Cache the observations passed into "partial_fit"

    def update_hat_H(self, y):
        self.curr_obs = y
        logger.info('Skipped, since we are doing a simple count estimation of A.')

    def update_hat_K(self):
        logger.info('Skipped, since we are doing a simple count estimation of A.')

    def find_closest_hidden_state(self, yi, return_min_value=False):
        curr_min = float('inf')
        curr_min_state = -1
        for state, val in enumerate(self.B_params):
            curr_centroid = val[0]
            curr_dist = self.compute_dist(curr_centroid, yi)
            if curr_dist < curr_min:
                curr_min = curr_dist
                curr_min_state = state
        if return_min_value:
            return curr_min_state, curr_min
        else:
            return curr_min_state

    def update_hat_A(self):
        y = self.curr_obs.copy()
        if self.prev_state is not None:
            y.insert(0, self.prev_state) 
        states = []
        for yi in tqdm(y, desc="  Match obs to mean"):
            states.append(self.find_closest_hidden_state(yi))

        logger.info('Counting the transitions and updating estimation of A accordingly.')

        for i, state in enumerate(states[:-1]):
            next_state = states[i+1]
            self.curr_sum_A[state, next_state] += 1

        for i in range(self.S):
            if self.curr_sum_A[i, :].sum() == 0:
                self.A_hat[i, :] = 1/self.S
            else:
                self.A_hat[i, :] = self.curr_sum_A[i, :]/self.curr_sum_A[i, :].sum()

        # Reset the internal state for the next partial_fit call:
        self.prev_state = y[-1]
        self.curr_obs = []


class SPD_GD_GaussianHMM(SPDGaussianHMM):
    """ 
    Learner the SPD-valued HMM with Gaussian emission probabilities,
    where the transition matrix is learned using Matila et al., 2020,
    and the Gaussian parameters are learned using Riemannian gradient descent.

    NOTE: currently the gradient descent step uses the Matlab code developed by Salem Said
    and others. Running this requires a Matlab license and an installation of the Matlab engine for Python. See:
    https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    """
    def __init__(self, 
                 max_lag=1,
                 S=3, 
                 N=0, 
                 init_B_params=None,
                 rng=None,

                 p=2, 
                 alpha=.25, 
                 num_samples_K=10000,
                 num_samples_sigma=400,
                 num_samples_sigma_prime=400,
                 num_samples_sigma_prime_prime=400,

                 min_sigma=np.spacing(1),
                 num_omit_MCMC=10000,
                 ):
        SPDGaussianHMM.__init__(
            self, 
            max_lag=max_lag, 
            S=S, 
            N=N, 
            init_B_params=init_B_params,
            rng=rng,

            p=p, 
            alpha=alpha, 
            num_samples_K=num_samples_K,
            num_samples_sigma=num_samples_sigma,
            num_samples_sigma_prime=num_samples_sigma_prime,
            num_samples_sigma_prime_prime=num_samples_sigma_prime_prime,

            min_sigma=min_sigma,
            num_omit_MCMC=num_omit_MCMC,
            )
        import matlab.engine
        import matlab
        self.eng = matlab.engine.start_matlab()
        self.matlab = matlab
        _ = self.eng.addpath("CodeForMixRiemGauss/SGD_test_dimGenerale/") # Assign to _ o/w prints out the paths

    def update_phi_B(self, y):
        inp_y = np.zeros((self.p, self.p, len(y)))
        for i in range(len(y)):
            inp_y[:, :, i] = y[i]
        eta, Ybar, w_sqrt = self.eng.gd(self.matlab.double(inp_y.tolist()), self.matlab.double(self.p), self.matlab.double(self.S), nargout=3)
        eta, Ybar, w_sqrt = np.asarray(eta), np.asarray(Ybar), np.asarray(w_sqrt)
        for i in range(self.S):
            sigma = np.sqrt(-2*eta[i, 0])
            self.B_params[i][0], self.B_params[i][1] = Ybar[:, :, i], sigma

        self.phi = w_sqrt**2
        self.phi = self.phi.flatten()


class SPD_EM_GaussianHMM(SPDGaussianHMM):
    """ 
    Learner the SPD-valued HMM with Gaussian emission probabilities,
    where the transition matrix is learned using Matila et al., 2020,
    and the Gaussian parameters are learned using expectation maximization.

    NOTE: currently the EM step uses the Matlab code developed by Salem Said
    and others. Running this requires a Matlab license and an installation of the Matlab engine for Python. See:
    https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    """
    def __init__(self, 
                 max_lag=1,
                 S=3, 
                 N=0, 
                 init_B_params=None,
                 rng=None,

                 p=2, 
                 num_samples_K=10000,
                 num_omit_MCMC=10000,

                 EM_max_iter=1000,
                 ):
        SPDGaussianHMM.__init__(
            self, 
            max_lag=max_lag, 
            S=S, 
            N=N, 
            init_B_params=init_B_params,
            rng=rng,

            p=p, 
            num_samples_K=num_samples_K,
            num_omit_MCMC=num_omit_MCMC,
            )
        import matlab.engine
        import matlab
        self.eng = matlab.engine.start_matlab()
        self.matlab = matlab
        _ = self.eng.addpath("CodeForMixRiemGauss/Code_de_M.LB/generation/dimension_p/centre_de_masse/facteur_normalisation_complex/")
        _ = self.eng.addpath("CodeForMixRiemGauss/Code_de_M.LB/Riemannian_Fisher_Vector/code/general")
        _ = self.eng.addpath("CodeForMixRiemGauss/Code_de_M.LB/estimation/mixture/centre_de_masse/EM")

        self.EM_max_iter = EM_max_iter

        self.Zeta_tabule = self.eng.choix_Zeta("gaussien", "notcomplex", "spline")

    def update_phi_B(self, y):
        inp_y = np.zeros((self.p, self.p, len(y)))
        for i in range(len(y)):
            inp_y[:, :, i] = y[i]
        w, sigma, Ybar = self.eng.estimateur_EM(
            self.matlab.double(inp_y.tolist()), 
            self.matlab.double(self.S), 
            self.matlab.double(self.EM_max_iter), 
            self.Zeta_tabule, 
            "notcomplex", 
            nargout=3,
            )
        w, sigma, Ybar = np.asarray(w), np.asarray(sigma), np.asarray(Ybar)
        for i in range(self.S):
            self.B_params[i][0], self.B_params[i][1] = Ybar[:, :, i], sigma[0, i]

        self.phi = w.flatten()


class SPD_Zanini_Simple_GaussianHMM(_Base_Simple_GaussianHMM, SPDGaussianHMM):
    """ 
    Learner for the SPD-valued HMM with Gaussian emission probabilities,
    where the transition matrix is learned using simple counting of transitions
    between hidden states which are inferred via nearest neighbor,
    and the Gaussian parameters are learned using the method described in
    Zanini et al., 2017.
    """
    def __init__(self, 
                 S=3, 
                 N=0, 
                 init_B_params=None,
                 rng=None,

                 curr_sum_A=None,
                 prev_state=None,

                 p=2, 
                 alpha=.25, 
                 num_samples_sigma=400,
                 num_samples_sigma_prime=400,
                 num_samples_sigma_prime_prime=400,

                 min_sigma=np.spacing(1),
                 ):
        SPDGaussianHMM.__init__(
            self, 
            max_lag=1, 
            S=S, 
            N=N, 
            init_B_params=init_B_params,
            rng=rng,

            p=p, 
            alpha=alpha, 
            num_samples_K=0,
            num_samples_sigma=num_samples_sigma,
            num_samples_sigma_prime=num_samples_sigma_prime,
            num_samples_sigma_prime_prime=num_samples_sigma_prime_prime,
            min_sigma=min_sigma)
        _Base_Simple_GaussianHMM.__init__(self, S=S, N=N, init_B_params=init_B_params, rng=rng, curr_sum_A=curr_sum_A, prev_state=prev_state,)


class SPD_EM_Simple_GaussianHMM(SPD_Zanini_Simple_GaussianHMM):
    """ 
    Learner the SPD-valued HMM with Gaussian emission probabilities,
    where the transition matrix is learned using simple counting of transitions
    between hidden states which are inferred via nearest neighbor,
    and the Gaussian parameters are learned using expectation maximization.

    NOTE: 
    - Currently the EM step uses the Matlab code developed by Salem Said
    and others. Running this requires a Matlab license and an installation of the Matlab engine for Python. See:
    https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    - Out of sheer convenience, the class inherits SPD_Zanini_Simple_GaussianHMM; however, a more logically
    appropriate parent would be _BaseGaussianHMM.
    """
    def __init__(self, 
                 S=3, 
                 N=0, 
                 init_B_params=None,
                 rng=None,

                 curr_sum_A=None,
                 prev_state=None,

                 p=2, 

                 EM_max_iter=1000,
                 ):
        self.EM_learner = SPD_EM_GaussianHMM(
                 S=S, 
                 N=N, 
                 init_B_params=init_B_params,
                 rng=rng,

                 p=p,

                 EM_max_iter=EM_max_iter,
            )
        SPD_Zanini_Simple_GaussianHMM.__init__(
                self,
                S=S, 
                N=N, 
                init_B_params=init_B_params,
                rng=rng,

                curr_sum_A=curr_sum_A,
                prev_state=prev_state,

                p=p, 
            )

    def update_phi_B(self, y):
        self.EM_learner.update_phi_B(y)
        EM_B, EM_phi = self.EM_learner.B_params, self.EM_learner.phi
        self.B_params, self.phi = [[B_i[0].copy(), B_i[1]] for B_i in EM_B], EM_phi.copy()


