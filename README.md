# `geomhmm`: Geometric Hidden Markov Models

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

This is an implementation of a learner for the generalized Hidden Markov Models, where the observed random variables are manifold-valued (we call such model a *geometric hidden Markov model*).

## Getting started

 Testing the code in your local environment is as simple as:
1. Clone this repository.
2. Navigate to the corresponding directory in the terminal.
3. Run `pip install -r requirements.txt` to install all dependencies.[^1]
4. Run `python func_tests.py` to do a smoke test of the code.

Of course, you may want to do the above in a virtual environment (see [here](https://realpython.com/python-virtual-environments-a-primer/) or [here](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) if you're not familiar with Python's virtual environment). We developed the code in Python 3.8.12, so we recommend running the code in a similar Python version.

Here is an example on how to use the code:

```python
from geomhmm import PoincareDiskGaussianHMM
from exp import gen_chain_Tupker2021

_, y, _ = gen_chain_Tupker2021() # Generate an example Poincare-disk-valued HMM
m = PoincareDiskGaussianHMM(S=3, max_lag=3, num_samples_K=500) # Initialize the learner
m.partial_fit(y) # Learning step
print(m.B_params, m.pi_inf_hat, m.P_hat) # Print the current estimates
```

## What's inside

* The file `geomhmm.py` is the meat of the code and contains the implementation of the learner itself. Currently we have `EuclideanGaussianHMM`, `PoincareDiskGaussianHMM`, and `SPDGaussianHMM`, which are learners for HMMs with observed values in Euclidean space / Poincare Disk / SPD matrices, respectively. The mixture estimation uses the approach outlined in Zanini et al., 2017, and the estimation for the transition matrix uses the method of moments algorithm which we adopted from Mattila et al., 2020. 
    * In addition, the file `extensions.py` contains the variants of the learner, such as `SPD_EM_GaussianHMM`, which uses the expectation-maximization (EM) algorithm for mixture estimation in place of the Zanini et al., 2017 approach.[^2] To run the code for most of the learners in this file (including `SPD_EM_GaussianHMM`), one will need a valid Matlab license in order to use the Matlab engine for Python (see [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) for more details).
    * More documentations on how to initialize/use these learners to come.
* The files `randSPDGauss.py`/`randPoincGauss.py` contain the code to sample a Gaussian distribution of SPD manifolds / Poincare Disk.
* The file `func_test.py` contains a suite of examples used to test the implementation.
* The file `exp.py` runs experiments that replicate the set-up used by Salem et al., 2021 and by Tupker et al., 2021.
    * For example, you can run `python exp.py --mode Salem2021 --oname output --opath ./out --seed=202 --givenTrue False` to replicate the set-up used by Salem et al., 2021 (the `givenTrue` flag controls whether the we want the learner to learn the emission probabilities/stationary distribution as well; `False` means we do want to learn those variables).
    * The directory `exp_config_templates` contains example config files used to run some of the experiments, such as hyperparameter tuning (`hyp_tuning`). More documentation about how to run each experiment to come.

## Acknowledgement
The algorithm was developed by Berlin Chen, Dr. Cyrus Mostajeran, and Dr. Salem Said. The development for the implementation is still ongoing. Any feedback is appreciated.

[^1]: The file `requirements-dev.txt` contains packages that are strictly not needed for the learners, but are useful if one were to further develop/experiment with our implementation. To install these packages, simply run `pip install -r requirements-dev.txt`.
[^2]:  The Matlab code for EM estimation and Riemannian gradient descent is currently proprietary and not publicly available. We are working to release a publicly available version.
