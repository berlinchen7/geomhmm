# `geohmm`: Geometric Hidden Markov Models

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

This is an implementation of a learner for the generalized Hidden Markov Models, where the observed random variables are manifold-valued (we call such model a *geometric hidden Markov model*).

## Getting started

 Testing the code in your local environment is as simple as:
1. Clone this repository.
2. Navigate to the corresponding directory in the terminal.
3. Run `pip install -r requirements.txt` to install all dependencies.
4. Run `python func_tests.py` to do a smoke test of the code.

Of course, you may want to do the above in a virtual environment (see [here](https://realpython.com/python-virtual-environments-a-primer/) or [here](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) if you're not familiar with Python's virtual environment). We developed the code in Python 3.8.12, so we recommend running the code in a similar Python version.

Here is an example on how to use the code:

```python
from geomhmm import PoincareDiskGaussianHMM
from exp import gen_chain_Tupker2021
y, _ = gen_chain_Tupker2021() # Generate an example Poincare-disk-valued HMM
m = PoincareDiskGaussianHMM(S=3, max_lag=3, num_samples_K=500) # Initialize the learner
m.partial_fit(y) # Learning step
print(m.B_params, m.phi, m.A_hat) # Print the current estimates
```

## What's inside

* The file `geomhmm.py` is the meat of the code and contains the implementation of the learner itself. Currently we have `EuclideanGaussianHMM`, `PoincareDiskGaussianHMM`, and `SPDGaussianHMM`, which are learners for HMMs with observed values in Euclidean space / Poincare Disk / SPD matrices, respectively. More documentations on how to initialize/use these learners to come.
* The files `randSPDGauss.py`/`randPoincGauss.py` contains the code to sample a Gaussian distribution of SPD manifolds / Poincare Disk.
* The file `func_test.py` contains a suite of examples used to test the implementation.
* The file `exp.py` runs experiments that replicate the set-up used by Salem et al., 2021 and by Tupker et al., 2021.
    * For example, you can run `python exp.py --mode Salem2021 --givenTrue False` to replicate the set-up used by Salem et al., 2021 (the `givenTrue` flag controls whether the we want the learner to learn the emission probabilities/stationary distribution as well; `False` means we do want to learn those variables).

## Acknowledgement
The algorithm was developed by Berlin Chen and Dr. Cyrus Mostajeran at the University of Cambridge, and the code is still at the initial stage of development. Any feedback is appreciated.
