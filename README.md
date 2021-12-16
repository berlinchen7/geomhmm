# `geohmm`: Geometric Hidden Markov Models

This is an implementation of a trainer for the generalized Hidden Markov Models, where the observed random variables are manifold-valued (we call such model a *geometric hidden Markov model*).

The file `geomhmm.py` is the meat of the code and contains the implementation of the learner for the geometric hidden Markov model. Simply run `python geomhmm.py` to train the model on a simple example (after installing the necessary packages). WARNING: the implementation uses the `cvxpy` package, which is unstable (see [this post](https://stackoverflow.com/questions/59843953/receiving-none-as-result-in-cvxpy-problem-solver)), so sometimes `python geomhmm.py` may crash.

The file `randSPDGauss.py` contains the code to sample a Gaussian distribution of SPD manifolds. It is used by `geomhmm.py`.  One can run a simple test of this code by running `python randSPDGauss.py`.

The file `test.py` contains some examples. To evaluate the trainer on these examples, simply run `python test.py`. The same warning for `python geomhmm.py` applies here.

This code is still under development and is quite brittle. Any info/feedback is appreciated.
