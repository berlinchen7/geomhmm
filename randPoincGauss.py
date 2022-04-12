import numpy as np
import math
import utils
import torch

def generate_r(sigma, num_samples, rng):
    ''' Generate the radius component of random samples.

    p(r) is proportional to exp(-r^2/(2*sigma^2))sinh(r), for r > 0.
    For more details, see, e.g., equation (24) of Said et al., 2017.
    '''
    k = 1 # df of the chi-square dist., which we use for the proposal dist
    pdf = lambda x: np.exp(-.5*np.square(x)/(sigma**2)) * np.sinh(x)
    proprnd = lambda x: rng.chisquare(k)
    proppdf = lambda x, y: x**(.5*k - 1)*np.exp(-.5*x) / (2**(.5*k)*math.gamma(.5*k))

    r, _ = utils.mhsample(1, num_samples, pdf, proppdf, proprnd, rng)
    return r


def randPoincGauss(Ybar, sigma, N, rng=None, omit=100):
    """ Generate N samples from a Poincare-disk-valued Gaussian with mean Ybar and dispersion gamma."""
    if rng is None:
        rng = np.random.default_rng()

    Ybar = np.array(Ybar) # Convert to numpy array in case Ybar is a torch tensor

    # Sample the polar coordinate representation of Poincare Disk centered at the origin:    
    r = generate_r(sigma, N, rng)   
    rho = (np.exp(r) - 1) / (np.exp(r) + 1) # Convert intrinsic radius to Poincare model representations   
    theta = 2*np.pi*rng.random(N)

    # Combine via the polar equation:
    z = rho*np.cos(theta) + 1j*rho*np.sin(theta)

    # Translate to mean Ybar:
    c = Ybar[0] + 1j*Ybar[1]
    ret = (z + c) / (1 + np.conjugate(c)*z)

    # Transform to a list of torch tensors
    ret = np.stack((ret.real, ret.imag), axis=-1)
    return [torch.tensor(x) for x in ret]


def plot(samples):
    import seaborn as sns
    import matplotlib.pyplot as plt

    ax = sns.scatterplot(x=samples[:,0], y=samples[:,1],marker='x', s=20)
    circle = plt.Circle(xy=(0, 0), radius=1, color='red', fill=False)
    ax.add_patch(circle)
    ax.set(xlim=(-1, 1))
    ax.set(ylim=(-1, 1));
    ax.set_aspect('equal')
    plt.show()

def tensor_list_to_numpy(tensor_list):
    ret = [np.array(t) for t in tensor_list]
    return np.array(ret)

def main():
    N = 10000

    centroid1 = np.array([0, 0])
    centroid2 = np.array([0.29, 0.82])
    centroid3 = np.array([-0.29, 0.82])

    disp1 = .1
    disp2 = .4
    disp3 = .4

    samples = randPoincGauss(centroid1, disp1, N)
    samples += randPoincGauss(centroid2, disp2, N)
    samples += randPoincGauss(centroid3, disp3, N)

    # samples = list(np.random.multivariate_normal(centroid1, np.eye(2)*disp1, N))
    # samples += list(np.random.multivariate_normal(centroid2, np.eye(2)*disp2, N))
    # samples += list(np.random.multivariate_normal(centroid3, np.eye(2)*disp3, N))

    plot(tensor_list_to_numpy(samples))

if __name__ == "__main__":
    main()
