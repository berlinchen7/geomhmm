import numpy as np
import math
import utils
import torch


def generate_r(sigma, num_samples, rng):
    ''' Generate the radius component of random samples.

    For more details, see, e.g., equation (24) of Said et al., 2017.
    '''
    Z_sigma2 = (np.pi*.5)**.5 * sigma*np.exp(sigma**2 * .5)*math.erf(sigma/(2**.5))

    delta = sigma
    pdf = lambda x: (np.exp(-.5*np.square(x)/(sigma**2))/Z_sigma2) * np.sinh(x)
    proppdf = lambda x, y: utils.unifpdf(y - x, -delta, delta)
    proprnd = lambda x: x + rng.random()*2*delta - delta

    r, _ = utils.mhsample_is(1, num_samples, pdf, proppdf, proprnd, rng)
    return r

def generate_z(sigma, centroid, num_samples, rng):
    ''' Generate Gaussian samples on the hyperbolic upper half plane.
    Returns a numpy array of complex values.
    '''
    r = generate_r(sigma, num_samples, rng)
    theta = 2*np.pi*rng.random(num_samples)
    z = np.cos(theta*.5)*np.exp(r*.5)*1j + np.sin(theta*.5)*np.exp(-r*.5)
    z /= (-np.sin(theta*.5)*np.exp(r*.5)*1j + np.cos(theta*.5)*np.exp(-r*.5))

    return centroid[0] + centroid[1]*z

def randPoincGauss(Ybar, sigma, N, rng=None, omit=50):
    """ Generate N samples from a Poincare-disk-valued Gaussian with mean Ybar and dispersion gamma.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert the centroid to an element of the hyperplane:
    Ybar_hp = Ybar[0] + Ybar[1]*1j
    Ybar_hp = (-1j*Ybar_hp - 1j)/(Ybar_hp - 1)
    Ybar_hp = np.array([Ybar_hp.real, Ybar_hp.imag])

    # Generate N hyperbolic halfplane valued Gaussian samples:
    zs = generate_z(sigma, Ybar_hp, N + omit, rng)
    zs = zs[omit:]

    # Transform them into Poincare disk valued samples:
    zs = (zs - 1j) / (zs + 1j)

    # Transform to a list of torch tensors
    ret = np.stack((zs.real, zs.imag), axis=-1)
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
    centroid = np.array([0, 0])
    disp = .1
    samples = randPoincGauss(centroid, disp, N)
    plot(tensor_list_to_numpy(samples))

if __name__ == "__main__":
    main()
