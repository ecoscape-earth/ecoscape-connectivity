import math
import numpy as np

"""In this module, we define probability distributions used to sample the spread."""

def constant(value):
    """Returns a function that always returns the same value.
    This is useful to allow a deterministic number of spreads."""
    def f():
        return value
    return f


def half_cauchy(median, truncation):
    """
    A distribution that has been found useful in modeling animal dispersal is the
    half-Cauchy distribution; see e.g. Paradis, Emmanuel, Stephen R. Baillie, and William J. Sutherland.
    “Modeling Large-Scale Dispersal Distances.” Ecological Modelling 151, no. 2–3 (June 2002): 279–92.
    https://doi.org/10.1016/S0304-3800(01)00487-2.

    This function returns a function that, when called, returns a random sample from
    a truncated half-Cauchy probability distribution.
    The function takes as input the mean of the desired samples, and the truncation,
    corresponding to the largest integer that can be returned.
    To generate a sample, the function will sample a half-Cauchy distribution p.
    If x ~ p is the sample, the function will return round(x), that is, x rounded to the
    nearest integer.  We do not return the value 0, since for dispersal distances,
    0 is not a useful value for a simulation.

    Given the truncation, we obtain p by considering a half-Cauchy distribution truncated
    to the interval [0 + 0.5, truncation + 0.5], where the +0.5 is there to accommodate for the
    rounding.  We select the parameter sigma of the half-Cauchy distribution such that the
    median, after such truncation, is equal to the input median.
    """

    # define a cumulative distribution of a half-Cauchy with parameter sigma.
    def cdf(sigma, y):
        return 2 / math.pi * math.atan(y / sigma)

    r'''
    Solving for sigma using cdf formula knowing the median probability
    CDF of Half-Cauchy Distribution - {\mu is distribution shift}
    F(\mu, \sigma : y) = 2 / \pi * arctan( (y - \mu) / \sigma )  
    
    CDF of H-C at the truncation threshold
    \theta = 2/ \pi * arctan( (c - \mu) / \sigma )
    where c = truncation_threshold

    CDF of H-C at the median
    \theta / 2 = 2/ \pi * arctan( (m - \mu) / \sigma )
    where m = median

    Solve for \sigma:
    arctan( (c - \mu) / \sigma ) = 2 * arctan( (m - \mu) / \sigma )

    let x = (m - \mu) / \sigma
    2 * arctan( (m - \mu) / \sigma ) = 2 * arctan(x)
    => arctan(1 - x^2)

    arctan( (c - \mu) / \sigma ) = arctan(1 - x^2)
    (c - \mu) / \sigma = 1 - x^2
    (c - \mu) / \sigma = 1 - ( (m - \mu) / \sigma )^2

    Simplify
    \sigma = +/- ( (m - \mu) * sqrt(c - \mu) ) / sqrt( c + \mu - 2m)
    '''

    # Due to the way we round; see above.
    c = truncation + 0.5
    assert c > 2 * median # Otherwise, we cannot find the parameters.

    sigma = (median * c ** 0.5) / (c - 2 * median) ** 0.5
    # We now pre-compute the probabilities of each integer in the
    # range [0, truncation], extremes included.
    cdf_dif = []
    prev_cdf = 0
    for i in range(truncation + 1):
        cur_cdf = cdf(sigma, i + 0.5)
        cdf_dif.append(cur_cdf - prev_cdf)
        prev_cdf = cur_cdf
    # Drops the 0.
    cdf_dif = cdf_dif[1:]
    probs = np.array(cdf_dif) * (1 / np.sum(cdf_dif))

    def f():
        """This is the function that does the actual sampling."""
        return 1 + int(np.random.choice(range(truncation), p=probs))

    return f
