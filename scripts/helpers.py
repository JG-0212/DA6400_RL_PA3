import numpy as np


def moving_average(arr, n=100):
    """The function returns a rolling average of  scores over a window
    of size n
    """
    csum = np.cumsum(arr)
    csum[n:] = csum[n:] - csum[:-n]
    return csum[n - 1:] / n


def compute_decay(param_start, param_end, frac_episodes_to_decay, num_episodes, decay_type):
    """The function identifies the decay parameter to decay a parameter from 
    start to end in a fixed number of episodes.
    """
    if decay_type == 'linear':
        param_decay = ((param_start-param_end) /
                       (frac_episodes_to_decay*num_episodes))
    elif decay_type == 'exponential':
        param_decay = 10 ** (np.log10(param_end/param_start) /
                             (frac_episodes_to_decay*num_episodes))

    return param_decay
