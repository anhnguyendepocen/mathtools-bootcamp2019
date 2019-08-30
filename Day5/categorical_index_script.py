import numpy as np
import os
from scipy.io import loadmat

def compute_shift(category_boundary, unique):
    return np.where((np.histogram(category_boundary, unique))[0] == 1)[0][0] - 4

def get_index_pairs(category_boundary, unique):

    l_between = [(4,5), (4,6), ((3,5)), (4,7), (3,6), (2,5), (3,7), (2,6), (2,7)]
    l_between = np.array([np.array(p) for p in l_between])
    r_between = (l_between + 6) % 12
    l_within = (l_between + 3) % 12
    r_within = (l_between - 3) % 12

    between_pairs = np.concatenate((l_between, r_between))
    within_pairs = np.concatenate((l_within, r_within))

    shift = compute_shift(category_boundary, unique)
    between_pairs = (between_pairs + shift) % 12
    within_pairs = (within_pairs + shift) % 12

    return between_pairs, within_pairs

def compute_mean_abs_difference(rates, pairs):
    absolute_differences = []
    for p in pairs:
        absolute_differences.append(np.abs(rates[p[0]] - rates[p[1]]))
    return np.array(absolute_differences).mean()

def category_index(rates, category_bound, udirs):
    between_pairs, within_pairs = get_index_pairs(category_bound, udirs)
    BCD = compute_mean_abs_difference(rates, between_pairs)
    WCD = compute_mean_abs_difference(rates, within_pairs)
    return (BCD - WCD) / (BCD + WCD)

def load_neuron(filename):
    """
    loads neuron and add its name and unique directions

    Parameter:
    filname: str
        path to the neuron

    Return:
    neuron: dict
        dictionary containing the neural data
    """
    neuron = loadmat(filename)
    neuron['name'] = filename[-11:-4]
    neuron['udirs'] = np.unique(neuron['samp_direction_this_trial'].squeeze())
    return neuron

def compute_direction_spike_rate(neuron_dict, count_start, count_stop):
    """
    Parameter:
    neuron_dict: dict
        dictionary containing the neural data
    count_start: int
        time in ms when we start counting
    count_stop: int
        time in ms when we stop counting spikes

    Returns:
    direction_spike_rate: list
        firing rate for each direction
    """

    conversion = 1000 / (count_stop - count_start)
    unique_dirs = neuron_dict['udirs']
    direction_spike_rate = []
    for d in unique_dirs:
        trials_in_d = neuron_dict['samp_direction_this_trial'].squeeze() == d
        raster_slice = neuron_dict['trial_raster'][trials_in_d, count_start:count_stop]
        spike_count = np.sum(raster_slice, axis=1)
        mean_spike_count = np.mean(spike_count)
        mean_spike_rate = mean_spike_count * conversion
        direction_spike_rate.append(mean_spike_rate)

    return direction_spike_rate


def get_category_index(path, category_bound, count_start, count_stop):
    neuron = load_neuron(path)
    rates = compute_direction_spike_rate(neuron, count_start, count_stop)
    return category_index(rates, category_bound, neuron['udirs'])

def get_category_index_list(folder, category_bound, count_start, count_stop):
    names = os.listdir(folder)
    ci_dist = []
    for n in names:
        path = os.path.join(folder, n)
        ci_dist.append(get_category_index(path, category_bound, count_start, count_stop))
    return ci_dist
