
def has_duplicated(list_):
    visited=set()
    has_duplicated=False
    i = 0
    while not has_duplicated and i<len(list_):
        has_duplicated = list_[i] in visited
        visited.add(list_[i])
        i+=1
    return has_duplicated

import importlib
def module_exists(module_name):
    loader = importlib.util.find_spec(module_name)
    return loader is not None



from scipy.stats import t
import numpy as np

def _confidence_interval(x, confidence=0.95):
    m = x.mean()
    s = x.std()
    dof = len(x)-1

    t_crit = np.abs(t.ppf((1-confidence)/2, dof))

    return (m-s*t_crit/np.sqrt(len(x)), m+s*t_crit/np.sqrt(len(x)))

def confidence_interval_from_matrix(matrix, confidence=0.95):
    matrix = matrix.astype('float64')
    m = np.average(matrix, axis=0)
    s = np.std(matrix, axis=0)
    features_count=matrix.shape[1]
    dof = features_count-1

    t_crit = np.abs(t.ppf((1-confidence)/2, dof))

    return (m-s*t_crit/np.sqrt(features_count), m+s*t_crit/np.sqrt(features_count))