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