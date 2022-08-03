import numpy as np
def Quant_estimator(Rr, Ur):
    """
    Estimated Recall
    Rr = <p0, p1, ...>
    Ur = <p0, p1, ...>
    """
#     print(f'Rr={np.sum(Rr)}')
#     print(f'Ur={np.sum(Ur>0.5)}')
#     return np.sum(Rr) / (np.sum(Rr) + np.sum(Ur>0.5))
    return np.sum(Rr) / (np.sum(Rr) + np.sum(Ur))