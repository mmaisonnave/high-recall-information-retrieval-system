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

def QuantCI_estimator(Rr, Ur):
    """
    Estimated Recall
    Rr = <p0, p1, ...>
    Ur = <p0, p1, ...>
    """
#     print(f'Rr={np.sum(Rr)}')
#     print(f'Ur={np.sum(Ur>0.5)}')
#     return np.sum(Rr) / (np.sum(Rr) + np.sum(Ur>0.5))
    var_dr = np.sum(Rr*(1-Rr))
    var_du = np.sum(Ur*(1-Ur))
    estimator =  np.sum(Rr) / (np.sum(Rr) + np.sum(Ur))
    Rrhat = np.sum(Rr)
    Urhat = np.sum(Ur)
    term1 = (1/(Rrhat+Urhat)**2)*var_dr
    term2 = ((Rrhat**2)/((Rrhat+Urhat)**4))*(var_dr + var_du )
    ci = 2*np.sqrt(term1+term2)
 
    return (estimator-ci, estimator+ci)