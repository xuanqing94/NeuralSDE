"""
Implements Milstein Scheme
"""
import math
import torch

def sdeint_milstein(f, g, dif_g, t, h, x0):
    """
    SDE integration from t0=0 to t1=t. Assume diagnoal noise.
    Args:
        f: drift function of (t, X), t is time and X is a d-dimensional vector.
           Outputs a d-dimensional vector
        g: diffusion function of (t, X), t is time and X is a d-dimensional vector.
           We assume G=g(t, X) is diagnoal matrix, so g(t, X) outputs the diagnoal
           vector (d-dimensional).
        dif_g: gradient of g on X.
        t: final time.
        h: step size of discritization (the real step size might be slightly smaller).
        x0: initial value of X.
        d: dimension of X.
    Returns:
        y: d-dimensional vector, storing the integration result X(t).
    """
    N = int(t / h) + 1
    h_real = t / N
    root_h = math.sqrt(h_real)
    # for storing the noise
    tt = 0
    x = x0
    for step in range(N):
        z = torch.randn_like(x0).to(x0)
        wiener = z * root_h
        x = x + f(tt, x) * h_real + g(tt, x) * wiener \
                + 0.5 * dif_g(tt, x) * g(tt, x) * (wiener * wiener - h)
        tt += h
    return x
