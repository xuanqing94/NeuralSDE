"""
Implements Euler Scheme
"""
import math
import torch

def sdeint_euler(f, g, dif_g, t, h, x0, mid_state=None):
    """
    SDE integration from t0=0 to t1=t. Assume diagnoal noise.
    Args:
        f: drift function of (t, X), t is time and X is a d-dimensional vector.
           Outputs a d-dimensional vector
        g: diffusion function of (t, X), t is time and X is a d-dimensional vector.
           We assume G=g(t, X) is diagnoal matrix, so g(t, X) outputs the diagnoal
           vector (d-dimensional).
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

    z = torch.randn_like(x0).to(x0)
    for step in range(N):
        z.normal_()
        x = x + f(tt, x) * h_real + g(tt, x) * z.detach() * root_h
        tt += h
        if mid_state is not None:
            mid_state.append(x.detach().clone())
    return x

def odeint_euler(f, g, dif_g, t, h, x0):
    """
    SDE integration from t0=0 to t1=t. Assume diagnoal noise.
    Args:
        f: drift function of (t, X), t is time and X is a d-dimensional vector.
           Outputs a d-dimensional vector
        g: diffusion function of (t, X), t is time and X is a d-dimensional vector.
           We assume G=g(t, X) is diagnoal matrix, so g(t, X) outputs the diagnoal
           vector (d-dimensional).
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
        #z = torch.randn_like(x0).to(x0)
        x = x + f(tt, x) * h_real
        tt += h
    return x
