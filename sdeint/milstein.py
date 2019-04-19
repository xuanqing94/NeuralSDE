"""
Implements Milstein Scheme
"""
import math
import torch

def sdeint_milstein(f, g, dif_g, t0, t1, h, x0):
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
    N = int(abs(t1-t0) / h) + 1
    h_real = (t1-t0) / N
    root_h = math.sqrt(abs(h_real))
    # for storing the noise
    tt = t0
    x = x0
    for step in range(N):
        z = torch.randn_like(x0).to(x0)
        wiener = z * root_h
        x = x + f(tt, x) * h_real + g(tt, x) * wiener \
                + 0.5 * dif_g(tt, x) * g(tt, x) * (wiener * wiener - abs(h_real))
        tt += h_real
    return x


def sdeint_joint_milstein(f, g, dif_g, t0, t1, h, x0):
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
    N = int(abs(t1-t0) / h) + 1
    h_real = (t1-t0) / N
    root_h = math.sqrt(abs(h_real))
    # for storing the noise
    tt = t0
    hidden_t, logp_t = x0
    z_hidden, z_logp = torch.randn_like(hidden_t).to(hidden_t), \
            torch.randn_like(logp_t).to(logp_t)
    for step in range(N):
        z_hidden.normal_()
        z_logp.normal_()
        wiener_hidden = z_hidden.detach() * root_h
        wiener_logp = z_logp.detach() * root_h
        drift_hidden, drift_logp = f(tt, (hidden_t, logp_t))
        diffusion_hidden, diffusion_logp = g(tt, (hidden_t, logp_t))
        grad_diffu_hidden, grad_diffu_logp = dif_g(tt, (hidden_t, logp_t))
        hidden_t = hidden_t + drift_hidden * h_real + diffusion_hidden * wiener_hidden \
                + 0.5 * grad_diffu_hidden * diffusion_hidden * (wiener_hidden * wiener_hidden - abs(h_real))
        logp_t = logp_t + drift_logp * h_real + diffusion_logp * wiener_logp \
                + 0.5 * grad_diffu_logp * diffusion_logp * (wiener_logp * wiener_logp - abs(h_real))
        tt += h_real
    return hidden_t, logp_t


