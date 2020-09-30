import torch
import torch.nn as nn

from sdeint.euler import sdeint_euler_with_jump

class IntegratedFlow(nn.Module):
    """This is essentially wrapping a flow function by an integration. This class
    works for both Neural ODE and Neural SDE, it also supports Bayesian Neural
    Networks (if self.flow_fn is Bayesian)."""
    def __init__(self, flow_fn, diffusion, jump, grid_size, T):
        super().__init__()
        self.flow_fn = flow_fn
        self.diffusion = diffusion
        self.jump = jump
        self.T = T
        self.grid_size = grid_size

    def forward(self, x):
        """We assume implicitly that if diffusion term is None, it is a NeuralODE
        so we do not run sdeint."""
        diffusion_fn = self.diffusion.diffusion if self.diffusion else None
        jump_fn = self.jump.jump if self.jump else None
        out = sdeint_euler_with_jump(
            self.flow_fn,
            diffusion_fn,
            jump_fn,
            self.T,
            self.grid_size,
            x,
        )
        return out
