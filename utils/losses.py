import torch
import torch.nn as nn

def wgan_discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return -(d_real.mean() - d_fake.mean())


def wgan_generator_loss(d_fake_s: torch.Tensor, d_fake_t: torch.Tensor) -> torch.Tensor:
    return -(d_fake_s.mean() + d_fake_t.mean())


def gradient_penalty(critic_fn, real: torch.Tensor, fake: torch.Tensor, device: str) -> torch.Tensor:
    B   = real.shape[0]
    eps = torch.rand(B, *([1] * (real.dim() - 1)), device=device)
    interp   = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interp = critic_fn(interp)
    grads    = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True,
    )[0]
    grads = grads.reshape(B, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def irreversibility_loss(irr_states: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
    """Penalizza transizioni 1->0 nelle variabili irreversibili."""
    if visit_mask.dim() == 3:
        vm = visit_mask.squeeze(-1)
    else:
        vm = visit_mask
    if irr_states.dim() == 2:
        irr_states = irr_states.unsqueeze(-1)
    diffs  = irr_states[:, 1:, :] - irr_states[:, :-1, :]
    flips  = torch.relu(-diffs)
    vm_mid = (vm[:, 1:] * vm[:, :-1]).unsqueeze(-1)
    return (flips * vm_mid).sum() / (vm_mid.sum().clamp(min=1))