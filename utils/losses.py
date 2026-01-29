#losses.py
import torch

def wgan_discriminator_loss(d_real, d_fake):
    return d_fake.mean() - d_real.mean()

def wgan_generator_loss(d_fake_static, d_fake_temporal):
    return -(d_fake_static.mean() + d_fake_temporal.mean())

def gradient_penalty(
    discriminator,
    real,
    fake,
    device,
    additional_inputs=None
):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    for _ in range(len(real.shape) - 2):
        alpha = alpha.unsqueeze(-1)

    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    if additional_inputs is None:
        out = discriminator(interpolated)
    else:
        out = discriminator(*additional_inputs, interpolated)

    gradients = torch.autograd.grad(
        outputs=out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def irreversibility_loss(states, mask):
    """
    states: [B,T] binari (0/1)
    mask:   [B,T]
    """
    diff = states[:, 1:] - states[:, :-1]
    viol = torch.clamp(-diff, min=0)   # solo 1→0
    return (viol * mask[:, 1:]).mean()
