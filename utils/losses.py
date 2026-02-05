import torch
import torch.nn as nn


def wgan_discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """
    WGAN discriminator loss: E[D(fake)] - E[D(real)]
    Il discriminatore cerca di massimizzare D(real) e minimizzare D(fake).
    """
    return d_fake.mean() - d_real.mean()


def wgan_generator_loss(d_fake_static: torch.Tensor, d_fake_temporal: torch.Tensor) -> torch.Tensor:
    """
    WGAN generator loss: -E[D(fake)]
    Il generatore cerca di massimizzare D(fake), equivalente a minimizzare -D(fake).
    
    In DoppelGANger ci sono due discriminatori: sommiamo entrambi i contributi.
    """
    return -(d_fake_static.mean() + d_fake_temporal.mean())


def gradient_penalty(
    discriminator_fn,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: str,
    lambda_gp: float = 20.0,
) -> torch.Tensor:
    """
    WGAN-GP: penalizza se ||∇_x D(x)|| si allontana da 1 sui punti interpolati.
    
    Funziona sia per tensori 2D [B, D] (static) che 3D [B, T, D] (temporal).
    
    Args:
        discriminator_fn: funzione che prende x e restituisce score [B, 1]
        real: dati reali [B, D] o [B, T, D]
        fake: dati fake (stessa forma di real)
        device: "cuda" o "cpu"
        lambda_gp: peso della penalità (default 10.0)
    
    Returns:
        scalar penalty
    """
    batch_size = real.size(0)
    
    # epsilon per l'interpolazione: [B, 1] per 2D, [B, 1, 1] per 3D
    if real.dim() == 2:
        eps_shape = (batch_size, 1)
    elif real.dim() == 3:
        eps_shape = (batch_size, 1, 1)
    else:
        raise ValueError(f"gradient_penalty: real deve essere 2D o 3D, ricevuto {real.dim()}D")
    
    eps = torch.rand(eps_shape, device=device, dtype=real.dtype)
    
    # interpolazione
    interpolated = eps * real + (1 - eps) * fake
    interpolated.requires_grad_(True)
    
    # score del discriminatore sui punti interpolati
    d_interpolated = discriminator_fn(interpolated)
    
    # gradienti rispetto agli input interpolati
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        #retain_graph=True,
        only_inputs=True,
    )[0]
    
    # flatten per calcolare la norma: [B, D] → [B, D],  [B, T, D] → [B, T*D]
    gradients = gradients.reshape(batch_size, -1)
    
    # norma L2 per sample
    grad_norm = gradients.norm(2, dim=1)  # [B]
    
    # penalità: (||∇|| - 1)²
    penalty = ((grad_norm - 1.0) ** 2).mean()
    
    return penalty


def irreversibility_loss(
    irr_states: torch.Tensor,    # [B, T, n_irr]
    visit_mask: torch.Tensor,    # [B, T]
) -> torch.Tensor:
    """
    Penalizza transizioni 1 → 0 nelle variabili irreversibili.
    
    Una variabile irreversibile può solo rimanere 0 o passare da 0 a 1.
    La transizione 1 → 0 è vietata.
    
    Args:
        irr_states: stati binari [B, T, n_irr] in {0, 1}
        visit_mask: [B, T]
    
    Returns:
        scalar loss (media su batch e variabili)
    """
    if irr_states.size(-1) == 0:  # nessuna variabile irreversibile
        return torch.tensor(0.0, device=irr_states.device)
    
    B, T, n_irr = irr_states.shape
    
    # differenza tra step consecutivi: [B, T-1, n_irr]
    diff = irr_states[:, 1:, :] - irr_states[:, :-1, :]
    
    # transizione illegale: diff < 0  (da 1 a 0)
    violations = torch.relu(-diff)  # [B, T-1, n_irr], positivo solo se diff < 0
    
    # applica visit_mask: considera solo transizioni tra visite presenti
    transition_mask = visit_mask[:, :-1] * visit_mask[:, 1:]  # [B, T-1]
    transition_mask = transition_mask.unsqueeze(-1)           # [B, T-1, 1]
    
    violations = violations * transition_mask
    
    # media normalizzata
    loss = violations.sum() / (transition_mask.sum() * n_irr + 1e-8)
    
    return loss