"""All loss components: KL, CLUB-MI, supervised contrastive, counterfactual,
exposure, and helpers."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


def kl_normal(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class CLUBMI(nn.Module):
    """CLUB upper bound on I(X;Y); minimize wrt encoder."""
    def __init__(self, x_dim, y_dim, hidden=128):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(x_dim, hidden), nn.ReLU(),
                                nn.Linear(hidden, y_dim))
        self.logvar = nn.Sequential(nn.Linear(x_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, y_dim), nn.Tanh())

    def variational(self, x, y):
        mu, logvar = self.mu(x), self.logvar(x)
        return -((mu - y).pow(2) / (2 * logvar.exp()) + logvar / 2).sum(-1).mean()

    def mi_estimate(self, x, y):
        mu, logvar = self.mu(x), self.logvar(x)
        pos = -(mu - y).pow(2) / (2 * logvar.exp())
        # randomly permute y for negatives
        idx = torch.randperm(y.size(0), device=y.device)
        neg = -(mu - y[idx]).pow(2) / (2 * logvar.exp())
        return (pos.sum(-1).mean() - neg.sum(-1).mean()) / 2.0

    def learning_loss(self, x, y):
        return -self.variational(x, y)


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """SupCon (Khosla et al. 2020). Pulls same-label samples together,
    pushes different-label apart. Positive only on the sens subspace."""
    device = features.device
    features = F.normalize(features, dim=1)
    n = features.size(0)
    if n < 2:
        return torch.tensor(0.0, device=device)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    logits = features @ features.T / temperature
    logits_max, _ = logits.max(dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    diag = torch.eye(n, device=device)
    mask = mask * (1 - diag)
    exp_logits = torch.exp(logits) * (1 - diag)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    pos_count = mask.sum(1).clamp(min=1)
    mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count
    return -mean_log_prob_pos.mean()


def counterfactual_swap_loss(model, u_idx, i_idx, u_attrs_batch, i_attrs_batch,
                              twin_encoder, w_u=1.0, w_i=1.0):
    """Force task representation to be invariant under swapping z_s with that of
    a random other user (often opposite gender). If task_dim is gender-clean,
    swapping sens has no effect on task or rating prediction."""
    device = u_idx.device
    bsz = u_idx.size(0)

    if twin_encoder:
        u_t_mu, _, u_s_mu, _ = model.encode_user(u_idx)
        i_t_mu, _, i_s_mu, _ = model.encode_item(i_idx)
    else:
        u_mu, _ = model.encode_user(u_idx)
        i_mu, _ = model.encode_item(i_idx)
        u_t_mu, u_s_mu = torch.split(u_mu, [model.task_dim, model.sens_dim], 1)
        i_t_mu, i_s_mu = torch.split(i_mu, [model.task_dim, model.sens_dim], 1)

    perm_u = torch.randperm(bsz, device=device)
    perm_i = torch.randperm(bsz, device=device)

    r_orig = model.task_decoder(torch.cat([u_t_mu, i_t_mu], 1)).squeeze(-1)
    # Swap sens with that of permuted users/items; task should stay invariant
    r_swap = model.task_decoder(torch.cat([u_t_mu, i_t_mu], 1)).squeeze(-1)
    # Stronger: also pass through encoder again with swapped sens. Since encode
    # is just embedding lookup, the cleanest formulation is to require that
    # `task_decoder` output is identical when sens swap is applied by replacing
    # the *full* encoder output and re-running through downstream heads.
    # Here task_decoder takes only task_mu, so equality is trivial. To make CF
    # meaningful we instead require that the rating produced when concatenating
    # swapped sens into the *full* latent is unchanged, by perturbing task with
    # information that flows from sens via permutation.
    # Implementation: feed perturbed user task_mu = u_t_mu + small * (u_s_mu_perm - u_s_mu_orig)
    # projected through a learned bridge -- but simplest faithful version is
    # to require model.predict to be stable under input-level CF: re-encode
    # but inject a different sens embedding at the latent level via the
    # twin encoder. Since both are mu-only here, we add a noise perturbation
    # along sens direction and require task invariance.
    eps_u = (u_s_mu[perm_u] - u_s_mu).detach()
    eps_i = (i_s_mu[perm_i] - i_s_mu).detach()

    # Project sens-difference into task space via encoder bridge: simulate by
    # nudging task_mu by an *adversarial* projection. Here we use a fixed
    # random projection that the encoder cannot game.
    if not hasattr(model, '_cf_proj_u'):
        model._cf_proj_u = torch.randn(model.sens_dim, model.task_dim,
                                       device=device) * 0.1
        model._cf_proj_i = torch.randn(model.sens_dim, model.task_dim,
                                       device=device) * 0.1
    u_t_perturbed = u_t_mu + eps_u @ model._cf_proj_u
    i_t_perturbed = i_t_mu + eps_i @ model._cf_proj_i

    r_perturbed = model.task_decoder(
        torch.cat([u_t_perturbed, i_t_perturbed], 1)).squeeze(-1)
    return w_u * F.mse_loss(r_orig, r_perturbed) + w_i * 0.0


def exposure_loss(scores, item_attrs_batch, temperature=0.5):
    """Differentiable group-exposure parity via softmax over candidate scores.
    `scores` is (B, C) with C candidates per user; item_attrs_batch is (B, C)
    in {0,1}. We compute the probability mass placed on each gender group
    via softmax(scores/temperature) and penalize the |M-F| gap."""
    probs = F.softmax(scores / temperature, dim=-1)  # (B, C)
    p_f = (probs * item_attrs_batch).sum(-1)         # exposure to female
    p_m = (probs * (1 - item_attrs_batch)).sum(-1)   # exposure to male
    return (p_f - p_m).abs().mean()
