"""Models: BaselineVAE, FairDisVAE (proposed). The single configurable model
covers all ablations via component flags."""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .losses import grad_reverse


class BaselineVAE(nn.Module):
    def __init__(self, n_users, n_items, cfg):
        super().__init__()
        d = cfg['task_dim'] + cfg['sens_dim']
        self.task_dim = cfg['task_dim']
        self.sens_dim = cfg['sens_dim']
        self.u_emb = nn.Embedding(n_users, d)
        self.i_emb = nn.Embedding(n_items, d)
        self.u_mu = nn.Linear(d, d); self.u_lv = nn.Linear(d, d)
        self.i_mu = nn.Linear(d, d); self.i_lv = nn.Linear(d, d)
        self.task_decoder = nn.Sequential(
            nn.Linear(d * 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def reparam(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)

    def encode_user(self, u):
        h = self.u_emb(u); return self.u_mu(h), self.u_lv(h)

    def encode_item(self, i):
        h = self.i_emb(i); return self.i_mu(h), self.i_lv(h)

    def forward(self, u, i):
        u_mu, u_lv = self.encode_user(u)
        i_mu, i_lv = self.encode_item(i)
        z_u = self.reparam(u_mu, u_lv); z_i = self.reparam(i_mu, i_lv)
        r = self.task_decoder(torch.cat([z_u, z_i], 1)).squeeze(-1)
        return {'r': r, 'u_mu': u_mu, 'u_lv': u_lv, 'i_mu': i_mu, 'i_lv': i_lv}

    def predict(self, u, i):
        u_mu, _ = self.encode_user(u); i_mu, _ = self.encode_item(i)
        return self.task_decoder(torch.cat([u_mu, i_mu], 1)).squeeze(-1)

    def get_user_embeddings(self, u):
        with torch.no_grad():
            return self.encode_user(u)[0].cpu().numpy()

    def get_item_embeddings(self, i):
        with torch.no_grad():
            return self.encode_item(i)[0].cpu().numpy()


class _AdvHead(nn.Module):
    def __init__(self, in_dim, kind='linear'):
        super().__init__()
        if kind == 'linear':
            self.net = nn.Linear(in_dim, 1)
        else:
            self.net = nn.Sequential(
                spectral_norm(nn.Linear(in_dim, 64)), nn.ReLU(),
                spectral_norm(nn.Linear(64, 32)), nn.ReLU(),
                spectral_norm(nn.Linear(32, 1)))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


class FairDisVAE(nn.Module):
    """Configurable two-sided fair disentangled VAE.

    Components toggled by `comp` dict (set in cfg['components']):
      twin_encoder, multi_head_adv, attribute_supervision, adversarial.
    Other components (mi, supcon, counterfactual, exposure, propensity) are
    enforced in the training loop, not the architecture.
    """
    def __init__(self, n_users, n_items, cfg, components):
        super().__init__()
        self.task_dim = cfg['task_dim']
        self.sens_dim = cfg['sens_dim']
        self.comp = components
        d_t, d_s = self.task_dim, self.sens_dim
        d_full = d_t + d_s

        if components['twin_encoder']:
            self.u_emb_t = nn.Embedding(n_users, d_t)
            self.u_emb_s = nn.Embedding(n_users, d_s)
            self.i_emb_t = nn.Embedding(n_items, d_t)
            self.i_emb_s = nn.Embedding(n_items, d_s)
            self.u_mu_t = nn.Linear(d_t, d_t); self.u_lv_t = nn.Linear(d_t, d_t)
            self.u_mu_s = nn.Linear(d_s, d_s); self.u_lv_s = nn.Linear(d_s, d_s)
            self.i_mu_t = nn.Linear(d_t, d_t); self.i_lv_t = nn.Linear(d_t, d_t)
            self.i_mu_s = nn.Linear(d_s, d_s); self.i_lv_s = nn.Linear(d_s, d_s)
        else:
            self.u_emb = nn.Embedding(n_users, d_full)
            self.i_emb = nn.Embedding(n_items, d_full)
            self.u_mu = nn.Linear(d_full, d_full); self.u_lv = nn.Linear(d_full, d_full)
            self.i_mu = nn.Linear(d_full, d_full); self.i_lv = nn.Linear(d_full, d_full)

        self.task_decoder = nn.Sequential(
            nn.Linear(d_t * 2, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())

        if components['attribute_supervision']:
            self.u_sens_clf = nn.Sequential(
                nn.Linear(d_s, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid())
            self.i_sens_clf = nn.Sequential(
                nn.Linear(d_s, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid())

        if components['adversarial']:
            kinds = (['linear', 'mlp', 'mlp']
                     if components['multi_head_adv'] else ['linear'])
            self.u_advs = nn.ModuleList([_AdvHead(d_t, k) for k in kinds])
            self.i_advs = nn.ModuleList([_AdvHead(d_t, k) for k in kinds])

    def _reparam(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)

    def encode_user(self, u):
        if self.comp['twin_encoder']:
            ht = self.u_emb_t(u); hs = self.u_emb_s(u)
            return (self.u_mu_t(ht), self.u_lv_t(ht),
                    self.u_mu_s(hs), self.u_lv_s(hs))
        h = self.u_emb(u)
        return self.u_mu(h), self.u_lv(h)

    def encode_item(self, i):
        if self.comp['twin_encoder']:
            ht = self.i_emb_t(i); hs = self.i_emb_s(i)
            return (self.i_mu_t(ht), self.i_lv_t(ht),
                    self.i_mu_s(hs), self.i_lv_s(hs))
        h = self.i_emb(i)
        return self.i_mu(h), self.i_lv(h)

    def _split_sample(self, mu_or_tuple, lv_or_None=None):
        if self.comp['twin_encoder']:
            mu_t, lv_t, mu_s, lv_s = mu_or_tuple
            z_t = self._reparam(mu_t, lv_t); z_s = self._reparam(mu_s, lv_s)
            return mu_t, lv_t, mu_s, lv_s, z_t, z_s
        mu, lv = mu_or_tuple, lv_or_None
        z = self._reparam(mu, lv)
        mu_t, mu_s = torch.split(mu, [self.task_dim, self.sens_dim], 1)
        lv_t, lv_s = torch.split(lv, [self.task_dim, self.sens_dim], 1)
        z_t, z_s = torch.split(z, [self.task_dim, self.sens_dim], 1)
        return mu_t, lv_t, mu_s, lv_s, z_t, z_s

    def forward(self, u, i, grl_lambda=1.0):
        if self.comp['twin_encoder']:
            u_enc = self.encode_user(u); i_enc = self.encode_item(i)
            u_mu_t, u_lv_t, u_mu_s, u_lv_s, zu_t, zu_s = self._split_sample(u_enc)
            i_mu_t, i_lv_t, i_mu_s, i_lv_s, zi_t, zi_s = self._split_sample(i_enc)
        else:
            u_mu, u_lv = self.encode_user(u); i_mu, i_lv = self.encode_item(i)
            u_mu_t, u_lv_t, u_mu_s, u_lv_s, zu_t, zu_s = self._split_sample(u_mu, u_lv)
            i_mu_t, i_lv_t, i_mu_s, i_lv_s, zi_t, zi_s = self._split_sample(i_mu, i_lv)

        r = self.task_decoder(torch.cat([zu_t, zi_t], 1)).squeeze(-1)

        out = {'r': r,
               'u_mu_t': u_mu_t, 'u_lv_t': u_lv_t,
               'u_mu_s': u_mu_s, 'u_lv_s': u_lv_s,
               'i_mu_t': i_mu_t, 'i_lv_t': i_lv_t,
               'i_mu_s': i_mu_s, 'i_lv_s': i_lv_s,
               'zu_t': zu_t, 'zu_s': zu_s, 'zi_t': zi_t, 'zi_s': zi_s}

        if self.comp['attribute_supervision']:
            out['au_pred'] = self.u_sens_clf(zu_s).squeeze(-1)
            out['ai_pred'] = self.i_sens_clf(zi_s).squeeze(-1)

        if self.comp['adversarial']:
            zu_t_rev = grad_reverse(zu_t, grl_lambda)
            zi_t_rev = grad_reverse(zi_t, grl_lambda)
            out['au_advs'] = [h(zu_t_rev) for h in self.u_advs]
            out['ai_advs'] = [h(zi_t_rev) for h in self.i_advs]
            out['au_advs_d'] = [h(zu_t.detach()) for h in self.u_advs]
            out['ai_advs_d'] = [h(zi_t.detach()) for h in self.i_advs]

        return out

    def predict(self, u, i):
        with torch.no_grad():
            if self.comp['twin_encoder']:
                u_mu_t = self.u_mu_t(self.u_emb_t(u))
                i_mu_t = self.i_mu_t(self.i_emb_t(i))
            else:
                u_mu = self.u_mu(self.u_emb(u))
                i_mu = self.i_mu(self.i_emb(i))
                u_mu_t, _ = torch.split(u_mu, [self.task_dim, self.sens_dim], 1)
                i_mu_t, _ = torch.split(i_mu, [self.task_dim, self.sens_dim], 1)
            return self.task_decoder(torch.cat([u_mu_t, i_mu_t], 1)).squeeze(-1)

    def predict_batch(self, u_idx, cand_idx):
        """Score (B, C) for differentiable exposure loss (no torch.no_grad)."""
        if self.comp['twin_encoder']:
            u_mu_t = self.u_mu_t(self.u_emb_t(u_idx))
            i_mu_t = self.i_mu_t(self.i_emb_t(cand_idx))
        else:
            u_mu = self.u_mu(self.u_emb(u_idx))
            i_mu = self.i_mu(self.i_emb(cand_idx))
            u_mu_t, _ = torch.split(u_mu, [self.task_dim, self.sens_dim], 1)
            i_mu_t, _ = torch.split(i_mu, [self.task_dim, self.sens_dim], 1)
        B, C = u_mu_t.size(0), cand_idx.size(1)
        u_exp = u_mu_t.unsqueeze(1).expand(-1, C, -1).reshape(B * C, -1)
        i_flat = i_mu_t.reshape(B * C, -1)
        s = self.task_decoder(torch.cat([u_exp, i_flat], 1)).squeeze(-1)
        return s.view(B, C)

    def get_user_task(self, u):
        with torch.no_grad():
            if self.comp['twin_encoder']:
                return self.u_mu_t(self.u_emb_t(u)).cpu().numpy()
            mu = self.u_mu(self.u_emb(u))
            t, _ = torch.split(mu, [self.task_dim, self.sens_dim], 1)
            return t.cpu().numpy()

    def get_item_task(self, i):
        with torch.no_grad():
            if self.comp['twin_encoder']:
                return self.i_mu_t(self.i_emb_t(i)).cpu().numpy()
            mu = self.i_mu(self.i_emb(i))
            t, _ = torch.split(mu, [self.task_dim, self.sens_dim], 1)
            return t.cpu().numpy()

    def get_user_full(self, u):
        with torch.no_grad():
            if self.comp['twin_encoder']:
                t = self.u_mu_t(self.u_emb_t(u))
                s = self.u_mu_s(self.u_emb_s(u))
                return torch.cat([t, s], 1).cpu().numpy()
            return self.u_mu(self.u_emb(u)).cpu().numpy()

    def get_item_full(self, i):
        with torch.no_grad():
            if self.comp['twin_encoder']:
                t = self.i_mu_t(self.i_emb_t(i))
                s = self.i_mu_s(self.i_emb_s(i))
                return torch.cat([t, s], 1).cpu().numpy()
            return self.i_mu(self.i_emb(i)).cpu().numpy()
