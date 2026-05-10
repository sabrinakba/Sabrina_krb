"""Training loops: baseline + FairDis-VAE with toggleable components."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .losses import (kl_normal, CLUBMI, supervised_contrastive_loss,
                     counterfactual_swap_loss, exposure_loss)


def _bce_weighted(pred, target, weight=None):
    eps = 1e-7
    pred = pred.clamp(eps, 1 - eps)
    loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def train_baseline(model, train_loader, val_loader, cfg):
    opt = optim.Adam(model.parameters(), lr=cfg['lr_gen'])
    sched = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    mse = nn.MSELoss()
    best = float('inf')
    print("\n--- Training Baseline VAE ---")
    for ep in range(cfg['epochs']):
        model.train(); tot = 0
        for u, i, r, _, _, w in train_loader:
            u, i, r = u.to(cfg['device']), i.to(cfg['device']), r.to(cfg['device'])
            opt.zero_grad()
            out = model(u, i)
            loss = mse(out['r'], r) + cfg['w_kl_u'] * kl_normal(out['u_mu'], out['u_lv']) \
                 + cfg['w_kl_i'] * kl_normal(out['i_mu'], out['i_lv'])
            loss.backward(); opt.step(); tot += loss.item()
        model.eval(); vtot = 0
        with torch.no_grad():
            for u, i, r, _, _, _ in val_loader:
                u, i, r = u.to(cfg['device']), i.to(cfg['device']), r.to(cfg['device'])
                out = model(u, i)
                vtot += (mse(out['r'], r)
                         + cfg['w_kl_u'] * kl_normal(out['u_mu'], out['u_lv'])
                         + cfg['w_kl_i'] * kl_normal(out['i_mu'], out['i_lv'])).item()
        sched.step()
        avg_v = vtot / max(1, len(val_loader))
        print(f"  Ep {ep+1}/{cfg['epochs']} Train {tot/len(train_loader):.4f} Val {avg_v:.4f}")
        if avg_v < best:
            best = avg_v
            torch.save(model.state_dict(), cfg.get('ckpt_path', 'baseline_best.pt'))


def train_fairdis(model, train_loader, val_loader, cfg, n_items,
                   i_attrs, components):
    """Train FairDis-VAE with all enabled components."""
    device = cfg['device']
    i_attrs_t = torch.from_numpy(i_attrs).to(device)

    gen_params = []
    disc_params = []
    for name, p in model.named_parameters():
        if 'advs' in name:
            disc_params.append(p)
        else:
            gen_params.append(p)

    opt_gen = optim.Adam(gen_params, lr=cfg['lr_gen'])
    opt_disc = (optim.Adam(disc_params, lr=cfg['lr_disc'])
                if components['adversarial'] and disc_params else None)
    sched_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=5, gamma=0.7)

    club_u = club_i = opt_club = None
    if components['mi_penalty']:
        club_u = CLUBMI(model.task_dim, model.sens_dim).to(device)
        club_i = CLUBMI(model.task_dim, model.sens_dim).to(device)
        opt_club = optim.Adam(list(club_u.parameters()) + list(club_i.parameters()),
                              lr=cfg['lr_club'])

    mse = nn.MSELoss(reduction='none')
    best = float('inf')

    print(f"\n--- Training FairDis-VAE [{cfg.get('tag', 'run')}] ---")
    for ep in range(cfg['epochs']):
        progress = min(1.0, (ep + 1) / max(1, cfg['gamma_warmup_epochs']))
        grl_lambda = cfg['gamma_max'] * progress

        model.train()
        tot_g = tot_d = tot_c = 0.0
        nb = 0

        for batch in train_loader:
            u, i, r, au, ai, iw = [t.to(device) for t in batch]

            # --- Discriminator step(s) ---
            if components['adversarial'] and opt_disc is not None:
                for _ in range(cfg['disc_steps']):
                    out = model(u, i, grl_lambda=grl_lambda)
                    l_d = 0.0
                    for p in out['au_advs_d']:
                        l_d = l_d + _bce_weighted(p, au)
                    for p in out['ai_advs_d']:
                        l_d = l_d + _bce_weighted(p, ai)
                    opt_disc.zero_grad(); l_d.backward(); opt_disc.step()
                    tot_d += float(l_d.detach())

            # --- CLUB variational update ---
            if components['mi_penalty']:
                for _ in range(cfg['club_steps']):
                    out = model(u, i, grl_lambda=grl_lambda)
                    lc = (club_u.learning_loss(out['zu_t'].detach(), out['zu_s'].detach())
                          + club_i.learning_loss(out['zi_t'].detach(), out['zi_s'].detach()))
                    opt_club.zero_grad(); lc.backward(); opt_club.step()
                    tot_c += float(lc.detach())

            # --- Generator step ---
            out = model(u, i, grl_lambda=grl_lambda)
            loss = cfg['w_rec'] * (mse(out['r'], r) * iw).mean() if components['propensity_weight'] \
                   else cfg['w_rec'] * mse(out['r'], r).mean()
            loss = loss + cfg['w_kl_u'] * kl_normal(out['u_mu_t'], out['u_lv_t'])
            loss = loss + cfg['w_kl_u'] * kl_normal(out['u_mu_s'], out['u_lv_s'])
            loss = loss + cfg['w_kl_i'] * kl_normal(out['i_mu_t'], out['i_lv_t'])
            loss = loss + cfg['w_kl_i'] * kl_normal(out['i_mu_s'], out['i_lv_s'])

            if components['attribute_supervision']:
                loss = loss + cfg['w_attr_u'] * _bce_weighted(out['au_pred'], au)
                loss = loss + cfg['w_attr_i'] * _bce_weighted(out['ai_pred'], ai)

            if components['adversarial']:
                # GRL flips encoder gradient; adversary heads in this pass are
                # frozen via small lr or we rely on opt_gen excluding their params.
                for p in out['au_advs']:
                    loss = loss + _bce_weighted(p, au)
                for p in out['ai_advs']:
                    loss = loss + _bce_weighted(p, ai)

            if components['mi_penalty']:
                mi_u = club_u.mi_estimate(out['zu_t'], out['zu_s'])
                mi_i = club_i.mi_estimate(out['zi_t'], out['zi_s'])
                loss = loss + cfg['w_mi_u'] * mi_u + cfg['w_mi_i'] * mi_i

            if components['supcon']:
                loss = loss + cfg['w_supcon_u'] * supervised_contrastive_loss(
                    out['zu_s'], au, cfg['supcon_temp'])
                loss = loss + cfg['w_supcon_i'] * supervised_contrastive_loss(
                    out['zi_s'], ai, cfg['supcon_temp'])

            if components['counterfactual']:
                cf = counterfactual_swap_loss(model, u, i, au, ai,
                                              components['twin_encoder'],
                                              cfg['w_cf_u'], cfg['w_cf_i'])
                loss = loss + cf

            if components['exposure_loss']:
                # Sample negatives uniformly to build candidate sets
                C = 32
                neg = torch.randint(0, n_items, (u.size(0), C - 1), device=device)
                cand = torch.cat([i.unsqueeze(1), neg], 1)
                scores = model.predict_batch(u, cand)
                cand_attrs = i_attrs_t[cand]
                loss = loss + cfg['w_exposure'] * exposure_loss(
                    scores, cand_attrs, cfg['exposure_temp'])

            opt_gen.zero_grad(); loss.backward(); opt_gen.step()
            tot_g += float(loss.detach()); nb += 1

        # --- Validation ---
        model.eval(); v = 0; nv = 0
        with torch.no_grad():
            for batch in val_loader:
                u, i, r, au, ai, iw = [t.to(device) for t in batch]
                out = model(u, i, grl_lambda=grl_lambda)
                base = mse(out['r'], r).mean()
                if components['attribute_supervision']:
                    base = base + cfg['w_attr_u'] * _bce_weighted(out['au_pred'], au)
                    base = base + cfg['w_attr_i'] * _bce_weighted(out['ai_pred'], ai)
                v += float(base); nv += 1
        sched_gen.step()
        avg_v = v / max(1, nv)
        print(f"  Ep {ep+1}/{cfg['epochs']} | Gen {tot_g/max(1,nb):.4f} | "
              f"Disc {tot_d/max(1,nb*cfg['disc_steps']):.4f} | "
              f"CLUB {tot_c/max(1,nb*max(1,cfg['club_steps'])):.4f} | "
              f"Val {avg_v:.4f} | λ {grl_lambda:.2f}")

        if avg_v < best:
            best = avg_v
            torch.save(model.state_dict(), cfg.get('ckpt_path', 'fairdis_best.pt'))
