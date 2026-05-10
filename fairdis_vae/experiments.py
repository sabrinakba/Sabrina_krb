"""Run a single variant for one seed; run ablation matrix; run Pareto sweep."""
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import VARIANTS, DEFAULT_COMPONENTS, PARETO_SWEEP
from .data import InteractionDataset
from .eval import evaluate_rec, run_attacks, summarize_attacks
from .models import BaselineVAE, FairDisVAE
from .train import train_baseline, train_fairdis


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _make_loaders(train_df, val_df, u_attrs, i_attrs, item_w, cfg):
    tr = InteractionDataset(train_df, u_attrs, i_attrs, item_w)
    vl = InteractionDataset(val_df, u_attrs, i_attrs, item_w)
    return (DataLoader(tr, batch_size=cfg['batch_size'], shuffle=True,
                       num_workers=cfg['num_workers'], pin_memory=True),
            DataLoader(vl, batch_size=cfg['batch_size'], shuffle=False,
                       num_workers=cfg['num_workers'], pin_memory=True))


def run_variant(variant_name, df, n_users, n_items, u_attrs, i_attrs,
                item_w, train_df, val_df, test_df, cfg, seed):
    set_seed(seed)
    variant = VARIANTS[variant_name]
    components = dict(variant['components'])
    if 'overrides' in variant:
        components.update(variant['overrides'])

    cfg = dict(cfg)
    cfg['tag'] = f"{variant_name}_s{seed}"
    ckpt = os.path.join(cfg['out_dir'], f"ckpt_{cfg['tag']}.pt")
    cfg['ckpt_path'] = ckpt

    train_loader, val_loader = _make_loaders(train_df, val_df, u_attrs, i_attrs,
                                              item_w, cfg)

    if variant_name == 'baseline_vae':
        model = BaselineVAE(n_users, n_items, cfg).to(cfg['device'])
        train_baseline(model, train_loader, val_loader, cfg)
        model.load_state_dict(torch.load(ckpt, map_location=cfg['device']))
        rec = evaluate_rec(model, test_df, df, n_items, u_attrs, i_attrs, cfg,
                           model_type='baseline')
        atk = run_attacks(model, n_users, n_items, u_attrs, i_attrs, cfg,
                          model_type='baseline')
    else:
        model = FairDisVAE(n_users, n_items, cfg, components).to(cfg['device'])
        train_fairdis(model, train_loader, val_loader, cfg, n_items, i_attrs,
                       components)
        model.load_state_dict(torch.load(ckpt, map_location=cfg['device']))
        rec = evaluate_rec(model, test_df, df, n_items, u_attrs, i_attrs, cfg)
        atk = run_attacks(model, n_users, n_items, u_attrs, i_attrs, cfg)

    return {'rec': rec, 'attack': atk, 'attack_summary': summarize_attacks(atk),
            'components': components, 'seed': seed, 'variant': variant_name}


def aggregate(results_list, variant_name):
    """Mean ± std across seeds for one variant."""
    keys = ['MAE', 'Precision@K', 'Recall@K', 'NDCG@K',
            'UGF_Precision', 'UGF_Recall', 'UGF_NDCG', 'RSP', 'REO']
    agg = {'variant': variant_name}
    for k in keys:
        vals = [r['rec'][k] for r in results_list]
        agg[f'{k}_mean'] = float(np.mean(vals))
        agg[f'{k}_std']  = float(np.std(vals))
    # Attack summary (worst-case AUC per representation)
    reps = list(results_list[0]['attack_summary'].keys())
    for rep in reps:
        vals = [r['attack_summary'][rep] for r in results_list]
        agg[f'AUCworst_{rep}_mean'] = float(np.nanmean(vals))
        agg[f'AUCworst_{rep}_std']  = float(np.nanstd(vals))
    # Per-attacker AUC on the most-relevant rep (user_task / item_task / user_emb)
    for rep in reps:
        for atk_name in ['LR', 'MLP', 'GBM', 'kNN']:
            vals = [r['attack'][rep][atk_name]['auc'] for r in results_list]
            agg[f'AUC_{rep}_{atk_name}_mean'] = float(np.nanmean(vals))
            agg[f'AUC_{rep}_{atk_name}_std']  = float(np.nanstd(vals))
    return agg


def run_ablation_matrix(df, n_users, n_items, u_attrs, i_attrs, item_w,
                        train_df, val_df, test_df, cfg, variant_names=None):
    variants = variant_names or list(VARIANTS.keys())
    all_results = {}
    rows = []
    for v in variants:
        print(f"\n{'='*60}\nVARIANT: {v}  ({VARIANTS[v]['description']})\n{'='*60}")
        seed_results = []
        for s in cfg['seeds']:
            print(f"\n--- {v} seed={s} ---")
            r = run_variant(v, df, n_users, n_items, u_attrs, i_attrs, item_w,
                            train_df, val_df, test_df, cfg, seed=s)
            seed_results.append(r)
        all_results[v] = seed_results
        rows.append(aggregate(seed_results, v))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(cfg['out_dir'], 'ablation_summary.csv'),
                   index=False)
    return all_results, summary


def run_pareto_sweep(df, n_users, n_items, u_attrs, i_attrs, item_w,
                     train_df, val_df, test_df, cfg,
                     variant='fairdis_full', sweep_key='gamma_max'):
    """Vary one hyperparam, fixing variant. Builds Pareto frontier."""
    rows = []
    for v_val in PARETO_SWEEP[sweep_key]:
        print(f"\n--- Pareto: {sweep_key}={v_val} ---")
        cfg_s = dict(cfg); cfg_s[sweep_key] = v_val
        seed_results = []
        for s in cfg_s['seeds']:
            r = run_variant(variant, df, n_users, n_items, u_attrs, i_attrs,
                            item_w, train_df, val_df, test_df, cfg_s, seed=s)
            seed_results.append(r)
        agg = aggregate(seed_results, f"{variant}_{sweep_key}={v_val}")
        agg[sweep_key] = v_val
        rows.append(agg)
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(os.path.join(cfg['out_dir'], f'pareto_{sweep_key}.csv'),
                    index=False)
    return sweep_df
