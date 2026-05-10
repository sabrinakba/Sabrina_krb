"""Plots: ablation bars, attacker AUC heatmap, Pareto frontiers, t-SNE."""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE


def plot_ablation_bars(summary_df, cfg):
    metrics = [('NDCG@K', 'Utility (NDCG@10) ↑'),
               ('UGF_NDCG', 'User-Group Fairness (UGF NDCG) ↓'),
               ('RSP', 'Item Exposure Parity (RSP) ↓'),
               ('REO', 'Item Equal Opportunity (REO) ↓'),
               ('AUCworst_user_task_mean', 'User Attack AUC (task subspace) ↓'),
               ('AUCworst_item_task_mean', 'Item Attack AUC (task subspace) ↓')]
    available = [(k, t) for k, t in metrics if any(
        (k + ('_mean' if not k.endswith('_mean') else '')) in summary_df.columns
        or k in summary_df.columns for k in [k])]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Ablation Study: FairDis-VAE Components', fontsize=15, fontweight='bold')
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, metrics):
        col = key if key in summary_df.columns else f'{key}_mean'
        col_std = col.replace('_mean', '_std') if col.endswith('_mean') else f'{key}_std'
        if col not in summary_df.columns:
            ax.axis('off'); continue
        vals = summary_df[col].values
        stds = summary_df[col_std].values if col_std in summary_df.columns else np.zeros_like(vals)
        names = summary_df['variant'].values
        ax.barh(range(len(vals)), vals, xerr=stds, capsize=4)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3); ax.invert_yaxis()
    plt.tight_layout()
    out = os.path.join(cfg['out_dir'], 'ablation_bars.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f"Saved: {out}")


def plot_attacker_heatmap(all_results, cfg):
    """Heatmap: rows = variants, cols = attackers; values = AUC on user_task."""
    variants = list(all_results.keys())
    attackers = ['LR', 'MLP', 'GBM', 'kNN']
    # Pick the most relevant rep per variant
    rep_for_user = lambda v: 'user_emb' if v == 'baseline_vae' else 'user_task'
    rep_for_item = lambda v: 'item_emb' if v == 'baseline_vae' else 'item_task'

    for who, rep_fn, fname in [('user', rep_for_user, 'attacker_heatmap_user.png'),
                                ('item', rep_for_item, 'attacker_heatmap_item.png')]:
        mat = np.zeros((len(variants), len(attackers)))
        for i, v in enumerate(variants):
            rep = rep_fn(v)
            seed_aucs = [[r['attack'][rep][a]['auc'] for a in attackers]
                          for r in all_results[v]]
            mat[i] = np.nanmean(seed_aucs, axis=0)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(variants))))
        im = ax.imshow(mat, cmap='RdYlGn_r', vmin=0.45, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(attackers))); ax.set_xticklabels(attackers)
        ax.set_yticks(range(len(variants))); ax.set_yticklabels(variants, fontsize=8)
        for i in range(len(variants)):
            for j in range(len(attackers)):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha='center', va='center',
                        fontsize=8, color='black')
        ax.set_title(f'{who.title()} Gender Attack AUC (lower = more private)',
                     fontweight='bold')
        plt.colorbar(im, ax=ax, label='AUC')
        plt.tight_layout()
        out = os.path.join(cfg['out_dir'], fname)
        plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
        print(f"Saved: {out}")


def plot_pareto(sweep_df, cfg, sweep_key='gamma_max',
                util_key='NDCG@K_mean',
                priv_key='AUCworst_user_task_mean',
                fair_key='RSP_mean'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Pareto Frontier ({sweep_key} sweep)', fontweight='bold')
    x_p = sweep_df[priv_key]; y_u = sweep_df[util_key]
    x_f = sweep_df[fair_key]; tags = sweep_df[sweep_key]
    axes[0].scatter(x_p, y_u, s=80)
    for t, x, y in zip(tags, x_p, y_u):
        axes[0].annotate(f"{sweep_key}={t}", (x, y), fontsize=8,
                         xytext=(5, 5), textcoords='offset points')
    axes[0].set_xlabel('User Attack AUC ↓ (privacy risk)')
    axes[0].set_ylabel('NDCG@10 ↑ (utility)')
    axes[0].set_title('Privacy–Utility frontier')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(x_f, y_u, s=80, color='darkorange')
    for t, x, y in zip(tags, x_f, y_u):
        axes[1].annotate(f"{sweep_key}={t}", (x, y), fontsize=8,
                         xytext=(5, 5), textcoords='offset points')
    axes[1].set_xlabel('RSP ↓ (item exposure disparity)')
    axes[1].set_ylabel('NDCG@10 ↑ (utility)')
    axes[1].set_title('Fairness–Utility frontier')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(cfg['out_dir'], f'pareto_{sweep_key}.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f"Saved: {out}")


def plot_tsne(baseline_model, fairdis_model, u_attrs, i_attrs, cfg,
              num_samples=2000):
    rng = np.random.default_rng(0)
    us = rng.choice(len(u_attrs), min(num_samples, len(u_attrs)), replace=False)
    is_ = rng.choice(len(i_attrs), min(num_samples, len(i_attrs)), replace=False)
    u_idx = torch.LongTensor(us).to(cfg['device'])
    i_idx = torch.LongTensor(is_).to(cfg['device'])

    Eu_b = baseline_model.get_user_embeddings(u_idx)
    Ei_b = baseline_model.get_item_embeddings(i_idx)
    Eu_f = fairdis_model.get_user_task(u_idx)
    Ei_f = fairdis_model.get_item_task(i_idx)

    tsne = TSNE(n_components=2, random_state=0, perplexity=30)
    points = [tsne.fit_transform(x) for x in (Eu_b, Eu_f, Ei_b, Ei_f)]
    labels = [u_attrs[us], u_attrs[us], i_attrs[is_], i_attrs[is_]]
    titles = ['User — Baseline', 'User — FairDis task subspace',
              'Item — Baseline', 'Item — FairDis task subspace']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, p, l, t in zip(axes.flatten(), points, labels, titles):
        sc = ax.scatter(p[:, 0], p[:, 1], c=l, cmap='coolwarm', alpha=0.6, s=8)
        ax.set_title(t, fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, label='Gender (0=M, 1=F)')
    plt.suptitle('Task subspace: gender mixing before/after debiasing',
                 fontweight='bold')
    plt.tight_layout()
    out = os.path.join(cfg['out_dir'], 'tsne_subspaces.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f"Saved: {out}")
