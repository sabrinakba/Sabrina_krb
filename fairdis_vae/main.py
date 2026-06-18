"""Entry point. Runs:
  1. Full ablation matrix (10 variants × N seeds)
  2. Pareto sweep on gamma_max
  3. Generates all CSVs and plots

Usage:
  python -m fairdis_vae.main --smoke    # quick test on subsample
  python -m fairdis_vae.main             # full run
"""
import argparse
import json
import os

from .config import get_cfg, VARIANTS
from .data import load_and_process_data, split_data_per_user
from .experiments import run_ablation_matrix, run_pareto_sweep, run_variant
from .viz import plot_ablation_bars, plot_attacker_heatmap, plot_pareto, plot_tsne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true',
                    help='Subsample users/items for fast testing')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--out', type=str, default='results')
    ap.add_argument('--ablation_only', action='store_true')
    ap.add_argument('--pareto_only', action='store_true')
    ap.add_argument('--variants', nargs='+', default=None,
                    help='Subset of ablation variants to run')
    args = ap.parse_args()

    overrides = {'out_dir': args.out, 'smoke_test': args.smoke}
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.seeds is not None:
        overrides['seeds'] = args.seeds
    if args.smoke:
        overrides.setdefault('epochs', 5)
    cfg = get_cfg(overrides)

    print(f"Config: device={cfg['device']} seeds={cfg['seeds']} "
          f"epochs={cfg['epochs']} smoke={cfg['smoke_test']}")

    df, n_users, n_items, u_attrs, i_attrs, item_w = load_and_process_data(cfg)
    train_df, val_df, test_df = split_data_per_user(df, seed=42)

    print(f"\nDataset: {n_users:,} users, {n_items:,} items, "
          f"{len(df):,} interactions")
    print(f"User gender split: M={int((u_attrs==0).sum())} F={int((u_attrs==1).sum())}")
    print(f"Item gender split: M={int((i_attrs==0).sum())} F={int((i_attrs==1).sum())}")

    if not args.pareto_only:
        print("\n" + "#" * 70)
        print("# ABLATION MATRIX")
        print("#" * 70)
        all_results, summary = run_ablation_matrix(
            df, n_users, n_items, u_attrs, i_attrs, item_w,
            train_df, val_df, test_df, cfg, variant_names=args.variants)
        print("\nAblation summary:")
        print(summary.to_string(index=False))

        plot_ablation_bars(summary, cfg)
        plot_attacker_heatmap(all_results, cfg)

        # t-SNE on first-seed baseline vs fairdis_full
        if ('baseline_vae' in all_results and 'fairdis_full' in all_results):
            import torch
            from .models import BaselineVAE, FairDisVAE
            bm = BaselineVAE(n_users, n_items, cfg).to(cfg['device'])
            fm = FairDisVAE(n_users, n_items, cfg,
                            VARIANTS['fairdis_full']['components']).to(cfg['device'])
            bm.load_state_dict(torch.load(
                os.path.join(cfg['out_dir'],
                             f"ckpt_baseline_vae_s{cfg['seeds'][0]}.pt"),
                map_location=cfg['device']))
            fm.load_state_dict(torch.load(
                os.path.join(cfg['out_dir'],
                             f"ckpt_fairdis_full_s{cfg['seeds'][0]}.pt"),
                map_location=cfg['device']))
            plot_tsne(bm, fm, u_attrs, i_attrs, cfg)

    if not args.ablation_only:
        print("\n" + "#" * 70)
        print("# PARETO SWEEP (gamma_max)")
        print("#" * 70)
        sweep_df = run_pareto_sweep(
            df, n_users, n_items, u_attrs, i_attrs, item_w,
            train_df, val_df, test_df, cfg)
        print(sweep_df.to_string(index=False))
        plot_pareto(sweep_df, cfg)

    print("\nAll done. Outputs in:", cfg['out_dir'])


if __name__ == '__main__':
    main()
