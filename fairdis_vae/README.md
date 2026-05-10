# FairDis-VAE: Twin-encoder Disentangled VAE for Two-sided Gender Fairness

A research codebase for a novel disentangled recommender that mitigates gender
bias on **both** the user side and the item-provider side, without any
post-hoc re-ranking or off-the-shelf fairness module.

## Proposed method (FairDis-VAE)

Seven components, each individually ablatable via flags:

| # | Component | Where |
|---|-----------|-------|
| 1 | **Twin encoders** — structurally separated `(u_emb_task, u_emb_sens)` and `(i_emb_task, i_emb_sens)` with independent projection heads. No shared backbone. | `models.py` |
| 2 | **CLUB mutual-information minimization** between task and sens subspaces (variational MLP critic, separately optimized). | `losses.py::CLUBMI` |
| 3 | **Multi-head ensemble adversary** (1 linear + 2 spectral-norm MLPs) under a gradient-reversal layer, with k-step discriminator training. | `models.py::_AdvHead`, `losses.py::grad_reverse` |
| 4 | **Supervised contrastive loss** on `sens_dim` — pulls same-gender close, pushes opposite apart. Replaces the trivial BCE attribute head as the sens-localization signal. | `losses.py::supervised_contrastive_loss` |
| 5 | **Counterfactual swap regularizer** — perturbs task latents along the sens difference between random pairs and forces rating prediction to be invariant. | `losses.py::counterfactual_swap_loss` |
| 6 | **Differentiable group-exposure loss** — softmax over candidate scores yields a soft top-K, then penalizes \|p(female) - p(male)\| in exposure. | `losses.py::exposure_loss` |
| 7 | **Inverse-propensity weighted reconstruction** — weights MSE by 1/popularity^α so rare (often female-coded) items aren't penalized. | `data.py`, `train.py` |

## Files

```
fairdis_vae/
├── config.py          # Hyperparams, ablation variant matrix, Pareto sweep grid
├── data.py            # Last.fm-360K loader, k-core filter, propensity scores
├── losses.py          # KL, CLUB-MI, SupCon, CF-swap, exposure, GRL
├── models.py          # BaselineVAE, FairDisVAE (configurable)
├── train.py           # Training loops for both
├── attackers.py       # 4 inference attackers: LR, MLP, GBM, kNN
├── eval.py            # Rating/ranking/fairness metrics + attacks
├── experiments.py     # Run-variant, ablation matrix, Pareto sweep
├── viz.py             # Bar charts, AUC heatmap, Pareto, t-SNE
├── main.py            # Entry point
└── README.md
```

## Required input files

- `lastfm-dataset-360K.tar.gz` (auto-downloaded by `data.py`)
- `lfm-360-gender.json` (artist gender labels; place in the working directory)

## Usage

```bash
# Smoke test (5k users, 3k items, 5 epochs) -- fast sanity check
python -m fairdis_vae.main --smoke

# Full run: ablation matrix + Pareto sweep, 3 seeds
python -m fairdis_vae.main

# Just the ablation matrix
python -m fairdis_vae.main --ablation_only

# Just the Pareto sweep
python -m fairdis_vae.main --pareto_only

# Single variant, custom seeds
python -m fairdis_vae.main --variants fairdis_full --seeds 0 1 2 3 4
```

## Ablation matrix (10 variants)

| Variant | What it tests |
|---|---|
| `baseline_vae` | Pure recommendation, no fairness mechanism |
| `orig_dsdvae` | Single encoder + BCE adversary (the original published baseline) |
| `fairdis_full` | All 7 components on (proposed method) |
| `fairdis_no_twin` | Remove twin encoder (split a shared backbone instead) |
| `fairdis_no_mi` | Remove CLUB MI penalty |
| `fairdis_no_multiadv` | Single linear adversary instead of 3-head ensemble |
| `fairdis_no_supcon` | Remove supervised contrastive on sens_dim |
| `fairdis_no_cf` | Remove counterfactual swap regularizer |
| `fairdis_no_exposure` | Remove differentiable exposure loss |
| `fairdis_no_propensity` | Plain MSE reconstruction (no IPW) |

## Pareto sweep

`gamma_max ∈ {0.0, 0.1, 0.5, 1.0, 2.0, 5.0}` — produces:
- **Privacy–utility frontier**: NDCG@10 vs. worst-case attacker AUC
- **Fairness–utility frontier**: NDCG@10 vs. RSP

## Inference attackers (4)

All run on each representation (`user_task`, `item_task`, `user_full`, `item_full`):
- **LR** — Logistic Regression (linear baseline)
- **MLP** — 2-layer non-linear network
- **GBM** — Gradient-boosted trees (captures interactions)
- **kNN** — distance-based; catches local clustering the others miss

Privacy summary is **worst-case AUC** across attackers — a robust upper bound
on what any linear-or-non-linear attacker can recover.

## Methodological notes

- **Binary gender only.** The Last.fm-360K dataset labels `m/f` only. This is
  a known limitation; non-binary users are excluded at preprocessing and
  artist-gender labels in `lfm-360-gender.json` should be treated as noisy.
- **Multiple seeds.** Each variant is run with 3 seeds (configurable);
  results are reported as mean ± std.
- **Sanity check.** The inference attack is run on **both** the task
  subspace and the full latent. The full latent should show *high* AUC
  (the model did learn gender), while the task subspace should show low
  AUC (the disentanglement and adversary worked). If both are low,
  the model didn't learn gender at all and there's nothing to debias.

## Expected outputs

In `results/`:

- `ablation_summary.csv` — every variant × every metric (mean ± std)
- `pareto_gamma_max.csv` — sweep results
- `ablation_bars.png` — variant comparison across 6 key metrics
- `attacker_heatmap_user.png`, `attacker_heatmap_item.png` — AUC heatmaps
- `pareto_gamma_max.png` — privacy/fairness vs. utility frontiers
- `tsne_subspaces.png` — qualitative gender mixing
- `ckpt_{variant}_s{seed}.pt` — model checkpoints
