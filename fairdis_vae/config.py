"""Experiment configuration. All ablations are toggled via component flags."""
import os
import torch

BASE_CFG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seeds': [0, 1, 2],
    'smoke_test': False,
    'smoke_users': 5000,
    'smoke_items': 3000,

    'task_dim': 32,
    'sens_dim': 16,
    'embed_dim': 64,

    'lr_gen': 1e-3,
    'lr_disc': 1e-3,
    'lr_club': 1e-3,
    'batch_size': 4096,
    'epochs': 20,
    'top_k': 10,
    'num_workers': 2,

    # Loss weights
    'w_rec': 1.0,
    'w_kl_u': 1e-3,
    'w_kl_i': 1e-3,
    'w_attr_u': 0.5,
    'w_attr_i': 0.5,
    'w_mi_u': 0.1,
    'w_mi_i': 0.1,
    'w_supcon_u': 0.5,
    'w_supcon_i': 0.5,
    'w_cf_u': 1.0,
    'w_cf_i': 1.0,
    'w_exposure': 1.0,
    'gamma_max': 1.0,
    'gamma_warmup_epochs': 5,
    'disc_steps': 2,
    'club_steps': 1,
    'supcon_temp': 0.1,
    'exposure_temp': 0.5,
    'propensity_alpha': 0.5,

    # Filtering
    'min_interactions': 10,

    # Data
    'dataset_url': 'http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz',
    'extract_folder': 'lastfm-dataset-360K',
    'json_path': 'lfm-360-gender.json',
    'file_plays': 'usersha1-artmbid-artname-plays.tsv',
    'file_profiles': 'usersha1-profile.tsv',

    # Output
    'out_dir': 'results',
}

DEFAULT_COMPONENTS = {
    'twin_encoder': True,
    'mi_penalty': True,
    'multi_head_adv': True,
    'supcon': True,
    'counterfactual': True,
    'exposure_loss': True,
    'propensity_weight': True,
    'attribute_supervision': True,
    'adversarial': True,
}

# Variants for the ablation study.
# Each variant overrides components / hyperparams from the full method.
VARIANTS = {
    'baseline_vae': {
        'components': {k: False for k in DEFAULT_COMPONENTS},
        'description': 'Vanilla VAE, no fairness mechanism',
    },
    'orig_dsdvae': {
        'components': {**{k: False for k in DEFAULT_COMPONENTS},
                       'attribute_supervision': True, 'adversarial': True},
        'description': 'Original DS-DVAE (single encoder, BCE adversary, no MI/supcon/CF/exposure)',
        'overrides': {'twin_encoder': False},
    },
    'fairdis_full': {
        'components': dict(DEFAULT_COMPONENTS),
        'description': 'Proposed FairDis-VAE (all components)',
    },
    'fairdis_no_twin': {
        'components': {**DEFAULT_COMPONENTS, 'twin_encoder': False},
        'description': 'Ours w/o twin encoder (shared backbone + split)',
    },
    'fairdis_no_mi': {
        'components': {**DEFAULT_COMPONENTS, 'mi_penalty': False},
        'description': 'Ours w/o MI minimization',
    },
    'fairdis_no_multiadv': {
        'components': {**DEFAULT_COMPONENTS, 'multi_head_adv': False},
        'description': 'Ours w/ single linear adversary',
    },
    'fairdis_no_supcon': {
        'components': {**DEFAULT_COMPONENTS, 'supcon': False},
        'description': 'Ours w/o supervised contrastive on sens_dim',
    },
    'fairdis_no_cf': {
        'components': {**DEFAULT_COMPONENTS, 'counterfactual': False},
        'description': 'Ours w/o counterfactual swap regularizer',
    },
    'fairdis_no_exposure': {
        'components': {**DEFAULT_COMPONENTS, 'exposure_loss': False},
        'description': 'Ours w/o differentiable exposure loss',
    },
    'fairdis_no_propensity': {
        'components': {**DEFAULT_COMPONENTS, 'propensity_weight': False},
        'description': 'Ours w/o inverse-propensity weighting',
    },
}

PARETO_SWEEP = {
    'gamma_max': [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
}


def get_cfg(overrides=None):
    cfg = dict(BASE_CFG)
    if overrides:
        cfg.update(overrides)
    os.makedirs(cfg['out_dir'], exist_ok=True)
    return cfg
