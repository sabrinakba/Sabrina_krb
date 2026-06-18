import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from collections import defaultdict
import json
import os
import requests
import tarfile
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Model Hyperparameters
    'latent_dim': 64,
    'task_dim': 32,
    'sens_dim': 32,
    'alpha_u': 0.5,          # Weight: User Attribute Loss (concentrates gender in sens_dim)
    'alpha_i': 0.5,          # Weight: Item Attribute Loss
    'beta_u': 1e-3,          # Small KL regularization (was 0.0 -> posterior unconstrained)
    'beta_i': 1e-3,          # Small KL regularization
    'gamma_max': 1.0,        # Max adversarial weight (ramped via GRL lambda)
    'gamma_warmup_epochs': 5,
    'disc_steps': 2,         # Disc updates per generator update
    'lr_gen': 0.001,
    'lr_disc': 0.001,
    'batch_size': 4096,
    'epochs': 20,
    'top_k': 10,

    # Filtering Constraints
    'min_interactions': 10,

    # Data Sources
    'dataset_url': 'http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz',
    'extract_folder': 'lastfm-dataset-360K',
    'json_path': 'lfm-360-gender.json',
    'file_plays': 'usersha1-artmbid-artname-plays.tsv',
    'file_profiles': 'usersha1-profile.tsv'
}

print(f"Running on device: {CONFIG['device']}")

# ==========================================
# 2. DATA DOWNLOADER
# ==========================================
def download_and_extract():
    tar_filename = "lastfm-dataset-360K.tar.gz"

    if os.path.exists(CONFIG['extract_folder']):
        print(f"[Info] Dataset directory found: {CONFIG['extract_folder']}")
        return

    if not os.path.exists(tar_filename):
        print(f"[Info] Downloading dataset from {CONFIG['dataset_url']}...")
        response = requests.get(CONFIG['dataset_url'], stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(tar_filename, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))

    print("[Info] Extracting tar.gz...")
    with tarfile.open(tar_filename, "r:gz") as tar:
        tar.extractall()
    print("[Info] Extraction complete.")

# ==========================================
# 3. PREPROCESSING
# ==========================================
def load_and_process_data():
    download_and_extract()

    path_plays = os.path.join(CONFIG['extract_folder'], CONFIG['file_plays'])
    path_profiles = os.path.join(CONFIG['extract_folder'], CONFIG['file_profiles'])

    print("Loading User Profiles...")
    profiles = pd.read_csv(path_profiles, sep='\t', header=None,
                           names=['user_sha1', 'gender', 'age', 'country', 'signup'],
                           usecols=['user_sha1', 'gender'])

    profiles = profiles[profiles['gender'].isin(['m', 'f'])]
    profiles['gender_bin'] = profiles['gender'].map({'m': 0, 'f': 1}).astype(int)
    valid_users_gender = dict(zip(profiles['user_sha1'], profiles['gender_bin']))

    print("Loading Artist Genders...")
    if not os.path.exists(CONFIG['json_path']):
        raise FileNotFoundError(f"Missing {CONFIG['json_path']}")

    with open(CONFIG['json_path'], 'r') as f:
        artist_json = json.load(f)

    valid_items_gender = {}
    for mbid, gender in artist_json.items():
        if gender.lower() == 'male':
            valid_items_gender[mbid] = 0
        elif gender.lower() == 'female':
            valid_items_gender[mbid] = 1

    print("Loading Plays...")
    df = pd.read_csv(path_plays, sep='\t', header=None,
                     names=['user_sha1', 'art_mbid', 'art_name', 'plays'],
                     usecols=['user_sha1', 'art_mbid', 'plays'],
                     on_bad_lines='skip')

    df = df.dropna(subset=['user_sha1', 'art_mbid'])

    print("Filtering by Gender info...")
    df = df[df['user_sha1'].isin(valid_users_gender.keys())]
    df = df[df['art_mbid'].isin(valid_items_gender.keys())]
    print(f"Interactions after gender filter: {len(df)}")

    print(f"Applying {CONFIG['min_interactions']}-core filter...")
    min_k = CONFIG['min_interactions']
    iteration = 0
    while True:
        start_len = len(df)
        u_counts = df['user_sha1'].value_counts()
        valid_u = u_counts[u_counts > min_k].index
        df = df[df['user_sha1'].isin(valid_u)]
        i_counts = df['art_mbid'].value_counts()
        valid_i = i_counts[i_counts > min_k].index
        df = df[df['art_mbid'].isin(valid_i)]
        end_len = len(df)
        print(f"  Iter {iteration}: {start_len} -> {end_len}")
        if start_len == end_len:
            break
        iteration += 1

    print("Mapping IDs...")
    user_list = df['user_sha1'].unique()
    item_list = df['art_mbid'].unique()
    user2idx = {u: i for i, u in enumerate(user_list)}
    item2idx = {i: j for j, i in enumerate(item_list)}
    df['u_idx'] = df['user_sha1'].map(user2idx)
    df['i_idx'] = df['art_mbid'].map(item2idx)

    u_attrs = np.zeros(len(user_list), dtype=np.float32)
    for u_sha1, idx in user2idx.items():
        u_attrs[idx] = valid_users_gender[u_sha1]

    i_attrs = np.zeros(len(item_list), dtype=np.float32)
    for i_mbid, idx in item2idx.items():
        i_attrs[idx] = valid_items_gender[i_mbid]

    df['plays'] = np.log1p(df['plays'])
    df['rating'] = df['plays'] / df['plays'].max()

    print(f"Final: {len(df)} interactions, {len(user_list)} users, {len(item_list)} items")
    return df, len(user_list), len(item_list), u_attrs, i_attrs

# ==========================================
# 4. DATA SPLITTING
# ==========================================
def split_data_per_user(df):
    print("\nSplitting data per user (70/10/20)...")
    train_list, val_list, test_list = [], [], []
    for user, group in df.groupby('u_idx'):
        user_data = group.sample(frac=1, random_state=42)
        n = len(user_data)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.8)
        train_list.append(user_data.iloc[:train_idx])
        val_list.append(user_data.iloc[train_idx:val_idx])
        test_list.append(user_data.iloc[val_idx:])
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

# ==========================================
# 5. BASELINE MODEL
# ==========================================
class BaselineVAE(nn.Module):
    def __init__(self, num_users, num_items, cfg):
        super().__init__()
        latent_dim = cfg['task_dim'] + cfg['sens_dim']
        self.u_emb = nn.Embedding(num_users, latent_dim)
        self.i_emb = nn.Embedding(num_items, latent_dim)
        self.u_mu = nn.Linear(latent_dim, latent_dim)
        self.u_lv = nn.Linear(latent_dim, latent_dim)
        self.i_mu = nn.Linear(latent_dim, latent_dim)
        self.i_lv = nn.Linear(latent_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, u_idx, i_idx):
        u_m, u_l = self.u_mu(self.u_emb(u_idx)), self.u_lv(self.u_emb(u_idx))
        i_m, i_l = self.i_mu(self.i_emb(i_idx)), self.i_lv(self.i_emb(i_idx))
        z_u = self.reparameterize(u_m, u_l)
        z_i = self.reparameterize(i_m, i_l)
        r_hat = self.decoder(torch.cat([z_u, z_i], dim=1)).squeeze()
        return r_hat, u_m, u_l, i_m, i_l

    def predict(self, u_idx, i_idx):
        u_m = self.u_mu(self.u_emb(u_idx))
        i_m = self.i_mu(self.i_emb(i_idx))
        return self.decoder(torch.cat([u_m, i_m], dim=1)).squeeze()

    def get_embeddings(self, u_idx, i_idx):
        with torch.no_grad():
            u_full = self.u_mu(self.u_emb(u_idx))
            i_full = self.i_mu(self.i_emb(i_idx))
        return u_full.cpu().numpy(), i_full.cpu().numpy()

# ==========================================
# 6. GRADIENT REVERSAL LAYER
# ==========================================
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

# ==========================================
# 7. DS-DVAE MODEL
# ==========================================
class DS_DVAE(nn.Module):
    def __init__(self, num_users, num_items, cfg):
        super().__init__()
        self.task_dim = cfg['task_dim']
        self.sens_dim = cfg['sens_dim']
        full_dim = self.task_dim + self.sens_dim

        self.u_emb = nn.Embedding(num_users, full_dim)
        self.i_emb = nn.Embedding(num_items, full_dim)
        self.u_mu = nn.Linear(full_dim, full_dim)
        self.u_lv = nn.Linear(full_dim, full_dim)
        self.i_mu = nn.Linear(full_dim, full_dim)
        self.i_lv = nn.Linear(full_dim, full_dim)

        self.task_decoder = nn.Sequential(
            nn.Linear(self.task_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.u_sens_decoder = nn.Sequential(
            nn.Linear(self.sens_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.i_sens_decoder = nn.Sequential(
            nn.Linear(self.sens_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        # Larger discriminators so they are a credible adversary
        self.u_adv = nn.Sequential(
            nn.Linear(self.task_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.i_adv = nn.Sequential(
            nn.Linear(self.task_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode(self, u_idx, i_idx):
        u_h = self.u_emb(u_idx)
        i_h = self.i_emb(i_idx)
        u_m, u_l = self.u_mu(u_h), self.u_lv(u_h)
        i_m, i_l = self.i_mu(i_h), self.i_lv(i_h)
        return u_m, u_l, i_m, i_l

    def forward(self, u_idx, i_idx, grl_lambda=1.0):
        u_m, u_l, i_m, i_l = self.encode(u_idx, i_idx)
        z_u = self.reparameterize(u_m, u_l)
        z_i = self.reparameterize(i_m, i_l)
        zu_t, zu_s = torch.split(z_u, [self.task_dim, self.sens_dim], dim=1)
        zi_t, zi_s = torch.split(z_i, [self.task_dim, self.sens_dim], dim=1)

        r_hat = self.task_decoder(torch.cat([zu_t, zi_t], dim=1)).squeeze()
        au_hat = self.u_sens_decoder(zu_s).squeeze()
        ai_hat = self.i_sens_decoder(zi_s).squeeze()

        # Gradient reversal: adv learns to predict gender,
        # encoder receives negated gradient -> learns to scrub gender from task_dim
        zu_t_rev = grad_reverse(zu_t, grl_lambda)
        zi_t_rev = grad_reverse(zi_t, grl_lambda)
        au_adv = self.u_adv(zu_t_rev).squeeze()
        ai_adv = self.i_adv(zi_t_rev).squeeze()
        return r_hat, au_hat, ai_hat, au_adv, ai_adv, u_m, u_l, i_m, i_l

    def predict(self, u_idx, i_idx):
        u_m = self.u_mu(self.u_emb(u_idx))
        i_m = self.i_mu(self.i_emb(i_idx))
        zu_t, _ = torch.split(u_m, [self.task_dim, self.sens_dim], dim=1)
        zi_t, _ = torch.split(i_m, [self.task_dim, self.sens_dim], dim=1)
        return self.task_decoder(torch.cat([zu_t, zi_t], dim=1)).squeeze()

    def get_task_embeddings(self, u_idx, i_idx):
        with torch.no_grad():
            u_m = self.u_mu(self.u_emb(u_idx))
            i_m = self.i_mu(self.i_emb(i_idx))
            zu_t, _ = torch.split(u_m, [self.task_dim, self.sens_dim], dim=1)
            zi_t, _ = torch.split(i_m, [self.task_dim, self.sens_dim], dim=1)
        return zu_t.cpu().numpy(), zi_t.cpu().numpy()

    def get_full_embeddings(self, u_idx, i_idx):
        """Sanity check: full latent including sens_dim."""
        with torch.no_grad():
            u_m = self.u_mu(self.u_emb(u_idx))
            i_m = self.i_mu(self.i_emb(i_idx))
        return u_m.cpu().numpy(), i_m.cpu().numpy()

# ==========================================
# 8. DATASET
# ==========================================
class InteractionDataset(Dataset):
    def __init__(self, df, u_attrs, i_attrs):
        self.users = torch.LongTensor(df['u_idx'].values)
        self.items = torch.LongTensor(df['i_idx'].values)
        self.ratings = torch.FloatTensor(df['rating'].values)
        self.u_attrs = torch.FloatTensor(u_attrs)
        self.i_attrs = torch.FloatTensor(i_attrs)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        i = self.items[idx]
        return u, i, self.ratings[idx], self.u_attrs[u], self.i_attrs[i]

# ==========================================
# 9. LOSSES
# ==========================================
def kl_loss(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

# ==========================================
# 10. TRAINING - BASELINE
# ==========================================
def train_baseline(model, train_loader, val_loader, cfg):
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr_gen'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    crit_mse = nn.MSELoss()
    best_loss = float('inf')

    print("\n--- Training Baseline VAE ---")
    for epoch in range(cfg['epochs']):
        model.train()
        total_train_loss = 0.0
        for u, i, r, _, _ in train_loader:
            u, i, r = u.to(cfg['device']), i.to(cfg['device']), r.to(cfg['device'])
            optimizer.zero_grad()
            r_pred, um, ul, im, il = model(u, i)
            l_rec = crit_mse(r_pred, r)
            l_kl = cfg['beta_u'] * kl_loss(um, ul) + cfg['beta_i'] * kl_loss(im, il)
            loss = l_rec + l_kl
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for u, i, r, _, _ in val_loader:
                u, i, r = u.to(cfg['device']), i.to(cfg['device']), r.to(cfg['device'])
                r_pred, um, ul, im, il = model(u, i)
                l_rec = crit_mse(r_pred, r)
                l_kl = cfg['beta_u'] * kl_loss(um, ul) + cfg['beta_i'] * kl_loss(im, il)
                total_val_loss += (l_rec + l_kl).item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        scheduler.step()
        print(f"Epoch {epoch+1}/{cfg['epochs']} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), 'baseline_best.pt')

# ==========================================
# 11. TRAINING - DS-DVAE (GRL + k-step discriminator)
# ==========================================
def quick_task_auc(model, u_attrs, i_attrs, cfg, sample=5000):
    """Cheap probe: linear AUC on task subspace. Lower = better debiasing."""
    model.eval()
    rng = np.random.default_rng(0)
    n_u = min(sample, len(u_attrs))
    n_i = min(sample, len(i_attrs))
    us = rng.choice(len(u_attrs), n_u, replace=False)
    is_ = rng.choice(len(i_attrs), n_i, replace=False)
    u_idx = torch.LongTensor(us).to(cfg['device'])
    i_idx = torch.LongTensor(is_).to(cfg['device'])
    u_t, i_t = model.get_task_embeddings(u_idx, i_idx)
    try:
        clf_u = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, solver='liblinear'))
        clf_u.fit(u_t, u_attrs[us])
        auc_u = roc_auc_score(u_attrs[us], clf_u.predict_proba(u_t)[:, 1])
    except Exception:
        auc_u = float('nan')
    try:
        clf_i = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, solver='liblinear'))
        clf_i.fit(i_t, i_attrs[is_])
        auc_i = roc_auc_score(i_attrs[is_], clf_i.predict_proba(i_t)[:, 1])
    except Exception:
        auc_i = float('nan')
    return auc_u, auc_i

def train_debiased(model, train_loader, val_loader, cfg, u_attrs, i_attrs):
    # Encoder + decoders + sens heads
    gen_params = (
        list(model.u_emb.parameters()) + list(model.i_emb.parameters()) +
        list(model.u_mu.parameters()) + list(model.u_lv.parameters()) +
        list(model.i_mu.parameters()) + list(model.i_lv.parameters()) +
        list(model.task_decoder.parameters()) +
        list(model.u_sens_decoder.parameters()) +
        list(model.i_sens_decoder.parameters())
    )
    # Discriminator gets a SEPARATE optimizer so we can do k disc steps
    disc_params = list(model.u_adv.parameters()) + list(model.i_adv.parameters())

    optimizer_gen = optim.Adam(gen_params, lr=cfg['lr_gen'])
    optimizer_disc = optim.Adam(disc_params, lr=cfg['lr_disc'])
    scheduler_gen = optim.lr_scheduler.StepLR(optimizer_gen, step_size=5, gamma=0.7)
    scheduler_disc = optim.lr_scheduler.StepLR(optimizer_disc, step_size=5, gamma=0.7)

    crit_mse = nn.MSELoss()
    crit_bce = nn.BCELoss()
    best_loss = float('inf')

    print("\n--- Training DS-DVAE (GRL adversarial) ---")
    for epoch in range(cfg['epochs']):
        # Ramp GRL lambda from 0 -> gamma_max
        progress = min(1.0, (epoch + 1) / max(1, cfg['gamma_warmup_epochs']))
        grl_lambda = cfg['gamma_max'] * progress

        model.train()
        total_gen = 0.0
        total_disc = 0.0
        n_batches = 0

        for u, i, r, au, ai in train_loader:
            u, i, r = u.to(cfg['device']), i.to(cfg['device']), r.to(cfg['device'])
            au, ai = au.to(cfg['device']), ai.to(cfg['device'])

            # --- k discriminator steps on detached features ---
            for _ in range(cfg['disc_steps']):
                with torch.no_grad():
                    u_m, u_l, i_m, i_l = model.encode(u, i)
                    z_u = model.reparameterize(u_m, u_l)
                    z_i = model.reparameterize(i_m, i_l)
                    zu_t, _ = torch.split(z_u, [model.task_dim, model.sens_dim], dim=1)
                    zi_t, _ = torch.split(z_i, [model.task_dim, model.sens_dim], dim=1)
                au_d = model.u_adv(zu_t).squeeze()
                ai_d = model.i_adv(zi_t).squeeze()
                l_disc = crit_bce(au_d, au) + crit_bce(ai_d, ai)
                optimizer_disc.zero_grad()
                l_disc.backward()
                optimizer_disc.step()
                total_disc += l_disc.item()

            # --- 1 generator step with GRL ---
            r_pred, au_pred, ai_pred, au_adv_g, ai_adv_g, um, ul, im, il = model(u, i, grl_lambda=grl_lambda)
            l_rec = crit_mse(r_pred, r)
            l_attr = cfg['alpha_u'] * crit_bce(au_pred, au) + cfg['alpha_i'] * crit_bce(ai_pred, ai)
            l_kl = cfg['beta_u'] * kl_loss(um, ul) + cfg['beta_i'] * kl_loss(im, il)
            # GRL flips encoder gradient automatically; just use true labels here
            l_adv = crit_bce(au_adv_g, au) + crit_bce(ai_adv_g, ai)
            loss_gen = l_rec + l_attr + l_kl + l_adv
            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()
            total_gen += loss_gen.item()
            n_batches += 1

        # --- Validation ---
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for u, i, r, au, ai in val_loader:
                u, i, r = u.to(cfg['device']), i.to(cfg['device']), r.to(cfg['device'])
                au, ai = au.to(cfg['device']), ai.to(cfg['device'])
                r_pred, au_pred, ai_pred, au_adv_g, ai_adv_g, um, ul, im, il = model(u, i, grl_lambda=grl_lambda)
                l_rec = crit_mse(r_pred, r)
                l_attr = cfg['alpha_u'] * crit_bce(au_pred, au) + cfg['alpha_i'] * crit_bce(ai_pred, ai)
                l_kl = cfg['beta_u'] * kl_loss(um, ul) + cfg['beta_i'] * kl_loss(im, il)
                l_adv = crit_bce(au_adv_g, au) + crit_bce(ai_adv_g, ai)
                total_val += (l_rec + l_attr + l_kl + l_adv).item()

        scheduler_gen.step()
        scheduler_disc.step()

        avg_gen = total_gen / n_batches
        avg_disc = total_disc / (n_batches * cfg['disc_steps'])
        avg_val = total_val / len(val_loader)

        # Probe: linear AUC on task subspace (this is the real debias signal)
        auc_u_probe, auc_i_probe = quick_task_auc(model, u_attrs, i_attrs, cfg)
        print(f"Epoch {epoch+1}/{cfg['epochs']} | Gen: {avg_gen:.4f} | Val: {avg_val:.4f} | "
              f"Disc: {avg_disc:.4f} | GRL λ: {grl_lambda:.2f} | "
              f"task-AUC u/i: {auc_u_probe:.3f}/{auc_i_probe:.3f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), 'dvae_best.pt')

# ==========================================
# 12. EVALUATION
# ==========================================
def dcg_at_k(relevance, k):
    relevance = np.asarray(relevance, dtype=np.float64)[:k]
    if relevance.size:
        return np.sum(relevance / np.log2(np.arange(2, relevance.size + 2)))
    return 0.0

def ndcg_at_k(relevance, k):
    dcg_max = dcg_at_k(sorted(relevance, reverse=True), k)
    return 0.0 if not dcg_max else dcg_at_k(relevance, k) / dcg_max

def evaluate_model(model, test_df, full_df, num_items, u_attrs, i_attrs, cfg):
    print("\n--- Evaluation ---")
    model.eval()

    u_t = torch.LongTensor(test_df['u_idx'].values).to(cfg['device'])
    i_t = torch.LongTensor(test_df['i_idx'].values).to(cfg['device'])
    r_t = torch.FloatTensor(test_df['rating'].values).to(cfg['device'])

    with torch.no_grad():
        preds = model.predict(u_t, i_t)
        mae = torch.mean(torch.abs(r_t - preds)).item()

    user_groups = defaultdict(list)
    all_user_items = defaultdict(set)
    for _, row in test_df.iterrows():
        user_groups[row['u_idx']].append(row['i_idx'])
    for _, row in full_df.iterrows():
        all_user_items[row['u_idx']].add(row['i_idx'])

    precisions, recalls, ndcgs = [], [], []
    u_precs = {0: [], 1: []}
    u_recs = {0: [], 1: []}
    u_ndcgs = {0: [], 1: []}
    exposure = {0: 0, 1: 0}
    rel_rec = {0: 0, 1: 0}
    rel_total = {0: 0, 1: 0}
    total_top_k_items = 0

    rng = np.random.default_rng(seed=42)

    with torch.no_grad():
        for u_id, pos_items in tqdm(user_groups.items(), desc="Ranking Eval"):
            neg_pool = list(set(range(num_items)) - all_user_items[u_id])
            if len(neg_pool) < 100:
                continue
            negs = rng.choice(neg_pool, 100, replace=False)
            candidates = np.concatenate([pos_items, negs])
            cand_t = torch.LongTensor(candidates).to(cfg['device'])
            user_t = torch.LongTensor([u_id] * len(candidates)).to(cfg['device'])
            scores = model.predict(user_t, cand_t).cpu().numpy()

            k = min(cfg['top_k'], len(candidates))
            top_k_idx = np.argsort(scores)[::-1][:k]
            top_k_items = candidates[top_k_idx]
            relevance = [1 if item in pos_items else 0 for item in top_k_items]

            hits = len(set(top_k_items) & set(pos_items))
            prec = hits / k
            rec = hits / len(pos_items) if pos_items else 0
            ndcg = ndcg_at_k(relevance, k)
            precisions.append(prec); recalls.append(rec); ndcgs.append(ndcg)

            u_g = int(u_attrs[u_id])
            u_precs[u_g].append(prec)
            u_recs[u_g].append(rec)
            u_ndcgs[u_g].append(ndcg)

            for itm in top_k_items:
                g_i = int(i_attrs[itm])
                exposure[g_i] += 1
                total_top_k_items += 1
            for itm in pos_items:
                g_i = int(i_attrs[itm])
                rel_total[g_i] += 1
                if itm in top_k_items:
                    rel_rec[g_i] += 1

    metrics = {
        'MAE': mae,
        'Precision@10': np.mean(precisions),
        'Recall@10': np.mean(recalls),
        'NDCG@10': np.mean(ndcgs),
        'UGF_Precision': abs(np.mean(u_precs[0]) - np.mean(u_precs[1])),
        'UGF_Recall': abs(np.mean(u_recs[0]) - np.mean(u_recs[1])),
        'UGF_NDCG': abs(np.mean(u_ndcgs[0]) - np.mean(u_ndcgs[1])),
        'RSP': abs((exposure[0] / total_top_k_items) - (exposure[1] / total_top_k_items)) if total_top_k_items > 0 else 0,
        'REO': abs((rel_rec[0] / (rel_total[0] + 1e-9)) - (rel_rec[1] / (rel_total[1] + 1e-9))),
        'user_metrics': {
            'male': {'precision': np.mean(u_precs[0]) if u_precs[0] else 0,
                     'recall': np.mean(u_recs[0]) if u_recs[0] else 0,
                     'ndcg': np.mean(u_ndcgs[0]) if u_ndcgs[0] else 0},
            'female': {'precision': np.mean(u_precs[1]) if u_precs[1] else 0,
                       'recall': np.mean(u_recs[1]) if u_recs[1] else 0,
                       'ndcg': np.mean(u_ndcgs[1]) if u_ndcgs[1] else 0},
        },
        'item_metrics': {
            'male': {'exposure': exposure[0] / total_top_k_items if total_top_k_items else 0,
                     'count': exposure[0]},
            'female': {'exposure': exposure[1] / total_top_k_items if total_top_k_items else 0,
                       'count': exposure[1]},
        }
    }
    return metrics

# ==========================================
# 13. INFERENCE ATTACK
# ==========================================
def _compute_auc(X, y, title):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced'))
    clf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    print(f"  {title}: AUC = {auc:.4f}")
    return auc

def inference_attack(model, u_attrs, i_attrs, is_debiased=False):
    print("\n--- Inference Attack Analysis ---")
    u_idx = torch.arange(len(u_attrs)).to(CONFIG['device'])
    i_idx = torch.arange(len(i_attrs)).to(CONFIG['device'])

    if is_debiased:
        u_emb, i_emb = model.get_task_embeddings(u_idx, i_idx)
    else:
        u_emb, i_emb = model.get_embeddings(u_idx, i_idx)

    auc_user = _compute_auc(u_emb, u_attrs, "User Gender Attack")
    auc_item = _compute_auc(i_emb, i_attrs, "Artist Gender Attack")

    # Sanity check on full latent (debiased only) - should be HIGH; confirms model
    # learned gender and adversary is the only thing pushing it out of task_dim
    if is_debiased:
        u_full, i_full = model.get_full_embeddings(u_idx, i_idx)
        print("  [sanity] AUC on full latent (task+sens) - expected HIGH:")
        _compute_auc(u_full, u_attrs, "  Full latent User")
        _compute_auc(i_full, i_attrs, "  Full latent Artist")

    return auc_user, auc_item

# ==========================================
# 14. VISUALIZATION
# ==========================================
def visualize_subspaces(baseline_model, debiased_model, u_attrs, i_attrs, num_samples=2000):
    print("\n--- Generating Subspace Visualizations ---")
    rng = np.random.default_rng(seed=42)
    u_sample = rng.choice(len(u_attrs), min(num_samples, len(u_attrs)), replace=False)
    i_sample = rng.choice(len(i_attrs), min(num_samples, len(i_attrs)), replace=False)

    u_idx = torch.LongTensor(u_sample).to(CONFIG['device'])
    i_idx = torch.LongTensor(i_sample).to(CONFIG['device'])

    u_base, i_base = baseline_model.get_embeddings(u_idx, i_idx)
    u_deb, i_deb = debiased_model.get_task_embeddings(u_idx, i_idx)

    print("Running t-SNE projections...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    u_base_2d = tsne.fit_transform(u_base)
    u_deb_2d = tsne.fit_transform(u_deb)
    i_base_2d = tsne.fit_transform(i_base)
    i_deb_2d = tsne.fit_transform(i_deb)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Embedding Space: Before vs After Debiasing', fontsize=15, fontweight='bold')
    for ax, data, label, cmap, title in [
        (axes[0, 0], u_base_2d, u_attrs[u_sample], 'coolwarm', 'User embeddings — Baseline (before)'),
        (axes[0, 1], u_deb_2d, u_attrs[u_sample], 'coolwarm', 'User task subspace — DS-DVAE (after)'),
        (axes[1, 0], i_base_2d, i_attrs[i_sample], 'viridis', 'Artist embeddings — Baseline (before)'),
        (axes[1, 1], i_deb_2d, i_attrs[i_sample], 'viridis', 'Artist task subspace — DS-DVAE (after)'),
    ]:
        sc = ax.scatter(data[:, 0], data[:, 1], c=label, cmap=cmap, alpha=0.6, s=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
        plt.colorbar(sc, ax=ax, label='Gender (0=M, 1=F)')
    plt.tight_layout()
    plt.savefig('subspace_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved: subspace_visualization.png")
    plt.show()

# ==========================================
# 15. FAIRNESS TABLES & PLOTS
# ==========================================
def create_detailed_fairness_analysis(baseline_metrics, debiased_metrics):
    print("\n" + "=" * 80)
    print("USER FAIRNESS BREAKDOWN (by Gender)")
    print("=" * 80)
    user_fairness_df = pd.DataFrame({
        'User Group': ['Male Users', 'Female Users', 'Disparity (Δ)'],
        'Precision@10 (Before)': [f"{baseline_metrics['user_metrics']['male']['precision']:.4f}",
                                  f"{baseline_metrics['user_metrics']['female']['precision']:.4f}",
                                  f"{baseline_metrics['UGF_Precision']:.4f}"],
        'Precision@10 (After)': [f"{debiased_metrics['user_metrics']['male']['precision']:.4f}",
                                 f"{debiased_metrics['user_metrics']['female']['precision']:.4f}",
                                 f"{debiased_metrics['UGF_Precision']:.4f}"],
        'Recall@10 (Before)': [f"{baseline_metrics['user_metrics']['male']['recall']:.4f}",
                               f"{baseline_metrics['user_metrics']['female']['recall']:.4f}",
                               f"{baseline_metrics['UGF_Recall']:.4f}"],
        'Recall@10 (After)': [f"{debiased_metrics['user_metrics']['male']['recall']:.4f}",
                              f"{debiased_metrics['user_metrics']['female']['recall']:.4f}",
                              f"{debiased_metrics['UGF_Recall']:.4f}"],
        'NDCG@10 (Before)': [f"{baseline_metrics['user_metrics']['male']['ndcg']:.4f}",
                             f"{baseline_metrics['user_metrics']['female']['ndcg']:.4f}",
                             f"{baseline_metrics['UGF_NDCG']:.4f}"],
        'NDCG@10 (After)': [f"{debiased_metrics['user_metrics']['male']['ndcg']:.4f}",
                            f"{debiased_metrics['user_metrics']['female']['ndcg']:.4f}",
                            f"{debiased_metrics['UGF_NDCG']:.4f}"],
    })
    print(user_fairness_df.to_string(index=False))
    user_fairness_df.to_csv('user_fairness_breakdown.csv', index=False)

    print("\n" + "=" * 80)
    print("ITEM FAIRNESS BREAKDOWN (Artist Gender)")
    print("=" * 80)
    item_fairness_df = pd.DataFrame({
        'Artist Group': ['Male Artists', 'Female Artists', 'Disparity (Δ)'],
        'Exposure % (Before)': [f"{baseline_metrics['item_metrics']['male']['exposure']*100:.2f}%",
                                f"{baseline_metrics['item_metrics']['female']['exposure']*100:.2f}%",
                                f"{baseline_metrics['RSP']*100:.2f}%"],
        'Exposure % (After)': [f"{debiased_metrics['item_metrics']['male']['exposure']*100:.2f}%",
                               f"{debiased_metrics['item_metrics']['female']['exposure']*100:.2f}%",
                               f"{debiased_metrics['RSP']*100:.2f}%"],
        'Count (Before)': [f"{baseline_metrics['item_metrics']['male']['count']:,}",
                           f"{baseline_metrics['item_metrics']['female']['count']:,}", "-"],
        'Count (After)': [f"{debiased_metrics['item_metrics']['male']['count']:,}",
                          f"{debiased_metrics['item_metrics']['female']['count']:,}", "-"],
    })
    print(item_fairness_df.to_string(index=False))
    item_fairness_df.to_csv('item_fairness_breakdown.csv', index=False)
    create_fairness_plots(baseline_metrics, debiased_metrics)

def create_fairness_plots(baseline_metrics, debiased_metrics):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Fairness Analysis: Before vs After Debiasing', fontsize=16, fontweight='bold', y=0.995)
    color_before = '#95a5a6'; color_after = '#27ae60'
    color_male = '#3498db'; color_female = '#e74c3c'
    x = np.arange(2); width = 0.35

    for ax, key, ylabel, title in [
        (axes[0, 0], 'precision', 'Precision@10', 'User Precision by Gender'),
        (axes[0, 1], 'recall', 'Recall@10', 'User Recall by Gender'),
        (axes[0, 2], 'ndcg', 'NDCG@10', 'User NDCG by Gender'),
    ]:
        before = [baseline_metrics['user_metrics']['male'][key], baseline_metrics['user_metrics']['female'][key]]
        after = [debiased_metrics['user_metrics']['male'][key], debiased_metrics['user_metrics']['female'][key]]
        ax.bar(x - width / 2, before, width, label='Before', color=color_before, alpha=0.8)
        ax.bar(x + width / 2, after, width, label='After', color=color_after, alpha=0.8)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(['Male Users', 'Female Users'])
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        ymax = max(before + after)
        ax.text(0.5, ymax * 0.95,
                f'Δ Before: {abs(before[0]-before[1]):.4f}\nΔ After:  {abs(after[0]-after[1]):.4f}',
                ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    before_exp = [baseline_metrics['item_metrics']['male']['exposure'] * 100,
                  baseline_metrics['item_metrics']['female']['exposure'] * 100]
    after_exp = [debiased_metrics['item_metrics']['male']['exposure'] * 100,
                 debiased_metrics['item_metrics']['female']['exposure'] * 100]
    axes[1, 0].bar(x - width / 2, before_exp, width, label='Before', color=color_before, alpha=0.8)
    axes[1, 0].bar(x + width / 2, after_exp, width, label='After', color=color_after, alpha=0.8)
    axes[1, 0].set_ylabel('Exposure %', fontweight='bold')
    axes[1, 0].set_title('Artist Exposure by Gender', fontweight='bold')
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(['Male Artists', 'Female Artists'])
    axes[1, 0].legend(); axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].text(0.5, max(before_exp + after_exp) * 0.95,
                    f'Δ Before: {abs(before_exp[0]-before_exp[1]):.2f}%\nΔ After:  {abs(after_exp[0]-after_exp[1]):.2f}%',
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[1, 1].axis('off')
    axes[1, 1].set_title('Exposure Distribution', fontweight='bold', pad=20)
    axes[1, 1].pie(before_exp, labels=['Male', 'Female'],
                   colors=[color_male, color_female], autopct='%1.1f%%',
                   startangle=90, center=(-0.5, 0), radius=0.4)
    axes[1, 1].text(-0.5, -0.6, 'Before', ha='center', fontweight='bold', fontsize=11)
    axes[1, 1].pie(after_exp, labels=['Male', 'Female'],
                   colors=[color_male, color_female], autopct='%1.1f%%',
                   startangle=90, center=(0.5, 0), radius=0.4)
    axes[1, 1].text(0.5, -0.6, 'After', ha='center', fontweight='bold', fontsize=11)

    axes[1, 2].axis('off')
    axes[1, 2].set_title('Fairness Metrics Summary', fontweight='bold')
    bm, dm = baseline_metrics, debiased_metrics
    txt = (
        f"USER GROUP FAIRNESS (UGF):\n{'-'*33}\n"
        f"UGF Precision : {bm['UGF_Precision']:.4f} → {dm['UGF_Precision']:.4f}\n"
        f"UGF Recall    : {bm['UGF_Recall']:.4f} → {dm['UGF_Recall']:.4f}\n"
        f"UGF NDCG      : {bm['UGF_NDCG']:.4f} → {dm['UGF_NDCG']:.4f}\n\n"
        f"ITEM FAIRNESS:\n{'-'*33}\n"
        f"RSP (Exposure): {bm['RSP']:.4f} → {dm['RSP']:.4f}\n"
        f"REO (Equality): {bm['REO']:.4f} → {dm['REO']:.4f}\n"
    )
    axes[1, 2].text(0.1, 0.5, txt, fontsize=10, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    plt.tight_layout()
    plt.savefig('fairness_analysis_plots.png', dpi=300, bbox_inches='tight')
    print("Saved: fairness_analysis_plots.png")
    plt.show()

# ==========================================
# 16. COMPARISON TABLE
# ==========================================
def print_comparison_table(baseline_metrics, debiased_metrics, baseline_auc, debiased_auc):
    print("\n" + "=" * 80)
    print("COMPARATIVE METRICS: BASELINE vs DS-DVAE")
    print("=" * 80)
    bm, dm = baseline_metrics, debiased_metrics
    comparison = pd.DataFrame({
        'Metric': ['MAE ↓', 'Precision@10 ↑', 'Recall@10 ↑', 'NDCG@10 ↑',
                   '─' * 13,
                   'UGF_Precision ↓', 'UGF_Recall ↓', 'UGF_NDCG ↓',
                   '─' * 13,
                   'RSP (Item) ↓', 'REO (Item) ↓',
                   '─' * 13,
                   'AUC_User ↓', 'AUC_Artist ↓'],
        'Baseline (Before)': [f"{bm['MAE']:.4f}", f"{bm['Precision@10']:.4f}",
                              f"{bm['Recall@10']:.4f}", f"{bm['NDCG@10']:.4f}", '─' * 13,
                              f"{bm['UGF_Precision']:.4f}", f"{bm['UGF_Recall']:.4f}",
                              f"{bm['UGF_NDCG']:.4f}", '─' * 13,
                              f"{bm['RSP']:.4f}", f"{bm['REO']:.4f}", '─' * 13,
                              f"{baseline_auc[0]:.4f}", f"{baseline_auc[1]:.4f}"],
        'DS-DVAE (After)': [f"{dm['MAE']:.4f}", f"{dm['Precision@10']:.4f}",
                            f"{dm['Recall@10']:.4f}", f"{dm['NDCG@10']:.4f}", '─' * 13,
                            f"{dm['UGF_Precision']:.4f}", f"{dm['UGF_Recall']:.4f}",
                            f"{dm['UGF_NDCG']:.4f}", '─' * 13,
                            f"{dm['RSP']:.4f}", f"{dm['REO']:.4f}", '─' * 13,
                            f"{debiased_auc[0]:.4f}", f"{debiased_auc[1]:.4f}"]
    })
    print(comparison.to_string(index=False))
    comparison.to_csv('comparison_metrics.csv', index=False)
    print("\nSaved: comparison_metrics.csv")

# ==========================================
# 17. MAIN
# ==========================================
if __name__ == "__main__":
    df, n_users, n_items, u_attrs, i_attrs = load_and_process_data()
    train_df, val_df, test_df = split_data_per_user(df)

    train_ds = InteractionDataset(train_df, u_attrs, i_attrs)
    val_ds = InteractionDataset(val_df, u_attrs, i_attrs)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=2, pin_memory=True)

    print("\n" + "=" * 60)
    print("TRAINING BASELINE VAE")
    print("=" * 60)
    baseline_model = BaselineVAE(n_users, n_items, CONFIG).to(CONFIG['device'])
    train_baseline(baseline_model, train_loader, val_loader, CONFIG)
    baseline_model.load_state_dict(torch.load('baseline_best.pt'))
    baseline_metrics = evaluate_model(baseline_model, test_df, df, n_items, u_attrs, i_attrs, CONFIG)
    baseline_auc = inference_attack(baseline_model, u_attrs, i_attrs, is_debiased=False)

    print("\n" + "=" * 60)
    print("TRAINING DS-DVAE")
    print("=" * 60)
    debiased_model = DS_DVAE(n_users, n_items, CONFIG).to(CONFIG['device'])
    train_debiased(debiased_model, train_loader, val_loader, CONFIG, u_attrs, i_attrs)
    debiased_model.load_state_dict(torch.load('dvae_best.pt'))
    debiased_metrics = evaluate_model(debiased_model, test_df, df, n_items, u_attrs, i_attrs, CONFIG)
    debiased_auc = inference_attack(debiased_model, u_attrs, i_attrs, is_debiased=True)

    visualize_subspaces(baseline_model, debiased_model, u_attrs, i_attrs)
    create_detailed_fairness_analysis(baseline_metrics, debiased_metrics)
    print_comparison_table(baseline_metrics, debiased_metrics, baseline_auc, debiased_auc)

    print("\nExperiment Complete!")
    print("Outputs: comparison_metrics.csv | subspace_visualization.png | fairness_analysis_plots.png")
