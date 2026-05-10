"""Data loading, k-core filtering, propensity scores, train/val/test split."""
import json
import os
import tarfile
import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def download_and_extract(cfg):
    tar_filename = "lastfm-dataset-360K.tar.gz"
    if os.path.exists(cfg['extract_folder']):
        print(f"[Info] Dataset directory found: {cfg['extract_folder']}")
        return
    if not os.path.exists(tar_filename):
        print(f"[Info] Downloading dataset from {cfg['dataset_url']}...")
        r = requests.get(cfg['dataset_url'], stream=True)
        total = int(r.headers.get('content-length', 0))
        with open(tar_filename, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
    print("[Info] Extracting tar.gz...")
    with tarfile.open(tar_filename, "r:gz") as tar:
        tar.extractall()
    print("[Info] Extraction complete.")


def load_and_process_data(cfg):
    download_and_extract(cfg)
    path_plays = os.path.join(cfg['extract_folder'], cfg['file_plays'])
    path_profiles = os.path.join(cfg['extract_folder'], cfg['file_profiles'])

    print("Loading user profiles...")
    profiles = pd.read_csv(path_profiles, sep='\t', header=None,
                           names=['user_sha1', 'gender', 'age', 'country', 'signup'],
                           usecols=['user_sha1', 'gender'])
    profiles = profiles[profiles['gender'].isin(['m', 'f'])]
    profiles['gender_bin'] = profiles['gender'].map({'m': 0, 'f': 1}).astype(int)
    user_gender = dict(zip(profiles['user_sha1'], profiles['gender_bin']))

    print("Loading artist genders...")
    if not os.path.exists(cfg['json_path']):
        raise FileNotFoundError(f"Missing {cfg['json_path']}")
    with open(cfg['json_path'], 'r') as f:
        artist_json = json.load(f)
    artist_gender = {}
    for mbid, g in artist_json.items():
        if g.lower() == 'male':
            artist_gender[mbid] = 0
        elif g.lower() == 'female':
            artist_gender[mbid] = 1

    print("Loading plays...")
    df = pd.read_csv(path_plays, sep='\t', header=None,
                     names=['user_sha1', 'art_mbid', 'art_name', 'plays'],
                     usecols=['user_sha1', 'art_mbid', 'plays'],
                     on_bad_lines='skip')
    df = df.dropna(subset=['user_sha1', 'art_mbid'])
    df = df[df['user_sha1'].isin(user_gender.keys())]
    df = df[df['art_mbid'].isin(artist_gender.keys())]

    print(f"Applying {cfg['min_interactions']}-core filter...")
    min_k = cfg['min_interactions']
    while True:
        n0 = len(df)
        u_counts = df['user_sha1'].value_counts()
        df = df[df['user_sha1'].isin(u_counts[u_counts > min_k].index)]
        i_counts = df['art_mbid'].value_counts()
        df = df[df['art_mbid'].isin(i_counts[i_counts > min_k].index)]
        if len(df) == n0:
            break

    if cfg.get('smoke_test'):
        users = df['user_sha1'].drop_duplicates().sample(
            min(cfg['smoke_users'], df['user_sha1'].nunique()), random_state=0)
        df = df[df['user_sha1'].isin(users)]
        items = df['art_mbid'].drop_duplicates().sample(
            min(cfg['smoke_items'], df['art_mbid'].nunique()), random_state=0)
        df = df[df['art_mbid'].isin(items)]
        # re-apply k-core lightly
        for _ in range(3):
            uc = df['user_sha1'].value_counts(); df = df[df['user_sha1'].isin(uc[uc > 5].index)]
            ic = df['art_mbid'].value_counts(); df = df[df['art_mbid'].isin(ic[ic > 5].index)]

    user_list = df['user_sha1'].unique()
    item_list = df['art_mbid'].unique()
    user2idx = {u: i for i, u in enumerate(user_list)}
    item2idx = {i: j for j, i in enumerate(item_list)}
    df['u_idx'] = df['user_sha1'].map(user2idx)
    df['i_idx'] = df['art_mbid'].map(item2idx)

    u_attrs = np.zeros(len(user_list), dtype=np.float32)
    for u, idx in user2idx.items():
        u_attrs[idx] = user_gender[u]
    i_attrs = np.zeros(len(item_list), dtype=np.float32)
    for i, idx in item2idx.items():
        i_attrs[idx] = artist_gender[i]

    df['plays'] = np.log1p(df['plays'])
    df['rating'] = df['plays'] / df['plays'].max()

    # Item propensity from popularity (used for IPW)
    item_counts = df['i_idx'].value_counts().sort_index().values.astype(np.float32)
    n_items = len(item_list)
    item_pop = np.zeros(n_items, dtype=np.float32)
    item_pop[df['i_idx'].value_counts().sort_index().index] = item_counts
    item_pop = item_pop / item_pop.sum()
    propensity = np.power(np.clip(item_pop, 1e-9, None), cfg['propensity_alpha'])
    item_weight = 1.0 / (propensity + 1e-9)
    item_weight = item_weight / item_weight.mean()  # normalize

    print(f"Final: {len(df):,} interactions, {len(user_list):,} users, {len(item_list):,} items")
    return df, len(user_list), len(item_list), u_attrs, i_attrs, item_weight


def split_data_per_user(df, seed=42):
    train_l, val_l, test_l = [], [], []
    for _, group in df.groupby('u_idx'):
        d = group.sample(frac=1, random_state=seed)
        n = len(d)
        a = int(n * 0.7); b = int(n * 0.8)
        train_l.append(d.iloc[:a]); val_l.append(d.iloc[a:b]); test_l.append(d.iloc[b:])
    return (pd.concat(train_l).reset_index(drop=True),
            pd.concat(val_l).reset_index(drop=True),
            pd.concat(test_l).reset_index(drop=True))


class InteractionDataset(Dataset):
    def __init__(self, df, u_attrs, i_attrs, item_weight):
        self.users = torch.LongTensor(df['u_idx'].values)
        self.items = torch.LongTensor(df['i_idx'].values)
        self.ratings = torch.FloatTensor(df['rating'].values)
        self.u_attrs = torch.FloatTensor(u_attrs)
        self.i_attrs = torch.FloatTensor(i_attrs)
        self.item_weight = torch.FloatTensor(item_weight)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]; i = self.items[idx]
        return (u, i, self.ratings[idx], self.u_attrs[u], self.i_attrs[i],
                self.item_weight[i])
