"""Evaluation: rating accuracy + ranking metrics + fairness metrics +
inference attack on multiple representations with multiple attackers."""
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

from .attackers import attack_all, worst_case_auc


def _dcg(rel, k):
    rel = np.asarray(rel, dtype=np.float64)[:k]
    return np.sum(rel / np.log2(np.arange(2, rel.size + 2))) if rel.size else 0.0


def _ndcg(rel, k):
    ideal = _dcg(sorted(rel, reverse=True), k)
    return 0.0 if not ideal else _dcg(rel, k) / ideal


def evaluate_rec(model, test_df, full_df, n_items, u_attrs, i_attrs, cfg,
                 num_neg=100, model_type='fairdis'):
    """Rating + ranking + group fairness metrics."""
    device = cfg['device']
    model.eval()

    u_t = torch.LongTensor(test_df['u_idx'].values).to(device)
    i_t = torch.LongTensor(test_df['i_idx'].values).to(device)
    r_t = torch.FloatTensor(test_df['rating'].values).to(device)

    with torch.no_grad():
        preds = model.predict(u_t, i_t)
        mae = torch.mean(torch.abs(r_t - preds)).item()

    user_pos = defaultdict(list)
    user_all = defaultdict(set)
    for _, row in test_df.iterrows():
        user_pos[row['u_idx']].append(row['i_idx'])
    for _, row in full_df.iterrows():
        user_all[row['u_idx']].add(row['i_idx'])

    P, R, N = [], [], []
    P_g = {0: [], 1: []}; R_g = {0: [], 1: []}; N_g = {0: [], 1: []}
    expo = {0: 0, 1: 0}; rel_rec = {0: 0, 1: 0}; rel_tot = {0: 0, 1: 0}
    n_in_topk = 0

    rng = np.random.default_rng(42)

    with torch.no_grad():
        for u_id, pos in tqdm(user_pos.items(), desc="Eval", leave=False):
            neg_pool = list(set(range(n_items)) - user_all[u_id])
            if len(neg_pool) < num_neg:
                continue
            negs = rng.choice(neg_pool, num_neg, replace=False)
            cand = np.concatenate([pos, negs])
            ct = torch.LongTensor(cand).to(device)
            ut = torch.LongTensor([u_id] * len(cand)).to(device)
            scores = model.predict(ut, ct).cpu().numpy()
            k = min(cfg['top_k'], len(cand))
            top_idx = np.argsort(scores)[::-1][:k]
            top = cand[top_idx]
            rel = [1 if x in pos else 0 for x in top]
            hits = len(set(top) & set(pos))
            p = hits / k
            r = hits / max(1, len(pos))
            n = _ndcg(rel, k)
            P.append(p); R.append(r); N.append(n)
            g = int(u_attrs[u_id])
            P_g[g].append(p); R_g[g].append(r); N_g[g].append(n)
            for it in top:
                gi = int(i_attrs[it])
                expo[gi] += 1; n_in_topk += 1
            for it in pos:
                gi = int(i_attrs[it])
                rel_tot[gi] += 1
                if it in top:
                    rel_rec[gi] += 1

    def safemean(x): return float(np.mean(x)) if x else 0.0

    rsp = (abs(expo[0] / n_in_topk - expo[1] / n_in_topk)
           if n_in_topk else 0.0)
    reo = abs(rel_rec[0] / (rel_tot[0] + 1e-9) - rel_rec[1] / (rel_tot[1] + 1e-9))

    return {
        'MAE': mae,
        'Precision@K': safemean(P), 'Recall@K': safemean(R), 'NDCG@K': safemean(N),
        'UGF_Precision': abs(safemean(P_g[0]) - safemean(P_g[1])),
        'UGF_Recall':    abs(safemean(R_g[0]) - safemean(R_g[1])),
        'UGF_NDCG':      abs(safemean(N_g[0]) - safemean(N_g[1])),
        'RSP': rsp, 'REO': reo,
        'user_m': {'precision': safemean(P_g[0]), 'recall': safemean(R_g[0]),
                   'ndcg': safemean(N_g[0])},
        'user_f': {'precision': safemean(P_g[1]), 'recall': safemean(R_g[1]),
                   'ndcg': safemean(N_g[1])},
        'item_m_expo': expo[0] / n_in_topk if n_in_topk else 0.0,
        'item_f_expo': expo[1] / n_in_topk if n_in_topk else 0.0,
    }


def run_attacks(model, n_users, n_items, u_attrs, i_attrs, cfg,
                model_type='fairdis'):
    """Run all attackers on task embeddings AND full embeddings for sanity.
    Returns nested dict {representation: {attacker: {auc, acc}}}."""
    u_idx = torch.arange(n_users).to(cfg['device'])
    i_idx = torch.arange(n_items).to(cfg['device'])

    results = {}
    if model_type == 'baseline':
        Eu = model.get_user_embeddings(u_idx)
        Ei = model.get_item_embeddings(i_idx)
        results['user_emb'] = attack_all(Eu, u_attrs)
        results['item_emb'] = attack_all(Ei, i_attrs)
    else:
        Eu_t = model.get_user_task(u_idx); Ei_t = model.get_item_task(i_idx)
        Eu_f = model.get_user_full(u_idx); Ei_f = model.get_item_full(i_idx)
        results['user_task'] = attack_all(Eu_t, u_attrs)
        results['item_task'] = attack_all(Ei_t, i_attrs)
        results['user_full'] = attack_all(Eu_f, u_attrs)
        results['item_full'] = attack_all(Ei_f, i_attrs)
    return results


def summarize_attacks(attack_results):
    """Worst-case AUC per representation (defensible privacy summary)."""
    return {rep: worst_case_auc(res) for rep, res in attack_results.items()}
