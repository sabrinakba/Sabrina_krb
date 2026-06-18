"""Multiple attacker models for the inference attack. Reports AUC + accuracy
on a held-out split for: Logistic Regression, MLP, Gradient Boosting, kNN."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ATTACKERS = {
    'LR':  lambda: make_pipeline(StandardScaler(),
              LogisticRegression(max_iter=1000, solver='liblinear',
                                 class_weight='balanced')),
    'MLP': lambda: make_pipeline(StandardScaler(),
              MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                            early_stopping=True, random_state=0)),
    'GBM': lambda: GradientBoostingClassifier(n_estimators=100,
                                              max_depth=3, random_state=0),
    'kNN': lambda: make_pipeline(StandardScaler(),
              KNeighborsClassifier(n_neighbors=15)),
}


def attack_all(X, y, seed=0):
    """Run all attackers on (X, y). Returns dict of {name: {auc, acc}}."""
    if len(np.unique(y)) < 2:
        return {n: {'auc': float('nan'), 'acc': float('nan')} for n in ATTACKERS}
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    results = {}
    for name, maker in ATTACKERS.items():
        try:
            clf = maker()
            clf.fit(X_tr, y_tr)
            if hasattr(clf, 'predict_proba'):
                p = clf.predict_proba(X_te)[:, 1]
                auc = roc_auc_score(y_te, p)
            else:
                auc = roc_auc_score(y_te, clf.decision_function(X_te))
            acc = accuracy_score(y_te, clf.predict(X_te))
        except Exception as e:
            auc = float('nan'); acc = float('nan')
            print(f"  [warn] attacker {name} failed: {e}")
        results[name] = {'auc': auc, 'acc': acc}
    return results


def worst_case_auc(results):
    """Return the largest attacker AUC -- a defensible privacy summary."""
    vals = [r['auc'] for r in results.values() if not np.isnan(r['auc'])]
    return max(vals) if vals else float('nan')
