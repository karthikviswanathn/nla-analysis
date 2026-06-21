"""Linear probe: is the true entity (Hannibal) linearly decodable from the layer-20 activation the
AV confabulated on (it said 'Columbus')?

If yes -> the identity IS present in the activation; the AV's confabulation is a decoding/expressivity
failure, not missing information. If no -> the binding is genuinely lost by layer 20.

Probes (L2 logistic regression on standardized features, stratified 5-fold CV):
  MAIN            Hannibal-named vs Columbus-named
  SHUFFLED ctrl   same, labels permuted        (overfit floor -> must be ~chance)
  POSITIVE ctrl   history vs biology           (easy -> pipeline sanity)
  NAME-MASKED     Hannibal-desc vs Columbus-desc (no name in text -> tests bound concept, not name echo)
DECISIVE: train on named H/C, apply to the exemplar confab vector -> does it say HANNIBAL?
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = "/gpfs/work5/0/gusr0688/fair_stuff/nla-analysis"
A = np.load(f"{BASE}/linear-probe/acts.npz")


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))


def run(Xpos, Xneg, name, shuffle=False):
    X = np.vstack([Xpos, Xneg])
    y = np.r_[np.ones(len(Xpos)), np.zeros(len(Xneg))]
    if shuffle:
        y = np.random.default_rng(0).permutation(y)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = cross_val_score(clf(), X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(clf(), X, y, cv=cv, scoring="roc_auc")
    print(f"  {name:38s} acc={acc.mean():.3f}±{acc.std():.3f}   auc={auc.mean():.3f}   "
          f"(n={len(Xpos)}/{len(Xneg)})")


H, C = A["hannibal_named"], A["columbus_named"]
print("=" * 78)
print("LINEAR PROBE — is entity identity decodable from Qwen layer-20 activations?")
print("=" * 78)
print("\nMAIN:")
run(H, C, "Hannibal vs Columbus (named)")
run(H, C, "  shuffled-label control", shuffle=True)
print("\nCONTROLS:")
run(np.vstack([H, C]), A["biology"], "positive: history vs biology")
run(A["hannibal_desc"], A["columbus_desc"], "name-masked: H-desc vs C-desc")

print("\n" + "=" * 78)
print("DECISIVE — apply the probe to the activation the AV confabulated on")
print("=" * 78)
model = clf().fit(np.vstack([H, C]), np.r_[np.ones(len(H)), np.zeros(len(C))])
ex = A["exemplar"]
pH = model.predict_proba(ex)[0, 1]
print(f"  source sentence: 'Hannibal led his army ... across the Alps in 218 BC ...'")
print(f"  AV's NLA explanation said:  COLUMBUS")
print(f"  P(Hannibal | the exact activation the AV read) = {pH:.3f}  ->  probe says "
      f"{'HANNIBAL ✓ (identity was present; AV failed to decode it)' if pH > 0.5 else 'COLUMBUS'}")
# raw geometric check: project onto standardized diff-of-means direction
sc = StandardScaler().fit(np.vstack([H, C]))
Hs, Cs, exs = sc.transform(H), sc.transform(C), sc.transform(ex)
w = Hs.mean(0) - Cs.mean(0)
w /= np.linalg.norm(w)
proj = float((exs[0] - (Hs.mean(0) + Cs.mean(0)) / 2) @ w)
print(f"  diff-of-means projection (standardized; >0 => Hannibal side): {proj:+.3f}")
