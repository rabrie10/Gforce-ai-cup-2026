import os, json, cv2, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from v5.pipeline import Config
from v5.visual_expert import VisualExpert
from v5.gpu.discovery_v54 import MergedDiscovery

d = np.load("/workspace/assets/bg_head_data.npz")
Xtr, ytr, Xho, yho = d["Xtr"], d["ytr"], d["Xho"], d["yho"]
cfg = Config()
expert = VisualExpert(cfg.assets, device=cfg.device)

# oracle hosted target crops (8 manual boxes) - NEVER trained on
manual = json.load(open("/workspace/results/manual_targets.json"))["targets"]
files = {60: "00060_80324c03bcc65ff1d04c510e_180324c03bcc65ff1d04c510e_60_0_1920_1080.png",
         125: "00125_80324c03bcc65ff1d04c510e_80324c03bcc65ff1d04c510e_125_0_1920_1080.png"}
oracle_feats = []
for t in manual:
    im = cv2.imread(f"/workspace/hosted-v3/images/{files[t['capture']]}")
    _, f, _ = expert.classify(im, [t["box"]])
    oracle_feats.append(f[0])
oracle_feats = np.array(oracle_feats)


def sig(x): return 1/(1+np.exp(-x))


def eval_head(scorer, name):
    pho = scorer(Xho)
    auc = roc_auc_score(yho, pho)
    # operating threshold: keep >=95% Helsinki-test target retention
    pos = pho[yho == 1]; neg = pho[yho == 0]
    thr = np.percentile(pos, 5)  # retains 95% of holdout positives
    ret = (pos >= thr).mean(); rej = (neg < thr).mean()
    orc = scorer(oracle_feats)
    orc_acc = (orc >= thr).mean()
    print(f"[{name}] holdoutAUC={auc:.3f} thr={thr:.3f} | holdout target-retain={ret:.2f} bg-reject={rej:.2f} | ORACLE hosted-target accept={orc_acc:.2f} ({int((orc>=thr).sum())}/8)")
    print(f"        oracle per-target P(target): " + " ".join(f"{t['id'].split('_')[-1][:4]}={p:.2f}" for t, p in zip(manual, orc)))
    return auc, thr, orc_acc


results = {}
# Logistic regression
lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
results["logreg"] = eval_head(lambda X: lr.predict_proba(X)[:, 1], "logreg C=1")
# MLP
mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=400, early_stopping=True,
                    random_state=0).fit(Xtr, ytr)
results["mlp"] = eval_head(lambda X: mlp.predict_proba(X)[:, 1], "mlp 256")

# detector-crop diagnostic on 60/125: how many selected crops rejected, and boats on 125
disc = MergedDiscovery(cfg.assets, budget=cfg.candidate_budget, device=cfg.device)
best = mlp if results["mlp"][0] >= results["logreg"][0] else lr
best_name = "mlp" if best is mlp else "logreg"
thr = results[best_name][1]
print(f"\n=== detector-crop diagnostic with {best_name} thr={thr:.3f} ===")
for cap, fn in files.items():
    im = cv2.imread(f"/workspace/hosted-v3/images/{fn}")
    raw, sel, _ = disc.propose(im, [0, 0, 3840, 2160])
    if not sel:
        print(f"  frame {cap}: no selected"); continue
    _, feats, _ = expert.classify(im, [c["local_box"] for c in sel])
    p = best.predict_proba(feats)[:, 1]
    print(f"  frame {cap}: selected={len(sel)} accepted-as-target={(p>=thr).sum()} rejected-as-bg={(p<thr).sum()}")

# export winner to npz for numpy runtime inference
if best_name == "logreg":
    np.savez("/workspace/assets/bg_head.npz", kind="logreg", w=lr.coef_[0].astype(np.float32),
             b=np.float32(lr.intercept_[0]), thr=np.float32(results["logreg"][1]))
else:
    np.savez("/workspace/assets/bg_head.npz", kind="mlp",
             W1=mlp.coefs_[0].astype(np.float32), b1=mlp.intercepts_[0].astype(np.float32),
             W2=mlp.coefs_[1].astype(np.float32), b2=mlp.intercepts_[1].astype(np.float32),
             thr=np.float32(thr))
json.dump({"winner": best_name, "holdout_auc": float(results[best_name][0]),
           "threshold": float(thr), "oracle_hosted_accept_frac": float(results[best_name][2])},
          open("/workspace/assets/bg_head_metrics.json", "w"), indent=2)
print(f"\nWROTE /workspace/assets/bg_head.npz (winner={best_name})")
